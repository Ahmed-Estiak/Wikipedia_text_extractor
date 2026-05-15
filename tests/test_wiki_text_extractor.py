from pathlib import Path
import unittest

from wiki_text_extractor import (
    PageRequest,
    add_topic_heading,
    clean_plain_text,
    clean_wikipedia_html_with_references,
    clean_wikipedia_html,
    compare_texts,
    comparison_output_path,
    extraction_output_path,
    output_path_for_method,
    page_request_from_url,
    partial_match_report_path,
    partial_output_path,
    extract_partial_text,
    PartialExtractionError,
    required_partial_anchor_score,
    runtime_label,
    strong_anchor_candidates,
    runtime_output_path,
    update_runtime_report,
    references_output_path,
    extract_note_section,
    note_section_has_body,
    remove_empty_note_section,
    raw_output_path,
    lower_alpha_label,
)


class CleanWikipediaHtmlTests(unittest.TestCase):
    def test_removes_references_infoboxes_and_tables(self):
        # Verifies noisy Wikipedia blocks and citation markers are removed from HTML output.
        html = """
        <div class="mw-parser-output">
          <table class="infobox"><tr><td>Hidden infobox</td></tr></table>
          <p>Python is a programming language.<sup class="reference">[1]</sup></p>
          <p>It emphasizes readability.</p>
          <div class="reflist">Hidden references</div>
        </div>
        """

        text = clean_wikipedia_html(html)

        self.assertEqual(
            text,
            "Python is a programming language.\n\nIt emphasizes readability.",
        )

    def test_preserves_non_reference_superscripts(self):
        # Verifies real superscripts are kept as caret notation instead of being treated as citations.
        html = """
        <div class="mw-parser-output">
          <p>There are 8 (=2<sup>3</sup>) examples at 90&deg;.</p>
        </div>
        """

        self.assertEqual(
            clean_wikipedia_html(html),
            "There are 8 (=2^3) examples at 90°.",
        )

    def test_preserves_space_after_removed_reference_superscripts(self):
        # Verifies removing citation superscripts does not join neighboring words or punctuation.
        html = """
        <div class="mw-parser-output">
          <p>That suggests not many are.<sup class="reference">[1]</sup> Orcus follows.</p>
          <p>Accepted: Pluto,<sup class="reference">[2]</sup> Haumea, Máni,<sup class="reference">[3]</sup> Aya.</p>
          <p>This spacing <sup class="reference">[4]</sup> already existed.</p>
        </div>
        """

        self.assertEqual(
            clean_wikipedia_html(html),
            (
                "That suggests not many are. Orcus follows.\n\n"
                "Accepted: Pluto, Haumea, Máni, Aya.\n\n"
                "This spacing already existed."
            ),
        )

    def test_preserves_space_between_adjacent_inline_link_text(self):
        # Verifies adjacent inline/link chunks do not merge separate words.
        html = """
        <div class="mw-parser-output">
          <p><a href="/wiki/Log-log">log-log</a> <a href="/wiki/Learning_rate">learning rate schedule</a> is common.</p>
          <p>Token <a href="/wiki/Embedding">embedding</a><span>space</span> stays readable.</p>
          <p><span>H</span><span>ello</span> and <span>e</span><span>mail</span> stay joined.</p>
        </div>
        """

        cleaned = clean_wikipedia_html(html)

        self.assertIn("log-log learning rate schedule is common.", cleaned)
        self.assertIn("Token embedding space stays readable.", cleaned)
        self.assertIn("Hello and email stay joined.", cleaned)
        self.assertNotIn("log-loglearning", cleaned)
        self.assertNotIn("embeddingspace", cleaned)
        self.assertNotIn("H ello", cleaned)
        self.assertNotIn("e mail", cleaned)

    def test_preserves_link_space_after_math_html(self):
        # Verifies a previous math container does not make later link whitespace disappear.
        html = """
        <div class="mw-parser-output">
          <span class="mwe-math-element"><span><math>x</math></span></span>
          <img class="mwe-math-fallback-image-inline" alt="{\\displaystyle y}" />
          <p>with a <a href="/wiki/Log-log_plot">log-log</a> <a href="/wiki/Learning_rate">learning rate</a> schedule.</p>
        </div>
        """

        cleaned = clean_wikipedia_html(html, math_mode="latex")

        self.assertIn("log-log learning rate schedule.", cleaned)
        self.assertNotIn("log-loglearning", cleaned)

    def test_does_not_split_normal_hyphenated_terms(self):
        # Verifies normal hyphen chains are not changed by cleanup.
        cleaned = clean_plain_text(
            "A state-of-the-art model used a fine-tuning schedule and log-loglearning remains literal."
        )

        self.assertEqual(
            cleaned,
            "A state-of-the-art model used a fine-tuning schedule and log-loglearning remains literal.",
        )

    def test_preserves_non_math_subscripts_with_underscores(self):
        # Verifies normal HTML subscripts become underscore notation and split following letters.
        html = """
        <div class="mw-parser-output">
          <p>Water is H<sub>2</sub>O and hydrogen sulfide is H<sub>2</sub>S.</p>
          <p>The isotope is C<sub>14</sub> gas.</p>
        </div>
        """

        self.assertEqual(
            clean_wikipedia_html(html),
            "Water is H_2_O and hydrogen sulfide is H_2_S.\n\nThe isotope is C_14 gas.",
        )

    def test_does_not_add_subscript_underscores_inside_math_html(self):
        # Verifies math containers keep their own formatting and do not get extra subscript markers.
        html = """
        <div class="mw-parser-output">
          <span class="mwe-math-element">x<sub>i</sub></span>
          <p>Water is H<sub>2</sub>O.</p>
        </div>
        """

        cleaned = clean_wikipedia_html(html, math_mode="keep")

        self.assertIn("xi", cleaned)
        self.assertNotIn("x_i", cleaned)
        self.assertIn("H_2_O", cleaned)

    def test_removes_image_captions_and_map_labels(self):
        # Verifies figures, thumbnails, map labels, and noexcerpt blocks stay out of clean text.
        html = """
        <div class="mw-parser-output">
          <figure>
            <figcaption>Hidden figure caption</figcaption>
          </figure>
          <div class="thumb">
            <div class="thumbcaption">Hidden thumb caption</div>
          </div>
          <div class="noexcerpt">Sun Jupiter trojans Giant planets</div>
          <p>The Kuiper belt is in the outer Solar System.</p>
        </div>
        """

        self.assertEqual(
            clean_wikipedia_html(html),
            "The Kuiper belt is in the outer Solar System.",
        )

    def test_preserves_inline_file_symbols_from_titles(self):
        # Verifies inline glyph images keep symbol titles while descriptive image titles are ignored.
        html = """
        <div class="mw-parser-output">
          <p>Ceres <span class="skin-invert" typeof="mw:File">
            <a href="/wiki/File:Ceres_symbol.svg" class="mw-file-description" title="⚳">
              <img alt="⚳" />
            </a>
          </span> and Pluto <span class="skin-invert" typeof="mw:File">
            <a href="/wiki/File:Pluto_symbol.svg" class="mw-file-description" title="♇">
              <img alt="♇" />
            </a>
          </span> received symbols.</p>
          <p>Photo <a class="mw-file-description" title="Photo of Ceres"><img alt="Ceres" /></a> stays quiet.</p>
          <figure>
            <a class="mw-file-description" title="Hidden figure symbol"><img alt="Hidden" /></a>
            <figcaption>Hidden caption</figcaption>
          </figure>
        </div>
        """

        text = clean_wikipedia_html(html)

        self.assertIn("received symbols.", text)
        self.assertIn("Photo stays quiet.", text)
        self.assertNotIn("Photo of Ceres", text)
        self.assertNotIn("Hidden figure symbol", text)

    def test_extracts_language_and_title_from_url(self):
        # Verifies Wikipedia URL parsing returns the language code and page title.
        page = page_request_from_url("https://en.wikipedia.org/wiki/Albert_Einstein")

        self.assertEqual(page.lang, "en")
        self.assertEqual(page.title, "Albert Einstein")

    def test_cleans_extracts_api_plain_text_noise(self):
        # Verifies plain extracts text cleanup removes refs, empty parentheses, and unwanted sections.
        text = """
        Saturn ( ) is a planet.[1]

        == History ==

        It has 8 (=23) rings.[2, 3]

        === Moons ===

        Titan is large.

        == References ==

        Reference text that should not be included.
        """

        self.assertEqual(
            clean_plain_text(text),
            (
                "Saturn is a planet.\n\n== History ==\n\n"
                "It has 8 (=2^3) rings.\n\n== Moons ==\n\nTitan is large."
            ),
        )

    def test_removes_selected_sections_but_keeps_notes(self):
        # Verifies See also/References/External links are removed while Notes content remains.
        text = """
        Article body.

        == See also ==

        Hidden related page.

        == Notes ==

        ^ a b This note should remain.
        ^c Another note should remain.
        ^a^bNo-space note should remain.

        == References ==

        Hidden reference.

        == External links ==

        Hidden link.
        """

        cleaned = clean_plain_text(text)

        self.assertIn("Article body.", cleaned)
        self.assertIn("== Notes ==", cleaned)
        self.assertIn("This note should remain.", cleaned)
        self.assertIn("Another note should remain.", cleaned)
        self.assertIn("No-space note should remain.", cleaned)
        self.assertNotIn("^ a b", cleaned)
        self.assertNotIn("^c", cleaned)
        self.assertNotIn("^a^b", cleaned)
        self.assertNotIn("Hidden related page.", cleaned)
        self.assertNotIn("Hidden reference.", cleaned)
        self.assertNotIn("Hidden link.", cleaned)

    def test_removes_plain_html_sections_but_keeps_plain_notes(self):
        # Verifies plain heading text from HTML is handled when section markers are not present.
        text = """
        Article body.

        See also

        Hidden related page.

        Notes

        ^a^bVisible note.

        References

        Hidden reference.
        """

        cleaned = clean_plain_text(text)

        self.assertIn("Article body.", cleaned)
        self.assertIn("Notes", cleaned)
        self.assertIn("Visible note.", cleaned)
        self.assertNotIn("Hidden related page.", cleaned)
        self.assertNotIn("Hidden reference.", cleaned)

    def test_note_section_helpers_detect_missing_body_and_keep_list_labels(self):
        # Verifies Note helper functions remove empty notes but preserve a./b. list labels.
        self.assertFalse(note_section_has_body("Article body.\n\nNote"))
        self.assertFalse(note_section_has_body("Article body.\n\n== Note =="))
        self.assertTrue(note_section_has_body("Article body.\n\n== Note ==\n\na. Visible note."))
        self.assertEqual(remove_empty_note_section("Article body.\n\n== Note =="), "Article body.")
        self.assertEqual(
            extract_note_section("Article body.\n\n== Note ==\n\na. Visible note.\n\nb. Other note."),
            "== Note ==\n\na. Visible note.\n\nb. Other note.",
        )

    def test_html_parser_removes_section_content_by_heading_id(self):
        # Verifies HTML heading IDs trigger full unwanted-section removal while Note remains.
        html = """
        <div class="mw-parser-output">
          <p>Article body.</p>
          <div class="mw-heading mw-heading2"><h2 id="See_also">See also</h2></div>
          <ul><li>Hidden related page</li></ul>
          <div class="mw-heading mw-heading2"><h2 id="Note">Note</h2></div>
          <p>a. Visible note stays.</p>
          <h2><span id="References">References</span></h2>
          <p>Hidden reference.</p>
        </div>
        """

        cleaned = clean_wikipedia_html(html)

        self.assertIn("Article body.", cleaned)
        self.assertIn("== Note ==", cleaned)
        self.assertIn("a. Visible note stays.", cleaned)
        self.assertNotIn("Hidden related page", cleaned)
        self.assertNotIn("Hidden reference.", cleaned)

    def test_html_parser_formats_any_heading_level(self):
        # Verifies all HTML heading levels are normalized to the readable == Heading == format.
        html = """
        <div class="mw-parser-output">
          <p>Article body.</p>
          <h2>Largest KBOs</h2>
          <p>Section body.</p>
          <h3>Discovery</h3>
          <p>Subsection body.</p>
        </div>
        """

        cleaned = clean_wikipedia_html(html)

        self.assertIn("Article body.\n\n== Largest KBOs ==\n\nSection body.", cleaned)
        self.assertIn("Section body.\n\n== Discovery ==\n\nSubsection body.", cleaned)

    def test_html_parser_generates_decimal_ordered_list_labels(self):
        # Verifies normal ordered lists produce numeric labels, including start offsets.
        html = """
        <div class="mw-parser-output">
          <p>In order of discovery, these bodies are:</p>
          <ol>
            <li>Ceres</li>
            <li>Pluto</li>
            <li>Eris</li>
          </ol>
          <ol start="4">
            <li>Haumea</li>
            <li>Makemake</li>
          </ol>
        </div>
        """

        cleaned = clean_wikipedia_html(html)

        self.assertIn("1. Ceres", cleaned)
        self.assertIn("2. Pluto", cleaned)
        self.assertIn("3. Eris", cleaned)
        self.assertIn("4. Haumea", cleaned)
        self.assertIn("5. Makemake", cleaned)

    def test_html_parser_generates_hyphen_bullet_list_labels(self):
        # Verifies unordered HTML lists keep visible bullet structure as hyphen labels.
        html = """
        <div class="mw-parser-output">
          <p>Examples include:</p>
          <ul>
            <li>reported arithmetics</li>
            <li>decoding the International Phonetic Alphabet</li>
            <li>unscrambling a word's letters</li>
            <li>disambiguating word-in-context datasets<sup class="reference">[79]</sup></li>
            <li>converting spatial words</li>
          </ul>
        </div>
        """

        cleaned = clean_wikipedia_html(html)

        self.assertIn("Examples include:\n\n- reported arithmetics", cleaned)
        self.assertIn("- decoding the International Phonetic Alphabet", cleaned)
        self.assertIn("- unscrambling a word's letters", cleaned)
        self.assertIn("- disambiguating word-in-context datasets", cleaned)
        self.assertIn("- converting spatial words", cleaned)
        self.assertNotIn("[79]", cleaned)

    def test_html_parser_generates_lower_alpha_note_labels(self):
        # Verifies lower-alpha note references keep a./b. labels and drop backlink markers.
        html = """
        <div class="mw-parser-output">
          <div class="mw-heading mw-heading2"><h2 id="Note">Note</h2></div>
          <ol class="references" data-mw-group="lower-alpha">
            <li><span class="mw-cite-backlink">^ a b</span>
              <span class="reference-text">First note.</span></li>
            <li><span class="mw-cite-backlink">^ c</span>
              <span class="reference-text">Second note.</span></li>
          </ol>
        </div>
        """

        cleaned = clean_wikipedia_html(html)

        self.assertIn("a. First note.", cleaned)
        self.assertIn("b. Second note.", cleaned)
        self.assertNotIn("^", cleaned)

    def test_html_references_export_keeps_markers_and_numeric_references(self):
        # Verifies optional references export keeps inline markers and appends numeric references.
        html = """
        <div class="mw-parser-output">
          <p>Article text.<sup class="reference"><a><span>[</span>1<span>]</span></a></sup> More text.</p>
          <div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
          <ol class="references">
            <li><span class="mw-cite-backlink">^</span>
              <span class="reference-text">First source. Retrieved 2024.</span></li>
            <li><span class="mw-cite-backlink">^</span>
              <span class="reference-text">Second source.</span></li>
          </ol>
          <div class="mw-heading mw-heading2"><h2 id="Notes">Notes</h2></div>
          <ol class="references" data-mw-group="lower-alpha">
            <li><span class="reference-text">A note, not a numeric reference.</span></li>
          </ol>
        </div>
        """

        text = clean_wikipedia_html_with_references(html)

        self.assertIn("Article text.[1] More text.", text)
        self.assertIn("== References ==\n\n1. First source. Retrieved 2024.", text)
        self.assertIn("2. Second source.", text)
        self.assertNotIn("3. A note, not a numeric reference.", text)

    def test_html_references_export_can_keep_sources_only_at_end(self):
        # Verifies references export can omit inline markers while keeping the final source list.
        html = """
        <div class="mw-parser-output">
          <p>Article text.<sup class="reference"><a><span>[</span>1<span>]</span></a></sup> More text.</p>
          <div class="mw-heading mw-heading2"><h2 id="References">References</h2></div>
          <ol class="references">
            <li><span class="mw-cite-backlink">^</span>
              <span class="reference-text">First source.</span></li>
          </ol>
        </div>
        """

        text = clean_wikipedia_html_with_references(
            html, include_inline_markers=False
        )

        self.assertIn("Article text. More text.", text)
        self.assertNotIn("Article text.[1]", text)
        self.assertIn("== References ==\n\n1. First source.", text)

    def test_partial_extraction_matches_exact_clean_section(self):
        # Verifies pasted text can extract the same section from already-clean HTML output.
        full_text = (
            "Lead paragraph.\n\n"
            "Target section starts with a distinctive sentence.\n\n"
            "It continues with enough detail to be a stable anchor.\n\n"
            "The target section ends with another distinctive sentence.\n\n"
            "Later article text."
        )
        pasted = (
            "Target section starts with a distinctive sentence.\n\n"
            "It continues with enough detail to be a stable anchor.\n\n"
            "The target section ends with another distinctive sentence."
        )

        result = extract_partial_text(full_text, pasted, threshold=0.92, anchor_size=120)

        self.assertEqual(result.text, pasted)
        self.assertGreaterEqual(result.start_match.score, 0.92)
        self.assertGreaterEqual(result.end_match.score, 0.92)

    def test_partial_extraction_tolerates_refs_and_spacing(self):
        # Verifies copied citation markers and whitespace differences do not block matching.
        full_text = (
            "Before text.\n\n"
            "Ceres was discovered January 1, 1801, and announced January 24.\n\n"
            "Pluto was discovered February 18, 1930, and announced March 13.\n\n"
            "Eris was discovered January 5, 2005, and announced July 29.\n\n"
            "After text."
        )
        pasted = (
            "Ceres was discovered January 1, 1801, and announced January 24.[56]\n"
            "Pluto was discovered February 18, 1930, and announced March 13.\n\n"
            "Eris was discovered January 5, 2005, and announced July 29."
        )

        result = extract_partial_text(full_text, pasted, threshold=0.90, anchor_size=150)

        self.assertIn("Ceres was discovered", result.text)
        self.assertIn("Eris was discovered", result.text)
        self.assertNotIn("[56]", result.text)

    def test_partial_extraction_skips_noisy_copied_edges(self):
        # Verifies table/caption-like copied edge lines are ignored when choosing anchors.
        full_text = (
            "Intro text.\n\n"
            "The clean section begins with a paragraph that has enough normal words to match.\n\n"
            "The middle paragraph keeps the selected content inside the extracted range.\n\n"
            "The clean section ends with another paragraph that has enough normal words to match.\n\n"
            "Footer text."
        )
        pasted = (
            "12 34 56 | image map\n"
            "Short caption\n\n"
            "The clean section begins with a paragraph that has enough normal words to match.\n\n"
            "The middle paragraph keeps the selected content inside the extracted range.\n\n"
            "The clean section ends with another paragraph that has enough normal words to match.\n\n"
            "90 88 77 | hidden table"
        )

        result = extract_partial_text(full_text, pasted, threshold=0.92, anchor_size=160)

        self.assertTrue(result.text.startswith("The clean section begins"))
        self.assertTrue(result.text.endswith("enough normal words to match."))
        self.assertNotIn("Intro text.", result.text)
        self.assertNotIn("Footer text.", result.text)

    def test_partial_extraction_can_use_short_valid_lines(self):
        # Verifies short meaningful lines are not discarded as copied-edge noise.
        full_text = (
            "Before text.\n\n"
            "Valid short sentence.\n\n"
            "The section body has enough surrounding detail to identify the partial range.\n\n"
            "Done here.\n\n"
            "After text."
        )
        pasted = (
            "Valid short sentence.\n\n"
            "The section body has enough surrounding detail to identify the partial range.\n\n"
            "Done here."
        )

        result = extract_partial_text(full_text, pasted, threshold=0.92, anchor_size=120)

        self.assertTrue(result.text.startswith("Valid short sentence."))
        self.assertTrue(result.text.endswith("Done here."))

    def test_short_partial_anchors_require_stricter_scores(self):
        # Verifies very short anchors use a higher score floor to avoid weak fuzzy matches.
        self.assertEqual(required_partial_anchor_score("Done here.", 0.92), 0.95)
        self.assertEqual(
            required_partial_anchor_score(
                "This is a longer and more distinctive anchor.", 0.92
            ),
            0.92,
        )
        self.assertEqual(required_partial_anchor_score("Done here.", 0.97), 0.97)

    def test_partial_anchor_candidates_prefer_prose_over_math_edges(self):
        # Verifies weird copied math at pasted edges is not preferred over nearby prose.
        pasted = (
            "N∝D1−q+a constant\n\n"
            "The size distribution is commonly described with a power law.\n\n"
            "This prose line should close the selected section.\n\n"
            "dN/dD ∝ D^-q"
        )

        start_anchor = strong_anchor_candidates(pasted, "start", anchor_size=120)[0]
        end_anchor = strong_anchor_candidates(pasted, "end", anchor_size=120)[0]

        self.assertIn("The size distribution", start_anchor)
        self.assertIn("This prose line", end_anchor)
        self.assertNotIn("N∝D", start_anchor)
        self.assertNotIn("dN/dD", end_anchor)

    def test_partial_extraction_reports_threshold_failure(self):
        # Verifies unreliable pasted text fails with a debuggable report instead of bad output.
        with self.assertRaises(PartialExtractionError) as context:
            extract_partial_text(
                "A clean article with one body paragraph.",
                "Completely unrelated pasted text that cannot match.",
                threshold=0.99,
            )

        self.assertIn("failed", context.exception.report_text)
        self.assertIn("Threshold: 0.990", context.exception.report_text)

    def test_partial_html_cli_defaults_to_latex_math(self):
        # Verifies partial extraction keeps LaTeX math by default unless the CLI overrides it.
        from extract_partial_html import build_parser

        args = build_parser().parse_args(["--url", "https://en.wikipedia.org/wiki/Kuiper_belt"])

        self.assertEqual(args.math, "latex")

    def test_lower_alpha_label_handles_multiple_letters(self):
        # Verifies lower-alpha label generation continues past z.
        self.assertEqual(lower_alpha_label(1), "a")
        self.assertEqual(lower_alpha_label(26), "z")
        self.assertEqual(lower_alpha_label(27), "aa")

    def test_output_path_for_both_methods_adds_method_suffix(self):
        # Verifies legacy method-splitting path helper appends the method suffix.
        path = output_path_for_method("output/saturn.txt", "html", split_methods=True)

        self.assertEqual(path, Path("output/saturn_html.txt"))

    def test_add_topic_heading_prefixes_clean_outputs_once(self):
        # Verifies generated clean text starts with the page title as a section heading.
        page = PageRequest("Large language model", "en")

        headed = add_topic_heading("Article body.", page)

        self.assertEqual(headed, "== Large language model ==\n\nArticle body.")
        self.assertEqual(add_topic_heading(headed, page), headed)

    def test_update_runtime_report_preserves_unrelated_entries(self):
        # Verifies single runs only replace their own timing and keep other runtime entries.
        page = PageRequest("Large language model", "en")
        path = Path("test_runtime_report.txt")
        try:
            update_runtime_report(
                path,
                page,
                {
                    runtime_label("Extracts API runtime", "remove", page): 1.0,
                    runtime_label("HTML parser runtime", "remove", page): 2.0,
                },
            )
            update_runtime_report(
                path,
                page,
                {runtime_label("Partial HTML runtime", "latex", page): 3.0},
            )
            update_runtime_report(
                path,
                page,
                {runtime_label("HTML parser runtime", "remove", page): 4.0},
            )

            text = path.read_text(encoding="utf-8")

            self.assertNotIn("Topic:", text)
            self.assertIn(
                "Extracts API runtime (remove, Large language model): 1.000 seconds",
                text,
            )
            self.assertIn(
                "HTML parser runtime (remove, Large language model): 4.000 seconds",
                text,
            )
            self.assertIn(
                "Partial HTML runtime (latex, Large language model): 3.000 seconds",
                text,
            )
            self.assertNotIn(
                "HTML parser runtime (remove, Large language model): 2.000 seconds",
                text,
            )
        finally:
            path.unlink(missing_ok=True)

    def test_nested_output_paths_include_topic_and_language(self):
        # Verifies generated output paths are grouped by topic and language folders.
        page = page_request_from_url("https://en.wikipedia.org/wiki/Kuiper_belt")

        self.assertEqual(
            extraction_output_path("output/kuiper_belt.txt", page, "extracts", "remove"),
            Path("output/Kuiper_belt/English/kuiper_belt_extracts_remove.txt"),
        )
        self.assertEqual(
            comparison_output_path("output/kuiper_belt.txt", page),
            Path("output/Kuiper_belt/English/kuiper_belt_comparison.txt"),
        )
        self.assertEqual(
            runtime_output_path("output/kuiper_belt.txt", page),
            Path("output/Kuiper_belt/English/kuiper_belt_runtime.txt"),
        )
        self.assertEqual(
            raw_output_path("output/kuiper_belt.txt", page, "extracts"),
            Path("output/Kuiper_belt/English/kuiper_belt_raw_extracts.txt"),
        )
        self.assertEqual(
            references_output_path("output/kuiper_belt.txt", page),
            Path("output/Kuiper_belt/English/kuiper_belt_html_references.txt"),
        )
        self.assertEqual(
            partial_output_path("output/kuiper_belt.txt", page),
            Path("output/Kuiper_belt/English/partial_text.txt"),
        )
        self.assertEqual(
            partial_match_report_path("output/kuiper_belt.txt", page),
            Path("output/Kuiper_belt/English/partial_match_report.txt"),
        )

    def test_compare_texts_reports_mismatch(self):
        # Verifies comparison reports include exact-match status and unified diff lines.
        report = compare_texts("Line one\nLine two", "Line one\nLine three")

        self.assertIn("Exact match: False", report)
        self.assertIn("-Line two", report)
        self.assertIn("+Line three", report)

    def test_math_remove_drops_displaystyle_and_fragments(self):
        # Verifies math remove mode drops displaystyle formulas and rendered math fragments.
        text = "\n\n".join(
            [
                "The objects follow this relation:",
                "d",
                "D",
                "\u221d",
                "{\\displaystyle {\\frac {dN}{dD}}\\propto D^{-q},}",
                "Normal text remains.",
            ]
        )

        cleaned = clean_plain_text(text, math_mode="remove")

        self.assertEqual(cleaned, "The objects follow this relation:\n\nNormal text remains.")

    def test_math_latex_keeps_clean_latex(self):
        # Verifies math latex mode keeps clean LaTeX while removing duplicated rendered math.
        text = "\n".join(
            [
                "The objects follow this relation:",
                "",
                "d",
                "N",
                "{\\displaystyle {\\frac {dN}{dD}}\\propto D^{-q},}",
                "which yields:N\u221dD1\u2212q+a constant.",
                "a constant.",
                "{\\displaystyle N\\propto D^{1-q}+{\\text{a constant}}.}",
            ]
        )

        cleaned = clean_plain_text(text, math_mode="latex")

        self.assertIn("${\\frac {dN}{dD}}\\propto D^{-q}$", cleaned)
        self.assertIn("$N\\propto D^{1-q}+{\\mathrm{a constant}}$", cleaned)
        self.assertIn("which yields:", cleaned)
        self.assertNotIn("N\u221dD1\u2212q+a constant", cleaned)
        self.assertNotIn("$which yields", cleaned)
        self.assertNotIn("\nd\nN", cleaned)
        self.assertNotIn("a constant.", cleaned)

    def test_math_latex_removes_rendered_lines_before_latex(self):
        # Verifies rendered HTML math fallback lines do not duplicate following LaTeX.
        text = "\n".join(
            [
                "y",
                "$y$, the post-processed vector f(E(y))",
                "f(E(y))",
                "$f(E(y))$",
                "α=0.34,β=0.28,A=406.4,B=410.7,L0=1.69",
                "$\\alpha =0.34,\\beta =0.28,A=406.4,B=410.7,L_{0}=1.69$",
                "y=average Pr(correct token)",
                "$y={\\mathrm{average }}\\Pr({\\mathrm{correct token}})$",
                "log\u2061(Perplexity)=−1N∑i=1Nlog\u2061(Pr(tokeni∣context for tokeni))",
                "$\\log({\\mathrm{Perplexity}})=-{\\frac {1}{N}}\\sum _{i=1}^{N}\\log(\\Pr({\\mathrm{token}}_{i}\\mid {\\mathrm{context for token}}_{i}))$",
            ]
        )

        cleaned = clean_plain_text(text, math_mode="latex")

        self.assertIn("$y$, the post-processed vector f(E(y))", cleaned)
        self.assertIn("$f(E(y))$", cleaned)
        self.assertIn("$\\alpha =0.34", cleaned)
        self.assertIn("$y={\\mathrm{average }}", cleaned)
        self.assertIn("$\\log({\\mathrm{Perplexity}})", cleaned)
        self.assertNotIn("\ny\n$y$", cleaned)
        self.assertNotIn("\nf(E(y))\n$f(E(y))$", cleaned)
        self.assertNotIn("α=0.34,β=0.28", cleaned)
        self.assertNotIn("y=average Pr(correct token)", cleaned)
        self.assertNotIn("log\u2061(Perplexity)=−1N", cleaned)

    def test_math_latex_rejoins_inline_symbols_with_prose(self):
        # Verifies inline math symbols do not force awkward line breaks in prose.
        text = "\n".join(
            [
                "Let x",
                "x",
                "$x$",
                "be the number of parameter count, and y",
                "y",
                "$y$",
                "be the performance of the model.",
            ]
        )

        cleaned = clean_plain_text(text, math_mode="latex")

        self.assertEqual(
            cleaned,
            (
                "Let $x$ be the number of parameter count, and $y$ "
                "be the performance of the model."
            ),
        )

    def test_math_latex_removes_same_line_rendered_symbol_duplicates(self):
        # Verifies rendered one-letter symbols beside inline LaTeX are not repeated.
        text = "\n".join(
            [
                'Here, N',
                "N",
                "$N$",
                'is the number of tokens in the text corpus, and "context for token i',
                "i",
                "$i$",
                '" depends on the specific type of LLM.',
                "If masked, then context surrounds token i",
                "$i$",
                "The vector is x_i",
                "$x_i$",
                "and the token id is token_i",
                "$token_i$",
                "while log",
                "$log$",
                "and NLL",
                "$NLL$",
                "are short math identifiers.",
                "The activation is softmax",
                "$softmax$",
                "or sigmoid",
                "$sigmoid$",
                "or ReLU",
                "$ReLU$",
                "in some models.",
                "But model",
                "$model$",
                "and language",
                "$language$",
                "stay as prose-plus-math.",
            ]
        )

        cleaned = clean_plain_text(text, math_mode="latex")

        self.assertIn("Here, $N$ is the number of tokens", cleaned)
        self.assertIn('"context for token $i$" depends', cleaned)
        self.assertIn("context surrounds token $i$", cleaned)
        self.assertIn("The vector is $x_i$ and the token id is $token_i$", cleaned)
        self.assertIn("while $log$ and $NLL$ are short math identifiers.", cleaned)
        self.assertIn(
            "The activation is $softmax$ or $sigmoid$ or $ReLU$ in some models.",
            cleaned,
        )
        self.assertIn("But model $model$ and language $language$", cleaned)
        self.assertNotIn("N $N$", cleaned)
        self.assertNotIn("i $i$", cleaned)
        self.assertNotIn("x_i $x_i$", cleaned)
        self.assertNotIn("token_i $token_i$", cleaned)
        self.assertNotIn("log $log$", cleaned)
        self.assertNotIn("NLL $NLL$", cleaned)
        self.assertNotIn("softmax $softmax$", cleaned)
        self.assertNotIn("sigmoid $sigmoid$", cleaned)
        self.assertNotIn("ReLU $ReLU$", cleaned)
        self.assertNotIn("\n$i$", cleaned)

    def test_math_keep_preserves_raw_displaystyle(self):
        # Verifies math keep mode leaves raw displaystyle markup untouched.
        text = "{\\displaystyle N\\propto D^{1-q}}"

        self.assertIn("{\\displaystyle", clean_plain_text(text, math_mode="keep"))


if __name__ == "__main__":
    unittest.main()
