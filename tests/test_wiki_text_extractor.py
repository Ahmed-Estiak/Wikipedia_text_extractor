from pathlib import Path
import unittest

from wiki_text_extractor import (
    clean_plain_text,
    clean_wikipedia_html,
    compare_texts,
    comparison_output_path,
    extraction_output_path,
    output_path_for_method,
    page_request_from_url,
    runtime_output_path,
    extract_note_section,
    note_section_has_body,
    remove_empty_note_section,
    raw_output_path,
    lower_alpha_label,
)


class CleanWikipediaHtmlTests(unittest.TestCase):
    def test_removes_references_infoboxes_and_tables(self):
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

    def test_preserves_non_math_subscripts_with_underscores(self):
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
          <figure>
            <a class="mw-file-description" title="Hidden figure symbol"><img alt="Hidden" /></a>
            <figcaption>Hidden caption</figcaption>
          </figure>
        </div>
        """

        text = clean_wikipedia_html(html)

        self.assertEqual(text, "Ceres ⚳ and Pluto ♇ received symbols.")
        self.assertNotIn("Hidden figure symbol", text)

    def test_extracts_language_and_title_from_url(self):
        page = page_request_from_url("https://en.wikipedia.org/wiki/Albert_Einstein")

        self.assertEqual(page.lang, "en")
        self.assertEqual(page.title, "Albert Einstein")

    def test_cleans_extracts_api_plain_text_noise(self):
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
        self.assertFalse(note_section_has_body("Article body.\n\nNote"))
        self.assertFalse(note_section_has_body("Article body.\n\n== Note =="))
        self.assertTrue(note_section_has_body("Article body.\n\n== Note ==\n\na. Visible note."))
        self.assertEqual(remove_empty_note_section("Article body.\n\n== Note =="), "Article body.")
        self.assertEqual(
            extract_note_section("Article body.\n\n== Note ==\n\na. Visible note.\n\nb. Other note."),
            "== Note ==\n\na. Visible note.\n\nb. Other note.",
        )

    def test_html_parser_removes_section_content_by_heading_id(self):
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

    def test_html_parser_generates_lower_alpha_note_labels(self):
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

    def test_lower_alpha_label_handles_multiple_letters(self):
        self.assertEqual(lower_alpha_label(1), "a")
        self.assertEqual(lower_alpha_label(26), "z")
        self.assertEqual(lower_alpha_label(27), "aa")

    def test_output_path_for_both_methods_adds_method_suffix(self):
        path = output_path_for_method("output/saturn.txt", "html", split_methods=True)

        self.assertEqual(path, Path("output/saturn_html.txt"))

    def test_nested_output_paths_include_topic_and_language(self):
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

    def test_compare_texts_reports_mismatch(self):
        report = compare_texts("Line one\nLine two", "Line one\nLine three")

        self.assertIn("Exact match: False", report)
        self.assertIn("-Line two", report)
        self.assertIn("+Line three", report)

    def test_math_remove_drops_displaystyle_and_fragments(self):
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

    def test_math_keep_preserves_raw_displaystyle(self):
        text = "{\\displaystyle N\\propto D^{1-q}}"

        self.assertIn("{\\displaystyle", clean_plain_text(text, math_mode="keep"))


if __name__ == "__main__":
    unittest.main()
