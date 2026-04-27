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

    def test_extracts_language_and_title_from_url(self):
        page = page_request_from_url("https://en.wikipedia.org/wiki/Albert_Einstein")

        self.assertEqual(page.lang, "en")
        self.assertEqual(page.title, "Albert Einstein")

    def test_cleans_extracts_api_plain_text_noise(self):
        text = """
        Saturn ( ) is a planet.[1]

        == History ==

        It has 8 (=23) rings.[2, 3]

        == References ==

        Reference text that should not be included.
        """

        self.assertEqual(
            clean_plain_text(text),
            "Saturn is a planet.\nHistory\nIt has 8 (=2^3) rings.",
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
        self.assertIn("Notes", cleaned)
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
        self.assertTrue(note_section_has_body("Article body.\n\nNote\n\na. Visible note."))
        self.assertEqual(remove_empty_note_section("Article body.\n\nNote"), "Article body.")
        self.assertEqual(
            extract_note_section("Article body.\n\nNote\n\na. Visible note.\n\nb. Other note."),
            "Note\n\na. Visible note.\n\nb. Other note.",
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
        self.assertIn("Note", cleaned)
        self.assertIn("a. Visible note stays.", cleaned)
        self.assertNotIn("Hidden related page", cleaned)
        self.assertNotIn("Hidden reference.", cleaned)

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
