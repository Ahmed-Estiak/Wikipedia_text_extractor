from pathlib import Path
import unittest

from wiki_text_extractor import (
    clean_plain_text,
    clean_wikipedia_html,
    compare_texts,
    output_path_for_method,
    page_request_from_url,
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

    def test_extracts_language_and_title_from_url(self):
        page = page_request_from_url("https://en.wikipedia.org/wiki/Albert_Einstein")

        self.assertEqual(page.lang, "en")
        self.assertEqual(page.title, "Albert Einstein")

    def test_cleans_extracts_api_plain_text_noise(self):
        text = """
        Saturn is a planet.[1]

        == History ==

        It has rings.[2, 3]

        == References ==

        Reference text that should not be included.
        """

        self.assertEqual(
            clean_plain_text(text),
            "Saturn is a planet.\nHistory\nIt has rings.",
        )

    def test_output_path_for_both_methods_adds_method_suffix(self):
        path = output_path_for_method("output/saturn.txt", "html", split_methods=True)

        self.assertEqual(path, Path("output/saturn_html.txt"))

    def test_compare_texts_reports_mismatch(self):
        report = compare_texts("Line one\nLine two", "Line one\nLine three")

        self.assertIn("Exact match: False", report)
        self.assertIn("-Line two", report)
        self.assertIn("+Line three", report)


if __name__ == "__main__":
    unittest.main()
