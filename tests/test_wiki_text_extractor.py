import unittest

from wiki_text_extractor import clean_wikipedia_html, page_request_from_url


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


if __name__ == "__main__":
    unittest.main()
