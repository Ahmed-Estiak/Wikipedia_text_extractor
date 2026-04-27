"""Extract clean plain text from Wikipedia pages."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


API_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "WikipediaTextExtractor/0.1"
EXTRACTION_METHODS = ("extracts", "html")
MATH_MODES = ("remove", "latex", "keep")
TAIL_SECTION_PATTERN = re.compile(
    r"\n\s*==\s*(See also|References|External links|Further reading|Notes)\s*==[\s\S]*$",
    re.IGNORECASE,
)
INLINE_REFERENCE_PATTERN = re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")
MATH_FRAGMENT_PATTERN = re.compile(r"^[\sA-Za-z0-9+\-–−=∝×*/^().,{}\\]+$")
MATH_SYMBOL_PATTERN = re.compile(r"[+\-–−=∝×*/^{}\\]")
LANGUAGE_FOLDER_NAMES = {
    "bn": "Bangla",
    "en": "English",
    "fi": "Suomi",
}


@dataclass(frozen=True)
class PageRequest:
    title: str
    lang: str = "en"


class WikipediaTextParser(HTMLParser):
    """Small HTML-to-text parser tuned for Wikipedia article HTML."""

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    }
    SKIP_TAGS = {"script", "style", "table", "sup"}
    SKIP_CLASSES = {
        "ambox",
        "asbox",
        "catlinks",
        "citation",
        "hatnote",
        "infobox",
        "metadata",
        "mw-editsection",
        "navbox",
        "noprint",
        "reference",
        "reflist",
        "sidebar",
        "toc",
        "vertical-navbox",
    }
    SKIP_IDS = {
        "References",
        "External_links",
        "Further_reading",
        "See_also",
        "Notes",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        classes = set(attr_map.get("class", "").split())
        element_id = attr_map.get("id", "")

        if (
            self._skip_depth
            or tag in self.SKIP_TAGS
            or classes.intersection(self.SKIP_CLASSES)
            or element_id in self.SKIP_IDS
        ):
            self._skip_depth += 1
            return

        if tag in self.BLOCK_TAGS:
            self._add_break()

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if tag in self.BLOCK_TAGS:
            self._add_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data)
        if text.strip():
            self._parts.append(text)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.strip()

    def _add_break(self) -> None:
        if not self._parts:
            return
        last = self._parts[-1]
        if last.endswith("\n\n"):
            return
        if last.endswith("\n"):
            self._parts[-1] = last + "\n"
            return
        self._parts.append("\n\n")


def page_request_from_url(url: str) -> PageRequest:
    parsed = urlparse(url)
    host_parts = parsed.netloc.split(".")
    if len(host_parts) < 3 or host_parts[-2:] != ["wikipedia", "org"]:
        raise ValueError("URL must be from a wikipedia.org domain")

    lang = host_parts[0]
    title = unquote(parsed.path.rsplit("/", 1)[-1]).replace("_", " ")
    if not title:
        raise ValueError("Could not find a page title in the URL")
    return PageRequest(title=title, lang=lang)


def fetch_page_html(page: PageRequest) -> str:
    query = (
        f"?action=parse&page={quote(page.title)}&prop=text&format=json"
        "&formatversion=2&redirects=1"
    )
    request = Request(API_TEMPLATE.format(lang=page.lang) + query)
    request.add_header("User-Agent", USER_AGENT)

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Wikipedia API returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Wikipedia API: {exc.reason}") from exc

    if "error" in payload:
        message = payload["error"].get("info", "Unknown Wikipedia API error")
        raise RuntimeError(message)
    return payload["parse"]["text"]


def fetch_page_extract(page: PageRequest) -> str:
    query = (
        f"?action=query&prop=extracts&explaintext=1&titles={quote(page.title)}"
        "&format=json&formatversion=2&redirects=1"
    )
    request = Request(API_TEMPLATE.format(lang=page.lang) + query)
    request.add_header("User-Agent", USER_AGENT)

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Wikipedia API returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Wikipedia API: {exc.reason}") from exc

    if "error" in payload:
        message = payload["error"].get("info", "Unknown Wikipedia API error")
        raise RuntimeError(message)

    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise RuntimeError(f"Wikipedia page not found: {page.title}")
    return pages[0].get("extract", "")


def normalize_latex(latex: str) -> str:
    latex = re.sub(r"\s+", " ", latex)
    latex = latex.replace(r"\text{", r"\mathrm{")
    return latex.strip().rstrip(",.;:")


def paragraph_looks_like_math_fragment(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return False
    if len(stripped) <= 2 and re.match(r"^[A-Za-z0-9]+$", stripped):
        return True
    if len(stripped) > 120:
        return False
    if not MATH_FRAGMENT_PATTERN.match(stripped):
        return False
    return bool(MATH_SYMBOL_PATTERN.search(stripped))


def find_displaystyle_latex(paragraph: str) -> list[str]:
    expressions: list[str] = []
    marker = r"{\displaystyle"
    start = 0
    while True:
        marker_index = paragraph.find(marker, start)
        if marker_index == -1:
            return expressions

        depth = 0
        end_index = None
        for index in range(marker_index, len(paragraph)):
            char = paragraph[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_index = index
                    break

        if end_index is None:
            return expressions

        latex = paragraph[marker_index + len(marker) : end_index]
        expressions.append(latex.strip())
        start = end_index + 1


def clean_math(text: str, math_mode: str) -> str:
    if math_mode not in {"remove", "latex", "keep"}:
        raise ValueError(f"Unsupported math mode: {math_mode}")
    if math_mode == "keep":
        return text

    paragraphs = re.split(r"\n{2,}", text)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        latex_expressions = find_displaystyle_latex(paragraph)
        if latex_expressions:
            if math_mode == "latex":
                latex_parts = [
                    f"${normalize_latex(expression)}$"
                    for expression in latex_expressions
                ]
                if latex_parts:
                    cleaned.append(" ".join(latex_parts))
            continue
        if paragraph_looks_like_math_fragment(paragraph):
            continue
        cleaned.append(paragraph)
    return "\n\n".join(cleaned)


def clean_plain_text(text: str, math_mode: str = "remove") -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TAIL_SECTION_PATTERN.sub("", text)
    text = INLINE_REFERENCE_PATTERN.sub("", text)
    text = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)
    text = clean_math(text, math_mode)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def clean_wikipedia_html(html: str, math_mode: str = "remove") -> str:
    parser = WikipediaTextParser()
    parser.feed(html)
    parser.close()
    return clean_plain_text(parser.get_text(), math_mode)


def extract_text_from_extracts_api(page: PageRequest, math_mode: str = "remove") -> str:
    return clean_plain_text(fetch_page_extract(page), math_mode)


def extract_text_from_html(page: PageRequest, math_mode: str = "remove") -> str:
    return clean_wikipedia_html(fetch_page_html(page), math_mode)


def extract_text(page: PageRequest, method: str = "extracts", math_mode: str = "remove") -> str:
    if method == "extracts":
        return extract_text_from_extracts_api(page, math_mode)
    if method == "html":
        return extract_text_from_html(page, math_mode)
    raise ValueError(f"Unsupported extraction method: {method}")


def output_path_for_method(output: str, method: str, split_methods: bool) -> Path:
    path = Path(output)
    if not split_methods:
        return path
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_{method}{suffix}")


def safe_filename_part(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "wikipedia_page"


def topic_folder_name(page: PageRequest) -> str:
    return safe_filename_part(page.title)


def topic_file_stem(page: PageRequest) -> str:
    return topic_folder_name(page).lower()


def language_folder_name(lang: str) -> str:
    return LANGUAGE_FOLDER_NAMES.get(lang.lower(), lang.lower())


def output_directory(output: str, page: PageRequest) -> Path:
    path = Path(output)
    root = path.parent if path.suffix else path
    return root / topic_folder_name(page) / language_folder_name(page.lang)


def extraction_output_path(output: str, page: PageRequest, method: str, math_mode: str) -> Path:
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_{method}_{math_mode}{suffix}"


def comparison_output_path(output: str, page: PageRequest) -> Path:
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_comparison{suffix}"


def runtime_output_path(output: str, page: PageRequest) -> Path:
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_runtime{suffix}"


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def run_extraction(
    page: PageRequest, method: str, math_mode: str = "remove"
) -> tuple[str, float]:
    started_at = time.perf_counter()
    text = extract_text(page, method, math_mode)
    return text, time.perf_counter() - started_at


def compare_texts(extracts_text: str, html_text: str) -> str:
    extracts_lines = extracts_text.splitlines()
    html_lines = html_text.splitlines()
    diff = list(
        difflib.unified_diff(
            extracts_lines,
            html_lines,
            fromfile="extracts_api",
            tofile="html_parser",
            lineterm="",
        )
    )
    report = [
        "Wikipedia Text Extraction Comparison",
        "",
        f"Extracts API characters: {len(extracts_text)}",
        f"HTML parser characters: {len(html_text)}",
        f"Character mismatch: {abs(len(extracts_text) - len(html_text))}",
        f"Extracts API lines: {len(extracts_lines)}",
        f"HTML parser lines: {len(html_lines)}",
        f"Line mismatch: {abs(len(extracts_lines) - len(html_lines))}",
        f"Exact match: {extracts_text == html_text}",
        "",
        "Unified Diff:",
        "",
    ]
    report.extend(diff or ["No mismatch found."])
    return "\n".join(report).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract clean plain text from a Wikipedia page."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "--method",
        choices=("extracts", "html", "both"),
        default="extracts",
        help="Extraction method to use",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="remove",
        help="How math equations should be handled",
    )
    parser.add_argument("-o", "--output", help="Write extracted text to this file")
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    methods = ("extracts", "html") if args.method == "both" else (args.method,)

    try:
        results = {method: extract_text(page, method, args.math) for method in methods}
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        for method, text in results.items():
            path = extraction_output_path(args.output, page, method, args.math)
            write_text_file(path, text)
            print(f"Saved {method} text: {path}")
    else:
        for method, text in results.items():
            if len(results) > 1:
                print(f"===== {method} =====")
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
