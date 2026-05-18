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
MATH_IDENTIFIER_ALLOWLIST = {
    "argmax",
    "argmin",
    "cos",
    "exp",
    "gelu",
    "ln",
    "log",
    "max",
    "min",
    "pr",
    "relu",
    "sigmoid",
    "sin",
    "softmax",
    "tan",
    "tanh",
}
REMOVED_SECTION_TITLES = {"see also", "references", "external links"}
REMOVED_SECTION_IDS = {"See_also", "References", "External_links"}
PLAIN_SECTION_TITLES = REMOVED_SECTION_TITLES | {"note", "notes"}
SECTION_HEADING_PATTERN = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")
INLINE_REFERENCE_PATTERN = re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")
LEADING_CARET_MARKER_PATTERN = re.compile(r"^[ \t]*(?:\^(?:[ \t]*[a-zA-Z0-9]+)?[ \t]*)+")
MATH_FRAGMENT_PATTERN = re.compile(r"^[\sA-Za-z0-9+\-–−=∝×*/^().,{}\\]+$")
MATH_SYMBOL_PATTERN = re.compile(r"[+\-–−=∝×*/^{}\\]")
EMPTY_PARENTHESES_PATTERN = re.compile(r"\s*\(\s*\)")
POWER_HINT_PATTERN = re.compile(r"(?P<value>\b\d+)\s+\(=(?P<compact>\d{2,6})\)")
LATEX_LINE_PATTERN = re.compile(r"^\s*\$[^$]+\$")
SIMPLE_LATEX_LINE_PATTERN = re.compile(r"^\s*\$[A-Za-z][A-Za-z0-9_]*\$\s*$")
INLINE_DUPLICATE_LATEX_SYMBOL_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s+\$\1\$"
)
MERGED_WORD_LATEX_SYMBOL_PATTERN = re.compile(r"\b([A-Za-z]{4,})([A-Za-z])\s+\$\2\$")
INLINE_DUPLICATE_LATEX_FUNCTION_PATTERN = re.compile(r"\b([A-Za-z]\([^$\n]+?\))\s+\$\1\$")
GREEK_LETTER_PATTERN = re.compile(r"[\u0370-\u03ff]")
RENDERED_MATH_FUNCTION_PATTERN = re.compile(r"\b(?:Pr|log|sin|cos|tan|exp|token)\s*\(")
SENTENCE_CONTINUATION_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:al|e\.g|i\.e|etc|vs|fig|no|vol|pp)\.$",
    re.IGNORECASE,
)
LANGUAGE_FOLDER_NAMES = {
    "bn": "Bangla",
    "en": "English",
    "fi": "Suomi",
}


@dataclass(frozen=True)
class PageRequest:
    # Carries the normalized Wikipedia page title and language code through the pipeline.
    title: str
    lang: str = "en"


@dataclass(frozen=True)
class FuzzyBoundaryMatch:
    # Records where one pasted-text boundary was found inside the cleaned article.
    anchor: str
    score: float
    start: int
    end: int


@dataclass(frozen=True)
class PartialExtractionResult:
    # Carries the extracted section and the boundary match diagnostics.
    text: str
    start_match: FuzzyBoundaryMatch
    end_match: FuzzyBoundaryMatch
    threshold: float


@dataclass(frozen=True)
class TextHeading:
    # Records one formatted heading and its character offset in cleaned text.
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class TextSentence:
    # Records one sentence-like unit and its character offsets in cleaned text.
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class CitationOccurrence:
    # Records one inline citation marker and the sentence it belongs to.
    number: str
    start: int
    end: int
    sentence_index: int


@dataclass(frozen=True)
class HybridTextIndex:
    # Lightweight structural index used by the heading/citation hybrid extractor.
    text: str
    headings: list[TextHeading]
    sentences: list[TextSentence]
    citations: list[CitationOccurrence]
    references_start: int | None


@dataclass(frozen=True)
class HybridExtractionResult:
    # Carries hybrid partial output and a readable decision report.
    text: str
    report: str
    start: int
    end: int
    confidence: str


class PartialExtractionError(ValueError):
    """Raised when pasted text cannot be matched reliably enough."""

    def __init__(self, message: str, report_text: str) -> None:
        super().__init__(message)
        self.report_text = report_text


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
    SKIP_TAGS = {"figcaption", "figure", "map", "script", "style", "table"}
    SKIP_CLASSES = {
        "ambox",
        "asbox",
        "catlinks",
        "hatnote",
        "infobox",
        "image",
        "imagemap",
        "locmap",
        "metadata",
        "mw-cite-backlink",
        "mw-file-description",
        "mw-kartographer-map",
        "mw-editsection",
        "navbox",
        "noexcerpt",
        "noprint",
        "noviewer",
        "reference",
        "reflist",
        "sidebar",
        "thumb",
        "thumbcaption",
        "thumbinner",
        "toc",
        "vertical-navbox",
    }
    SKIP_IDS: set[str] = set()
    MATH_CLASSES = {
        "mwe-math-element",
        "mwe-math-fallback-image-display",
        "mwe-math-fallback-image-inline",
        "mwe-math-mathml-display",
        "mwe-math-mathml-inline",
    }
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self, include_reference_markers: bool = False) -> None:
        # Initializes parser state for clean article text and optional inline citation markers.
        super().__init__(convert_charrefs=True)
        self._include_reference_markers = include_reference_markers
        self._parts: list[str] = []
        self._skip_depth = 0
        self._sup_depth = 0
        self._reference_marker_buffer: list[str] | None = None
        self._reference_marker_depth = 0
        self._sub_depth = 0
        self._reference_skip_depth = 0
        self._pending_reference_separator = False
        self._pending_inline_space = False
        self._pending_subscript_separator = False
        self._math_depth = 0
        self._math_tag_stack: list[str] = []
        self._heading_level: int | None = None
        self._heading_buffer: list[str] | None = None
        self._skip_section_level: int | None = None
        self._list_stack: list[dict[str, int | str]] = []
        self._pending_prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Handles HTML opening tags, skip rules, headings, lists, refs, and inline math markers.
        if self._reference_marker_buffer is not None:
            self._reference_marker_depth += 1
            return

        attr_map = {name: value or "" for name, value in attrs}
        classes = set(attr_map.get("class", "").split())
        element_id = attr_map.get("id", "")
        heading_level = int(tag[1]) if re.fullmatch(r"h[1-6]", tag) else None
        in_math = tag == "math" or bool(classes.intersection(self.MATH_CLASSES))
        if in_math and tag not in self.VOID_TAGS:
            self._math_depth += 1
            self._math_tag_stack.append(tag)

        if heading_level is not None:
            if self._skip_section_level is not None and heading_level <= self._skip_section_level:
                self._skip_section_level = None
            self._heading_level = heading_level

        if self._heading_level is not None and element_id in REMOVED_SECTION_IDS:
            self._skip_section_level = self._heading_level
            self._heading_buffer = None
            self._skip_depth += 1
            return

        if self._skip_depth:
            self._skip_depth += 1
            return

        if tag == "ol":
            group = attr_map.get("data-mw-group", "")
            if group == "lower-alpha" and "references" in classes:
                self._list_stack.append({"style": "lower-alpha", "index": 0})
            else:
                start = attr_map.get("start", "1")
                start_index = int(start) - 1 if start.isdigit() else 0
                self._list_stack.append({"style": "decimal", "index": start_index})
        elif tag == "ul":
            self._list_stack.append({"style": "bullet", "index": 0})

        if tag == "li" and self._list_stack and self._skip_section_level is None:
            current_list = self._list_stack[-1]
            if current_list["style"] in {"decimal", "lower-alpha"}:
                current_list["index"] = int(current_list["index"]) + 1
                self._add_break()
                if current_list["style"] == "lower-alpha":
                    label = lower_alpha_label(int(current_list["index"]))
                else:
                    label = str(current_list["index"])
                self._pending_prefix = f"{label}. "
            elif current_list["style"] == "bullet":
                self._add_break()
                self._pending_prefix = "- "

        if tag == "sup":
            if classes.intersection(self.SKIP_CLASSES) or "reference" in classes:
                if self._include_reference_markers and "reference" in classes:
                    self._reference_marker_buffer = []
                    self._reference_marker_depth = 1
                else:
                    self._skip_depth += 1
                    if "reference" in classes:
                        self._reference_skip_depth += 1
            else:
                self._sup_depth += 1
                self._parts.append("^")
            return

        if tag == "sub":
            if not self._math_depth:
                self._sub_depth += 1
                self._append_inline_marker("_")
            return

        if tag == "a" and "mw-file-description" in classes:
            file_title = attr_map.get("title", "").strip()
            if is_non_word_glyph(file_title):
                self._append_inline_text(file_title)
            self._skip_depth += 1
            return

        if (
            self._skip_depth
            or tag in self.SKIP_TAGS
            or classes.intersection(self.SKIP_CLASSES)
            or element_id in self.SKIP_IDS
        ):
            self._skip_depth += 1
            return

        if heading_level is not None:
            self._add_break()
            self._heading_buffer = []
            return

        if tag in self.BLOCK_TAGS:
            self._add_break()

    def handle_endtag(self, tag: str) -> None:
        # Closes parser state for skipped blocks, lists, headings, superscripts, and math spans.
        if self._reference_marker_buffer is not None:
            self._reference_marker_depth -= 1
            if self._reference_marker_depth <= 0:
                self._flush_reference_marker()
            return

        if self._skip_depth:
            self._skip_depth -= 1
            if tag == "sup" and self._reference_skip_depth:
                self._reference_skip_depth -= 1
                self._pending_reference_separator = True
            self._close_math_tag(tag)
            return

        if tag in {"ol", "ul"} and self._list_stack:
            self._list_stack.pop()

        if re.fullmatch(r"h[1-6]", tag):
            self._flush_heading()
            self._heading_level = None
            return

        if tag == "sup" and self._sup_depth:
            self._sup_depth -= 1
            return

        if tag == "sub":
            if self._sub_depth:
                self._sub_depth -= 1
                if not self._sub_depth:
                    self._pending_subscript_separator = True
            self._close_math_tag(tag)
            return

        if self._close_math_tag(tag):
            return

        if tag in self.BLOCK_TAGS:
            self._add_break()

    def handle_data(self, data: str) -> None:
        # Adds visible text nodes while preserving needed separators after refs/subscripts.
        if self._reference_marker_buffer is not None:
            self._reference_marker_buffer.append(data)
            return
        if self._skip_depth or self._skip_section_level is not None:
            return
        text = re.sub(r"\s+", " ", data)
        if not text.strip():
            if self._pending_reference_separator:
                self._append_pending_space()
                self._pending_reference_separator = False
            elif data and any(char.isspace() for char in data):
                self._pending_inline_space = True
            self._pending_subscript_separator = False
            return
        self._pending_reference_separator = False
        if self._pending_subscript_separator:
            if text[0].isalpha():
                text = "_" + text
            self._pending_subscript_separator = False
        if self._heading_buffer is not None:
            self._heading_buffer.append(text)
            return
        if self._pending_prefix:
            text = self._pending_prefix + text.lstrip()
            self._pending_prefix = ""
        self._append_inline_text(text)

    def get_text(self) -> str:
        # Returns the joined parser output with basic whitespace and punctuation cleanup.
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.strip()

    def _add_break(self) -> None:
        # Inserts a paragraph break without producing duplicate blank lines.
        self._pending_reference_separator = False
        self._pending_inline_space = False
        self._pending_subscript_separator = False
        if not self._parts:
            return
        last = self._parts[-1]
        if last.endswith("\n\n"):
            return
        if last.endswith("\n"):
            self._parts[-1] = last + "\n"
            return
        self._parts.append("\n\n")

    def _flush_heading(self) -> None:
        # Emits a collected heading in the shared == Heading == format.
        if self._heading_buffer is None:
            return
        heading = re.sub(r"\s+", " ", "".join(self._heading_buffer)).strip()
        self._heading_buffer = None
        if not heading:
            return
        self._add_break()
        self._parts.append(f"== {heading} ==")
        self._add_break()

    def _flush_reference_marker(self) -> None:
        # Emits a collected inline citation marker such as [1] for references export mode.
        if self._reference_marker_buffer is None:
            return
        marker = re.sub(r"\s+", "", "".join(self._reference_marker_buffer))
        self._reference_marker_buffer = None
        self._reference_marker_depth = 0
        if not marker:
            return
        if not marker.startswith("["):
            marker = f"[{marker}]"
        self._append_inline_text(marker)
        self._pending_reference_separator = True

    def _append_inline_marker(self, marker: str) -> None:
        # Appends generated inline markers like ^ or _ to the active output buffer.
        if self._heading_buffer is not None:
            self._heading_buffer.append(marker)
            return
        self._parts.append(marker)

    def _append_inline_text(self, text: str) -> None:
        # Appends normal text to either a heading buffer or the main output parts.
        if self._heading_buffer is not None:
            self._heading_buffer.append(text)
            return
        previous_chunk = self._parts[-1] if self._parts else ""
        previous_word = previous_chunk.strip()
        current_word = text.strip()
        if (
            self._pending_inline_space
            and not self._math_depth
            and self._parts
            and previous_chunk
            and not previous_chunk.endswith((" ", "\n"))
            and text
            and not text.startswith((" ", "\n"))
        ):
            self._parts.append(" ")
            self._pending_inline_space = False
        if (
            not self._math_depth
            and self._parts
            and text
            and previous_chunk
            and previous_chunk[-1].isalnum()
            and text[0].isalnum()
            and len(previous_word) >= 2
            and len(current_word) >= 2
        ):
            self._parts.append(" ")
        self._pending_inline_space = False
        self._parts.append(text)

    def _append_pending_space(self) -> None:
        # Restores meaningful whitespace that appears after a removed reference marker.
        target = self._heading_buffer if self._heading_buffer is not None else self._parts
        if not target:
            return
        last = target[-1]
        if last.endswith((" ", "\n")):
            return
        target.append(" ")

    def _close_math_tag(self, tag: str) -> bool:
        # Tracks when a math container closes so normal subscript behavior can resume.
        if tag in self._math_tag_stack:
            index = len(self._math_tag_stack) - 1 - self._math_tag_stack[::-1].index(tag)
            self._math_tag_stack.pop(index)
            self._math_depth -= 1
            return True
        return False


def is_non_word_glyph(value: str) -> bool:
    # Allows only symbol-like file titles, preventing descriptive image titles from leaking.
    return bool(value.strip()) and not re.search(r"\w", value)


class WikipediaReferencesParser(HTMLParser):
    """Extract numbered citation text from Wikipedia reference lists."""

    SKIP_CLASSES = {"mw-cite-backlink", "mw-editsection", "noprint"}
    BLOCK_TAGS = {"br", "div", "li", "p"}

    def __init__(self) -> None:
        # Initializes parser state for collecting only numeric reference-list entries.
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self._in_reference_list = False
        self._reference_list_depth = 0
        self._li_depth = 0
        self._skip_depth = 0
        self._current_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Enters numeric References lists and skips backlink/editing noise inside citations.
        attr_map = {name: value or "" for name, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if self._skip_depth:
            self._skip_depth += 1
            return

        if tag in {"script", "style"} or classes.intersection(self.SKIP_CLASSES):
            self._skip_depth += 1
            return

        if tag == "ol" and "references" in classes:
            group = attr_map.get("data-mw-group", "")
            if group != "lower-alpha":
                self._in_reference_list = True
                self._reference_list_depth = 1
                return

        if self._in_reference_list and tag == "ol":
            self._reference_list_depth += 1

        if self._in_reference_list and tag == "li" and self._li_depth == 0:
            self._li_depth = 1
            self._current_parts = []
            return

        if self._li_depth:
            if tag == "li":
                self._li_depth += 1
            if tag in self.BLOCK_TAGS:
                self._append_reference_text(" ")

    def handle_endtag(self, tag: str) -> None:
        # Leaves citation list/items and flushes completed references into numbered text.
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if self._li_depth and tag == "li":
            self._li_depth -= 1
            if self._li_depth == 0:
                self._flush_reference()
            return

        if self._in_reference_list and tag == "ol":
            self._reference_list_depth -= 1
            if self._reference_list_depth <= 0:
                self._in_reference_list = False

        if self._li_depth and tag in self.BLOCK_TAGS:
            self._append_reference_text(" ")

    def handle_data(self, data: str) -> None:
        # Collects visible citation text while inside a numeric reference item.
        if self._skip_depth or not self._li_depth:
            return
        self._append_reference_text(data)

    def _append_reference_text(self, text: str) -> None:
        # Adds raw citation fragments before final reference text normalization.
        if self._current_parts is not None:
            self._current_parts.append(text)

    def _flush_reference(self) -> None:
        # Normalizes one completed citation and assigns its output number.
        if self._current_parts is None:
            return
        text = clean_reference_text("".join(self._current_parts))
        self._current_parts = None
        if text:
            self.references.append(f"{len(self.references) + 1}. {text}")


def clean_reference_text(text: str) -> str:
    # Collapses citation whitespace and fixes punctuation spacing.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def page_request_from_url(url: str) -> PageRequest:
    # Converts a Wikipedia URL into the internal title/language request object.
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
    # Fetches rendered article HTML through the Wikipedia parse API.
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
    # Fetches plain extract text through the Wikipedia extracts API.
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
    # Cleans LaTeX spacing and normalizes text macros for exported math.
    latex = re.sub(r"\s+", " ", latex)
    latex = latex.replace(r"\text{", r"\mathrm{")
    return latex.strip().rstrip(",.;:")


def repair_compact_power_notation(text: str) -> str:
    # Repairs compact exponent text like 8 (=23) into 8 (=2^3) when inferable.
    def replace(match: re.Match[str]) -> str:
        # Tests possible base/exponent splits and returns the first exact match.
        value = int(match.group("value"))
        compact = match.group("compact")
        for split_at in range(1, len(compact)):
            base = int(compact[:split_at])
            exponent = int(compact[split_at:])
            if exponent > 1 and base**exponent == value:
                return f"{value} (={base}^{exponent})"
        return match.group(0)

    return POWER_HINT_PATTERN.sub(replace, text)


def lower_alpha_label(index: int) -> str:
    # Converts 1-based indexes into lower-alpha labels: a, b, ..., z, aa.
    label = ""
    while index > 0:
        index -= 1
        label = chr(ord("a") + index % 26) + label
        index //= 26
    return label


def remove_unwanted_sections(text: str) -> str:
    # Drops unwanted terminal sections while keeping Notes available for HTML output.
    lines = text.splitlines()
    kept: list[str] = []
    skip_level: int | None = None
    skip_plain_section = False

    for line in lines:
        heading = SECTION_HEADING_PATTERN.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip().casefold()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            skip_plain_section = False
            if title in REMOVED_SECTION_TITLES:
                skip_level = level
                continue

        plain_title = line.strip().casefold()
        if plain_title in PLAIN_SECTION_TITLES:
            skip_plain_section = plain_title in REMOVED_SECTION_TITLES
            if not skip_plain_section:
                kept.append(line)
            continue

        if skip_level is None and not skip_plain_section:
            kept.append(line)

    return "\n".join(kept)


def clean_leading_caret_markers(text: str) -> str:
    # Removes leading backlink markers such as ^, ^a, or ^a^b from note lines.
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("^"):
            cleaned_lines.append(line)
            continue

        index = 1
        while index < len(stripped):
            while index < len(stripped) and stripped[index].isspace():
                index += 1
            if index < len(stripped) and stripped[index] == "^":
                index += 1
                continue
            if index < len(stripped) and stripped[index].isalnum():
                next_char = stripped[index + 1] if index + 1 < len(stripped) else ""
                if not next_char or next_char.isspace() or next_char == "^" or next_char.isupper():
                    index += 1
                    continue
            break
        cleaned_lines.append(stripped[index:].lstrip())
    return "\n".join(cleaned_lines)


def normalize_section_headings(text: str) -> str:
    # Normalizes any wiki heading level into the shared == Heading == output format.
    lines: list[str] = []
    for line in text.splitlines():
        heading = SECTION_HEADING_PATTERN.match(line)
        if heading:
            lines.append(f"== {heading.group(2).strip()} ==")
        else:
            lines.append(line)
    return "\n".join(lines)


def section_title_from_line(line: str) -> str:
    # Extracts a casefolded section title from either formatted or plain heading text.
    heading = SECTION_HEADING_PATTERN.match(line)
    if heading:
        return heading.group(2).strip().casefold()
    return line.strip().casefold()


def format_heading_spacing(text: str) -> str:
    # Ensures formatted headings have one blank line above for readability.
    lines = text.splitlines()
    formatted: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"==\s+.+?\s+==", stripped):
            while formatted and formatted[-1] == "":
                formatted.pop()
            if formatted:
                formatted.append("")
            formatted.append(stripped)
            continue
        formatted.append(line)
    return "\n".join(formatted)


def paragraph_looks_like_math_fragment(paragraph: str) -> bool:
    # Detects rendered math fragments that should disappear in remove/latex modes.
    stripped = paragraph.strip()
    if not stripped:
        return False
    if SECTION_HEADING_PATTERN.match(stripped):
        return False
    if len(stripped) <= 2 and re.match(r"^[A-Za-z0-9]+$", stripped):
        return True
    if len(stripped) <= 40 and "\n" in stripped and re.match(r"^[A-Za-z0-9\s.]+$", stripped):
        return True
    if stripped == "a constant.":
        return True
    if len(re.findall(r"\b[A-Za-z]{3,}\b", stripped)) >= 2:
        return False
    if len(re.findall(r"[A-Za-z]{2,}", stripped)) >= 4:
        return False
    if len(stripped) > 120:
        return False
    if not MATH_FRAGMENT_PATTERN.match(stripped):
        return False
    return bool(MATH_SYMBOL_PATTERN.search(stripped))


def line_looks_like_rendered_math(line: str) -> bool:
    # Detects short rendered-math lines used to clean duplicated LaTeX context.
    stripped = line.strip().rstrip(",.;:")
    if not stripped:
        return True
    if stripped == "a constant":
        return True
    if len(stripped) <= 2 and re.match(r"^[A-Za-z0-9]+$", stripped):
        return True
    if re.match(r"^[+\-–−=∝×*/^(),.{}\\]+$", stripped):
        return True
    if len(stripped) <= 30 and MATH_SYMBOL_PATTERN.search(stripped):
        return True
    if len(stripped) <= 80 and " " not in stripped and re.search(r"[A-Za-z]\(", stripped):
        return True
    math_indicators = 0
    if any(char in stripped for char in "=∝×÷√∑∫≤≥≈≠−|"):
        math_indicators += 1
    if GREEK_LETTER_PATTERN.search(stripped):
        math_indicators += 1
    if "\u2061" in stripped or RENDERED_MATH_FUNCTION_PATTERN.search(stripped):
        math_indicators += 1
    if math_indicators and len(stripped) <= 180:
        return True
    return False


def remove_rendered_math_before_latex(text: str) -> str:
    # Drops rendered fallback equation lines that immediately duplicate following LaTeX.
    lines: list[str] = []
    for line in text.splitlines():
        if (
            LATEX_LINE_PATTERN.match(line)
            and lines
            and line_looks_like_rendered_math(lines[-1])
        ):
            lines.pop()
            while lines and not lines[-1].strip():
                lines.pop()
        lines.append(line)
    return "\n".join(lines)


def join_inline_latex_lines(text: str) -> str:
    # Rejoins short inline LaTeX lines that HTML math extraction split from prose.
    lines = text.splitlines()
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            LATEX_LINE_PATTERN.match(line)
            and joined
            and joined[-1].strip()
            and index + 1 < len(lines)
            and lines[index + 1].strip()
            and not SECTION_HEADING_PATTERN.match(lines[index + 1].strip())
        ):
            joined[-1] = f"{joined[-1].rstrip()} {line.strip()} {lines[index + 1].lstrip()}"
            index += 2
            continue
        if (
            SIMPLE_LATEX_LINE_PATTERN.match(line)
            and joined
            and joined[-1].strip()
            and not SECTION_HEADING_PATTERN.match(joined[-1].strip())
        ):
            joined[-1] = f"{joined[-1].rstrip()} {line.strip()}"
            index += 1
            continue
        joined.append(line)
        index += 1
    return "\n".join(joined)


def remove_inline_duplicate_latex_symbols(text: str) -> str:
    # Removes rendered one-letter symbols that duplicate adjacent inline LaTeX.
    text = MERGED_WORD_LATEX_SYMBOL_PATTERN.sub(r"\1 $\2$", text)
    text = INLINE_DUPLICATE_LATEX_FUNCTION_PATTERN.sub(r"$\1$", text)

    def replace_duplicate(match: re.Match[str]) -> str:
        identifier = match.group(1)
        if (
            len(identifier) == 1
            or len(identifier) <= 3
            or re.search(r"[\d_]", identifier)
            or identifier.casefold() in MATH_IDENTIFIER_ALLOWLIST
        ):
            return f"${identifier}$"
        return match.group(0)

    text = INLINE_DUPLICATE_LATEX_SYMBOL_PATTERN.sub(replace_duplicate, text)
    return re.sub(r"(\$[A-Za-z][A-Za-z0-9_]*\$)\s+([\"”])", r"\1\2", text)


def clean_latex_context_segment(segment: str) -> str:
    # Keeps prose around LaTeX formulas while dropping duplicated rendered math lines.
    lines: list[str] = []
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            prefix, suffix = stripped.split(":", 1)
            if suffix and line_looks_like_rendered_math(suffix):
                lines.append(f"{prefix}:")
                continue
        if not line_looks_like_rendered_math(stripped):
            lines.append(stripped)
    return "\n".join(lines)


def replace_displaystyle_latex(paragraph: str, math_mode: str) -> tuple[str, bool]:
    # Replaces MediaWiki displaystyle blocks with clean LaTeX or removes them.
    parts: list[str] = []
    marker = r"{\displaystyle"
    start = 0
    found = False
    while True:
        marker_index = paragraph.find(marker, start)
        if marker_index == -1:
            segment = paragraph[start:]
            parts.append(
                clean_latex_context_segment(segment)
                if found and math_mode == "latex"
                else segment
            )
            return "".join(parts), found

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
            parts.append(paragraph[start:])
            return "".join(parts), found

        found = True
        segment = paragraph[start:marker_index]
        if math_mode == "latex":
            context = clean_latex_context_segment(segment)
            if context:
                parts.append(context)
                parts.append("\n")
        latex = paragraph[marker_index + len(marker) : end_index]
        if math_mode == "latex":
            parts.append(f"${normalize_latex(latex)}$")
            parts.append("\n")
        start = end_index + 1


def clean_math(text: str, math_mode: str) -> str:
    # Applies the selected math policy: remove, latex, or keep.
    if math_mode not in {"remove", "latex", "keep"}:
        raise ValueError(f"Unsupported math mode: {math_mode}")
    if math_mode == "keep":
        return text

    paragraphs = re.split(r"\n{2,}", text)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        paragraph, had_latex = replace_displaystyle_latex(paragraph, math_mode)
        if had_latex and not paragraph.strip():
            continue
        if paragraph_looks_like_math_fragment(paragraph):
            continue
        cleaned.append(paragraph)
    text = "\n\n".join(cleaned)
    if math_mode == "latex":
        text = remove_rendered_math_before_latex(text)
        text = join_inline_latex_lines(text)
        text = remove_inline_duplicate_latex_symbols(text)
    return text


def clean_plain_text(
    text: str, math_mode: str = "remove", remove_inline_references: bool = True
) -> str:
    # Runs shared text cleanup used by both extracts and HTML paths.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_unwanted_sections(text)
    text = clean_leading_caret_markers(text)
    if remove_inline_references:
        text = INLINE_REFERENCE_PATTERN.sub("", text)
    text = normalize_section_headings(text)
    text = clean_math(text, math_mode)
    text = repair_compact_power_notation(text)
    text = EMPTY_PARENTHESES_PATTERN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = format_heading_spacing(text)
    return text.strip()


def clean_wikipedia_html(html: str, math_mode: str = "remove") -> str:
    # Converts Wikipedia HTML to clean article text without inline citation numbers.
    parser = WikipediaTextParser()
    parser.feed(html)
    parser.close()
    return clean_plain_text(parser.get_text(), math_mode)


def extract_references_from_html(html: str) -> str:
    # Extracts numeric References entries into a separate formatted section.
    parser = WikipediaReferencesParser()
    parser.feed(html)
    parser.close()
    if not parser.references:
        return ""
    return "== References ==\n\n" + "\n\n".join(parser.references)


def clean_wikipedia_html_with_references(
    html: str, math_mode: str = "remove", include_inline_markers: bool = True
) -> str:
    # Builds an HTML-derived article copy with appended References.
    if include_inline_markers:
        parser = WikipediaTextParser(include_reference_markers=True)
        parser.feed(html)
        parser.close()
        body = clean_plain_text(
            parser.get_text(), math_mode, remove_inline_references=False
        )
    else:
        body = clean_wikipedia_html(html, math_mode)
    references = extract_references_from_html(html)
    return "\n\n".join(part for part in (body, references) if part).strip()


def normalize_for_match(text: str) -> str:
    # Builds a forgiving comparison string for pasted browser text and cleaned output.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INLINE_REFERENCE_PATTERN.sub("", text)
    text = normalize_math_for_match(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def simplify_math_for_match(value: str) -> str:
    # Reduces LaTeX/rendered math to comparable identifier-like text for fuzzy matching.
    value = re.sub(r"\\(?:displaystyle|mathrm|text|operatorname)\s*", " ", value)
    value = re.sub(r"\\(?:frac|begin|end|sum|log|Pr|mid|left|right)\b", " ", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = re.sub(r"[_^{}$\\]", " ", value)
    value = re.sub(r"[^0-9A-Za-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def replace_displaystyle_for_match(text: str) -> str:
    # Converts MediaWiki displaystyle fragments in pasted text into comparable math tokens.
    marker = r"{\displaystyle"
    parts: list[str] = []
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index == -1:
            parts.append(text[start:])
            return "".join(parts)

        parts.append(text[start:marker_index])
        depth = 0
        end_index = None
        for index in range(marker_index, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_index = index
                    break
        if end_index is None:
            parts.append(text[marker_index:])
            return "".join(parts)

        latex = text[marker_index + len(marker) : end_index]
        simplified = simplify_math_for_match(latex)
        if simplified:
            parts.append(f" {simplified} ")
        start = end_index + 1


def normalize_math_for_match(text: str) -> str:
    # Makes copied rendered math and cleaned LaTeX close enough for boundary matching.
    text = replace_displaystyle_for_match(text)
    text = re.sub(
        r"\$([^$]+)\$",
        lambda match: f" {simplify_math_for_match(match.group(1))} ",
        text,
    )
    text = re.sub(r"\b([A-Za-z][A-Za-z0-9_]*)\s+\1\b", r"\1", text)
    return text


def normalize_match_index_map(text: str) -> tuple[str, list[int]]:
    # Normalizes text while keeping a map back to original character offsets.
    normalized: list[str] = []
    index_map: list[int] = []
    previous_was_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and not previous_was_space:
                normalized.append(" ")
                index_map.append(index)
            previous_was_space = True
            continue
        for folded_char in char.casefold():
            normalized.append(folded_char)
            index_map.append(index)
        previous_was_space = False
    return "".join(normalized), index_map


def line_looks_like_partial_noise(line: str) -> bool:
    # Flags copied edge lines that commonly come from tables, captions, maps, or controls.
    stripped = line.strip()
    if not stripped:
        return True
    letters = sum(1 for char in stripped if char.isalpha())
    visible = sum(1 for char in stripped if not char.isspace())
    if visible and letters / visible < 0.30:
        return True
    lower = stripped.casefold()
    if lower in {"edit", "source", "map", "image"}:
        return True
    if len(stripped) < 40 and re.fullmatch(r"[\W\d_]+", stripped):
        return True
    return False


def line_looks_like_math_heavy(line: str) -> bool:
    # Detects equation-heavy copied lines so prose can be preferred for fuzzy anchors.
    stripped = line.strip()
    if not stripped:
        return False
    math_symbols = sum(1 for char in stripped if char in "+-=*/^_{}\\∝×÷√∑∫≤≥≈≠−")
    letters = sum(1 for char in stripped if char.isalpha())
    visible = sum(1 for char in stripped if not char.isspace())
    if visible == 0:
        return False
    if math_symbols >= 2 and math_symbols / visible >= 0.12:
        return True
    if visible >= 3 and letters / visible < 0.35 and math_symbols >= 1:
        return True
    if re.search(r"\b[a-zA-Z]\s*[=∝]\s*[A-Za-z0-9]", stripped):
        return True
    return False


def meaningful_partial_units(text: str) -> list[str]:
    # Keeps paragraph-like pasted chunks that are useful as fuzzy anchors.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", text)
    prose_units: list[str] = []
    math_units: list[str] = []
    for paragraph in paragraphs:
        lines = [
            line.strip()
            for line in paragraph.splitlines()
            if not line_looks_like_partial_noise(line)
        ]
        if not lines:
            continue
        prose_lines = [line for line in lines if not line_looks_like_math_heavy(line)]
        unit = " ".join(prose_lines or lines)
        if len(normalize_for_match(unit)) >= 8:
            if prose_lines:
                prose_units.append(unit)
            else:
                math_units.append(unit)
    if prose_units:
        return prose_units
    if math_units:
        return math_units

    fallback = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return [fallback] if fallback else []


def strong_anchor_candidates(
    pasted_text: str,
    side: str,
    anchor_size: int = 300,
    max_candidates: int = 5,
) -> list[str]:
    # Takes several strong start/end anchors so noisy copied edges can be skipped.
    units = meaningful_partial_units(pasted_text)
    if side == "end":
        units = list(reversed(units))
    candidates: list[str] = []
    for unit in units:
        normalized = normalize_for_match(unit)
        if not normalized:
            continue
        lengths = [180, 120, 240, anchor_size] if side == "start" else [anchor_size, 240, 180, 120]
        for length in lengths:
            length = min(length, len(unit))
            if length < 80 and len(unit) >= 80:
                continue
            anchor = unit[:length] if side == "start" else unit[-length:]
            if normalize_for_match(anchor) and anchor not in candidates:
                candidates.append(anchor)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        stripped = pasted_text.strip()
        anchor = stripped[:anchor_size] if side == "start" else stripped[-anchor_size:]
        if anchor:
            candidates.append(anchor)
    return candidates


def fuzzy_ratio(left: str, right: str) -> float:
    # Uses the standard library to avoid adding a dependency for boundary matching.
    return difflib.SequenceMatcher(None, left, right).ratio()


def required_partial_anchor_score(anchor: str, threshold: float) -> float:
    # Requires stricter matches for very short anchors because they are less unique.
    if len(normalize_for_match(anchor)) < 14:
        return max(threshold, 0.95)
    tail_tokens = re.findall(r"\b[A-Za-z]\b", anchor[-140:])
    if (
        line_looks_like_math_heavy(anchor)
        or "{\\displaystyle" in anchor
        or "$" in anchor
        or len(tail_tokens) >= 5
    ):
        return max(0.87, threshold - 0.05)
    return threshold


def original_span_from_normalized(
    index_map: list[int], normalized_start: int, normalized_end: int
) -> tuple[int, int]:
    # Converts a normalized match span back to offsets in the cleaned article text.
    if not index_map:
        return 0, 0
    normalized_start = max(0, min(normalized_start, len(index_map) - 1))
    normalized_end = max(normalized_start + 1, min(normalized_end, len(index_map)))
    return index_map[normalized_start], index_map[normalized_end - 1] + 1


def find_fuzzy_boundary(full_text: str, anchor: str) -> FuzzyBoundaryMatch | None:
    # Finds the best approximate location of one anchor in the cleaned article text.
    normalized_anchor = normalize_for_match(anchor)
    if not normalized_anchor:
        return None

    normalized_full, index_map = normalize_match_index_map(full_text)
    exact_start = normalized_full.find(normalized_anchor)
    if exact_start >= 0:
        exact_end = exact_start + len(normalized_anchor)
        start, end = original_span_from_normalized(index_map, exact_start, exact_end)
        return FuzzyBoundaryMatch(anchor, 1.0, start, end)

    base_size = len(normalized_anchor)
    deltas = (-80, -40, 0, 40, 80)
    window_sizes = sorted(
        {
            max(40, base_size + delta)
            for delta in deltas
            if max(40, base_size + delta) <= len(normalized_full)
        }
    )
    if not window_sizes:
        return None

    step = max(15, min(80, base_size // 4 or 15))
    best_score = -1.0
    best_span = (0, 0)
    for window_size in window_sizes:
        limit = len(normalized_full) - window_size
        for start_index in range(0, limit + 1, step):
            window = normalized_full[start_index : start_index + window_size]
            score = fuzzy_ratio(normalized_anchor, window)
            if score > best_score:
                best_score = score
                best_span = (start_index, start_index + window_size)
        if limit > 0:
            window = normalized_full[limit : limit + window_size]
            score = fuzzy_ratio(normalized_anchor, window)
            if score > best_score:
                best_score = score
                best_span = (limit, limit + window_size)

    start, end = original_span_from_normalized(index_map, best_span[0], best_span[1])
    return FuzzyBoundaryMatch(anchor, best_score, start, end)


def shorten_report_anchor(anchor: str, limit: int = 180) -> str:
    # Keeps debug reports readable even when anchors are long.
    anchor = re.sub(r"\s+", " ", anchor).strip()
    if len(anchor) <= limit:
        return anchor
    return anchor[: limit - 3] + "..."


def format_partial_match_report(
    result: PartialExtractionResult | None,
    threshold: float,
    message: str = "success",
    start_candidates: list[str] | None = None,
    end_candidates: list[str] | None = None,
) -> str:
    # Writes a human-readable report so failed fuzzy matches can be debugged quickly.
    lines = [
        "Wikipedia Partial HTML Extraction Match Report",
        "",
        f"Status: {message}",
        f"Threshold: {threshold:.3f}",
    ]
    if result:
        lines.extend(
            [
                f"Start score: {result.start_match.score:.3f}",
                f"End score: {result.end_match.score:.3f}",
                f"Output characters: {len(result.text)}",
                "",
                f"Start anchor: {shorten_report_anchor(result.start_match.anchor)}",
                f"End anchor: {shorten_report_anchor(result.end_match.anchor)}",
            ]
        )
    if start_candidates is not None:
        lines.extend(["", "Start candidates:"])
        lines.extend(f"- {shorten_report_anchor(candidate)}" for candidate in start_candidates)
    if end_candidates is not None:
        lines.extend(["", "End candidates:"])
        lines.extend(f"- {shorten_report_anchor(candidate)}" for candidate in end_candidates)
    return "\n".join(lines).strip()


def extract_partial_text(
    full_text: str,
    pasted_text: str,
    threshold: float = 0.92,
    anchor_size: int = 300,
    max_candidates: int = 5,
) -> PartialExtractionResult:
    # Extracts a clean section by matching pasted start/end anchors against clean HTML text.
    start_candidates = strong_anchor_candidates(
        pasted_text, "start", anchor_size, max_candidates
    )
    end_candidates = strong_anchor_candidates(
        pasted_text, "end", anchor_size, max_candidates
    )
    start_matches = [
        (index, match)
        for index, anchor in enumerate(start_candidates)
        if (match := find_fuzzy_boundary(full_text, anchor))
    ]
    end_matches = [
        (index, match)
        for index, anchor in enumerate(end_candidates)
        if (match := find_fuzzy_boundary(full_text, anchor))
    ]

    valid_pairs: list[
        tuple[int, int, float, int, FuzzyBoundaryMatch, FuzzyBoundaryMatch]
    ] = []
    for start_index, start_match in start_matches:
        for end_index, end_match in end_matches:
            if start_match.start > end_match.end:
                continue
            start_required_score = required_partial_anchor_score(
                start_match.anchor, threshold
            )
            end_required_score = required_partial_anchor_score(end_match.anchor, threshold)
            if (
                start_match.score < start_required_score
                or end_match.score < end_required_score
            ):
                continue
            average_score = (start_match.score + end_match.score) / 2
            extracted_length = end_match.end - start_match.start
            valid_pairs.append(
                (
                    -start_index,
                    -end_index,
                    average_score,
                    extracted_length,
                    start_match,
                    end_match,
                )
            )

    if not valid_pairs:
        best_start = max((match.score for _, match in start_matches), default=0.0)
        best_end = max((match.score for _, match in end_matches), default=0.0)
        message = (
            "failed: could not find reliable start/end boundaries "
            f"(best start {best_start:.3f}, best end {best_end:.3f})"
        )
        report_text = format_partial_match_report(
            None, threshold, message, start_candidates, end_candidates
        )
        raise PartialExtractionError(message, report_text)

    _, _, _, _, start_match, end_match = max(
        valid_pairs, key=lambda item: (item[0], item[1], item[2], item[3])
    )
    text = full_text[start_match.start : end_match.end].strip()
    return PartialExtractionResult(text, start_match, end_match, threshold)


def sentence_spans(text: str) -> list[TextSentence]:
    # Splits text into sentence-like units while preserving offsets for slicing.
    citation_tail = r"(?:\s*\[\d+(?:\s*[,\u2013-]\s*\d+)*\])*"
    pattern = re.compile(rf"[^.!?\n]+(?:[.!?]+{citation_tail}[\"')\]]*)?", re.MULTILINE)
    sentences: list[TextSentence] = []
    for match in pattern.finditer(text):
        sentence = re.sub(r"\s+", " ", match.group(0)).strip()
        if not sentence or SECTION_HEADING_PATTERN.match(sentence):
            continue
        if len(normalize_for_match(sentence)) < 8:
            continue
        if sentences and SENTENCE_CONTINUATION_ABBREVIATION_PATTERN.search(sentences[-1].text):
            previous = sentences.pop()
            joined = re.sub(r"\s+", " ", text[previous.start : match.end()]).strip()
            sentences.append(TextSentence(joined, previous.start, match.end()))
            continue
        sentences.append(TextSentence(sentence, match.start(), match.end()))
    return sentences


def extract_heading_spans(text: str) -> list[TextHeading]:
    # Finds normalized == Heading == lines in cleaned text.
    headings: list[TextHeading] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = SECTION_HEADING_PATTERN.match(stripped)
        if match:
            start = offset + line.find(stripped)
            headings.append(TextHeading(match.group(2).strip(), start, start + len(stripped)))
        offset += len(line)
    return headings


def sentence_index_at(sentences: list[TextSentence], char_index: int) -> int:
    # Finds the sentence containing a character offset, or the nearest previous sentence.
    for index, sentence in enumerate(sentences):
        if sentence.start <= char_index < sentence.end:
            return index
        if char_index < sentence.start:
            return max(0, index - 1)
    return max(0, len(sentences) - 1)


def build_hybrid_text_index(text: str) -> HybridTextIndex:
    # Builds headings, sentences, citations, and References boundary for hybrid matching.
    headings = extract_heading_spans(text)
    sentences = sentence_spans(text)
    references_start = next(
        (heading.start for heading in headings if heading.title.casefold() == "references"),
        None,
    )
    citations: list[CitationOccurrence] = []
    body_limit = references_start if references_start is not None else len(text)
    for match in INLINE_REFERENCE_PATTERN.finditer(text[:body_limit]):
        citations.append(
            CitationOccurrence(
                match.group(0).strip("[]"),
                match.start(),
                match.end(),
                sentence_index_at(sentences, match.start()),
            )
        )
    return HybridTextIndex(text, headings, sentences, citations, references_start)


def copied_heading_candidates(copied_text: str, clean_headings: list[TextHeading]) -> list[TextHeading]:
    # Detects copied heading lines by comparing them to known clean headings.
    candidates: list[TextHeading] = []
    clean_by_norm = {normalize_for_match(heading.title): heading for heading in clean_headings}
    offset = 0
    for line in copied_text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True):
        stripped = line.strip(" =\t\r\n")
        normalized = normalize_for_match(stripped)
        if normalized in clean_by_norm:
            start = offset + line.find(stripped)
            candidates.append(TextHeading(stripped, start, start + len(stripped)))
        offset += len(line)
    return candidates


def citation_numbers(text: str) -> list[str]:
    # Extracts inline citation numbers from copied or cleaned text.
    return [match.group(0).strip("[]") for match in INLINE_REFERENCE_PATTERN.finditer(text)]


def citation_sequence_matches(
    citations: list[CitationOccurrence], sequence: list[str]
) -> list[tuple[CitationOccurrence, CitationOccurrence]]:
    # Finds ordered contiguous citation-number sequences in the clean citation index.
    if not sequence:
        return []
    matches: list[tuple[CitationOccurrence, CitationOccurrence]] = []
    size = len(sequence)
    for index in range(0, len(citations) - size + 1):
        if [citation.number for citation in citations[index : index + size]] == sequence:
            matches.append((citations[index], citations[index + size - 1]))
    return matches


def text_tokens(text: str) -> set[str]:
    # Tokenizes normalized text for cheap overlap filtering.
    return set(re.findall(r"[a-z0-9]+", normalize_for_match(text)))


def token_overlap_score(left: str, right: str) -> float:
    # Scores token overlap without caring about exact punctuation or spacing.
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def hybrid_window_score(left: str, right: str) -> float:
    # Blends token overlap and character sequence similarity.
    left_normalized = normalize_for_match(left)
    right_normalized = normalize_for_match(right)
    if not left_normalized or not right_normalized:
        return 0.0
    token_score = token_overlap_score(left, right)
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if left_tokens and right_tokens:
        token_score = max(
            token_score,
            len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)),
        )
    char_score = fuzzy_ratio(left_normalized, right_normalized)
    shorter, longer = sorted((left_normalized, right_normalized), key=len)
    if len(shorter) >= 50 and longer.startswith(shorter):
        char_score = max(char_score, 0.95)
    return 0.60 * token_score + 0.40 * char_score


def sentence_window_chunks(
    sentences: list[TextSentence], reverse: bool = False, window_size: int = 3
) -> list[tuple[int, int, str]]:
    # Builds copied sentence chunks with step 3 and a final backfilled chunk.
    if window_size < 1 or len(sentences) < window_size:
        return []
    starts = list(range(0, len(sentences) - window_size + 1, 3))
    final_start = max(0, len(sentences) - window_size)
    if final_start not in starts:
        starts.append(final_start)
    if reverse:
        starts = list(reversed(starts))
    return [
        (
            start,
            start + window_size,
            " ".join(sentence.text for sentence in sentences[start : start + window_size]),
        )
        for start in starts
    ]


def find_sentence_window_match(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    clean_start: int,
    clean_end: int,
    reverse: bool = False,
    threshold: float = 0.84,
) -> tuple[int, int, int, int, float] | None:
    # Finds the first copied 3-sentence chunk that has a strong clean sliding-window match.
    clean_indexes = [
        index
        for index, sentence in enumerate(clean_sentences)
        if sentence.start >= clean_start and sentence.end <= clean_end
    ]
    if not clean_indexes:
        return None
    window_size = 3 if len(clean_indexes) >= 3 else len(clean_indexes)
    if len(copied_sentences) < window_size:
        return None
    clean_window_starts = clean_indexes[: len(clean_indexes) - window_size + 1]
    if reverse:
        clean_window_starts = list(reversed(clean_window_starts))
    for _copied_start, _copied_end, copied_text in sentence_window_chunks(
        copied_sentences, reverse, window_size
    ):
        best: tuple[int, int, int, int, float] | None = None
        for clean_index in clean_window_starts:
            window = clean_sentences[clean_index : clean_index + window_size]
            clean_text = " ".join(sentence.text for sentence in window)
            if token_overlap_score(copied_text, clean_text) < 0.30:
                continue
            score = hybrid_window_score(copied_text, clean_text)
            if score >= threshold and (best is None or score > best[4]):
                best = (_copied_start, _copied_end, clean_index, clean_index + window_size, score)
        if best is not None:
            return best
    return None


def refine_sentence_start(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    copied_anchor_start: int,
    clean_anchor_start: int,
    threshold: float = 0.88,
) -> int:
    # Expands a confirmed start anchor backward one sentence at a time until two failures.
    copied_index = copied_anchor_start - 1
    clean_index = clean_anchor_start - 1
    failures = 0
    start_index = clean_anchor_start
    while copied_index >= 0 and clean_index >= 0 and failures < 2:
        score = hybrid_window_score(copied_sentences[copied_index].text, clean_sentences[clean_index].text)
        if score >= threshold:
            start_index = clean_index
            failures = 0
        else:
            failures += 1
        copied_index -= 1
        clean_index -= 1
    return start_index


def refine_sentence_end(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    copied_anchor_end: int,
    clean_anchor_end: int,
    threshold: float = 0.88,
) -> int:
    # Expands a confirmed end anchor forward one sentence at a time until two failures.
    copied_index = copied_anchor_end
    clean_index = clean_anchor_end
    failures = 0
    end_index = clean_anchor_end
    while copied_index < len(copied_sentences) and clean_index < len(clean_sentences) and failures < 2:
        score = hybrid_window_score(copied_sentences[copied_index].text, clean_sentences[clean_index].text)
        if score >= threshold:
            end_index = clean_index + 1
            failures = 0
        else:
            failures += 1
        copied_index += 1
        clean_index += 1
    return end_index


def heading_position_matches(
    copied_headings: list[TextHeading], clean_headings: list[TextHeading]
) -> list[TextHeading]:
    # Maps copied heading titles to clean heading positions while preserving copied order.
    clean_by_norm = {normalize_for_match(heading.title): heading for heading in clean_headings}
    return [
        clean_by_norm[normalize_for_match(heading.title)]
        for heading in copied_headings
        if normalize_for_match(heading.title) in clean_by_norm
    ]


def references_heading_in_text(text: str) -> bool:
    # Detects whether the pasted input itself reached a References heading.
    return any(
        normalize_for_match(line.strip(" =\t")) == "references"
        for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    )


def next_heading_start(headings: list[TextHeading], heading: TextHeading, default: int) -> int:
    # Returns the next clean heading start after a matched heading.
    for candidate in headings:
        if candidate.start > heading.start:
            return candidate.start
    return default


def normalize_copied_text_for_hybrid_sentences(
    copied_text: str, clean_headings: list[TextHeading]
) -> str:
    # Removes structural/noisy copied lines before sentence-window matching.
    heading_titles = {normalize_for_match(heading.title) for heading in clean_headings}
    kept_lines: list[str] = []
    for line in copied_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        normalized = normalize_for_match(stripped.strip(" =\t"))
        if not stripped:
            kept_lines.append(line)
            continue
        if normalized in heading_titles:
            continue
        if re.match(r"^(?:main article|see also)\s*:", stripped, flags=re.IGNORECASE):
            continue
        if line_looks_like_partial_noise(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def extract_partial_hybrid_text(
    clean_text: str,
    copied_text: str,
    threshold: float = 0.84,
) -> HybridExtractionResult:
    # Uses heading/citation structural anchors plus sentence-window matching to slice clean text.
    clean_index = build_hybrid_text_index(clean_text)
    copied_headings = copied_heading_candidates(copied_text, clean_index.headings)
    matched_headings = heading_position_matches(copied_headings, clean_index.headings)
    copied_clean = normalize_copied_text_for_hybrid_sentences(copied_text, clean_index.headings)
    copied_sentences = sentence_spans(copied_clean)
    copied_citations = citation_numbers(copied_text)
    body_end = clean_index.references_start if clean_index.references_start is not None else len(clean_text)

    report_lines = [
        "Wikipedia Hybrid Partial Extraction Report",
        "",
        f"Copied headings: {', '.join(h.title for h in copied_headings) or 'none'}",
        f"Matched headings: {', '.join(h.title for h in matched_headings) or 'none'}",
        f"Copied citations: {', '.join(copied_citations) or 'none'}",
    ]

    coarse_start = 0
    coarse_end = body_end
    confidence = "medium"
    first_heading = matched_headings[0] if matched_headings else None
    last_heading = matched_headings[-1] if matched_headings else None
    start_anchor_locked = False
    if matched_headings:
        coarse_start = first_heading.start
        coarse_end = last_heading.end
        confidence = "high"

    if copied_citations:
        if len(copied_citations) >= 6:
            start_sequence = copied_citations[:3]
            end_sequence = copied_citations[-3:]
        else:
            start_sequence = copied_citations
            end_sequence = copied_citations
        start_matches = citation_sequence_matches(clean_index.citations, start_sequence)
        end_matches = citation_sequence_matches(clean_index.citations, end_sequence)
        if start_matches:
            start_candidates = (
                [
                    match
                    for match in start_matches
                    if first_heading is None or match[0].start < first_heading.start
                ]
                if matched_headings
                else start_matches
            )
            if start_candidates:
                citation_start = clean_index.sentences[start_candidates[0][0].sentence_index].start
                coarse_start = min(coarse_start, citation_start) if matched_headings else citation_start
                start_anchor_locked = True
                report_lines.append(f"Start citation sequence: {', '.join(start_sequence)}")
            elif matched_headings:
                report_lines.append("Start citation sequence ignored: all matches are below first heading.")
        if end_matches:
            end_candidates = (
                [
                    match
                    for match in end_matches
                    if last_heading is None or match[1].end > last_heading.end
                ]
                if matched_headings
                else end_matches
            )
            if end_candidates:
                citation_end = clean_index.sentences[end_candidates[-1][1].sentence_index].end
                coarse_end = max(coarse_end, citation_end) if matched_headings else citation_end
                report_lines.append(f"End citation sequence: {', '.join(end_sequence)}")
            elif matched_headings:
                report_lines.append("End citation sequence ignored: all matches are above last heading.")

    copied_has_references_heading = references_heading_in_text(copied_text)
    if copied_has_references_heading:
        coarse_end = body_end
        report_lines.append("References heading in copied input: yes; end set before clean References.")

    if not copied_sentences:
        message = "failed: no usable copied sentences found"
        raise PartialExtractionError(message, "\n".join(report_lines + ["", message]))

    start_search_end = first_heading.start if first_heading is not None else coarse_end
    start_search_begin = coarse_start if start_anchor_locked else max(0, coarse_start - 4000)
    start_match = find_sentence_window_match(
        copied_sentences,
        clean_index.sentences,
        start_search_begin,
        start_search_end,
        False,
        threshold,
    )

    if start_match:
        clean_start_index = refine_sentence_start(
            copied_sentences,
            clean_index.sentences,
            start_match[0],
            start_match[2],
        )
        start = clean_index.sentences[clean_start_index].start
        report_lines.append(f"Start sentence-window score: {start_match[4]:.3f}")
    else:
        start = coarse_start
        confidence = "low"
        report_lines.append("Start sentence-window match failed; used structural fallback.")

    end_search_start = last_heading.end if last_heading is not None else start
    end_search_limit = (
        next_heading_start(clean_index.headings, last_heading, min(len(clean_text), coarse_end + 4000))
        if last_heading is not None
        else min(len(clean_text), coarse_end + 4000)
    )
    end_copied_sentences = copied_sentences
    if last_heading is not None and copied_headings:
        copied_tail = copied_text[copied_headings[-1].end :]
        copied_tail_clean = normalize_copied_text_for_hybrid_sentences(
            copied_tail, clean_index.headings
        )
        tail_sentences = sentence_spans(copied_tail_clean)
        if tail_sentences:
            end_copied_sentences = tail_sentences
            report_lines.append("End copied sentence range: after last matched heading.")

    end_match = (
        None
        if copied_has_references_heading
        else find_sentence_window_match(
            end_copied_sentences,
            clean_index.sentences,
            end_search_start,
            end_search_limit,
            True,
            threshold,
        )
    )

    if copied_has_references_heading:
        end = body_end
    elif end_match:
        clean_end_index = refine_sentence_end(
            end_copied_sentences,
            clean_index.sentences,
            end_match[1],
            end_match[3],
        )
        end = clean_index.sentences[min(clean_end_index, len(clean_index.sentences)) - 1].end
        report_lines.append(f"End sentence-window score: {end_match[4]:.3f}")
    else:
        end = coarse_end
        confidence = "low"
        report_lines.append("End sentence-window match failed; used structural fallback.")

    if start >= end:
        message = "failed: hybrid boundaries are invalid"
        raise PartialExtractionError(message, "\n".join(report_lines + ["", message]))

    report_lines.extend(
        [
            f"Confidence: {confidence}",
            f"Start offset: {start}",
            f"End offset: {end}",
            f"Output characters: {end - start}",
        ]
    )
    return HybridExtractionResult(clean_text[start:end].strip(), "\n".join(report_lines), start, end, confidence)


def extract_note_section(text: str) -> str:
    # Returns the Note/Notes section text if present in formatted or plain form.
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if section_title_from_line(line) in {"note", "notes"}:
            return "\n".join(lines[index:]).strip()
    return ""


def note_section_has_body(text: str) -> bool:
    # Checks whether a Note/Notes section has content beyond just the heading.
    note_section = extract_note_section(text)
    if not note_section:
        return False
    lines = [line.strip() for line in note_section.splitlines() if line.strip()]
    return len(lines) > 1


def remove_empty_note_section(text: str) -> str:
    # Removes a trailing empty Note/Notes heading from extracts output.
    if note_section_has_body(text):
        return text
    return re.sub(
        r"\n{0,2}(?:==\s*)?(Note|Notes)(?:\s*==)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def topic_heading(page: PageRequest) -> str:
    # Formats the page title like the rest of the cleaned section headings.
    return f"== {page.title} =="


def add_topic_heading(text: str, page: PageRequest) -> str:
    # Prefixes generated clean extraction output with the requested topic title.
    heading = topic_heading(page)
    text = text.strip()
    if not text:
        return heading
    if text.casefold().startswith(heading.casefold()):
        return text
    return f"{heading}\n\n{text}"


def extract_text_from_extracts_api(page: PageRequest, math_mode: str = "remove") -> str:
    # Fetches and cleans text from the extracts API, then drops empty Notes.
    text = clean_plain_text(fetch_page_extract(page), math_mode)
    return remove_empty_note_section(text)



def extract_text_from_html(page: PageRequest, math_mode: str = "remove") -> str:
    # Fetches and cleans text from rendered HTML parsing.
    return clean_wikipedia_html(fetch_page_html(page), math_mode)


def extract_text(page: PageRequest, method: str = "extracts", math_mode: str = "remove") -> str:
    # Dispatches to the selected extraction method with the requested math policy.
    if method == "extracts":
        return add_topic_heading(extract_text_from_extracts_api(page, math_mode), page)
    if method == "html":
        return add_topic_heading(extract_text_from_html(page, math_mode), page)
    raise ValueError(f"Unsupported extraction method: {method}")


def output_path_for_method(output: str, method: str, split_methods: bool) -> Path:
    # Legacy helper that optionally appends a method suffix beside an output path.
    path = Path(output)
    if not split_methods:
        return path
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_{method}{suffix}")


def safe_filename_part(value: str) -> str:
    # Converts page titles/language values into filesystem-safe path components.
    value = value.strip().replace(" ", "_")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "wikipedia_page"


def topic_folder_name(page: PageRequest) -> str:
    # Uses the safe page title as the topic folder name.
    return safe_filename_part(page.title)


def topic_file_stem(page: PageRequest) -> str:
    # Uses a lowercase safe page title as the output file stem.
    return topic_folder_name(page).lower()


def language_folder_name(lang: str) -> str:
    # Maps known language codes to readable folder names and falls back to the code.
    return LANGUAGE_FOLDER_NAMES.get(lang.lower(), lang.lower())


def output_directory(output: str, page: PageRequest) -> Path:
    # Builds the shared output/<Topic>/<Language> directory path.
    path = Path(output)
    root = path.parent if path.suffix else path
    return root / topic_folder_name(page) / language_folder_name(page.lang)


def extraction_output_path(output: str, page: PageRequest, method: str, math_mode: str) -> Path:
    # Builds the text output path for a method/math combination.
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_{method}_{math_mode}{suffix}"


def comparison_output_path(output: str, page: PageRequest) -> Path:
    # Builds the remove-mode comparison report path.
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_comparison{suffix}"


def runtime_output_path(output: str, page: PageRequest) -> Path:
    # Builds the runtime report path shared by single and combined runners.
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_runtime{suffix}"


def raw_output_path(output: str, page: PageRequest, source: str) -> Path:
    # Builds the raw API debug output path for extracts or HTML.
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_raw_{source}{suffix}"


def references_output_path(output: str, page: PageRequest) -> Path:
    # Builds the optional HTML references export path.
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_html_references{suffix}"


def partial_output_path(output: str, page: PageRequest) -> Path:
    # Builds the fixed partial extraction output path; each run overwrites this file.
    return output_directory(output, page) / "partial_text.txt"


def partial_match_report_path(output: str, page: PageRequest) -> Path:
    # Builds the partial extraction debug report path.
    return output_directory(output, page) / "partial_match_report.txt"


def partial_hybrid_output_path(output: str, page: PageRequest) -> Path:
    # Builds the hybrid partial extraction output path.
    return output_directory(output, page) / "partial_hybrid_text.txt"


def partial_hybrid_report_path(output: str, page: PageRequest) -> Path:
    # Builds the hybrid partial extraction report path.
    return output_directory(output, page) / "partial_hybrid_match_report.txt"


def write_text_file(path: Path, text: str) -> None:
    # Writes UTF-8 text after creating any missing output directories.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def runtime_label_order(label: str) -> tuple[int, str]:
    # Keeps runtime reports stable while allowing old entries to survive single runs.
    if label.startswith("Extracts API runtime"):
        return 0, label
    if label.startswith("HTML parser runtime"):
        return 1, label
    if label.startswith("Runtime mismatch"):
        return 2, label
    if label.startswith("Partial HTML runtime"):
        return 3, label
    if label.startswith("Partial hybrid runtime"):
        return 4, label
    return 9, label


def runtime_label(base_label: str, math_mode: str, page: PageRequest) -> str:
    # Stores the topic beside each timing so mixed-topic runtime files stay unambiguous.
    return f"{base_label} ({math_mode}, {page.title})"


def update_runtime_report(path: Path, page: PageRequest, updates: dict[str, float]) -> None:
    # Updates only the runtime entries from the current command and preserves the rest.
    entries: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ": " not in line or not line.endswith(" seconds"):
                continue
            label, value = line.split(": ", 1)
            entries[label] = value

    for label, seconds in updates.items():
        entries[label] = f"{seconds:.3f} seconds"

    lines = ["Wikipedia Text Extraction Runtime"]
    if entries:
        lines.append("")
        for label in sorted(entries, key=runtime_label_order):
            lines.append(f"{label}: {entries[label]}")
    write_text_file(path, "\n".join(lines))


def run_extraction(
    page: PageRequest, method: str, math_mode: str = "remove"
) -> tuple[str, float]:
    # Runs one extraction and returns both its text and elapsed seconds.
    started_at = time.perf_counter()
    text = extract_text(page, method, math_mode)
    return text, time.perf_counter() - started_at


def compare_texts(extracts_text: str, html_text: str) -> str:
    # Produces a character/line mismatch summary plus a unified diff.
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
    # Defines the legacy shared CLI that can run extracts, HTML, or both.
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
    # Parses legacy CLI arguments, allowing tests to pass an explicit argv.
    return build_parser().parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    # Runs the legacy CLI, writes requested outputs, or prints text to stdout.
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
