"""Extract clean plain text from Wikipedia pages."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
PARTIAL_REFERENCE_MODES = ("none", "original", "smart")
TOKEN_CONFIRM_MODES = ("none", "fuzzy")
TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
}
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
HYBRID_IGNORED_HEADING_TITLES = {"see also", "references", "external links"}
SECTION_HEADING_PATTERN = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")
INLINE_REFERENCE_PATTERN = re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")
MAINTENANCE_MARKER_PATTERN = re.compile(
    r"\[\s*(?:citation needed|better source needed|clarification needed|dubious\s*[\u2013\u2014-]\s*discuss)\s*\]",
    re.IGNORECASE,
)
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
STRONG_CONTINUATION_ABBREVIATION_PATTERN = re.compile(r"\b(?:al|e\.g|i\.e)\.$", re.IGNORECASE)
CONTEXT_CONTINUATION_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:etc|vs|fig|no|vol|pp)\.$",
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


@dataclass(frozen=True)
class TextToken:
    # Records one normalized token and its original character offsets.
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenAnchorMatch:
    # Records one token-anchor match and its diagnostic scores.
    score: float
    overlap_score: float
    ordered_score: float
    fuzzy_score: float | None
    start_token: int
    end_token: int
    start: int
    end: int
    candidates_checked: int
    confirm_used: bool


@dataclass(frozen=True)
class TokenExtractionResult:
    # Carries token partial output and a readable decision report.
    text: str
    report: str
    start: int
    end: int
    start_match: TokenAnchorMatch
    end_match: TokenAnchorMatch


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

    def __init__(
        self,
        include_reference_markers: bool = False,
        reference_section_text: str = "",
    ) -> None:
        # Initializes parser state for clean article text and optional inline citation markers.
        super().__init__(convert_charrefs=True)
        self._include_reference_markers = include_reference_markers
        self._reference_section_text = reference_section_text
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
            if element_id == "References" and self._reference_section_text:
                self._add_break()
                self._parts.append(self._reference_section_text)
                self._add_break()
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


class WikipediaCaptionParser(HTMLParser):
    """Extract figure/image captions from Wikipedia article HTML."""

    CAPTION_CLASSES = {
        "gallerytext",
        "mw-kartographer-caption",
        "thumbcaption",
    }
    SKIP_CLASSES = {"mw-cite-backlink", "mw-editsection", "noprint", "reference"}
    BLOCK_TAGS = {"br", "div", "li", "p"}
    VOID_TAGS = WikipediaTextParser.VOID_TAGS

    def __init__(self) -> None:
        # Initializes state for collecting visible caption text only.
        super().__init__(convert_charrefs=True)
        self.captions: list[str] = []
        self._capture_depth = 0
        self._skip_depth = 0
        self._current_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Starts capture for caption containers and skips citation/editing noise.
        attr_map = {name: value or "" for name, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if self._skip_depth:
            self._skip_depth += 1
            return

        if tag in {"script", "style"} or classes.intersection(self.SKIP_CLASSES):
            self._skip_depth += 1
            return

        if self._capture_depth:
            if tag in self.BLOCK_TAGS:
                self._append_caption_text(" ")
            if tag not in self.VOID_TAGS:
                self._capture_depth += 1
            return

        if tag == "figcaption" or classes.intersection(self.CAPTION_CLASSES):
            self._capture_depth = 1
            self._current_parts = []

    def handle_endtag(self, tag: str) -> None:
        # Flushes a caption when its container closes.
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if not self._capture_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._append_caption_text(" ")
        self._capture_depth -= 1
        if self._capture_depth <= 0:
            self._flush_caption()

    def handle_data(self, data: str) -> None:
        # Collects visible caption text while inside a caption container.
        if self._skip_depth or not self._capture_depth:
            return
        self._append_caption_text(data)

    def _append_caption_text(self, text: str) -> None:
        # Adds caption fragments before final normalization.
        if self._current_parts is not None:
            self._current_parts.append(text)

    def _flush_caption(self) -> None:
        # Normalizes one completed caption and stores it if useful.
        if self._current_parts is None:
            return
        caption = clean_caption_text("".join(self._current_parts))
        self._current_parts = None
        self._capture_depth = 0
        if caption:
            self.captions.append(caption)


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


def clean_caption_text(text: str) -> str:
    # Collapses caption whitespace and removes citation/editing residue.
    text = INLINE_REFERENCE_PATTERN.sub("", text)
    text = re.sub(r"\bedit\b", "", text, flags=re.IGNORECASE)
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


def remove_unwanted_sections(text: str, remove_references_section: bool = True) -> str:
    # Drops unwanted terminal sections while keeping Notes available for HTML output.
    removed_section_titles = set(REMOVED_SECTION_TITLES)
    plain_section_titles = set(PLAIN_SECTION_TITLES)
    if not remove_references_section:
        removed_section_titles.discard("references")
        plain_section_titles.discard("references")
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
            if title in removed_section_titles:
                skip_level = level
                continue

        plain_title = line.strip().casefold()
        if plain_title in plain_section_titles:
            skip_plain_section = plain_title in removed_section_titles
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
    text: str,
    math_mode: str = "remove",
    remove_inline_references: bool = True,
    remove_references_section: bool = True,
) -> str:
    # Runs shared text cleanup used by both extracts and HTML paths.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_unwanted_sections(text, remove_references_section)
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


def normalize_final_text_spacing(text: str) -> str:
    # Applies final whitespace/heading cleanup without re-running math or section removal.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return format_heading_spacing(text).strip()


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


def extract_image_captions_from_html(html: str) -> list[str]:
    # Extracts visible image/figure captions from rendered Wikipedia HTML.
    parser = WikipediaCaptionParser()
    parser.feed(html)
    parser.close()
    captions: list[str] = []
    seen: set[str] = set()
    for caption in parser.captions:
        normalized = normalize_for_match(caption)
        if len(normalized) < 25 or normalized in seen:
            continue
        seen.add(normalized)
        captions.append(caption)
    return captions


def extract_reference_entries_from_html(html: str) -> dict[str, str]:
    # Extracts numeric References entries keyed by their original citation number.
    parser = WikipediaReferencesParser()
    parser.feed(html)
    parser.close()
    entries: dict[str, str] = {}
    for reference in parser.references:
        number, separator, text = reference.partition(". ")
        if separator and number.isdigit():
            entries[number] = text
    return entries


def clean_wikipedia_html_with_references(
    html: str, math_mode: str = "remove", include_inline_markers: bool = True
) -> str:
    # Re-inserts rebuilt numeric References at the original References heading position.
    references = extract_references_from_html(html)
    parser = WikipediaTextParser(
        include_reference_markers=include_inline_markers,
        reference_section_text=references,
    )
    parser.feed(html)
    parser.close()
    return clean_plain_text(
        parser.get_text(),
        math_mode,
        remove_inline_references=not include_inline_markers,
        remove_references_section=not bool(references),
    )


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


def line_looks_like_copied_rendered_math_word(line: str) -> bool:
    # Detects word-only rendered math fragments near copied displaystyle blocks.
    stripped = line.strip()
    if not stripped:
        return True
    if not re.fullmatch(r"[A-Za-z ]+", stripped):
        return False
    if stripped != stripped.casefold():
        return False
    return len(stripped) <= 60


def clean_copied_latex_context_segment(segment: str) -> str:
    # Keeps prose context while dropping rendered fallback lines before displaystyle math.
    kept: list[str] = []
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line_looks_like_rendered_math(stripped):
            continue
        if line_looks_like_copied_rendered_math_word(stripped):
            continue
        kept.append(stripped)
    return " ".join(kept)


def replace_copied_displaystyle_latex(text: str) -> str:
    # Converts copied MediaWiki displaystyle math into inline LaTeX before sentence splitting.
    marker = r"{\displaystyle"
    parts: list[str] = []
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index == -1:
            parts.append(text[start:])
            break

        context = clean_copied_latex_context_segment(text[start:marker_index])
        if context:
            parts.append(context)
            parts.append(" ")

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
            break

        latex = text[marker_index + len(marker) : end_index]
        parts.append(f"${normalize_latex(latex)}$")
        parts.append(" ")
        start = end_index + 1

    text = "".join(parts)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_copied_math_for_hybrid(text: str) -> str:
    # Normalizes pasted Wikipedia math before hybrid sentence detection.
    text = MAINTENANCE_MARKER_PATTERN.sub("", text)
    if r"{\displaystyle" in text:
        text = replace_copied_displaystyle_latex(text)
        text = join_inline_latex_lines(text)
        text = remove_inline_duplicate_latex_symbols(text)
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
    sentences: list[TextSentence] = []
    start = 0
    index = 0
    length = len(text)
    spans: list[tuple[int, int]] = []
    while index < length:
        char = text[index]
        if char == "\n":
            if start < index:
                spans.append((start, index))
            start = index + 1
            index += 1
            continue
        if char in ".!?" and not is_decimal_point(text, index):
            end = index + 1
            while end < length and text[end] in "\"')]":
                end += 1
            citation_match = INLINE_REFERENCE_PATTERN.match(text, end)
            while citation_match:
                end = citation_match.end()
                citation_match = INLINE_REFERENCE_PATTERN.match(text, end)
            spans.append((start, end))
            start = end
            index = end
            continue
        index += 1
    if start < length:
        spans.append((start, length))

    for span_start, span_end in spans:
        raw_sentence = text[span_start:span_end]
        sentence = re.sub(r"\s+", " ", raw_sentence).strip()
        if not sentence or SECTION_HEADING_PATTERN.match(sentence):
            continue
        if len(normalize_for_match(sentence)) < 8:
            continue
        if sentences and sentence_should_join_after_abbreviation(sentences[-1].text, sentence):
            previous = sentences.pop()
            joined = re.sub(r"\s+", " ", text[previous.start : span_end]).strip()
            sentences.append(TextSentence(joined, previous.start, span_end))
            continue
        sentences.append(TextSentence(sentence, span_start, span_end))
    return sentences


def is_decimal_point(text: str, index: int) -> bool:
    # Keeps numeric decimals such as 1.34 and 0.58 inside the same sentence.
    return (
        text[index] == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def sentence_should_join_after_abbreviation(previous: str, current: str) -> bool:
    # Decides whether an abbreviation period is a true sentence break or continuation.
    previous = previous.strip()
    current = current.strip()
    if STRONG_CONTINUATION_ABBREVIATION_PATTERN.search(previous):
        return True
    if not CONTEXT_CONTINUATION_ABBREVIATION_PATTERN.search(previous) or not current:
        return False
    return current[0].islower() or current[0].isdigit() or current.startswith(("(", "[", "{", ",", ";", ":"))


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


def heading_section_spans(
    headings: list[TextHeading], titles: set[str], text_length: int
) -> list[tuple[int, int]]:
    # Returns character spans covered by headings whose titles should be ignored.
    spans: list[tuple[int, int]] = []
    for index, heading in enumerate(headings):
        if heading.title.casefold() not in titles:
            continue
        end = headings[index + 1].start if index + 1 < len(headings) else text_length
        spans.append((heading.start, end))
    return spans


def span_is_inside_ignored_sections(
    start: int, end: int, ignored_spans: list[tuple[int, int]]
) -> bool:
    # Checks whether a sentence/list item is fully inside an ignored section.
    return any(start >= span_start and end <= span_end for span_start, span_end in ignored_spans)


def build_hybrid_text_index(text: str) -> HybridTextIndex:
    # Builds headings, sentences, citations, and References boundary for hybrid matching.
    headings = extract_heading_spans(text)
    ignored_spans = heading_section_spans(
        headings, HYBRID_IGNORED_HEADING_TITLES, len(text)
    )
    sentences = [
        sentence
        for sentence in sentence_spans(text)
        if not span_is_inside_ignored_sections(
            sentence.start, sentence.end, ignored_spans
        )
    ]
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
    ignored_titles = {
        normalize_for_match(title) for title in HYBRID_IGNORED_HEADING_TITLES
    }
    offset = 0
    for line in copied_text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True):
        stripped = line.strip(" =\t\r\n")
        normalized = normalize_for_match(stripped)
        if normalized in ignored_titles:
            offset += len(line)
            continue
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


def token_excluded(index: int, excluded_spans: list[tuple[int, int]]) -> bool:
    # Checks whether one token offset sits inside an excluded text span.
    return any(start <= index < end for start, end in excluded_spans)


def tokenize_with_offsets(
    text: str, excluded_spans: list[tuple[int, int]] | None = None
) -> list[TextToken]:
    # Tokenizes English text while preserving original character offsets.
    excluded_spans = excluded_spans or []
    reference_spans = [(match.start(), match.end()) for match in INLINE_REFERENCE_PATTERN.finditer(text)]
    all_excluded = excluded_spans + reference_spans
    tokens: list[TextToken] = []
    for match in re.finditer(r"[A-Za-z0-9]+", text):
        if token_excluded(match.start(), all_excluded):
            continue
        value = match.group(0).casefold()
        if value:
            tokens.append(TextToken(value, match.start(), match.end()))
    return tokens


def build_token_inverted_index(tokens: list[TextToken]) -> dict[str, list[int]]:
    # Builds token -> token positions for fast candidate generation.
    index: dict[str, list[int]] = defaultdict(list)
    for position, token in enumerate(tokens):
        index[token.value].append(position)
    return dict(index)


def token_weight(token: str, frequencies: Counter[str]) -> float:
    # Gives rare/content tokens more influence than common function words.
    frequency = max(1, frequencies[token])
    weight = 1.0 / (frequency ** 0.5)
    if token in TOKEN_STOPWORDS or len(token) <= 2:
        weight *= 0.25
    return weight


def weighted_token_overlap(
    anchor_tokens: list[str], candidate_tokens: list[str], frequencies: Counter[str]
) -> float:
    # Scores weighted multiset overlap from the pasted anchor's perspective.
    if not anchor_tokens or not candidate_tokens:
        return 0.0
    anchor_counts = Counter(anchor_tokens)
    candidate_counts = Counter(candidate_tokens)
    denominator = sum(
        count * token_weight(token, frequencies)
        for token, count in anchor_counts.items()
    )
    if denominator <= 0:
        return 0.0
    numerator = sum(
        min(count, candidate_counts.get(token, 0)) * token_weight(token, frequencies)
        for token, count in anchor_counts.items()
    )
    return numerator / denominator


def ordered_token_coverage(anchor_tokens: list[str], candidate_tokens: list[str]) -> float:
    # Measures how much of the pasted anchor appears in the same order.
    if not anchor_tokens or not candidate_tokens:
        return 0.0
    anchor_index = 0
    matched = 0
    for token in candidate_tokens:
        if token == anchor_tokens[anchor_index]:
            matched += 1
            anchor_index += 1
            if anchor_index >= len(anchor_tokens):
                break
    return matched / len(anchor_tokens)


def meaningful_anchor_tokens(tokens: list[TextToken], side: str, window_tokens: int) -> list[str]:
    # Selects start/end token anchors from the pasted input.
    values = [token.value for token in tokens]
    if len(values) <= window_tokens:
        return values
    return values[:window_tokens] if side == "start" else values[-window_tokens:]


def rare_anchor_token_positions(
    anchor_tokens: list[str],
    inverted_index: dict[str, list[int]],
    total_clean_tokens: int,
    rare_token_limit: int = 28,
) -> list[tuple[str, list[int]]]:
    # Chooses useful low-frequency anchor tokens for candidate generation.
    unique_tokens = sorted(
        set(anchor_tokens),
        key=lambda token: (len(inverted_index.get(token, [])) or total_clean_tokens, -len(token), token),
    )
    selected: list[tuple[str, list[int]]] = []
    max_frequency = max(20, total_clean_tokens // 35)
    for token in unique_tokens:
        positions = inverted_index.get(token, [])
        if not positions or token in TOKEN_STOPWORDS or len(token) <= 2:
            continue
        if len(positions) > max_frequency:
            continue
        selected.append((token, positions))
        if len(selected) >= rare_token_limit:
            break
    if selected:
        return selected
    for token in unique_tokens:
        positions = inverted_index.get(token, [])
        if positions:
            selected.append((token, positions))
        if len(selected) >= rare_token_limit:
            break
    return selected


def token_candidate_starts(
    anchor_tokens: list[str],
    clean_tokens: list[TextToken],
    inverted_index: dict[str, list[int]],
    max_candidates: int,
) -> list[int]:
    # Uses rare shared tokens to propose aligned clean token-window starts.
    if not anchor_tokens or not clean_tokens:
        return []
    anchor_positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(anchor_tokens):
        anchor_positions[token].append(index)
    votes: Counter[int] = Counter()
    rare_tokens = rare_anchor_token_positions(
        anchor_tokens,
        inverted_index,
        len(clean_tokens),
    )
    for token, clean_positions in rare_tokens:
        for clean_position in clean_positions:
            for anchor_position in anchor_positions[token][:3]:
                candidate_start = clean_position - anchor_position
                if candidate_start < 0:
                    continue
                if candidate_start >= len(clean_tokens):
                    continue
                votes[candidate_start] += 1
    if not votes:
        return []
    return [
        candidate
        for candidate, _count in votes.most_common(max_candidates)
    ]


def score_token_candidate(
    anchor_tokens: list[str],
    clean_tokens: list[TextToken],
    candidate_start: int,
    frequencies: Counter[str],
) -> tuple[float, float, float, int]:
    # Scores one clean candidate window using token overlap and ordered coverage.
    window_size = min(len(anchor_tokens), len(clean_tokens) - candidate_start)
    if window_size <= 0:
        return 0.0, 0.0, 0.0, candidate_start
    candidate_values = [
        token.value for token in clean_tokens[candidate_start : candidate_start + window_size]
    ]
    overlap = weighted_token_overlap(anchor_tokens, candidate_values, frequencies)
    ordered = ordered_token_coverage(anchor_tokens, candidate_values)
    score = 0.65 * overlap + 0.35 * ordered
    return score, overlap, ordered, candidate_start + window_size


def fuzzy_confirm_score(anchor_tokens: list[str], candidate_tokens: list[str]) -> float:
    # Optional close-candidate confirmation without using fuzzy as the primary scorer.
    return fuzzy_ratio(" ".join(anchor_tokens), " ".join(candidate_tokens))


def find_token_anchor_match(
    clean_tokens: list[TextToken],
    anchor_tokens: list[str],
    inverted_index: dict[str, list[int]],
    frequencies: Counter[str],
    min_score: float,
    max_candidates: int = 240,
    confirm_mode: str = "none",
    tie_margin: float = 0.03,
    min_start_token: int = 0,
    tie_preference: str = "first",
) -> TokenAnchorMatch | None:
    # Finds one token anchor using inverted-index candidates and ordered-token scoring.
    if confirm_mode not in TOKEN_CONFIRM_MODES:
        raise ValueError(f"Unsupported token confirm mode: {confirm_mode}")
    candidate_starts = [
        start
        for start in token_candidate_starts(
            anchor_tokens,
            clean_tokens,
            inverted_index,
            max_candidates,
        )
        if start >= min_start_token
    ]
    scored: list[tuple[float, float, float, int, int, float | None, bool]] = []
    for candidate_start in candidate_starts:
        score, overlap, ordered, candidate_end = score_token_candidate(
            anchor_tokens,
            clean_tokens,
            candidate_start,
            frequencies,
        )
        if score >= min_score:
            scored.append((score, overlap, ordered, candidate_start, candidate_end, None, False))
    if not scored:
        return None
    position_key = (lambda item: item[3]) if tie_preference == "first" else (lambda item: -item[3])
    scored.sort(key=lambda item: (-item[0], -item[2], position_key(item)))
    close = [
        item for item in scored[:5] if scored[0][0] - item[0] <= tie_margin
    ]
    if confirm_mode == "fuzzy" and len(close) > 1:
        confirmed: list[tuple[float, float, float, int, int, float | None, bool]] = []
        for score, overlap, ordered, start, end, _fuzzy, _used in close:
            candidate_values = [token.value for token in clean_tokens[start:end]]
            fuzzy_score = fuzzy_confirm_score(anchor_tokens, candidate_values)
            confirmed.append((score, overlap, ordered, start, end, fuzzy_score, True))
        confirmed.sort(key=lambda item: (-item[5], -item[0], -item[2], position_key(item)))
        best = confirmed[0]
    else:
        best = scored[0]
    score, overlap, ordered, start, end, fuzzy_score, confirm_used = best
    return TokenAnchorMatch(
        score=score,
        overlap_score=overlap,
        ordered_score=ordered,
        fuzzy_score=fuzzy_score,
        start_token=start,
        end_token=end,
        start=clean_tokens[start].start,
        end=clean_tokens[end - 1].end,
        candidates_checked=len(candidate_starts),
        confirm_used=confirm_used,
    )


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
    sentences: list[TextSentence],
    reverse: bool = False,
    window_size: int = 3,
    step: int = 3,
) -> list[tuple[int, int, str]]:
    # Builds copied sentence chunks with step 3 and a final backfilled chunk.
    if window_size < 1 or len(sentences) < window_size:
        return []
    starts = list(range(0, len(sentences) - window_size + 1, max(1, step)))
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
    window_size: int = 3,
    copied_step: int = 3,
) -> tuple[int, int, int, int, float] | None:
    # Finds the first copied sentence chunk that has a strong clean sliding-window match.
    clean_indexes = [
        index
        for index, sentence in enumerate(clean_sentences)
        if sentence.start >= clean_start and sentence.end <= clean_end
    ]
    if not clean_indexes or window_size < 1:
        return None
    if len(clean_indexes) < window_size or len(copied_sentences) < window_size:
        return None
    clean_window_starts = clean_indexes[: len(clean_indexes) - window_size + 1]
    if reverse:
        clean_window_starts = list(reversed(clean_window_starts))
    for _copied_start, _copied_end, copied_text in sentence_window_chunks(
        copied_sentences, reverse, window_size, copied_step
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


def capped_sentence_char_range(
    clean_sentences: list[TextSentence],
    clean_start: int,
    clean_end: int,
    reverse: bool = False,
    max_sentences: int = 12,
) -> tuple[int, int]:
    # Caps a fallback search range to the nearest clean sentences at the boundary side.
    indexes = [
        index
        for index, sentence in enumerate(clean_sentences)
        if sentence.start >= clean_start and sentence.end <= clean_end
    ]
    if not indexes:
        return clean_start, clean_end
    selected = indexes[-max_sentences:] if reverse else indexes[:max_sentences]
    return clean_sentences[selected[0]].start, clean_sentences[selected[-1]].end


def bounded_start_search_begin(
    clean_sentences: list[TextSentence],
    anchor_start: int,
    max_sentences: int = 15,
    max_chars: int = 4000,
) -> int:
    # Opens a limited range before a start anchor so uncited pre-anchor text can match.
    if not clean_sentences:
        return max(0, anchor_start - max_chars)
    anchor_index = sentence_index_at(clean_sentences, anchor_start)
    start_index = max(0, anchor_index - max_sentences)
    return max(clean_sentences[start_index].start, max(0, anchor_start - max_chars))


def nonempty_paragraph_spans(text: str) -> list[tuple[int, int]]:
    # Returns non-empty paragraph spans in already-cleaned text.
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
        start = match.start()
        end = match.end()
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def nonempty_paragraph_count(text: str) -> int:
    # Counts paragraph-like copied chunks after noise/section stripping.
    return len(nonempty_paragraph_spans(text.replace("\r\n", "\n").replace("\r", "\n")))


def paragraph_range_before(
    text: str,
    anchor_start: int,
    anchor_end: int,
    paragraph_count: int,
    max_chars: int = 4000,
) -> tuple[int, int]:
    # Builds the first start-side range from copied paragraph count + 1.
    paragraph_count = max(1, paragraph_count)
    candidates = [
        span for span in nonempty_paragraph_spans(text) if span[0] < anchor_end
    ]
    selected = candidates[-paragraph_count:] if candidates else []
    start = selected[0][0] if selected else max(0, anchor_start - max_chars)
    return max(0, max(start, anchor_start - max_chars)), anchor_end


def paragraph_range_after(
    text: str,
    anchor_start: int,
    anchor_end: int,
    paragraph_count: int,
    max_chars: int = 4000,
) -> tuple[int, int]:
    # Builds the first end-side range from copied paragraph count + 1.
    paragraph_count = max(1, paragraph_count)
    candidates = [
        span for span in nonempty_paragraph_spans(text) if span[1] > anchor_start
    ]
    selected = candidates[:paragraph_count] if candidates else []
    end = selected[-1][1] if selected else min(len(text), anchor_end + max_chars)
    return anchor_start, min(len(text), min(end, anchor_end + max_chars))


def valid_boundary_headings(headings: list[TextHeading]) -> list[TextHeading]:
    # Keeps headings that can act as boundary-range fallbacks.
    return [
        heading
        for heading in headings
        if heading.title.casefold() not in HYBRID_IGNORED_HEADING_TITLES
    ]


def sentence_indexes_in_char_range(
    sentences: list[TextSentence], start: int, end: int
) -> list[int]:
    # Finds clean sentence indexes fully inside one candidate search range.
    return [
        index
        for index, sentence in enumerate(sentences)
        if sentence.start >= start and sentence.end <= end
    ]


def first_sentence_overlap_end(
    sentences: list[TextSentence], start: int, end: int, count: int = 2
) -> int:
    # Returns the end offset of the first N sentences for start-side overlap.
    indexes = sentence_indexes_in_char_range(sentences, start, end)
    if not indexes:
        return min(end, start)
    return sentences[indexes[min(count, len(indexes)) - 1]].end


def last_sentence_overlap_start(
    sentences: list[TextSentence], start: int, end: int, count: int = 2
) -> int:
    # Returns the start offset of the last N sentences for end-side overlap.
    indexes = sentence_indexes_in_char_range(sentences, start, end)
    if not indexes:
        return max(start, end)
    return sentences[indexes[-min(count, len(indexes))]].start


def dedupe_start_search_ranges(
    sentences: list[TextSentence], ranges: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    # Expands start ranges backward while retaining only a two-sentence overlap.
    deduped: list[tuple[int, int, str]] = []
    checked_start: int | None = None
    checked_end: int | None = None
    for start, end, label in ranges:
        if end <= start:
            continue
        if checked_start is None or checked_end is None:
            deduped.append((start, end, label))
            checked_start, checked_end = start, end
            continue
        if start >= checked_start:
            continue
        overlap_end = first_sentence_overlap_end(
            sentences, checked_start, checked_end
        )
        adjusted_end = min(end, max(overlap_end, start))
        if adjusted_end > start:
            deduped.append((start, adjusted_end, label))
            checked_start = start
            checked_end = max(checked_end, end)
    return deduped


def dedupe_end_search_ranges(
    sentences: list[TextSentence], ranges: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    # Expands end ranges forward while retaining only a two-sentence overlap.
    deduped: list[tuple[int, int, str]] = []
    checked_start: int | None = None
    checked_end: int | None = None
    for start, end, label in ranges:
        if end <= start:
            continue
        if checked_start is None or checked_end is None:
            deduped.append((start, end, label))
            checked_start, checked_end = start, end
            continue
        if end <= checked_end:
            continue
        overlap_start = last_sentence_overlap_start(
            sentences, checked_start, checked_end
        )
        adjusted_start = max(start, min(overlap_start, end))
        if end > adjusted_start:
            deduped.append((adjusted_start, end, label))
            checked_start = min(checked_start, start)
            checked_end = end
    return deduped


def staged_start_search_ranges(
    text: str,
    headings: list[TextHeading],
    sentences: list[TextSentence],
    anchor_start: int,
    anchor_end: int,
    paragraph_count: int,
) -> list[tuple[int, int, str]]:
    # Builds start-side paragraph, previous-heading, and previous-2-heading ranges.
    raw_ranges = [
        (*paragraph_range_before(text, anchor_start, anchor_end, paragraph_count), "paragraphs+1")
    ]
    previous_headings = [
        heading
        for heading in valid_boundary_headings(headings)
        if heading.start < anchor_start
    ]
    if previous_headings:
        raw_ranges.append((previous_headings[-1].start, anchor_end, "previous heading"))
    if len(previous_headings) >= 2:
        raw_ranges.append((previous_headings[-2].start, anchor_end, "previous 2 headings"))
    return dedupe_start_search_ranges(sentences, raw_ranges)


def staged_end_search_ranges(
    text: str,
    headings: list[TextHeading],
    sentences: list[TextSentence],
    anchor_start: int,
    anchor_end: int,
    paragraph_count: int,
) -> list[tuple[int, int, str]]:
    # Builds end-side paragraph, next-heading, and next-2-heading ranges.
    raw_ranges = [
        (*paragraph_range_after(text, anchor_start, anchor_end, paragraph_count), "paragraphs+1")
    ]
    next_headings = [
        heading
        for heading in valid_boundary_headings(headings)
        if heading.start > anchor_end
    ]
    if next_headings:
        raw_ranges.append((anchor_start, next_headings[0].start, "next heading"))
    if len(next_headings) >= 2:
        raw_ranges.append((anchor_start, next_headings[1].start, "next 2 headings"))
    return dedupe_end_search_ranges(sentences, raw_ranges)


def find_sentence_window_match_in_ranges(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    ranges: list[tuple[int, int, str]],
    reverse: bool,
    threshold: float,
    allow_short_fallback: bool,
) -> tuple[tuple[int, int, int, int, float], tuple[int, int, str]] | None:
    # Runs sentence-window matching through staged ranges in order.
    for range_start, range_end, label in ranges:
        match = find_sentence_window_match_with_short_fallback(
            copied_sentences,
            clean_sentences,
            range_start,
            range_end,
            reverse,
            threshold,
            allow_short_fallback=allow_short_fallback,
        )
        if match is not None:
            return match, (range_start, range_end, label)
    return None


def find_sentence_window_match_with_short_fallback(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    clean_start: int,
    clean_end: int,
    reverse: bool = False,
    threshold: float = 0.84,
    allow_short_fallback: bool = False,
) -> tuple[int, int, int, int, float] | None:
    # Tries the normal 3-sentence window, then 2/1 only for short boundary sides.
    match = find_sentence_window_match(
        copied_sentences,
        clean_sentences,
        clean_start,
        clean_end,
        reverse,
        threshold,
        window_size=3,
        copied_step=3,
    )
    if match is not None or not allow_short_fallback or len(copied_sentences) >= 7:
        return match
    capped_start, capped_end = capped_sentence_char_range(
        clean_sentences,
        clean_start,
        clean_end,
        False,
        max_sentences=12,
    )
    for window_size, fallback_threshold in ((2, max(threshold, 0.89)), (1, max(threshold, 0.94))):
        match = find_sentence_window_match(
            copied_sentences,
            clean_sentences,
            capped_start,
            capped_end,
            reverse,
            fallback_threshold,
            window_size=window_size,
            copied_step=1,
        )
        if match is not None:
            return match
    return None


def refine_sentence_start(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    copied_anchor_start: int,
    clean_anchor_start: int,
    threshold: float = 0.88,
    clean_min: int | None = None,
    clean_max: int | None = None,
    max_search_sentences: int = 15,
) -> int:
    # Expands a start anchor backward inside rolling clean sentence windows until two failures.
    copied_index = copied_anchor_start - 1
    before_clean_index = clean_anchor_start
    failures = 0
    start_index = clean_anchor_start
    while copied_index >= 0 and before_clean_index > 0 and failures < 2:
        candidates = [
            index
            for index in range(0, before_clean_index)
            if (clean_min is None or clean_sentences[index].start >= clean_min)
            and (clean_max is None or clean_sentences[index].end <= clean_max)
        ]
        candidates = candidates[-max_search_sentences:]
        match = best_ordered_sentence_match(
            copied_sentences[copied_index].text,
            clean_sentences,
            candidates,
            threshold,
            tie_preference="last",
        )
        if match is not None:
            start_index = match[0]
            before_clean_index = match[0]
            failures = 0
        else:
            failures += 1
        copied_index -= 1
    return start_index


def refine_sentence_end(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    copied_anchor_end: int,
    clean_anchor_end: int,
    threshold: float = 0.88,
    clean_min: int | None = None,
    clean_max: int | None = None,
    max_search_sentences: int = 15,
) -> int:
    # Expands an end anchor forward inside rolling clean sentence windows until two failures.
    copied_index = copied_anchor_end
    after_clean_index = clean_anchor_end
    failures = 0
    end_index = clean_anchor_end
    while copied_index < len(copied_sentences) and after_clean_index < len(clean_sentences) and failures < 2:
        candidates = [
            index
            for index in range(after_clean_index, len(clean_sentences))
            if (clean_min is None or clean_sentences[index].start >= clean_min)
            and (clean_max is None or clean_sentences[index].end <= clean_max)
        ]
        candidates = candidates[:max_search_sentences]
        match = best_ordered_sentence_match(
            copied_sentences[copied_index].text,
            clean_sentences,
            candidates,
            threshold,
            tie_preference="first",
        )
        if match is not None:
            after_clean_index = match[0] + 1
            end_index = after_clean_index
            failures = 0
        else:
            failures += 1
        copied_index += 1
    return end_index


def best_ordered_sentence_match(
    copied_sentence: str,
    clean_sentences: list[TextSentence],
    candidate_indexes: list[int],
    threshold: float,
    tie_preference: str = "first",
) -> tuple[int, float] | None:
    # Finds the best clean sentence for one copied sentence with deterministic tie handling.
    best: tuple[int, float] | None = None
    for index in candidate_indexes:
        clean_text = clean_sentences[index].text
        if token_overlap_score(copied_sentence, clean_text) < 0.30:
            continue
        score = hybrid_window_score(copied_sentence, clean_text)
        if score < threshold:
            continue
        if best is None or score > best[1]:
            best = (index, score)
        elif score == best[1] and (
            (tie_preference == "first" and index < best[0])
            or (tie_preference == "last" and index > best[0])
        ):
            best = (index, score)
    return best


def ordered_tokens(text: str) -> list[str]:
    # Keeps normalized token order for directional partial boundary matching.
    return re.findall(r"[a-z0-9]+", normalize_for_match(MAINTENANCE_MARKER_PATTERN.sub("", text)))


def partial_boundary_score(copied_fragment: str, clean_sentence: str, side: str) -> float:
    # Scores a boundary fragment against directional windows from one clean sentence.
    copied_normalized = normalize_for_match(MAINTENANCE_MARKER_PATTERN.sub("", copied_fragment))
    if len(copied_normalized) < 40:
        return 0.0
    copied_tokens = ordered_tokens(copied_fragment)
    clean_tokens = ordered_tokens(clean_sentence)
    if not copied_tokens or not clean_tokens:
        return 0.0
    copied_count = len(copied_tokens)
    min_size = max(1, int(copied_count * 0.80))
    max_size = min(len(clean_tokens), max(min_size, int(copied_count * 1.30) + 1))
    if min_size > len(clean_tokens):
        return 0.0

    windows: list[list[str]] = []
    for size in range(min_size, max_size + 1):
        if side == "start":
            latest_start = len(clean_tokens) - size
            earliest_start = max(0, int(len(clean_tokens) * 0.35) - size)
            starts = range(max(0, earliest_start), latest_start + 1)
        else:
            latest_start = min(len(clean_tokens) - size, int(len(clean_tokens) * 0.65))
            starts = range(0, latest_start + 1)
        windows.extend(clean_tokens[start : start + size] for start in starts)

    best = 0.0
    for window_tokens in windows:
        window_text = " ".join(window_tokens)
        token_score = token_overlap_score(" ".join(copied_tokens), window_text)
        if token_score < 0.60:
            continue
        char_score = fuzzy_ratio(copied_normalized, window_text)
        score = 0.55 * token_score + 0.45 * char_score
        best = max(best, score)
    return best


def resolve_partial_start_boundary(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    clean_end: int,
    current_start: int,
    clean_min: int | None = None,
    max_search_sentences: int = 15,
    threshold: float = 0.85,
) -> tuple[int, float] | None:
    # Finds a clean full sentence when the copied first sentence starts mid-sentence.
    if not copied_sentences:
        return None
    candidates = [
        index
        for index, sentence in enumerate(clean_sentences)
        if sentence.end <= clean_end
        and sentence.start < current_start
        and (clean_min is None or sentence.start >= clean_min)
    ][-max_search_sentences:]
    best: tuple[int, float] | None = None
    for index in candidates:
        score = partial_boundary_score(copied_sentences[0].text, clean_sentences[index].text, "start")
        if score >= threshold and (best is None or score > best[1] or (score == best[1] and index > best[0])):
            best = (index, score)
    if best is None:
        return None
    return clean_sentences[best[0]].start, best[1]


def resolve_partial_end_boundary(
    copied_sentences: list[TextSentence],
    clean_sentences: list[TextSentence],
    clean_start: int,
    current_end: int,
    clean_max: int | None = None,
    max_search_sentences: int = 15,
    threshold: float = 0.85,
) -> tuple[int, float] | None:
    # Finds a clean full sentence when the copied last sentence ends mid-sentence.
    if not copied_sentences:
        return None
    candidates = [
        index
        for index, sentence in enumerate(clean_sentences)
        if sentence.start >= clean_start
        and sentence.end > current_end
        and (clean_max is None or sentence.end <= clean_max)
    ][:max_search_sentences]
    best: tuple[int, float] | None = None
    for index in candidates:
        score = partial_boundary_score(copied_sentences[-1].text, clean_sentences[index].text, "end")
        if score >= threshold and (best is None or score > best[1] or (score == best[1] and index < best[0])):
            best = (index, score)
    if best is None:
        return None
    return clean_sentences[best[0]].end, best[1]


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


def starts_at_line_boundary(text: str, start: int) -> bool:
    # Keeps caption removal conservative by requiring a line-start match.
    line_start = text.rfind("\n", 0, start) + 1
    return not text[line_start:start].strip()


def remove_span_preserving_boundaries(text: str, start: int, end: int) -> str:
    # Removes one matched copied-noise span while leaving surrounding prose intact.
    while start > 0 and text[start - 1] in " \t":
        start -= 1
    while end < len(text) and text[end] in " \t":
        end += 1
    return text[:start] + text[end:]


def strip_exact_caption_occurrences(text: str, caption: str) -> tuple[str, int]:
    # Removes exact normalized caption matches that begin at a copied line boundary.
    caption_normalized = normalize_for_match(caption)
    if len(caption_normalized) < 40:
        return text, 0
    removed = 0
    while True:
        normalized_text, index_map = normalize_match_index_map(text)
        normalized_start = normalized_text.find(caption_normalized)
        matched_span: tuple[int, int] | None = None
        while normalized_start >= 0:
            start, end = original_span_from_normalized(
                index_map,
                normalized_start,
                normalized_start + len(caption_normalized),
            )
            if starts_at_line_boundary(text, start):
                matched_span = (start, end)
                break
            normalized_start = normalized_text.find(
                caption_normalized, normalized_start + 1
            )
        if matched_span is None:
            break
        text = remove_span_preserving_boundaries(text, matched_span[0], matched_span[1])
        removed += 1
    return text, removed


def line_spans(text: str) -> list[tuple[int, int]]:
    # Returns line spans including their trailing newline where present.
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    if not text.endswith(("\n", "\r")) and offset < len(text):
        spans.append((offset, len(text)))
    return spans


def strip_fuzzy_caption_lines(text: str, caption: str) -> tuple[str, int]:
    # Removes standalone lines that strongly match a known HTML caption.
    caption_normalized = normalize_for_match(caption)
    if len(caption_normalized) < 40:
        return text, 0
    removed = 0
    for start, end in reversed(line_spans(text)):
        line = text[start:end]
        stripped = line.strip()
        if len(normalize_for_match(stripped)) < 40:
            continue
        if token_overlap_score(stripped, caption) < 0.80:
            continue
        if hybrid_window_score(stripped, caption) < 0.94:
            continue
        text = remove_span_preserving_boundaries(text, start, end)
        removed += 1
    return text, removed


def strip_copied_caption_phrases(
    copied_text: str, captions: list[str] | None
) -> tuple[str, int]:
    # Removes known image captions from copied input before hybrid matching.
    if not captions:
        return copied_text, 0
    unique_captions: list[str] = []
    seen: set[str] = set()
    for caption in sorted(captions, key=lambda value: len(normalize_for_match(value)), reverse=True):
        normalized = normalize_for_match(caption)
        if len(normalized) < 40 or normalized in seen:
            continue
        seen.add(normalized)
        unique_captions.append(caption)

    removed_total = 0
    text = copied_text
    for caption in unique_captions:
        text, removed = strip_exact_caption_occurrences(text, caption)
        if removed == 0:
            text, removed = strip_fuzzy_caption_lines(text, caption)
        removed_total += removed
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed_total


def reference_section_span(text: str) -> tuple[int, int] | None:
    # Finds a formatted References section inside already-cleaned text.
    headings = extract_heading_spans(text)
    for index, heading in enumerate(headings):
        if heading.title.casefold() != "references":
            continue
        end = headings[index + 1].start if index + 1 < len(headings) else len(text)
        return heading.start, end
    return None


def remove_reference_section(text: str) -> str:
    # Removes only the formatted References section while preserving later sections.
    span = reference_section_span(text)
    if span is None:
        return text.strip()
    start, end = span
    return (text[:start].rstrip() + "\n\n" + text[end:].lstrip()).strip()


def expand_citation_numbers(value: str) -> list[str]:
    # Expands one citation marker payload such as "1, 3-5" into individual numbers.
    numbers: list[str] = []
    for part in re.split(r"\s*,\s*", value):
        part = part.strip()
        range_match = re.fullmatch(r"(\d+)\s*[\u2013-]\s*(\d+)", part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start <= end and end - start <= 100:
                numbers.extend(str(number) for number in range(start, end + 1))
                continue
        if part.isdigit():
            numbers.append(part)
    return numbers


def citation_reference_numbers(text: str) -> list[str]:
    # Extracts individual citation numbers in first-seen order for reference output.
    numbers: list[str] = []
    for match in INLINE_REFERENCE_PATTERN.finditer(text):
        numbers.extend(expand_citation_numbers(match.group(0).strip("[]")))
    return numbers


def unique_in_order(values: Iterable[str]) -> list[str]:
    # Keeps first occurrence order while dropping duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_reference_block(entries: list[tuple[str, str]]) -> str:
    # Formats selected reference entries as a clean References section.
    if not entries:
        return ""
    return "== References ==\n\n" + "\n\n".join(
        f"{number}. {text}" for number, text in entries
    )


def remap_inline_reference_markers(text: str, mapping: dict[str, str]) -> str:
    # Rewrites inline citation markers using the smart reference number map.
    def replace(match: re.Match[str]) -> str:
        numbers = expand_citation_numbers(match.group(0).strip("[]"))
        remapped = [mapping[number] for number in numbers if number in mapping]
        if not remapped:
            return ""
        return "[" + ", ".join(remapped) + "]"

    return INLINE_REFERENCE_PATTERN.sub(replace, text)


def selected_reference_entries(
    reference_entries: dict[str, str],
    citation_numbers_for_output: list[str],
    mode: str,
) -> tuple[list[tuple[str, str]], dict[str, str], list[str]]:
    # Selects and optionally renumbers references requested for partial hybrid output.
    unique_numbers = unique_in_order(citation_numbers_for_output)
    present_numbers = [number for number in unique_numbers if number in reference_entries]
    missing_numbers = [number for number in unique_numbers if number not in reference_entries]
    if mode == "smart":
        sorted_numbers = sorted(present_numbers, key=lambda value: int(value))
        mapping = {
            original_number: str(index)
            for index, original_number in enumerate(sorted_numbers, start=1)
        }
        entries = [
            (mapping[original_number], reference_entries[original_number])
            for original_number in sorted_numbers
        ]
        return entries, mapping, missing_numbers
    entries = [
        (original_number, reference_entries[original_number])
        for original_number in present_numbers
    ]
    return entries, {number: number for number in present_numbers}, missing_numbers


def insert_reference_block(text: str, reference_block: str) -> str:
    # Replaces an existing References section or appends one when the slice ended before it.
    if not reference_block:
        return remove_reference_section(text)
    span = reference_section_span(text)
    if span is None:
        return (text.rstrip() + "\n\n" + reference_block).strip()
    start, end = span
    return (text[:start].rstrip() + "\n\n" + reference_block + "\n\n" + text[end:].lstrip()).strip()


def finalize_partial_hybrid_output(
    text: str,
    references_mode: str,
    copied_reference_numbers: list[str],
    reference_entries: dict[str, str] | None,
) -> tuple[str, list[str], list[str]]:
    # Applies the requested partial-hybrid reference policy after matching is complete.
    if references_mode not in PARTIAL_REFERENCE_MODES:
        raise ValueError(f"Unsupported partial references mode: {references_mode}")
    if references_mode == "none":
        text = remove_reference_section(text)
        text = INLINE_REFERENCE_PATTERN.sub("", text)
        return normalize_final_text_spacing(text), [], []

    entries, mapping, missing = selected_reference_entries(
        reference_entries or {},
        copied_reference_numbers,
        references_mode,
    )
    if references_mode == "smart":
        text = remap_inline_reference_markers(text, mapping)
    reference_block = build_reference_block(entries)
    text = insert_reference_block(text, reference_block)
    return normalize_final_text_spacing(text), [number for number, _ in entries], missing


def strip_copied_ignored_sections(
    copied_text: str, clean_headings: list[TextHeading]
) -> str:
    # Removes copied References/See also/External links sections while keeping later valid sections.
    clean_heading_titles = {normalize_for_match(heading.title) for heading in clean_headings}
    ignored_titles = {
        normalize_for_match(title) for title in HYBRID_IGNORED_HEADING_TITLES
    }
    kept_lines: list[str] = []
    skipping_ignored_section = False
    for line in copied_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        normalized = normalize_for_match(stripped.strip(" =\t"))
        if normalized in ignored_titles:
            skipping_ignored_section = True
            continue
        if skipping_ignored_section:
            if normalized in clean_heading_titles and normalized not in ignored_titles:
                skipping_ignored_section = False
            else:
                continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


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
    copied_text = strip_copied_ignored_sections(copied_text, clean_headings)
    copied_text = normalize_copied_math_for_hybrid(copied_text)
    heading_titles = {normalize_for_match(heading.title) for heading in clean_headings}
    ignored_titles = {
        normalize_for_match(title) for title in HYBRID_IGNORED_HEADING_TITLES
    }
    kept_lines: list[str] = []
    for line in copied_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        stripped = line.strip()
        normalized = normalize_for_match(stripped.strip(" =\t"))
        if not stripped:
            kept_lines.append(line)
            continue
        if normalized in ignored_titles:
            continue
        if normalized in heading_titles:
            continue
        if re.match(r"^(?:main article|see also)\s*:", stripped, flags=re.IGNORECASE):
            continue
        if line_looks_like_partial_noise(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def snap_start_to_sentence(sentences: list[TextSentence], start: int) -> int:
    # Moves a token start to the containing sentence start when available.
    for sentence in sentences:
        if sentence.start <= start < sentence.end:
            return sentence.start
        if start < sentence.start:
            return sentence.start
    return start


def snap_end_to_sentence(sentences: list[TextSentence], end: int) -> int:
    # Moves a token end to the containing sentence end when available.
    for sentence in sentences:
        if sentence.start < end <= sentence.end:
            return sentence.end
        if end < sentence.start:
            return sentence.start
    return end


def extract_partial_token_text(
    clean_text: str,
    copied_text: str,
    window_tokens: int = 120,
    min_score: float = 0.72,
    confirm_mode: str = "none",
    max_candidates: int = 240,
    copied_image_captions: list[str] | None = None,
) -> TokenExtractionResult:
    # Extracts partial text using inverted-index token matching instead of fuzzy matching.
    if confirm_mode not in TOKEN_CONFIRM_MODES:
        raise ValueError(f"Unsupported token confirm mode: {confirm_mode}")
    if window_tokens < 20:
        raise ValueError("window_tokens must be at least 20")

    clean_index = build_hybrid_text_index(clean_text)
    copied_text, stripped_caption_count = strip_copied_caption_phrases(
        copied_text,
        copied_image_captions,
    )
    copied_matching_text = strip_copied_ignored_sections(
        copied_text,
        clean_index.headings,
    )
    copied_clean = normalize_copied_text_for_hybrid_sentences(
        copied_matching_text,
        clean_index.headings,
    )
    copied_tokens = tokenize_with_offsets(copied_clean)
    if len(copied_tokens) < 8:
        message = "failed: not enough copied tokens for token matching"
        report = "\n".join(
            [
                "Wikipedia Token Partial Extraction Report",
                "",
                message,
                f"Copied tokens: {len(copied_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)

    ignored_spans = heading_section_spans(
        clean_index.headings,
        HYBRID_IGNORED_HEADING_TITLES,
        len(clean_text),
    )
    clean_tokens = tokenize_with_offsets(clean_text, ignored_spans)
    if len(clean_tokens) < 8:
        message = "failed: not enough clean tokens for token matching"
        report = "\n".join(
            [
                "Wikipedia Token Partial Extraction Report",
                "",
                message,
                f"Clean tokens: {len(clean_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)

    inverted_index = build_token_inverted_index(clean_tokens)
    frequencies: Counter[str] = Counter(token.value for token in clean_tokens)
    start_anchor_tokens = meaningful_anchor_tokens(copied_tokens, "start", window_tokens)
    end_anchor_tokens = meaningful_anchor_tokens(copied_tokens, "end", window_tokens)

    start_match = find_token_anchor_match(
        clean_tokens,
        start_anchor_tokens,
        inverted_index,
        frequencies,
        min_score,
        max_candidates=max_candidates,
        confirm_mode=confirm_mode,
        tie_preference="first",
    )
    if start_match is None:
        message = "failed: token start boundary did not meet the minimum score"
        report = "\n".join(
            [
                "Wikipedia Token Partial Extraction Report",
                "",
                message,
                f"Minimum score: {min_score:.3f}",
                f"Copied tokens: {len(copied_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)

    end_match = find_token_anchor_match(
        clean_tokens,
        end_anchor_tokens,
        inverted_index,
        frequencies,
        min_score,
        max_candidates=max_candidates,
        confirm_mode=confirm_mode,
        min_start_token=start_match.start_token,
        tie_preference="last",
    )
    if end_match is None:
        message = "failed: token end boundary did not meet the minimum score"
        report = "\n".join(
            [
                "Wikipedia Token Partial Extraction Report",
                "",
                message,
                f"Minimum score: {min_score:.3f}",
                f"Start token: {start_match.start_token}",
            ]
        )
        raise PartialExtractionError(message, report)

    start = snap_start_to_sentence(clean_index.sentences, start_match.start)
    end = snap_end_to_sentence(clean_index.sentences, end_match.end)
    if start >= end:
        message = "failed: token boundaries are invalid"
        report = "\n".join(
            [
                "Wikipedia Token Partial Extraction Report",
                "",
                message,
                f"Start offset: {start}",
                f"End offset: {end}",
            ]
        )
        raise PartialExtractionError(message, report)

    text, _added_references, _missing_references = finalize_partial_hybrid_output(
        clean_text[start:end].strip(),
        "none",
        [],
        None,
    )
    report_lines = [
        "Wikipedia Token Partial Extraction Report",
        "",
        f"Copied tokens: {len(copied_tokens)}",
        f"Clean tokens: {len(clean_tokens)}",
        f"Window tokens: {min(window_tokens, len(copied_tokens))}",
        f"Minimum score: {min_score:.3f}",
        f"Confirm mode: {confirm_mode}",
        f"Copied image captions stripped: {stripped_caption_count}",
        f"Start token score: {start_match.score:.3f}",
        f"Start overlap: {start_match.overlap_score:.3f}",
        f"Start ordered coverage: {start_match.ordered_score:.3f}",
        f"Start candidates checked: {start_match.candidates_checked}",
        f"Start fuzzy confirm: {start_match.fuzzy_score:.3f}" if start_match.fuzzy_score is not None else "Start fuzzy confirm: none",
        f"End token score: {end_match.score:.3f}",
        f"End overlap: {end_match.overlap_score:.3f}",
        f"End ordered coverage: {end_match.ordered_score:.3f}",
        f"End candidates checked: {end_match.candidates_checked}",
        f"End fuzzy confirm: {end_match.fuzzy_score:.3f}" if end_match.fuzzy_score is not None else "End fuzzy confirm: none",
        f"Start offset: {start}",
        f"End offset: {end}",
        f"Output characters: {len(text)}",
    ]
    return TokenExtractionResult(
        text=text,
        report="\n".join(report_lines),
        start=start,
        end=end,
        start_match=start_match,
        end_match=end_match,
    )


def extract_partial_hybrid_text(
    clean_text: str,
    copied_text: str,
    threshold: float = 0.84,
    references_mode: str = "none",
    reference_entries: dict[str, str] | None = None,
    copied_image_captions: list[str] | None = None,
) -> HybridExtractionResult:
    # Uses heading/citation structural anchors plus sentence-window matching to slice clean text.
    if references_mode not in PARTIAL_REFERENCE_MODES:
        raise ValueError(f"Unsupported partial references mode: {references_mode}")
    clean_index = build_hybrid_text_index(clean_text)
    copied_text, stripped_caption_count = strip_copied_caption_phrases(
        copied_text,
        copied_image_captions,
    )
    copied_had_references_heading = references_heading_in_text(copied_text)
    copied_matching_text = strip_copied_ignored_sections(
        copied_text, clean_index.headings
    )
    copied_headings = copied_heading_candidates(copied_matching_text, clean_index.headings)
    matched_headings = heading_position_matches(copied_headings, clean_index.headings)
    copied_clean = normalize_copied_text_for_hybrid_sentences(
        copied_matching_text, clean_index.headings
    )
    copied_sentences = sentence_spans(copied_clean)
    copied_citations = citation_numbers(copied_matching_text)
    copied_reference_numbers = citation_reference_numbers(copied_matching_text)
    copied_citation_occurrences = [
        (match.group(0).strip("[]"), match.start(), match.end())
        for match in INLINE_REFERENCE_PATTERN.finditer(copied_matching_text)
    ]
    body_end = len(clean_text)

    report_lines = [
        "Wikipedia Hybrid Partial Extraction Report",
        "",
        f"Copied headings: {', '.join(h.title for h in copied_headings) or 'none'}",
        f"Matched headings: {', '.join(h.title for h in matched_headings) or 'none'}",
        f"Copied citations: {', '.join(copied_citations) or 'none'}",
        f"Copied image captions stripped: {stripped_caption_count}",
        f"Copied References section ignored: {'yes' if copied_had_references_heading else 'no'}",
        f"References mode: {references_mode}",
    ]

    coarse_start = 0
    coarse_end = body_end
    confidence = "medium"
    first_heading = matched_headings[0] if matched_headings else None
    last_heading = matched_headings[-1] if matched_headings else None
    start_citation_clean_start: int | None = None
    start_citation_clean_end: int | None = None
    start_citation_copied_end: int | None = None
    end_citation_clean_start: int | None = None
    end_citation_clean_end: int | None = None
    end_citation_copied_start: int | None = None
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
                citation_start_sentence = clean_index.sentences[
                    start_candidates[0][0].sentence_index
                ]
                citation_end_sentence = clean_index.sentences[
                    start_candidates[0][1].sentence_index
                ]
                citation_start = citation_start_sentence.start
                coarse_start = min(coarse_start, citation_start) if matched_headings else citation_start
                start_citation_clean_start = citation_start_sentence.start
                start_citation_clean_end = citation_end_sentence.end
                if len(copied_citation_occurrences) >= len(start_sequence):
                    start_citation_copied_end = copied_citation_occurrences[
                        len(start_sequence) - 1
                    ][2]
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
                citation_start_sentence = clean_index.sentences[
                    end_candidates[-1][0].sentence_index
                ]
                citation_end_sentence = clean_index.sentences[
                    end_candidates[-1][1].sentence_index
                ]
                citation_end = citation_end_sentence.end
                coarse_end = max(coarse_end, citation_end) if matched_headings else citation_end
                end_citation_clean_start = citation_start_sentence.start
                end_citation_clean_end = citation_end_sentence.end
                if len(copied_citation_occurrences) >= len(end_sequence):
                    end_citation_copied_start = copied_citation_occurrences[
                        -len(end_sequence)
                    ][1]
                report_lines.append(f"End citation sequence: {', '.join(end_sequence)}")
            elif matched_headings:
                report_lines.append("End citation sequence ignored: all matches are above last heading.")

    if not copied_sentences:
        message = "failed: no usable copied sentences found"
        raise PartialExtractionError(message, "\n".join(report_lines + ["", message]))

    start_copied_sentences = copied_sentences
    copied_head = ""
    if first_heading is not None and copied_headings:
        copied_head = copied_matching_text[: copied_headings[0].start]
        copied_head_clean = normalize_copied_text_for_hybrid_sentences(
            copied_head, clean_index.headings
        )
        head_sentences = sentence_spans(copied_head_clean)
        if head_sentences:
            start_copied_sentences = head_sentences
            report_lines.append("Start copied sentence range: before first matched heading.")

    if first_heading is not None and copied_head:
        start_anchor_start = first_heading.start
        start_anchor_end = first_heading.start
        start_paragraph_count = nonempty_paragraph_count(copied_head) + 1
    elif start_citation_clean_start is not None and start_citation_clean_end is not None:
        start_anchor_start = start_citation_clean_start
        start_anchor_end = start_citation_clean_end
        copied_prefix = copied_matching_text[: start_citation_copied_end or 0]
        start_paragraph_count = nonempty_paragraph_count(copied_prefix) + 1
    elif first_heading is not None:
        start_anchor_start = first_heading.start
        start_anchor_end = first_heading.start
        start_paragraph_count = 1
    else:
        start_anchor_start = max(0, coarse_start)
        start_anchor_end = min(len(clean_text), max(coarse_end, start_anchor_start + 1))
        start_paragraph_count = nonempty_paragraph_count(copied_matching_text) + 1

    start_ranges = staged_start_search_ranges(
        clean_text,
        clean_index.headings,
        clean_index.sentences,
        start_anchor_start,
        start_anchor_end,
        start_paragraph_count,
    )
    start_match_result = find_sentence_window_match_in_ranges(
        start_copied_sentences,
        clean_index.sentences,
        start_ranges,
        False,
        threshold,
        allow_short_fallback=len(start_copied_sentences) < 7,
    )

    if start_match_result:
        start_match, (start_search_begin, start_search_end, start_range_label) = start_match_result
        clean_start_index = refine_sentence_start(
            start_copied_sentences,
            clean_index.sentences,
            start_match[0],
            start_match[2],
            clean_min=start_search_begin,
            clean_max=start_search_end,
        )
        start = clean_index.sentences[clean_start_index].start
        report_lines.append(f"Start sentence-window score: {start_match[4]:.3f}")
        report_lines.append(f"Start search range: {start_range_label}")
    else:
        start = coarse_start
        start_search_begin = start_ranges[0][0] if start_ranges else 0
        start_search_end = start_ranges[0][1] if start_ranges else coarse_end
        confidence = "low"
        report_lines.append("Start sentence-window match failed; used structural fallback.")

    partial_start = resolve_partial_start_boundary(
        start_copied_sentences,
        clean_index.sentences,
        start_search_end,
        start,
        clean_min=start_search_begin,
    )
    if partial_start is not None:
        start = partial_start[0]
        if confidence == "low":
            confidence = "medium"
        report_lines.append(f"Start partial sentence score: {partial_start[1]:.3f}")

    end_copied_sentences = copied_sentences
    copied_tail = ""
    if last_heading is not None and copied_headings:
        copied_tail = copied_matching_text[copied_headings[-1].end :]
        copied_tail_clean = normalize_copied_text_for_hybrid_sentences(
            copied_tail, clean_index.headings
        )
        tail_sentences = sentence_spans(copied_tail_clean)
        if tail_sentences:
            end_copied_sentences = tail_sentences
            report_lines.append("End copied sentence range: after last matched heading.")

    if last_heading is not None and copied_tail:
        end_anchor_start = last_heading.end
        end_anchor_end = max(last_heading.end, end_citation_clean_end or last_heading.end)
        end_paragraph_count = nonempty_paragraph_count(copied_tail) + 1
    elif end_citation_clean_start is not None and end_citation_clean_end is not None:
        end_anchor_start = end_citation_clean_start
        end_anchor_end = end_citation_clean_end
        copied_suffix = copied_matching_text[end_citation_copied_start or 0 :]
        end_paragraph_count = nonempty_paragraph_count(copied_suffix) + 1
    elif last_heading is not None:
        end_anchor_start = last_heading.end
        end_anchor_end = last_heading.end
        end_paragraph_count = 1
    else:
        end_anchor_start = start
        end_anchor_end = min(len(clean_text), max(coarse_end, end_anchor_start + 1))
        end_paragraph_count = nonempty_paragraph_count(copied_matching_text) + 1

    end_ranges = staged_end_search_ranges(
        clean_text,
        clean_index.headings,
        clean_index.sentences,
        end_anchor_start,
        end_anchor_end,
        end_paragraph_count,
    )
    end_match_result = find_sentence_window_match_in_ranges(
        end_copied_sentences,
        clean_index.sentences,
        end_ranges,
        True,
        threshold,
        allow_short_fallback=len(end_copied_sentences) < 7,
    )

    if end_match_result:
        end_match, (end_search_start, end_search_limit, end_range_label) = end_match_result
        clean_end_index = refine_sentence_end(
            end_copied_sentences,
            clean_index.sentences,
            end_match[1],
            end_match[3],
            clean_min=end_search_start,
            clean_max=end_search_limit,
        )
        end = clean_index.sentences[min(clean_end_index, len(clean_index.sentences)) - 1].end
        report_lines.append(f"End sentence-window score: {end_match[4]:.3f}")
        report_lines.append(f"End search range: {end_range_label}")
    else:
        end = coarse_end
        end_search_start = end_ranges[0][0] if end_ranges else start
        end_search_limit = end_ranges[0][1] if end_ranges else coarse_end
        confidence = "low"
        report_lines.append("End sentence-window match failed; used structural fallback.")

    partial_end = resolve_partial_end_boundary(
        end_copied_sentences,
        clean_index.sentences,
        end_search_start,
        end,
        clean_max=end_search_limit,
    )
    if partial_end is not None:
        end = partial_end[0]
        if confidence == "low":
            confidence = "medium"
        report_lines.append(f"End partial sentence score: {partial_end[1]:.3f}")

    if start >= end:
        message = "failed: hybrid boundaries are invalid"
        raise PartialExtractionError(message, "\n".join(report_lines + ["", message]))

    extracted_text, added_references, missing_references = finalize_partial_hybrid_output(
        clean_text[start:end].strip(),
        references_mode,
        copied_reference_numbers,
        reference_entries,
    )
    report_lines.extend(
        [
            f"References added: {', '.join(added_references) or 'none'}",
            f"Missing reference entries: {', '.join(missing_references) or 'none'}",
            f"Confidence: {confidence}",
            f"Start offset: {start}",
            f"End offset: {end}",
            f"Output characters: {len(extracted_text)}",
        ]
    )
    return HybridExtractionResult(extracted_text, "\n".join(report_lines), start, end, confidence)


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


def partial_token_output_path(output: str, page: PageRequest) -> Path:
    # Builds the token partial extraction output path.
    return output_directory(output, page) / "partial_token_text.txt"


def partial_token_report_path(output: str, page: PageRequest) -> Path:
    # Builds the token partial extraction report path.
    return output_directory(output, page) / "partial_token_match_report.txt"


def partial_dmp_output_path(output: str, page: PageRequest) -> Path:
    # Builds the Diff Match Patch partial extraction output path.
    return output_directory(output, page) / "partial_dmp_text.txt"


def partial_dmp_report_path(output: str, page: PageRequest) -> Path:
    # Builds the Diff Match Patch partial extraction report path.
    return output_directory(output, page) / "partial_dmp_match_report.txt"


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
    if label.startswith("Partial token runtime"):
        return 5, label
    if label.startswith("Partial DMP runtime"):
        return 6, label
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
