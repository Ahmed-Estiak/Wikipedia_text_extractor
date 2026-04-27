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
    SKIP_TAGS = {"figcaption", "figure", "map", "script", "style", "table"}
    SKIP_CLASSES = {
        "ambox",
        "asbox",
        "catlinks",
        "citation",
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
    SKIP_IDS = {
        "Further_reading",
    }
    MATH_CLASSES = {
        "mwe-math-element",
        "mwe-math-fallback-image-display",
        "mwe-math-fallback-image-inline",
        "mwe-math-mathml-display",
        "mwe-math-mathml-inline",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._sup_depth = 0
        self._sub_depth = 0
        self._reference_skip_depth = 0
        self._pending_reference_separator = False
        self._pending_subscript_separator = False
        self._math_depth = 0
        self._math_tag_stack: list[str] = []
        self._heading_level: int | None = None
        self._heading_buffer: list[str] | None = None
        self._skip_section_level: int | None = None
        self._list_stack: list[dict[str, int | str]] = []
        self._pending_prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        classes = set(attr_map.get("class", "").split())
        element_id = attr_map.get("id", "")
        heading_level = int(tag[1]) if re.fullmatch(r"h[1-6]", tag) else None
        in_math = tag == "math" or bool(classes.intersection(self.MATH_CLASSES))
        if in_math:
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

        if tag == "sup":
            if classes.intersection(self.SKIP_CLASSES) or "reference" in classes:
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
            if file_title:
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
        if self._skip_depth:
            self._skip_depth -= 1
            if tag == "sup" and self._reference_skip_depth:
                self._reference_skip_depth -= 1
                self._pending_reference_separator = True
            self._close_math_tag(tag)
            return

        if tag == "ol" and self._list_stack:
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
        if self._skip_depth or self._skip_section_level is not None:
            return
        text = re.sub(r"\s+", " ", data)
        if not text.strip():
            if self._pending_reference_separator:
                self._append_pending_space()
                self._pending_reference_separator = False
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
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.strip()

    def _add_break(self) -> None:
        self._pending_reference_separator = False
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
        if self._heading_buffer is None:
            return
        heading = re.sub(r"\s+", " ", "".join(self._heading_buffer)).strip()
        self._heading_buffer = None
        if not heading:
            return
        self._add_break()
        self._parts.append(f"== {heading} ==")
        self._add_break()

    def _append_inline_marker(self, marker: str) -> None:
        if self._heading_buffer is not None:
            self._heading_buffer.append(marker)
            return
        self._parts.append(marker)

    def _append_inline_text(self, text: str) -> None:
        if self._heading_buffer is not None:
            self._heading_buffer.append(text)
            return
        self._parts.append(text)

    def _append_pending_space(self) -> None:
        target = self._heading_buffer if self._heading_buffer is not None else self._parts
        if not target:
            return
        last = target[-1]
        if last.endswith((" ", "\n")):
            return
        target.append(" ")

    def _close_math_tag(self, tag: str) -> bool:
        if self._math_tag_stack and self._math_tag_stack[-1] == tag:
            self._math_tag_stack.pop()
            self._math_depth -= 1
            return True
        return False


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


def repair_compact_power_notation(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
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
    label = ""
    while index > 0:
        index -= 1
        label = chr(ord("a") + index % 26) + label
        index //= 26
    return label


def remove_unwanted_sections(text: str) -> str:
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
    lines: list[str] = []
    for line in text.splitlines():
        heading = SECTION_HEADING_PATTERN.match(line)
        if heading:
            lines.append(f"== {heading.group(2).strip()} ==")
        else:
            lines.append(line)
    return "\n".join(lines)


def section_title_from_line(line: str) -> str:
    heading = SECTION_HEADING_PATTERN.match(line)
    if heading:
        return heading.group(2).strip().casefold()
    return line.strip().casefold()


def format_heading_spacing(text: str) -> str:
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
    return False


def clean_latex_context_segment(segment: str) -> str:
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
    return "\n\n".join(cleaned)


def clean_plain_text(text: str, math_mode: str = "remove") -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_unwanted_sections(text)
    text = clean_leading_caret_markers(text)
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
    parser = WikipediaTextParser()
    parser.feed(html)
    parser.close()
    return clean_plain_text(parser.get_text(), math_mode)


def extract_note_section(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if section_title_from_line(line) in {"note", "notes"}:
            return "\n".join(lines[index:]).strip()
    return ""


def note_section_has_body(text: str) -> bool:
    note_section = extract_note_section(text)
    if not note_section:
        return False
    lines = [line.strip() for line in note_section.splitlines() if line.strip()]
    return len(lines) > 1


def remove_empty_note_section(text: str) -> str:
    if note_section_has_body(text):
        return text
    return re.sub(
        r"\n{0,2}(?:==\s*)?(Note|Notes)(?:\s*==)?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def extract_text_from_extracts_api(page: PageRequest, math_mode: str = "remove") -> str:
    text = clean_plain_text(fetch_page_extract(page), math_mode)
    return remove_empty_note_section(text)



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


def raw_output_path(output: str, page: PageRequest, source: str) -> Path:
    suffix = Path(output).suffix or ".txt"
    return output_directory(output, page) / f"{topic_file_stem(page)}_raw_{source}{suffix}"


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
