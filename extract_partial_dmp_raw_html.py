"""Extract a clean partial section by matching pasted text against raw HTML."""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from html import unescape
from pathlib import Path
from typing import Iterable

from extract_partial_dmp import (
    find_token_dmp_boundary_search,
    refine_token_dmp_end_boundary,
    refine_token_dmp_start_boundary,
    load_diff_match_patch,
    read_pasted_text,
)
from wiki_text_extractor import (
    MAINTENANCE_MARKER_PATTERN,
    PageRequest,
    PartialExtractionError,
    build_token_inverted_index,
    clean_wikipedia_html_with_references,
    fetch_page_html,
    finalize_partial_hybrid_output,
    output_directory,
    page_request_from_url,
    runtime_label,
    runtime_output_path,
    tokenize_with_offsets,
    update_runtime_report,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"
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
    "td",
    "th",
    "tr",
    "ul",
}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
TAG_NAME_PATTERN = re.compile(r"^</?\s*([A-Za-z0-9:-]+)")
ATTR_PATTERN = re.compile(
    r"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+))"
)
SENTENCE_END_PATTERN = re.compile(r"[.!?][\"')\]]*\s+")
CONTINUATION_ABBREVIATIONS = {"al", "e.g", "i.e", "etc", "fig", "vs"}


def build_parser() -> argparse.ArgumentParser:
    # Defines the CLI for the experimental raw-HTML DMP partial matcher.
    parser = argparse.ArgumentParser(
        description="Extract a partial clean Wikipedia section by matching pasted text against raw HTML."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Pasted source text file used to identify the partial section",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/partial_dmp_raw_html_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled after the raw HTML slice is found",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.50,
        help="Minimum local DMP coverage required for each raw-HTML boundary anchor",
    )
    parser.add_argument(
        "--anchor-chars",
        type=int,
        default=600,
        help="Number of normalized copied characters used for start/end boundary scoring",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="DMP match_main chunk size; keep this at or below 32",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=24,
        help="Maximum start/end DMP chunks to try per boundary",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.45,
        help="DMP match_main threshold; lower is stricter",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Local Diff Match Patch diff timeout in seconds",
    )
    parser.add_argument(
        "--locator-window-tokens",
        type=int,
        default=60,
        help="Token chunk size used to locate raw-HTML DMP start/end before local verification",
    )
    parser.add_argument(
        "--locator-refine-tokens",
        type=int,
        default=20,
        help="Token chunk size used to refine raw-HTML DMP boundaries after an anchor is found",
    )
    parser.add_argument(
        "--locator-min-score",
        type=float,
        default=0.72,
        help="Minimum token score required before raw-HTML DMP local verification",
    )
    parser.add_argument(
        "--locator-max-candidates",
        type=int,
        default=240,
        help="Maximum token candidate windows checked for each raw-HTML DMP locator chunk",
    )
    return parser


def partial_dmp_raw_output_path(output: str, page: PageRequest) -> Path:
    # Stores the latest raw-HTML DMP clean partial output beside other partial outputs.
    return output_directory(output, page) / "partial_dmp_raw_html_text.txt"


def partial_dmp_raw_report_path(output: str, page: PageRequest) -> Path:
    # Stores raw-HTML DMP boundary diagnostics.
    return output_directory(output, page) / "partial_dmp_raw_html_match_report.txt"


def append_visible_char(parts: list[str], index_map: list[int], value: str, html_index: int) -> None:
    # Adds visible text while preserving the source HTML offset for every emitted character.
    for char in value:
        parts.append(char)
        index_map.append(html_index)


def append_separator(parts: list[str], index_map: list[int], html_index: int, separator: str = " ") -> None:
    # Adds a single separator so adjacent raw tags do not merge words during matching.
    if parts and parts[-1].isspace():
        return
    append_visible_char(parts, index_map, separator, html_index)


def tag_name(tag_text: str) -> str:
    # Extracts the raw tag name without building a full HTML DOM.
    match = TAG_NAME_PATTERN.search(tag_text)
    return match.group(1).casefold() if match else ""


def tag_attrs(tag_text: str) -> dict[str, str]:
    # Extracts simple HTML attributes so math image alt text can participate in raw matching.
    attrs: dict[str, str] = {}
    for match in ATTR_PATTERN.finditer(tag_text):
        attrs[match.group(1).casefold()] = unescape(match.group(2) or match.group(3) or match.group(4) or "")
    return attrs


def raw_html_visible_text_with_map(html: str) -> tuple[str, list[int]]:
    # Converts raw HTML into a visible-ish text stream without removing tables/captions.
    parts: list[str] = []
    index_map: list[int] = []
    index = 0
    length = len(html)
    while index < length:
        char = html[index]
        if char == "<":
            if html.startswith("<!--", index):
                end = html.find("-->", index + 4)
                index = length if end == -1 else end + 3
                continue
            end = html.find(">", index + 1)
            if end == -1:
                append_visible_char(parts, index_map, char, index)
                index += 1
                continue
            tag_text = html[index : end + 1]
            name = tag_name(tag_text)
            attrs = tag_attrs(tag_text)
            classes = set(attrs.get("class", "").split())
            is_math_image = name == "img" and (
                attrs.get("alt", "").startswith("{\\displaystyle")
                or any(value.startswith("mwe-math") for value in classes)
            )
            if is_math_image and attrs.get("alt"):
                append_separator(parts, index_map, index)
                append_visible_char(parts, index_map, attrs["alt"], index)
                append_separator(parts, index_map, end)
            elif name in BLOCK_TAGS:
                append_separator(parts, index_map, index, "\n")
            elif name:
                append_separator(parts, index_map, index)
            index = end + 1
            continue
        if char == "&":
            end = html.find(";", index + 1, min(index + 32, length))
            if end != -1:
                entity = html[index : end + 1]
                decoded = unescape(entity)
                if decoded != entity:
                    append_visible_char(parts, index_map, decoded, index)
                    index = end + 1
                    continue
        append_visible_char(parts, index_map, char, index)
        index += 1
    return "".join(parts), index_map


def normalize_raw_dmp_text_with_map(
    text: str,
    source_map: list[int] | None = None,
) -> tuple[str, list[int]]:
    # Normalizes raw-visible/copied text while keeping offsets back to raw HTML when present.
    spans = [(match.start(), match.end()) for match in MAINTENANCE_MARKER_PATTERN.finditer(text)]
    span_index = 0
    normalized: list[str] = []
    index_map: list[int] = []
    previous_was_space = False
    source_map = source_map or list(range(len(text)))
    for index, char in enumerate(text):
        while span_index < len(spans) and index >= spans[span_index][1]:
            span_index += 1
        if span_index < len(spans) and spans[span_index][0] <= index < spans[span_index][1]:
            continue
        if char == "\r":
            char = "\n"
        if char.isspace() or char == "\xa0":
            if normalized and not previous_was_space:
                normalized.append(" ")
                index_map.append(source_map[index])
            previous_was_space = True
            continue
        if char in ",.;:!?" and normalized and normalized[-1] == " ":
            normalized.pop()
            index_map.pop()
        for folded_char in char.casefold():
            normalized.append(folded_char)
            index_map.append(source_map[index])
        previous_was_space = False

    while normalized and normalized[0] == " ":
        normalized.pop(0)
        index_map.pop(0)
    while normalized and normalized[-1] == " ":
        normalized.pop()
        index_map.pop()
    return "".join(normalized), index_map


def raw_html_span_from_normalized_map(
    index_map: list[int], normalized_start: int, normalized_end: int
) -> tuple[int, int]:
    # Converts normalized raw-visible offsets back to raw HTML source offsets.
    if not index_map:
        return 0, 0
    normalized_start = max(0, min(normalized_start, len(index_map) - 1))
    normalized_end = max(normalized_start + 1, min(normalized_end, len(index_map)))
    return index_map[normalized_start], index_map[normalized_end - 1] + 1


def punctuation_word_before(text: str, punctuation_index: int) -> str:
    # Gets the preceding word so common abbreviations do not force a sentence break.
    before = text[:punctuation_index].rstrip()
    if not before:
        return ""
    return before.split()[-1].strip("\"'()[]{}").casefold()


def snap_visible_start_to_sentence(text: str, visible_index: int, max_back: int = 1200) -> int:
    # Expands a raw-visible start backward to the likely sentence boundary.
    if visible_index <= 0:
        return 0
    search_start = max(0, visible_index - max_back)
    segment = text[search_start:visible_index]
    best = segment.rfind("\n\n")
    best = best + 2 if best != -1 else 0
    for match in SENTENCE_END_PATTERN.finditer(segment):
        word = punctuation_word_before(segment, match.start())
        if word in CONTINUATION_ABBREVIATIONS:
            continue
        best = max(best, match.end())
    return search_start + best


def snap_visible_end_to_sentence(text: str, visible_index: int, max_forward: int = 1200) -> int:
    # Expands a raw-visible end forward to the likely sentence boundary.
    visible_index = max(0, min(visible_index, len(text)))
    segment = text[visible_index : min(len(text), visible_index + max_forward)]
    for match in SENTENCE_END_PATTERN.finditer(segment):
        word = punctuation_word_before(segment, match.start())
        if word in CONTINUATION_ABBREVIATIONS:
            continue
        return visible_index + match.end()
    return visible_index


def html_offset_from_visible_index(raw_visible_map: list[int], visible_index: int, end: bool = False) -> int:
    # Maps a raw-visible character index back to a raw HTML source offset.
    if not raw_visible_map:
        return 0
    if end:
        visible_index = max(0, min(visible_index - 1, len(raw_visible_map) - 1))
        return raw_visible_map[visible_index] + 1
    visible_index = max(0, min(visible_index, len(raw_visible_map) - 1))
    return raw_visible_map[visible_index]


def avoid_cutting_inside_tag(html: str, start: int, end: int) -> tuple[int, int]:
    # Moves slice edges out of partial tag text so HTMLParser sees a usable fragment.
    previous_open = html.rfind("<", 0, start)
    previous_close = html.rfind(">", 0, start)
    if previous_open > previous_close:
        close = html.find(">", start)
        start = len(html) if close == -1 else close + 1

    previous_open = html.rfind("<", 0, end)
    previous_close = html.rfind(">", 0, end)
    if previous_open > previous_close:
        close = html.find(">", end)
        end = len(html) if close == -1 else close + 1
    return max(0, min(start, len(html))), max(0, min(end, len(html)))


def raw_dmp_partial_match(
    html: str,
    copied_text: str,
    math_mode: str = "latex",
    min_coverage: float = 0.50,
    anchor_chars: int = 600,
    chunk_size: int = 32,
    max_chunks: int = 24,
    match_threshold: float = 0.45,
    timeout: float = 1.0,
    locator_window_tokens: int = 60,
    locator_refine_tokens: int = 20,
    locator_min_score: float = 0.72,
    locator_max_candidates: int = 240,
    raw_visible_data: tuple[str, list[int]] | None = None,
) -> tuple[str, str]:
    # Finds pasted boundaries in raw-visible HTML, then cleans only the matched raw slice.
    if chunk_size > 32:
        raise ValueError("chunk_size must be 32 or less for diff-match-patch match_main")
    if locator_window_tokens < 8:
        raise ValueError("locator_window_tokens must be at least 8")
    if locator_refine_tokens < 8:
        raise ValueError("locator_refine_tokens must be at least 8")
    diff_match_patch = load_diff_match_patch()
    raw_visible, raw_visible_map = raw_visible_data or raw_html_visible_text_with_map(html)
    raw_normalized, raw_map = normalize_raw_dmp_text_with_map(raw_visible, raw_visible_map)
    _raw_normalized_visible, raw_visible_normalized_map = normalize_raw_dmp_text_with_map(raw_visible)
    copied_normalized, _copied_map = normalize_raw_dmp_text_with_map(copied_text)
    if len(copied_normalized) < max(40, chunk_size):
        message = "failed: not enough copied text for raw-HTML DMP matching"
        report = "\n".join(
            [
                "Wikipedia Raw HTML DMP Partial Extraction Report",
                "",
                message,
                f"Copied normalized characters: {len(copied_normalized)}",
            ]
        )
        raise PartialExtractionError(message, report)
    raw_tokens = tokenize_with_offsets(raw_normalized)
    copied_tokens = tokenize_with_offsets(copied_normalized)
    if len(copied_tokens) < 8:
        message = "failed: not enough copied tokens for raw-HTML DMP matching"
        report = "\n".join(
            [
                "Wikipedia Raw HTML DMP Partial Extraction Report",
                "",
                message,
                f"Copied tokens: {len(copied_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)
    inverted_index = build_token_inverted_index(raw_tokens)
    frequencies: Counter[str] = Counter(token.value for token in raw_tokens)

    match_started_at = time.perf_counter()
    start_boundary = find_token_dmp_boundary_search(
        diff_match_patch,
        raw_normalized,
        copied_normalized,
        raw_tokens,
        copied_tokens,
        inverted_index,
        frequencies,
        "start",
        locator_window_tokens,
        min_coverage,
        locator_min_score,
        locator_max_candidates,
        timeout,
    )
    if start_boundary is None:
        message = "failed: raw-HTML DMP start boundary did not meet the minimum coverage"
        report = "\n".join(
            [
                "Wikipedia Raw HTML DMP Partial Extraction Report",
                "",
                message,
                f"Minimum coverage: {min_coverage:.3f}",
                f"Locator window tokens: {min(locator_window_tokens, len(copied_tokens))}",
            ]
        )
        raise PartialExtractionError(message, report)
    start_boundary = refine_token_dmp_start_boundary(
        start_boundary,
        diff_match_patch,
        raw_normalized,
        copied_normalized,
        raw_tokens,
        copied_tokens,
        inverted_index,
        frequencies,
        locator_refine_tokens,
        min_coverage,
        locator_min_score,
        locator_max_candidates,
        timeout,
        2,
    )

    end_boundary = find_token_dmp_boundary_search(
        diff_match_patch,
        raw_normalized,
        copied_normalized,
        raw_tokens,
        copied_tokens,
        inverted_index,
        frequencies,
        "end",
        locator_window_tokens,
        min_coverage,
        locator_min_score,
        locator_max_candidates,
        timeout,
        min_start=start_boundary.match.normalized_start,
    )
    match_seconds = time.perf_counter() - match_started_at
    if end_boundary is None:
        message = "failed: raw-HTML DMP end boundary did not meet the minimum coverage"
        report = "\n".join(
            [
                "Wikipedia Raw HTML DMP Partial Extraction Report",
                "",
                message,
                f"Minimum coverage: {min_coverage:.3f}",
                f"Start score: {start_boundary.match.score:.3f}",
                f"Locator window tokens: {min(locator_window_tokens, len(copied_tokens))}",
            ]
        )
        raise PartialExtractionError(message, report)
    end_boundary = refine_token_dmp_end_boundary(
        end_boundary,
        diff_match_patch,
        raw_normalized,
        copied_normalized,
        raw_tokens,
        copied_tokens,
        inverted_index,
        frequencies,
        locator_refine_tokens,
        min_coverage,
        locator_min_score,
        locator_max_candidates,
        timeout,
        2,
    )
    start_match = start_boundary.match
    end_match = end_boundary.match

    start_visible, _ = raw_html_span_from_normalized_map(
        raw_visible_normalized_map,
        start_match.normalized_start,
        max(start_match.normalized_start + 1, start_match.normalized_start + chunk_size),
    )
    _, end_visible = raw_html_span_from_normalized_map(
        raw_visible_normalized_map,
        max(0, end_match.normalized_end - chunk_size),
        end_match.normalized_end,
    )
    snapped_start_visible = snap_visible_start_to_sentence(raw_visible, start_visible)
    snapped_end_visible = snap_visible_end_to_sentence(raw_visible, end_visible)
    raw_start = html_offset_from_visible_index(raw_visible_map, snapped_start_visible)
    raw_end = html_offset_from_visible_index(raw_visible_map, snapped_end_visible, end=True)
    raw_start, raw_end = avoid_cutting_inside_tag(html, raw_start, raw_end)
    if raw_start >= raw_end:
        message = "failed: raw-HTML DMP boundaries are invalid"
        report = "\n".join(
            [
                "Wikipedia Raw HTML DMP Partial Extraction Report",
                "",
                message,
                f"Raw start offset: {raw_start}",
                f"Raw end offset: {raw_end}",
            ]
        )
        raise PartialExtractionError(message, report)

    raw_slice = html[raw_start:raw_end]
    cleaned_slice = clean_wikipedia_html_with_references(
        raw_slice,
        math_mode,
        include_inline_markers=True,
    )
    text, _added_references, _missing_references = finalize_partial_hybrid_output(
        cleaned_slice.strip(),
        "none",
        [],
        None,
    )
    if not text:
        message = "failed: raw-HTML DMP slice cleaned to empty text"
        report = "\n".join(
            [
                "Wikipedia Raw HTML DMP Partial Extraction Report",
                "",
                message,
                f"Raw start offset: {raw_start}",
                f"Raw end offset: {raw_end}",
            ]
        )
        raise PartialExtractionError(message, report)

    report_lines = [
        "Wikipedia Raw HTML DMP Partial Extraction Report",
        "",
        f"Raw HTML characters: {len(html)}",
        f"Raw visible characters: {len(raw_visible)}",
        f"Raw normalized characters: {len(raw_normalized)}",
        f"Copied normalized characters: {len(copied_normalized)}",
        f"Raw normalized tokens: {len(raw_tokens)}",
        f"Copied tokens: {len(copied_tokens)}",
        f"Start coverage: {start_match.score:.3f}",
        f"End coverage: {end_match.score:.3f}",
        f"Minimum coverage: {min_coverage:.3f}",
        f"Anchor characters: {min(anchor_chars, len(copied_normalized))}",
        f"Locator window tokens: {min(locator_window_tokens, len(copied_tokens))}",
        f"Locator refine tokens: {min(locator_refine_tokens, len(copied_tokens))}",
        f"Locator minimum token score: {locator_min_score:.3f}",
        f"Locator max candidates: {locator_max_candidates}",
        f"Chunk size: {min(chunk_size, 32)}",
        f"Max chunks: {max_chunks}",
        f"Match threshold: {match_threshold:.3f}",
        f"Local DMP timeout: {timeout:.3f} seconds",
        f"DMP match runtime: {match_seconds:.3f} seconds",
        f"Start locator: {start_boundary.locator}",
        f"Start copied token chunk: {start_boundary.copied_start_token}-{start_boundary.copied_end_token}",
        f"Start candidates checked: {start_match.candidates_checked}",
        f"Start chunk offset: {start_match.chunk_offset}",
        f"Start matched at normalized raw-visible offset: {start_match.matched_at}",
        f"End locator: {end_boundary.locator}",
        f"End copied token chunk: {end_boundary.copied_start_token}-{end_boundary.copied_end_token}",
        f"End candidates checked: {end_match.candidates_checked}",
        f"End chunk offset: {end_match.chunk_offset}",
        f"End matched at normalized raw-visible offset: {end_match.matched_at}",
        f"Start visible offset: {start_visible}",
        f"Start snapped visible offset: {snapped_start_visible}",
        f"End visible offset: {end_visible}",
        f"End snapped visible offset: {snapped_end_visible}",
        f"Raw start offset: {raw_start}",
        f"Raw end offset: {raw_end}",
        f"Raw slice characters: {len(raw_slice)}",
        f"Output characters: {len(text)}",
    ]
    return text, "\n".join(report_lines)


def main(argv: Iterable[str] | None = None) -> int:
    # Fetches raw HTML once, matches pasted text against it, and cleans the matched raw slice.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_dmp_raw_output_path(args.output, page)
    report_path = partial_dmp_raw_report_path(args.output, page)
    runtime_path = runtime_output_path(args.output, page)

    try:
        pasted_text = read_pasted_text(input_path)
        started_at = time.perf_counter()
        fetch_started_at = time.perf_counter()
        html = fetch_page_html(page)
        fetch_seconds = time.perf_counter() - fetch_started_at
        match_started_at = time.perf_counter()
        text, report = raw_dmp_partial_match(
            html,
            pasted_text,
            math_mode=args.math,
            min_coverage=args.min_coverage,
            anchor_chars=args.anchor_chars,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            match_threshold=args.match_threshold,
            timeout=args.timeout,
            locator_window_tokens=args.locator_window_tokens,
            locator_refine_tokens=args.locator_refine_tokens,
            locator_min_score=args.locator_min_score,
            locator_max_candidates=args.locator_max_candidates,
        )
        match_seconds = time.perf_counter() - match_started_at
        seconds = time.perf_counter() - started_at
        report_text = "\n".join(
            [
                report,
                "",
                f"Total runtime: {seconds:.3f} seconds",
                f"Fetch runtime: {fetch_seconds:.3f} seconds",
                "Clean runtime: included in match runtime",
                f"Match+slice-clean runtime: {match_seconds:.3f} seconds",
                f"Input file: {input_path}",
                f"Math mode: {args.math}",
            ]
        )
        write_text_file(output_path, text)
        write_text_file(report_path, report_text)
        update_runtime_report(
            runtime_path,
            page,
            {runtime_label("Partial DMP raw HTML runtime", args.math, page): seconds},
        )
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated raw-HTML DMP match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved raw-HTML DMP partial text: {output_path}")
    print(f"Updated raw-HTML DMP match report: {report_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
