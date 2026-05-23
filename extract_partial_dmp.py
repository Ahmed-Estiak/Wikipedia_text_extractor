"""Extract a clean partial section using Google's Diff Match Patch."""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from wiki_text_extractor import (
    HYBRID_IGNORED_HEADING_TITLES,
    INLINE_REFERENCE_PATTERN,
    PageRequest,
    PartialExtractionError,
    build_hybrid_text_index,
    build_token_inverted_index,
    clean_wikipedia_html_with_references,
    extract_image_captions_from_html,
    fetch_page_html,
    finalize_partial_hybrid_output,
    find_token_anchor_match,
    heading_section_spans,
    page_request_from_url,
    partial_dmp_output_path,
    partial_dmp_report_path,
    runtime_label,
    runtime_output_path,
    snap_end_to_sentence,
    snap_start_to_sentence,
    strip_copied_caption_phrases,
    strip_copied_ignored_sections,
    normalize_copied_text_for_hybrid_sentences,
    token_anchor_chunks,
    token_chunks_in_range,
    tokenize_with_offsets,
    update_runtime_report,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"


@dataclass(frozen=True)
class DmpBoundaryMatch:
    # Records one DMP anchor-chunk boundary match.
    score: float
    normalized_start: int
    normalized_end: int
    chunk_offset: int
    matched_at: int
    candidates_checked: int


@dataclass(frozen=True)
class DmpBoundarySearch:
    # Records which copied character chunk produced the final DMP boundary.
    match: DmpBoundaryMatch
    copied_start: int
    copied_end: int
    chunks_checked: int
    copied_start_token: int = 0
    copied_end_token: int = 0
    locator: str = "dmp"


def load_diff_match_patch():
    # Keeps the experimental dependency optional until this script is run.
    try:
        from diff_match_patch import diff_match_patch
    except ImportError as exc:
        raise RuntimeError(
            "The diff-match-patch package is required for this method. "
            "Install it with: .venv\\Scripts\\python.exe -m pip install diff-match-patch"
        ) from exc
    return diff_match_patch


def build_parser() -> argparse.ArgumentParser:
    # Defines the CLI for the experimental Diff Match Patch partial matcher.
    parser = argparse.ArgumentParser(
        description="Extract a partial clean Wikipedia section using Google's Diff Match Patch."
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
        default="output/partial_dmp_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled before partial matching",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.72,
        help="Minimum local DMP coverage required for each boundary anchor",
    )
    parser.add_argument(
        "--anchor-chars",
        type=int,
        default=300,
        help="Number of normalized copied characters used per start/end boundary window",
    )
    parser.add_argument(
        "--refine-chars",
        type=int,
        default=100,
        help="Number of normalized copied characters used per backward/forward refinement window",
    )
    parser.add_argument(
        "--max-refine-failures",
        type=int,
        default=2,
        help="Stop DMP boundary refinement after this many consecutive failed windows",
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
        default=16,
        help="Maximum start/end DMP chunks to try per boundary",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.35,
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
        help="Token chunk size used to locate DMP start/end before local verification",
    )
    parser.add_argument(
        "--locator-refine-tokens",
        type=int,
        default=20,
        help="Token chunk size used to refine DMP boundaries after an anchor is found",
    )
    parser.add_argument(
        "--locator-min-score",
        type=float,
        default=0.72,
        help="Minimum token score required before DMP local verification",
    )
    parser.add_argument(
        "--locator-max-candidates",
        type=int,
        default=240,
        help="Maximum token candidate windows checked for each DMP locator chunk",
    )
    return parser


def read_pasted_text(path: Path) -> str:
    # Reads user-pasted copied text and reports missing/empty input clearly.
    if not path.exists():
        raise FileNotFoundError(
            f"Input text file not found: {path}. Create it and paste the Wikipedia text there."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input text file is empty: {path}")
    return text


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    # Merges excluded spans so character filtering stays linear.
    ordered = sorted((start, end) for start, end in spans if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def normalize_for_dmp_with_map(
    text: str, excluded_spans: list[tuple[int, int]] | None = None
) -> tuple[str, list[int]]:
    # Normalizes text for DMP while keeping offsets back to the original clean text.
    reference_spans = [(match.start(), match.end()) for match in INLINE_REFERENCE_PATTERN.finditer(text)]
    spans = merge_spans(reference_spans + (excluded_spans or []))
    span_index = 0
    normalized: list[str] = []
    index_map: list[int] = []
    previous_was_space = False
    for index, char in enumerate(text):
        while span_index < len(spans) and index >= spans[span_index][1]:
            span_index += 1
        if span_index < len(spans) and spans[span_index][0] <= index < spans[span_index][1]:
            continue
        if char == "\r":
            char = "\n"
        if char.isspace():
            if normalized and not previous_was_space:
                normalized.append(" ")
                index_map.append(index)
            previous_was_space = True
            continue
        if char in ",.;:!?" and normalized and normalized[-1] == " ":
            normalized.pop()
            index_map.pop()
        for folded_char in char.casefold():
            normalized.append(folded_char)
            index_map.append(index)
        previous_was_space = False

    while normalized and normalized[0] == " ":
        normalized.pop(0)
        index_map.pop(0)
    while normalized and normalized[-1] == " ":
        normalized.pop()
        index_map.pop()
    return "".join(normalized), index_map


def original_span_from_dmp_map(
    index_map: list[int], normalized_start: int, normalized_end: int
) -> tuple[int, int]:
    # Converts normalized DMP offsets back to the original clean text offsets.
    if not index_map:
        return 0, 0
    normalized_start = max(0, min(normalized_start, len(index_map) - 1))
    normalized_end = max(normalized_start + 1, min(normalized_end, len(index_map)))
    return index_map[normalized_start], index_map[normalized_end - 1] + 1


def dmp_chunk_offsets(
    text_length: int,
    side: str,
    anchor_chars: int,
    chunk_size: int,
    max_chunks: int,
) -> list[int]:
    # Chooses short DMP-safe chunks across the copied start/end anchor area.
    if text_length <= 0:
        return []
    chunk_size = max(1, min(32, chunk_size, text_length))
    anchor_chars = max(chunk_size, min(anchor_chars, text_length))
    if side == "start":
        first = 0
        last = anchor_chars - chunk_size
    else:
        first = text_length - anchor_chars
        last = text_length - chunk_size
    if last <= first or max_chunks <= 1:
        return [first]
    step = max(1, (last - first) // (max_chunks - 1))
    offsets = list(range(first, last + 1, step))
    if offsets[-1] != last:
        offsets.append(last)
    return offsets[:max_chunks]


def chunk_is_meaningful(chunk: str) -> bool:
    # Avoids matching tiny punctuation/common fragments as DMP anchors.
    tokens = re.findall(r"[a-z0-9]{3,}", chunk)
    return len(tokens) >= 2 or (len(chunk.strip()) >= 18 and bool(tokens))


def dmp_equal_coverage(diff_match_patch, left: str, right: str, timeout: float) -> float:
    # Scores two small local windows by equal-character coverage using DMP.
    if not left or not right:
        return 0.0
    dmp = diff_match_patch()
    dmp.Diff_Timeout = timeout
    diffs = dmp.diff_main(left, right, checklines=False)
    dmp.diff_cleanupEfficiency(diffs)
    equal_chars = sum(len(data) for operation, data in diffs if operation == 0)
    return equal_chars / len(right)


def dmp_equal_coverage_and_span(
    diff_match_patch, left: str, right: str, timeout: float
) -> tuple[float, int, int]:
    # Scores local windows and returns the meaningful equal span inside the clean window.
    if not left or not right:
        return 0.0, 0, 0
    dmp = diff_match_patch()
    dmp.Diff_Timeout = timeout
    diffs = dmp.diff_main(left, right, checklines=False)
    dmp.diff_cleanupEfficiency(diffs)
    equal_chars = 0
    left_index = 0
    meaningful_spans: list[tuple[int, int]] = []
    for operation, data in diffs:
        if operation == 0:
            equal_chars += len(data)
            if chunk_is_meaningful(data):
                meaningful_spans.append((left_index, left_index + len(data)))
            left_index += len(data)
        elif operation == -1:
            left_index += len(data)
    score = equal_chars / len(right)
    if meaningful_spans:
        return score, meaningful_spans[0][0], meaningful_spans[-1][1]
    return score, 0, len(left)


def find_dmp_boundary(
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    side: str,
    min_score: float,
    anchor_chars: int,
    chunk_size: int,
    max_chunks: int,
    match_threshold: float,
    timeout: float,
    min_start: int = 0,
) -> DmpBoundaryMatch | None:
    # Uses DMP match_main chunks, then verifies each candidate with local DMP coverage.
    if not clean_normalized or not copied_normalized:
        return None
    anchor_chars = min(anchor_chars, len(copied_normalized))
    dmp = diff_match_patch()
    dmp.Match_Distance = max(len(clean_normalized), 1000)
    dmp.Match_Threshold = match_threshold
    candidates_checked = 0
    best: DmpBoundaryMatch | None = None
    offsets = dmp_chunk_offsets(
        len(copied_normalized),
        side,
        anchor_chars,
        chunk_size,
        max_chunks,
    )
    preferred_loc = min_start if side == "start" else min(len(clean_normalized), min_start + len(copied_normalized))
    for offset in offsets:
        chunk = copied_normalized[offset : offset + min(chunk_size, len(copied_normalized) - offset)]
        if not chunk_is_meaningful(chunk):
            continue
        matched_at = dmp.match_main(clean_normalized, chunk, preferred_loc)
        candidates_checked += 1
        if matched_at < 0:
            continue
        candidate_start = max(0, matched_at - offset)
        if candidate_start < min_start:
            continue
        candidate_end = min(len(clean_normalized), candidate_start + len(copied_normalized))
        if side == "start":
            clean_window = clean_normalized[candidate_start : candidate_start + anchor_chars]
            copied_window = copied_normalized[:anchor_chars]
        else:
            clean_window = clean_normalized[max(0, candidate_end - anchor_chars) : candidate_end]
            copied_window = copied_normalized[-anchor_chars:]
        score = dmp_equal_coverage(diff_match_patch, clean_window, copied_window, timeout)
        if score < min_score:
            continue
        match = DmpBoundaryMatch(
            score=score,
            normalized_start=candidate_start,
            normalized_end=candidate_end,
            chunk_offset=offset,
            matched_at=matched_at,
            candidates_checked=candidates_checked,
        )
        if best is None:
            best = match
            continue
        if score > best.score:
            best = match
            continue
        if score == best.score:
            if side == "start" and candidate_start < best.normalized_start:
                best = match
            elif side == "end" and candidate_end > best.normalized_end:
                best = match
    if best is None and candidates_checked:
        return None
    return best


def dmp_char_chunks_in_range(
    text_length: int,
    start: int,
    end: int,
    chunk_chars: int,
    chunk_size: int,
    reverse: bool = False,
) -> list[tuple[int, int]]:
    # Builds non-overlapping copied-character windows for staged DMP matching.
    start = max(0, min(start, text_length))
    end = max(start, min(end, text_length))
    min_chars = max(40, min(chunk_size, chunk_chars))
    if end - start < min_chars:
        return []
    chunks: list[tuple[int, int]] = []
    if reverse:
        cursor = end
        while cursor > start:
            chunk_start = max(start, cursor - chunk_chars)
            if cursor - chunk_start >= min_chars:
                chunks.append((chunk_start, cursor))
            cursor = chunk_start
        return chunks

    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + chunk_chars)
        if chunk_end - cursor >= min_chars:
            chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def find_dmp_anchor_match(
    diff_match_patch,
    clean_normalized: str,
    copied_window: str,
    min_score: float,
    chunk_size: int,
    max_chunks: int,
    match_threshold: float,
    timeout: float,
    min_start: int = 0,
    max_end: int | None = None,
    tie_preference: str = "first",
) -> DmpBoundaryMatch | None:
    # Finds one copied character window in clean text using short DMP-safe chunks.
    if not clean_normalized or not copied_window:
        return None
    dmp = diff_match_patch()
    dmp.Match_Distance = max(len(clean_normalized), 1000)
    dmp.Match_Threshold = match_threshold
    if tie_preference == "last":
        preferred_loc = max_end if max_end is not None else len(clean_normalized)
    else:
        preferred_loc = min_start
    preferred_loc = max(0, min(preferred_loc, len(clean_normalized)))
    offsets = dmp_chunk_offsets(
        len(copied_window),
        "start",
        len(copied_window),
        chunk_size,
        max_chunks,
    )
    candidates_checked = 0
    best: DmpBoundaryMatch | None = None
    for offset in offsets:
        chunk = copied_window[offset : offset + min(chunk_size, len(copied_window) - offset)]
        if not chunk_is_meaningful(chunk):
            continue
        matched_at = dmp.match_main(clean_normalized, chunk, preferred_loc)
        candidates_checked += 1
        if matched_at < 0:
            continue
        candidate_start = matched_at - offset
        candidate_end = candidate_start + len(copied_window)
        if candidate_start < min_start:
            continue
        if max_end is not None and candidate_end > max_end:
            continue
        if candidate_start < 0 or candidate_end > len(clean_normalized):
            continue
        clean_window = clean_normalized[candidate_start:candidate_end]
        score, span_start, span_end = dmp_equal_coverage_and_span(
            diff_match_patch,
            clean_window,
            copied_window,
            timeout,
        )
        if score < min_score:
            continue
        match = DmpBoundaryMatch(
            score=score,
            normalized_start=candidate_start + span_start,
            normalized_end=candidate_start + span_end,
            chunk_offset=offset,
            matched_at=matched_at,
            candidates_checked=candidates_checked,
        )
        if best is None or score > best.score:
            best = match
            continue
        if abs(score - best.score) <= 1e-9:
            if tie_preference == "last" and candidate_end > best.normalized_end:
                best = match
            elif tie_preference != "last" and candidate_start < best.normalized_start:
                best = match
    return best


def find_dmp_boundary_search(
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    side: str,
    window_chars: int,
    min_score: float,
    chunk_size: int,
    max_chunks: int,
    match_threshold: float,
    timeout: float,
    min_start: int = 0,
) -> DmpBoundarySearch | None:
    # Scans copied start/end in non-overlapping windows until one window matches clean text.
    copied_chunks = dmp_char_chunks_in_range(
        len(copied_normalized),
        0,
        len(copied_normalized),
        window_chars,
        chunk_size,
        reverse=(side == "end"),
    )
    tie_preference = "last" if side == "end" else "first"
    for index, (copied_start, copied_end) in enumerate(copied_chunks):
        copied_window = copied_normalized[copied_start:copied_end]
        match = find_dmp_anchor_match(
            diff_match_patch,
            clean_normalized,
            copied_window,
            min_score,
            chunk_size,
            max_chunks,
            match_threshold,
            timeout,
            min_start=min_start,
            tie_preference=tie_preference,
        )
        if match is None:
            continue
        return DmpBoundarySearch(
            match=match,
            copied_start=copied_start,
            copied_end=copied_end,
            chunks_checked=index + 1,
        )
    return None


def token_index_at_or_after(tokens, char_offset: int) -> int:
    # Converts a normalized character offset into the first token index at/after it.
    for index, token in enumerate(tokens):
        if token.end > char_offset:
            return index
    return len(tokens)


def token_index_ending_at_or_before(tokens, char_offset: int) -> int:
    # Converts a normalized character offset into an exclusive token limit.
    for index, token in enumerate(tokens):
        if token.end > char_offset:
            return index
    return len(tokens)


def token_chunk_char_span(tokens, start_token: int, end_token: int) -> tuple[int, int]:
    # Maps copied token chunk indexes back to normalized character offsets.
    if not tokens or start_token >= end_token:
        return 0, 0
    return tokens[start_token].start, tokens[end_token - 1].end


def exact_dmp_window_match(
    clean_normalized: str,
    copied_window: str,
    copied_start: int,
    min_start: int = 0,
    max_end: int | None = None,
    tie_preference: str = "first",
) -> DmpBoundaryMatch | None:
    # Fast path: exact normalized substring matches are better than approximate DMP search.
    if not copied_window:
        return None
    search_end = len(clean_normalized) if max_end is None else max_end
    search_start = max(0, min_start)
    if search_end <= search_start:
        return None
    if tie_preference == "last":
        matched_at = clean_normalized.rfind(copied_window, search_start, search_end)
    else:
        matched_at = clean_normalized.find(copied_window, search_start, search_end)
    if matched_at < 0:
        return None
    return DmpBoundaryMatch(
        score=1.0,
        normalized_start=matched_at,
        normalized_end=matched_at + len(copied_window),
        chunk_offset=copied_start,
        matched_at=matched_at,
        candidates_checked=0,
    )


def token_dmp_window_match(
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    clean_tokens,
    copied_tokens,
    inverted_index: dict[str, list[int]],
    frequencies: Counter[str],
    copied_start_token: int,
    copied_end_token: int,
    anchor_tokens: list[str],
    min_coverage: float,
    locator_min_score: float,
    locator_max_candidates: int,
    timeout: float,
    min_start: int = 0,
    max_end: int | None = None,
    tie_preference: str = "first",
) -> DmpBoundaryMatch | None:
    # Uses token overlap to find a local candidate, then verifies it with DMP coverage.
    copied_start, copied_end = token_chunk_char_span(
        copied_tokens,
        copied_start_token,
        copied_end_token,
    )
    copied_window = copied_normalized[copied_start:copied_end]
    exact_match = exact_dmp_window_match(
        clean_normalized,
        copied_window,
        copied_start,
        min_start=min_start,
        max_end=max_end,
        tie_preference=tie_preference,
    )
    if exact_match is not None:
        return exact_match

    min_start_token = token_index_at_or_after(clean_tokens, min_start)
    max_end_token = (
        token_index_ending_at_or_before(clean_tokens, max_end)
        if max_end is not None
        else None
    )
    token_match = find_token_anchor_match(
        clean_tokens,
        anchor_tokens,
        inverted_index,
        frequencies,
        locator_min_score,
        max_candidates=locator_max_candidates,
        confirm_mode="none",
        min_start_token=min_start_token,
        max_end_token=max_end_token,
        tie_preference=tie_preference,
    )
    if token_match is None:
        return None
    clean_window = clean_normalized[token_match.start:token_match.end]
    score, span_start, span_end = dmp_equal_coverage_and_span(
        diff_match_patch,
        clean_window,
        copied_window,
        timeout,
    )
    if score < min_coverage:
        return None
    return DmpBoundaryMatch(
        score=score,
        normalized_start=token_match.start + span_start,
        normalized_end=token_match.start + span_end,
        chunk_offset=copied_start,
        matched_at=token_match.start,
        candidates_checked=token_match.candidates_checked,
    )


def find_token_dmp_boundary_search(
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    clean_tokens,
    copied_tokens,
    inverted_index: dict[str, list[int]],
    frequencies: Counter[str],
    side: str,
    window_tokens: int,
    min_coverage: float,
    locator_min_score: float,
    locator_max_candidates: int,
    timeout: float,
    min_start: int = 0,
) -> DmpBoundarySearch | None:
    # Progressively checks copied 60-token chunks until token+DMP finds a boundary.
    effective_window_tokens = window_tokens
    if len(copied_tokens) <= window_tokens and len(copied_tokens) >= 16:
        effective_window_tokens = max(8, len(copied_tokens) // 2)
    chunks = token_anchor_chunks(copied_tokens, side, effective_window_tokens)
    tie_preference = "last" if side == "end" else "first"
    for index, (copied_start_token, copied_end_token, anchor_tokens) in enumerate(chunks):
        match = token_dmp_window_match(
            diff_match_patch,
            clean_normalized,
            copied_normalized,
            clean_tokens,
            copied_tokens,
            inverted_index,
            frequencies,
            copied_start_token,
            copied_end_token,
            anchor_tokens,
            min_coverage,
            locator_min_score,
            locator_max_candidates,
            timeout,
            min_start=min_start,
            tie_preference=tie_preference,
        )
        if match is None:
            continue
        copied_start, copied_end = token_chunk_char_span(
            copied_tokens,
            copied_start_token,
            copied_end_token,
        )
        return DmpBoundarySearch(
            match=match,
            copied_start=copied_start,
            copied_end=copied_end,
            chunks_checked=index + 1,
            copied_start_token=copied_start_token,
            copied_end_token=copied_end_token,
            locator="token+dmp",
        )
    return None


def dmp_boundary_with_chunks(
    boundary: DmpBoundarySearch, chunks_checked: int
) -> DmpBoundarySearch:
    # Preserves the final match while recording failed refinement attempts.
    return DmpBoundarySearch(
        match=boundary.match,
        copied_start=boundary.copied_start,
        copied_end=boundary.copied_end,
        chunks_checked=chunks_checked,
        copied_start_token=boundary.copied_start_token,
        copied_end_token=boundary.copied_end_token,
        locator=boundary.locator,
    )


def refine_dmp_start_boundary(
    boundary: DmpBoundarySearch,
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    refine_chars: int,
    min_score: float,
    chunk_size: int,
    max_chunks: int,
    match_threshold: float,
    timeout: float,
    max_failures: int,
) -> DmpBoundarySearch:
    # Walks backward over earlier copied character windows to recover skipped start text.
    current = boundary
    chunks_checked = boundary.chunks_checked
    failures = 0
    refine_chunks = dmp_char_chunks_in_range(
        len(copied_normalized),
        0,
        current.copied_start,
        refine_chars,
        chunk_size,
        reverse=True,
    )
    for copied_start, copied_end in refine_chunks:
        if failures >= max_failures:
            break
        chunks_checked += 1
        copied_window = copied_normalized[copied_start:copied_end]
        match = find_dmp_anchor_match(
            diff_match_patch,
            clean_normalized,
            copied_window,
            min_score,
            chunk_size,
            max_chunks,
            match_threshold,
            timeout,
            max_end=current.match.normalized_start,
            tie_preference="last",
        )
        if match is None:
            failures += 1
            continue
        current = DmpBoundarySearch(match, copied_start, copied_end, chunks_checked)
        failures = 0
    return dmp_boundary_with_chunks(current, chunks_checked)


def refine_dmp_end_boundary(
    boundary: DmpBoundarySearch,
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    refine_chars: int,
    min_score: float,
    chunk_size: int,
    max_chunks: int,
    match_threshold: float,
    timeout: float,
    max_failures: int,
) -> DmpBoundarySearch:
    # Walks forward over later copied character windows to recover matched tail text.
    current = boundary
    chunks_checked = boundary.chunks_checked
    failures = 0
    refine_chunks = dmp_char_chunks_in_range(
        len(copied_normalized),
        current.copied_end,
        len(copied_normalized),
        refine_chars,
        chunk_size,
    )
    for copied_start, copied_end in refine_chunks:
        if failures >= max_failures:
            break
        chunks_checked += 1
        copied_window = copied_normalized[copied_start:copied_end]
        match = find_dmp_anchor_match(
            diff_match_patch,
            clean_normalized,
            copied_window,
            min_score,
            chunk_size,
            max_chunks,
            match_threshold,
            timeout,
            min_start=current.match.normalized_end,
            tie_preference="first",
        )
        if match is None:
            failures += 1
            continue
        current = DmpBoundarySearch(match, copied_start, copied_end, chunks_checked)
        failures = 0
    return dmp_boundary_with_chunks(current, chunks_checked)


def refine_token_dmp_start_boundary(
    boundary: DmpBoundarySearch,
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    clean_tokens,
    copied_tokens,
    inverted_index: dict[str, list[int]],
    frequencies: Counter[str],
    refine_tokens: int,
    min_coverage: float,
    locator_min_score: float,
    locator_max_candidates: int,
    timeout: float,
    max_failures: int,
) -> DmpBoundarySearch:
    # Walks backward over earlier copied token chunks and verifies each with DMP.
    current = boundary
    chunks_checked = boundary.chunks_checked
    failures = 0
    previous_chunks = token_chunks_in_range(
        copied_tokens,
        0,
        current.copied_start_token,
        refine_tokens,
    )
    for copied_start_token, copied_end_token, anchor_tokens in reversed(previous_chunks):
        if failures >= max_failures:
            break
        chunks_checked += 1
        match = token_dmp_window_match(
            diff_match_patch,
            clean_normalized,
            copied_normalized,
            clean_tokens,
            copied_tokens,
            inverted_index,
            frequencies,
            copied_start_token,
            copied_end_token,
            anchor_tokens,
            min_coverage,
            locator_min_score,
            locator_max_candidates,
            timeout,
            max_end=current.match.normalized_start,
            tie_preference="last",
        )
        if match is None:
            failures += 1
            continue
        copied_start, copied_end = token_chunk_char_span(
            copied_tokens,
            copied_start_token,
            copied_end_token,
        )
        current = DmpBoundarySearch(
            match=match,
            copied_start=copied_start,
            copied_end=copied_end,
            chunks_checked=chunks_checked,
            copied_start_token=copied_start_token,
            copied_end_token=copied_end_token,
            locator="token+dmp",
        )
        failures = 0
    return dmp_boundary_with_chunks(current, chunks_checked)


def refine_token_dmp_end_boundary(
    boundary: DmpBoundarySearch,
    diff_match_patch,
    clean_normalized: str,
    copied_normalized: str,
    clean_tokens,
    copied_tokens,
    inverted_index: dict[str, list[int]],
    frequencies: Counter[str],
    refine_tokens: int,
    min_coverage: float,
    locator_min_score: float,
    locator_max_candidates: int,
    timeout: float,
    max_failures: int,
) -> DmpBoundarySearch:
    # Walks forward over later copied token chunks and verifies each with DMP.
    current = boundary
    chunks_checked = boundary.chunks_checked
    failures = 0
    next_chunks = token_chunks_in_range(
        copied_tokens,
        current.copied_end_token,
        len(copied_tokens),
        refine_tokens,
    )
    for copied_start_token, copied_end_token, anchor_tokens in next_chunks:
        if failures >= max_failures:
            break
        chunks_checked += 1
        match = token_dmp_window_match(
            diff_match_patch,
            clean_normalized,
            copied_normalized,
            clean_tokens,
            copied_tokens,
            inverted_index,
            frequencies,
            copied_start_token,
            copied_end_token,
            anchor_tokens,
            min_coverage,
            locator_min_score,
            locator_max_candidates,
            timeout,
            min_start=current.match.normalized_end,
            tie_preference="first",
        )
        if match is None:
            failures += 1
            continue
        copied_start, copied_end = token_chunk_char_span(
            copied_tokens,
            copied_start_token,
            copied_end_token,
        )
        current = DmpBoundarySearch(
            match=match,
            copied_start=copied_start,
            copied_end=copied_end,
            chunks_checked=chunks_checked,
            copied_start_token=copied_start_token,
            copied_end_token=copied_end_token,
            locator="token+dmp",
        )
        failures = 0
    return dmp_boundary_with_chunks(current, chunks_checked)


def dmp_partial_match(
    clean_text: str,
    copied_text: str,
    min_coverage: float = 0.72,
    anchor_chars: int = 300,
    refine_chars: int = 100,
    chunk_size: int = 32,
    max_chunks: int = 16,
    match_threshold: float = 0.35,
    timeout: float = 1.0,
    max_refine_failures: int = 2,
    copied_image_captions: list[str] | None = None,
    locator_window_tokens: int = 60,
    locator_refine_tokens: int = 20,
    locator_min_score: float = 0.72,
    locator_max_candidates: int = 240,
) -> tuple[str, str]:
    # Uses DMP anchor chunks to infer the clean span corresponding to pasted text.
    if chunk_size > 32:
        raise ValueError("chunk_size must be 32 or less for diff-match-patch match_main")
    if refine_chars < max(40, chunk_size):
        raise ValueError("refine_chars must be at least 40 and no smaller than chunk_size")
    if locator_window_tokens < 8:
        raise ValueError("locator_window_tokens must be at least 8")
    if locator_refine_tokens < 8:
        raise ValueError("locator_refine_tokens must be at least 8")
    diff_match_patch = load_diff_match_patch()
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
    ignored_spans = heading_section_spans(
        clean_index.headings,
        HYBRID_IGNORED_HEADING_TITLES,
        len(clean_text),
    )
    clean_normalized, clean_map = normalize_for_dmp_with_map(clean_text, ignored_spans)
    copied_normalized, _copied_map = normalize_for_dmp_with_map(copied_clean)
    if len(copied_normalized) < max(40, chunk_size):
        message = "failed: not enough copied text for DMP matching"
        report = "\n".join(
            [
                "Wikipedia DMP Partial Extraction Report",
                "",
                message,
                f"Copied normalized characters: {len(copied_normalized)}",
            ]
        )
        raise PartialExtractionError(message, report)
    clean_tokens = tokenize_with_offsets(clean_normalized)
    copied_tokens = tokenize_with_offsets(copied_normalized)
    if len(copied_tokens) < 8:
        message = "failed: not enough copied tokens for DMP matching"
        report = "\n".join(
            [
                "Wikipedia DMP Partial Extraction Report",
                "",
                message,
                f"Copied tokens: {len(copied_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)
    inverted_index = build_token_inverted_index(clean_tokens)
    frequencies: Counter[str] = Counter(token.value for token in clean_tokens)

    match_started_at = time.perf_counter()
    start_boundary = find_token_dmp_boundary_search(
        diff_match_patch,
        clean_normalized,
        copied_normalized,
        clean_tokens,
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
        message = "failed: DMP start boundary did not meet the minimum coverage"
        report = "\n".join(
            [
                "Wikipedia DMP Partial Extraction Report",
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
        clean_normalized,
        copied_normalized,
        clean_tokens,
        copied_tokens,
        inverted_index,
        frequencies,
        locator_refine_tokens,
        min_coverage,
        locator_min_score,
        locator_max_candidates,
        timeout,
        max_refine_failures,
    )

    end_boundary = find_token_dmp_boundary_search(
        diff_match_patch,
        clean_normalized,
        copied_normalized,
        clean_tokens,
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
    if end_boundary is None:
        message = "failed: DMP end boundary did not meet the minimum coverage"
        report = "\n".join(
            [
                "Wikipedia DMP Partial Extraction Report",
                "",
                message,
                f"Minimum coverage: {min_coverage:.3f}",
                f"Start score: {start_boundary.match.score:.3f}",
                f"Locator window tokens: {min(locator_window_tokens, len(copied_tokens))}",
                f"Start copied char chunk: {start_boundary.copied_start}-{start_boundary.copied_end}",
            ]
        )
        raise PartialExtractionError(message, report)
    end_boundary = refine_token_dmp_end_boundary(
        end_boundary,
        diff_match_patch,
        clean_normalized,
        copied_normalized,
        clean_tokens,
        copied_tokens,
        inverted_index,
        frequencies,
        locator_refine_tokens,
        min_coverage,
        locator_min_score,
        locator_max_candidates,
        timeout,
        max_refine_failures,
    )
    match_seconds = time.perf_counter() - match_started_at
    start_match = start_boundary.match
    end_match = end_boundary.match

    start, _start_end = original_span_from_dmp_map(
        clean_map,
        start_match.normalized_start,
        max(start_match.normalized_start + 1, start_match.normalized_start + chunk_size),
    )
    _end_start, end = original_span_from_dmp_map(
        clean_map,
        max(0, end_match.normalized_end - chunk_size),
        end_match.normalized_end,
    )
    start = snap_start_to_sentence(clean_index.sentences, start)
    end = snap_end_to_sentence(clean_index.sentences, end)
    if start >= end:
        message = "failed: DMP boundaries are invalid"
        report = "\n".join(
            [
                "Wikipedia DMP Partial Extraction Report",
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
        "Wikipedia DMP Partial Extraction Report",
        "",
        f"Clean normalized characters: {len(clean_normalized)}",
        f"Copied normalized characters: {len(copied_normalized)}",
        f"Clean tokens: {len(clean_tokens)}",
        f"Copied tokens: {len(copied_tokens)}",
        f"Start coverage: {start_match.score:.3f}",
        f"End coverage: {end_match.score:.3f}",
        f"Minimum coverage: {min_coverage:.3f}",
        f"Anchor characters: {min(anchor_chars, len(copied_normalized))}",
        f"Refine characters: {min(refine_chars, len(copied_normalized))}",
        f"Locator window tokens: {min(locator_window_tokens, len(copied_tokens))}",
        f"Locator refine tokens: {min(locator_refine_tokens, len(copied_tokens))}",
        f"Locator minimum token score: {locator_min_score:.3f}",
        f"Locator max candidates: {locator_max_candidates}",
        f"Chunk size: {min(chunk_size, 32)}",
        f"Max chunks: {max_chunks}",
        f"Max refine failures: {max_refine_failures}",
        f"Match threshold: {match_threshold:.3f}",
        f"Local DMP timeout: {timeout:.3f} seconds",
        f"DMP match runtime: {match_seconds:.3f} seconds",
        f"Start locator: {start_boundary.locator}",
        f"Start copied char chunk: {start_boundary.copied_start}-{start_boundary.copied_end}",
        f"Start copied token chunk: {start_boundary.copied_start_token}-{start_boundary.copied_end_token}",
        f"Start chunks checked/refined: {start_boundary.chunks_checked}",
        f"Start head unmatched: {'yes' if start_boundary.copied_start > 0 else 'no'}",
        f"Start candidates checked: {start_match.candidates_checked}",
        f"Start chunk offset: {start_match.chunk_offset}",
        f"Start matched at: {start_match.matched_at}",
        f"End locator: {end_boundary.locator}",
        f"End copied char chunk: {end_boundary.copied_start}-{end_boundary.copied_end}",
        f"End copied token chunk: {end_boundary.copied_start_token}-{end_boundary.copied_end_token}",
        f"End chunks checked/refined: {end_boundary.chunks_checked}",
        f"End tail unmatched: {'yes' if end_boundary.copied_end < len(copied_normalized) else 'no'}",
        f"End candidates checked: {end_match.candidates_checked}",
        f"End chunk offset: {end_match.chunk_offset}",
        f"End matched at: {end_match.matched_at}",
        f"Copied image captions stripped: {stripped_caption_count}",
        f"Start offset: {start}",
        f"End offset: {end}",
        f"Output characters: {len(text)}",
    ]
    return text, "\n".join(report_lines)


def main(argv: Iterable[str] | None = None) -> int:
    # Fetches refs-enabled clean HTML text, then slices it with DMP alignment.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_dmp_output_path(args.output, page)
    report_path = partial_dmp_report_path(args.output, page)
    runtime_path = runtime_output_path(args.output, page)

    try:
        pasted_text = read_pasted_text(input_path)
        started_at = time.perf_counter()
        fetch_started_at = time.perf_counter()
        html = fetch_page_html(page)
        fetch_seconds = time.perf_counter() - fetch_started_at
        clean_started_at = time.perf_counter()
        full_text = clean_wikipedia_html_with_references(
            html,
            args.math,
            include_inline_markers=True,
        )
        captions = extract_image_captions_from_html(html)
        clean_seconds = time.perf_counter() - clean_started_at
        match_started_at = time.perf_counter()
        text, report = dmp_partial_match(
            full_text,
            pasted_text,
            min_coverage=args.min_coverage,
            anchor_chars=args.anchor_chars,
            refine_chars=args.refine_chars,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            match_threshold=args.match_threshold,
            timeout=args.timeout,
            max_refine_failures=args.max_refine_failures,
            copied_image_captions=captions,
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
                f"Clean runtime: {clean_seconds:.3f} seconds",
                f"Match runtime: {match_seconds:.3f} seconds",
                f"Input file: {input_path}",
                f"Math mode: {args.math}",
            ]
        )
        write_text_file(output_path, text)
        write_text_file(report_path, report_text)
        update_runtime_report(
            runtime_path,
            page,
            {runtime_label("Partial DMP runtime", args.math, page): seconds},
        )
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated DMP match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved DMP partial text: {output_path}")
    print(f"Updated DMP match report: {report_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
