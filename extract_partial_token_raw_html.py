"""Extract a clean partial section by token-matching pasted text against raw HTML."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from extract_partial_dmp import read_pasted_text
from extract_partial_dmp_raw_html import (
    avoid_cutting_inside_tag,
    html_offset_from_visible_index,
    raw_html_visible_text_with_map,
    snap_visible_end_to_sentence,
    snap_visible_start_to_sentence,
)
from wiki_text_extractor import (
    MAINTENANCE_MARKER_PATTERN,
    PageRequest,
    PartialExtractionError,
    TOKEN_CONFIRM_MODES,
    build_token_inverted_index,
    clean_wikipedia_html_with_references,
    fetch_page_html,
    finalize_partial_hybrid_output,
    find_token_anchor_match,
    output_directory,
    page_request_from_url,
    runtime_label,
    runtime_output_path,
    tokenize_with_offsets,
    update_runtime_report,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"


def build_parser() -> argparse.ArgumentParser:
    # Defines the CLI for the experimental raw-HTML token matcher.
    parser = argparse.ArgumentParser(
        description="Extract a partial clean Wikipedia section by token-matching pasted text against raw HTML."
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
        default="output/partial_token_raw_html_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled after the raw HTML slice is found",
    )
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=60,
        help="Number of pasted start/end tokens used as one-shot raw HTML anchors",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.72,
        help="Minimum combined token score for start/end raw HTML anchor matches",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=240,
        help="Maximum inverted-index candidate windows to score per boundary",
    )
    parser.add_argument(
        "--confirm",
        choices=TOKEN_CONFIRM_MODES,
        default="none",
        help="Optional close-candidate confirmation mode",
    )
    parser.add_argument(
        "--no-snap-sentences",
        action="store_true",
        help="Disable final sentence-boundary snapping after raw token anchors are found",
    )
    return parser


def partial_token_raw_output_path(output: str, page: PageRequest) -> Path:
    # Stores the latest raw-HTML token clean partial output beside other partial outputs.
    return output_directory(output, page) / "partial_token_raw_html_text.txt"


def partial_token_raw_report_path(output: str, page: PageRequest) -> Path:
    # Stores raw-HTML token boundary diagnostics.
    return output_directory(output, page) / "partial_token_raw_html_match_report.txt"


def token_values(tokens) -> list[str]:
    # Keeps anchor selection readable without changing the shared token model.
    return [token.value for token in tokens]


def raw_html_offset_for_token(raw_visible_map: list[int], visible_offset: int, end: bool = False) -> int:
    # Maps a raw-visible token offset back to raw HTML source text.
    return html_offset_from_visible_index(raw_visible_map, visible_offset, end=end)


def raw_token_partial_match(
    html: str,
    copied_text: str,
    math_mode: str = "latex",
    window_tokens: int = 60,
    min_score: float = 0.72,
    max_candidates: int = 240,
    confirm_mode: str = "none",
    snap_sentences: bool = True,
) -> tuple[str, str]:
    # Uses one start anchor and one end anchor against raw-visible HTML, then cleans the raw slice.
    if confirm_mode not in TOKEN_CONFIRM_MODES:
        raise ValueError(f"Unsupported token confirm mode: {confirm_mode}")
    if window_tokens < 8:
        raise ValueError("window_tokens must be at least 8")

    raw_visible, raw_visible_map = raw_html_visible_text_with_map(html)
    copied_matching_text = MAINTENANCE_MARKER_PATTERN.sub("", copied_text)
    raw_tokens = tokenize_with_offsets(raw_visible)
    copied_tokens = tokenize_with_offsets(copied_matching_text)
    if len(copied_tokens) < 8:
        message = "failed: not enough copied tokens for raw-HTML token matching"
        report = "\n".join(
            [
                "Wikipedia Raw HTML Token Partial Extraction Report",
                "",
                message,
                f"Copied tokens: {len(copied_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)
    if len(raw_tokens) < 8:
        message = "failed: not enough raw HTML visible tokens for matching"
        report = "\n".join(
            [
                "Wikipedia Raw HTML Token Partial Extraction Report",
                "",
                message,
                f"Raw visible tokens: {len(raw_tokens)}",
            ]
        )
        raise PartialExtractionError(message, report)

    anchor_size = min(window_tokens, len(copied_tokens))
    copied_values = token_values(copied_tokens)
    start_anchor = copied_values[:anchor_size]
    end_anchor = copied_values[-anchor_size:]
    inverted_index = build_token_inverted_index(raw_tokens)
    frequencies: Counter[str] = Counter(token.value for token in raw_tokens)

    match_started_at = time.perf_counter()
    start_match = find_token_anchor_match(
        raw_tokens,
        start_anchor,
        inverted_index,
        frequencies,
        min_score,
        max_candidates=max_candidates,
        confirm_mode=confirm_mode,
        tie_preference="first",
    )
    if start_match is None:
        message = "failed: raw-HTML token start anchor did not meet the minimum score"
        report = "\n".join(
            [
                "Wikipedia Raw HTML Token Partial Extraction Report",
                "",
                message,
                f"Minimum score: {min_score:.3f}",
                f"Anchor tokens: {anchor_size}",
            ]
        )
        raise PartialExtractionError(message, report)

    end_match = find_token_anchor_match(
        raw_tokens,
        end_anchor,
        inverted_index,
        frequencies,
        min_score,
        max_candidates=max_candidates,
        confirm_mode=confirm_mode,
        min_start_token=start_match.start_token,
        tie_preference="last",
    )
    match_seconds = time.perf_counter() - match_started_at
    if end_match is None:
        message = "failed: raw-HTML token end anchor did not meet the minimum score"
        report = "\n".join(
            [
                "Wikipedia Raw HTML Token Partial Extraction Report",
                "",
                message,
                f"Minimum score: {min_score:.3f}",
                f"Start token score: {start_match.score:.3f}",
                f"Anchor tokens: {anchor_size}",
            ]
        )
        raise PartialExtractionError(message, report)

    start_visible = raw_tokens[start_match.start_token].start
    end_visible = raw_tokens[end_match.end_token - 1].end
    snapped_start_visible = snap_visible_start_to_sentence(raw_visible, start_visible) if snap_sentences else start_visible
    snapped_end_visible = snap_visible_end_to_sentence(raw_visible, end_visible) if snap_sentences else end_visible
    raw_start = raw_html_offset_for_token(raw_visible_map, snapped_start_visible)
    raw_end = raw_html_offset_for_token(raw_visible_map, snapped_end_visible, end=True)
    raw_start, raw_end = avoid_cutting_inside_tag(html, raw_start, raw_end)
    if raw_start >= raw_end:
        message = "failed: raw-HTML token boundaries are invalid"
        report = "\n".join(
            [
                "Wikipedia Raw HTML Token Partial Extraction Report",
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
        message = "failed: raw-HTML token slice cleaned to empty text"
        report = "\n".join(
            [
                "Wikipedia Raw HTML Token Partial Extraction Report",
                "",
                message,
                f"Raw start offset: {raw_start}",
                f"Raw end offset: {raw_end}",
            ]
        )
        raise PartialExtractionError(message, report)

    report_lines = [
        "Wikipedia Raw HTML Token Partial Extraction Report",
        "",
        f"Raw HTML characters: {len(html)}",
        f"Raw visible characters: {len(raw_visible)}",
        f"Raw visible tokens: {len(raw_tokens)}",
        f"Copied tokens: {len(copied_tokens)}",
        f"Anchor tokens: {anchor_size}",
        f"Minimum score: {min_score:.3f}",
        f"Confirm mode: {confirm_mode}",
        f"Sentence snap: {'yes' if snap_sentences else 'no'}",
        f"Token match runtime: {match_seconds:.3f} seconds",
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
        f"Start raw-visible token: {start_match.start_token}",
        f"End raw-visible token: {end_match.end_token}",
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
    # Fetches raw HTML once, token-matches pasted text against it, and cleans the matched slice.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_token_raw_output_path(args.output, page)
    report_path = partial_token_raw_report_path(args.output, page)
    runtime_path = runtime_output_path(args.output, page)

    try:
        pasted_text = read_pasted_text(input_path)
        started_at = time.perf_counter()
        fetch_started_at = time.perf_counter()
        html = fetch_page_html(page)
        fetch_seconds = time.perf_counter() - fetch_started_at
        match_started_at = time.perf_counter()
        text, report = raw_token_partial_match(
            html,
            pasted_text,
            math_mode=args.math,
            window_tokens=args.window_tokens,
            min_score=args.min_score,
            max_candidates=args.max_candidates,
            confirm_mode=args.confirm,
            snap_sentences=not args.no_snap_sentences,
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
            {runtime_label("Partial token raw HTML runtime", args.math, page): seconds},
        )
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated raw-HTML token match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved raw-HTML token partial text: {output_path}")
    print(f"Updated raw-HTML token match report: {report_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
