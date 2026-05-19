"""Extract a clean partial section using fast token matching."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    PartialExtractionError,
    TOKEN_CONFIRM_MODES,
    clean_wikipedia_html_with_references,
    extract_image_captions_from_html,
    extract_partial_token_text,
    fetch_page_html,
    page_request_from_url,
    partial_token_output_path,
    partial_token_report_path,
    runtime_label,
    runtime_output_path,
    update_runtime_report,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"


def build_parser() -> argparse.ArgumentParser:
    # Defines the CLI for the experimental inverted-index token partial matcher.
    parser = argparse.ArgumentParser(
        description="Extract a partial clean Wikipedia section using token overlap and ordered coverage."
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
        default="output/partial_token_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled before partial matching",
    )
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=120,
        help="Number of pasted start/end tokens used as boundary anchors",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.72,
        help="Minimum combined token score for start/end matches",
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


def main(argv: Iterable[str] | None = None) -> int:
    # Fetches refs-enabled clean HTML text, then slices it with token matching.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_token_output_path(args.output, page)
    report_path = partial_token_report_path(args.output, page)
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
        result = extract_partial_token_text(
            full_text,
            pasted_text,
            window_tokens=args.window_tokens,
            min_score=args.min_score,
            confirm_mode=args.confirm,
            max_candidates=args.max_candidates,
            copied_image_captions=captions,
        )
        match_seconds = time.perf_counter() - match_started_at
        seconds = time.perf_counter() - started_at
        report_text = "\n".join(
            [
                result.report,
                "",
                f"Total runtime: {seconds:.3f} seconds",
                f"Fetch runtime: {fetch_seconds:.3f} seconds",
                f"Clean runtime: {clean_seconds:.3f} seconds",
                f"Match runtime: {match_seconds:.3f} seconds",
                f"Input file: {input_path}",
                f"Math mode: {args.math}",
            ]
        )
        write_text_file(output_path, result.text)
        write_text_file(report_path, report_text)
        update_runtime_report(
            runtime_path,
            page,
            {runtime_label("Partial token runtime", args.math, page): seconds},
        )
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated token match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved token partial text: {output_path}")
    print(f"Updated token match report: {report_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
