"""Extract a clean partial section using heading/citation hybrid matching."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    PartialExtractionError,
    clean_wikipedia_html_with_references,
    extract_partial_hybrid_text,
    fetch_page_html,
    page_request_from_url,
    partial_hybrid_output_path,
    partial_hybrid_report_path,
    runtime_label,
    runtime_output_path,
    update_runtime_report,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"


def build_parser() -> argparse.ArgumentParser:
    # Defines the CLI for the newer heading/citation-aware partial extraction method.
    parser = argparse.ArgumentParser(
        description="Extract a partial clean Wikipedia section using heading/citation anchors."
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
        default="output/partial_hybrid_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled before partial matching",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.84,
        help="Minimum sentence-window score for structural boundary refinement",
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
    # Fetches refs-enabled clean HTML text, then slices it with the hybrid matcher.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_hybrid_output_path(args.output, page)
    report_path = partial_hybrid_report_path(args.output, page)
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
        clean_seconds = time.perf_counter() - clean_started_at
        match_started_at = time.perf_counter()
        result = extract_partial_hybrid_text(
            full_text,
            pasted_text,
            threshold=args.threshold,
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
            {runtime_label("Partial hybrid runtime", args.math, page): seconds},
        )
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved hybrid partial text: {output_path}")
    print(f"Updated hybrid match report: {report_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
