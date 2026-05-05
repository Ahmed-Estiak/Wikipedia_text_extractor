"""Extract a clean partial section from a Wikipedia page using pasted text."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    PartialExtractionError,
    clean_wikipedia_html,
    extract_partial_text,
    fetch_page_html,
    format_partial_match_report,
    page_request_from_url,
    partial_match_report_path,
    partial_output_path,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"


def build_parser() -> argparse.ArgumentParser:
    # Defines the CLI for extracting only the pasted section from cleaned HTML text.
    parser = argparse.ArgumentParser(
        description="Extract a fuzzy-matched partial clean text section from Wikipedia HTML."
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
        default="output/partial_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="remove",
        help="How math equations should be handled before partial matching",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Minimum fuzzy score required for both start and end anchors",
    )
    parser.add_argument(
        "--anchor-size",
        type=int,
        default=300,
        help="Maximum character length for each fuzzy boundary anchor",
    )
    parser.add_argument(
        "--anchor-candidates",
        type=int,
        default=5,
        help="How many meaningful start/end pasted chunks to try",
    )
    return parser


def read_pasted_text(path: Path) -> str:
    # Reads user-pasted copied text and gives a clear error if the input is missing.
    if not path.exists():
        raise FileNotFoundError(
            f"Input text file not found: {path}. Create it and paste the Wikipedia text there."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input text file is empty: {path}")
    return text


def main(argv: Iterable[str] | None = None) -> int:
    # Fetches the page, cleans full HTML text, then extracts only the matched section.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_output_path(args.output, page)
    report_path = partial_match_report_path(args.output, page)

    try:
        pasted_text = read_pasted_text(input_path)
        started_at = time.perf_counter()
        html = fetch_page_html(page)
        full_text = clean_wikipedia_html(html, args.math)
        result = extract_partial_text(
            full_text,
            pasted_text,
            threshold=args.threshold,
            anchor_size=args.anchor_size,
            max_candidates=args.anchor_candidates,
        )
        seconds = time.perf_counter() - started_at
        report_text = "\n".join(
            [
                format_partial_match_report(result, args.threshold),
                "",
                f"Runtime: {seconds:.3f} seconds",
                f"Input file: {input_path}",
                f"Math mode: {args.math}",
            ]
        )
        write_text_file(output_path, result.text)
        write_text_file(report_path, report_text)
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved partial text: {output_path}")
    print(f"Updated match report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
