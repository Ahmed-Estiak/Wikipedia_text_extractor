"""Run both Wikipedia text extraction methods and compare their outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    compare_texts,
    output_path_for_method,
    page_request_from_url,
    run_extraction,
    write_text_file,
)


def comparison_path_for_output(output: str) -> Path:
    path = Path(output)
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_comparison{suffix}")


def runtime_path_for_output(output: str) -> Path:
    path = Path(output)
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}_runtime{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run both Wikipedia extractors and compare their text output."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Base text path, e.g. output/saturn.txt",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)

    try:
        extracts_text, extracts_seconds = run_extraction(page, "extracts")
        html_text, html_seconds = run_extraction(page, "html")
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    extracts_path = output_path_for_method(args.output, "extracts", split_methods=True)
    html_path = output_path_for_method(args.output, "html", split_methods=True)
    comparison_path = comparison_path_for_output(args.output)
    runtime_path = runtime_path_for_output(args.output)

    write_text_file(extracts_path, extracts_text)
    write_text_file(html_path, html_text)
    write_text_file(comparison_path, compare_texts(extracts_text, html_text))

    runtime_report = "\n".join(
        [
            "Wikipedia Text Extraction Runtime",
            "",
            f"Extracts API runtime: {extracts_seconds:.3f} seconds",
            f"HTML parser runtime: {html_seconds:.3f} seconds",
            f"Runtime mismatch: {abs(extracts_seconds - html_seconds):.3f} seconds",
        ]
    )
    write_text_file(runtime_path, runtime_report)

    print(f"Saved extracts API text: {extracts_path}")
    print(f"Saved HTML parser text: {html_path}")
    print(f"Saved comparison report: {comparison_path}")
    print(f"Saved runtime report: {runtime_path}")
    print(f"Extracts API runtime: {extracts_seconds:.3f} seconds")
    print(f"HTML parser runtime: {html_seconds:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
