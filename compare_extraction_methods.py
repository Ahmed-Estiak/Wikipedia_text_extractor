"""Run both Wikipedia text extraction methods and compare their outputs."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from wiki_text_extractor import (
    EXTRACTION_METHODS,
    MATH_MODES,
    PageRequest,
    compare_texts,
    comparison_output_path,
    extraction_output_path,
    page_request_from_url,
    run_extraction,
    runtime_output_path,
    write_text_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run both Wikipedia extractors and compare their text output."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "--math",
        choices=("all", "remove", "latex", "keep"),
        default="all",
        help="Math modes to run; all creates six extractor output files",
    )
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
    math_modes = MATH_MODES if args.math == "all" else (args.math,)

    try:
        results = {
            (method, math_mode): run_extraction(page, method, math_mode)
            for math_mode in math_modes
            for method in EXTRACTION_METHODS
        }
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for (method, math_mode), (text, _seconds) in results.items():
        path = extraction_output_path(args.output, page, method, math_mode)
        write_text_file(path, text)
        print(f"Saved {method} text ({math_mode}): {path}")

    comparison_path = comparison_output_path(args.output, page)
    if "remove" in math_modes:
        extracts_remove = results[("extracts", "remove")][0]
        html_remove = results[("html", "remove")][0]
        write_text_file(comparison_path, compare_texts(extracts_remove, html_remove))
        print(f"Saved remove-mode comparison report: {comparison_path}")

    runtime_lines = ["Wikipedia Text Extraction Runtime", "", f"Math mode request: {args.math}"]
    for math_mode in math_modes:
        extracts_seconds = results[("extracts", math_mode)][1]
        html_seconds = results[("html", math_mode)][1]
        runtime_lines.extend(
            [
                "",
                f"Math mode: {math_mode}",
                f"Extracts API runtime: {extracts_seconds:.3f} seconds",
                f"HTML parser runtime: {html_seconds:.3f} seconds",
                f"Runtime mismatch: {abs(extracts_seconds - html_seconds):.3f} seconds",
            ]
        )
    runtime_report = "\n".join(runtime_lines)
    runtime_path = runtime_output_path(args.output, page)
    write_text_file(runtime_path, runtime_report)

    print(f"Saved runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
