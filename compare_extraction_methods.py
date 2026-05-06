"""Run both Wikipedia text extraction methods and compare their outputs."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from wiki_text_extractor import (
    EXTRACTION_METHODS,
    MATH_MODES,
    PageRequest,
    add_topic_heading,
    clean_wikipedia_html_with_references,
    compare_texts,
    comparison_output_path,
    extraction_output_path,
    fetch_page_extract,
    fetch_page_html,
    page_request_from_url,
    raw_output_path,
    references_output_path,
    run_extraction,
    runtime_output_path,
    update_runtime_report,
    write_text_file,
)


def build_parser() -> argparse.ArgumentParser:
    # Defines the combined CLI for running both extractors, debug outputs, and reports.
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
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw Wikipedia API responses for debugging",
    )
    parser.add_argument(
        "--save-references",
        action="store_true",
        help="Save an HTML-only text file with citation numbers and References",
    )
    parser.add_argument(
        "--references-end-only",
        action="store_true",
        help="When saving references, omit inline citation markers and keep numbered sources only at the end",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    # Parses command arguments, resolves the requested page, and expands math=all.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    math_modes = MATH_MODES if args.math == "all" else (args.math,)

    try:
        # Runs every requested method/math combination and keeps both text and runtime.
        results = {
            (method, math_mode): run_extraction(page, method, math_mode)
            for math_mode in math_modes
            for method in EXTRACTION_METHODS
        }
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Writes the normal extractor outputs, e.g. extracts/html with remove/latex/keep.
    for (method, math_mode), (text, _seconds) in results.items():
        path = extraction_output_path(args.output, page, method, math_mode)
        write_text_file(path, text)
        print(f"Saved {method} text ({math_mode}): {path}")

    if args.save_raw:
        try:
            # Saves raw API responses so parser/cleaning issues can be inspected later.
            raw_extracts_path = raw_output_path(args.output, page, "extracts")
            raw_html_path = raw_output_path(args.output, page, "html")
            write_text_file(raw_extracts_path, fetch_page_extract(page))
            write_text_file(raw_html_path, fetch_page_html(page))
            print(f"Saved raw extracts API text: {raw_extracts_path}")
            print(f"Saved raw parse HTML text: {raw_html_path}")
        except RuntimeError as exc:
            print(f"Error saving raw API responses: {exc}", file=sys.stderr)
            return 1

    if args.save_references:
        try:
            # Saves a separate HTML-only output with inline citation numbers and References.
            references_path = references_output_path(args.output, page)
            write_text_file(
                references_path,
                add_topic_heading(
                    clean_wikipedia_html_with_references(
                        fetch_page_html(page),
                        "remove",
                        include_inline_markers=not args.references_end_only,
                    ),
                    page,
                ),
            )
            print(f"Saved HTML references text: {references_path}")
        except RuntimeError as exc:
            print(f"Error saving HTML references text: {exc}", file=sys.stderr)
            return 1

    comparison_path = comparison_output_path(args.output, page)
    if "remove" in math_modes:
        # Comparison is intentionally limited to remove mode, the cleanest text baseline.
        extracts_remove = results[("extracts", "remove")][0]
        html_remove = results[("html", "remove")][0]
        write_text_file(comparison_path, compare_texts(extracts_remove, html_remove))
        print(f"Saved remove-mode comparison report: {comparison_path}")

    # Updates timing entries for each requested math mode and both extraction methods.
    runtime_updates: dict[str, float] = {}
    for math_mode in math_modes:
        extracts_seconds = results[("extracts", math_mode)][1]
        html_seconds = results[("html", math_mode)][1]
        runtime_updates[f"Extracts API runtime ({math_mode})"] = extracts_seconds
        runtime_updates[f"HTML parser runtime ({math_mode})"] = html_seconds
        runtime_updates[f"Runtime mismatch ({math_mode})"] = abs(
            extracts_seconds - html_seconds
        )
    runtime_path = runtime_output_path(args.output, page)
    update_runtime_report(runtime_path, page, runtime_updates)

    print(f"Saved runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
