"""Run the Wikipedia parse HTML text extractor."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    add_topic_heading,
    clean_wikipedia_html,
    clean_wikipedia_html_with_references,
    extraction_output_path,
    fetch_page_html,
    page_request_from_url,
    references_output_path,
    run_extraction,
    runtime_label,
    runtime_output_path,
    update_runtime_report,
    write_text_file,
)


def build_parser() -> argparse.ArgumentParser:
    # Defines the HTML-only CLI, including optional citation/reference export.
    parser = argparse.ArgumentParser(
        description="Extract clean Wikipedia text by parsing page HTML."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="remove",
        help="How math equations should be handled",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Text file path where the cleaned output will be saved",
    )
    parser.add_argument(
        "--save-references",
        action="store_true",
        help="Also save an HTML-only text file with citation numbers and References",
    )
    parser.add_argument(
        "--references-end-only",
        action="store_true",
        help="When saving references, omit inline citation markers and keep numbered sources only at the end",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    # Parses command arguments and resolves either a URL or title into a PageRequest.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)

    try:
        references_path = None
        if args.save_references:
            # Fetches HTML once so both normal clean text and references output use the same source.
            started_at = time.perf_counter()
            html = fetch_page_html(page)
            text = add_topic_heading(clean_wikipedia_html(html, args.math), page)
            references_text = clean_wikipedia_html_with_references(
                html,
                args.math,
                include_inline_markers=not args.references_end_only,
            )
            references_text = add_topic_heading(references_text, page)
            seconds = time.perf_counter() - started_at
        else:
            # Runs the standard HTML extraction path when no references file is requested.
            text, seconds = run_extraction(page, "html", args.math)
            references_text = ""
        # Uses the shared topic/language folder layout for the cleaned HTML output.
        output_path = extraction_output_path(args.output, page, "html", args.math)
        runtime_path = runtime_output_path(args.output, page)
        write_text_file(output_path, text)
        if args.save_references:
            # Writes the citation-numbered article plus appended References as a separate file.
            references_path = references_output_path(args.output, page)
            write_text_file(references_path, references_text)
        # Updates the shared runtime file with the latest HTML-only run.
        update_runtime_report(
            runtime_path,
            page,
            {runtime_label("HTML parser runtime", args.math, page): seconds},
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved HTML parser text: {output_path}")
    if references_path:
        print(f"Saved HTML references text: {references_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
