"""Run the Wikipedia extracts API text extractor."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    extraction_output_path,
    page_request_from_url,
    run_extraction,
    runtime_output_path,
    write_text_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract clean Wikipedia text using prop=extracts."
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
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)

    try:
        text, seconds = run_extraction(page, "extracts", args.math)
        output_path = extraction_output_path(args.output, page, "extracts", args.math)
        runtime_path = runtime_output_path(args.output, page)
        write_text_file(output_path, text)
        write_text_file(
            runtime_path,
            "\n".join(
                [
                    "Wikipedia Text Extraction Runtime",
                    "",
                    f"Latest run: extracts API / math {args.math}",
                    f"Extracts API runtime ({args.math}): {seconds:.3f} seconds",
                ]
            ),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved extracts API text: {output_path}")
    print(f"Updated runtime report: {runtime_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
