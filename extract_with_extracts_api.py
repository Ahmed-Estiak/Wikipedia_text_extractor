"""Run the Wikipedia extracts API text extractor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from wiki_text_extractor import (
    PageRequest,
    extract_text_from_extracts_api,
    page_request_from_url,
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
        text = extract_text_from_extracts_api(page)
        write_text_file(Path(args.output), text)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved extracts API text: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
