"""Extract partial text with the default shared fallback chain."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from extract_partial_dmp_raw_html import raw_dmp_partial_match, raw_html_visible_text_with_map
from extract_partial_hybrid import read_pasted_text
from extract_partial_token_raw_html import raw_token_partial_match
from wiki_text_extractor import (
    PARTIAL_REFERENCE_MODES,
    PageRequest,
    PartialExtractionError,
    clean_wikipedia_html_with_references,
    extract_image_captions_from_html,
    extract_partial_hybrid_text,
    extract_reference_entries_from_html,
    fetch_page_html,
    output_directory,
    page_request_from_url,
    runtime_label,
    runtime_output_path,
    update_runtime_report,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"
FALLBACK_ORDER = ("dmp_raw_html", "hybrid", "token_raw_html")


@dataclass
class PartialBestContext:
    # Lazily caches expensive shared work so fallback methods do not repeat it.
    page: PageRequest
    input_path: Path
    math_mode: str
    timings: dict[str, float] = field(default_factory=dict)
    _pasted_text: str | None = None
    _html: str | None = None
    _raw_visible_data: tuple[str, list[int]] | None = None
    _full_clean_text: str | None = None
    _reference_entries: dict[str, str] | None = None
    _image_captions: list[str] | None = None

    @property
    def pasted_text(self) -> str:
        if self._pasted_text is None:
            started_at = time.perf_counter()
            self._pasted_text = read_pasted_text(self.input_path)
            self.timings["input_read_seconds"] = time.perf_counter() - started_at
        return self._pasted_text

    @property
    def html(self) -> str:
        if self._html is None:
            started_at = time.perf_counter()
            self._html = fetch_page_html(self.page)
            self.timings["fetch_seconds"] = time.perf_counter() - started_at
        return self._html

    @property
    def raw_visible_data(self) -> tuple[str, list[int]]:
        if self._raw_visible_data is None:
            started_at = time.perf_counter()
            self._raw_visible_data = raw_html_visible_text_with_map(self.html)
            self.timings["raw_visible_seconds"] = time.perf_counter() - started_at
        return self._raw_visible_data

    @property
    def full_clean_text(self) -> str:
        if self._full_clean_text is None:
            started_at = time.perf_counter()
            self._full_clean_text = clean_wikipedia_html_with_references(
                self.html,
                self.math_mode,
                include_inline_markers=True,
            )
            self.timings["full_clean_seconds"] = time.perf_counter() - started_at
        return self._full_clean_text

    @property
    def reference_entries(self) -> dict[str, str]:
        if self._reference_entries is None:
            started_at = time.perf_counter()
            self._reference_entries = extract_reference_entries_from_html(self.html)
            self.timings["reference_parse_seconds"] = time.perf_counter() - started_at
        return self._reference_entries

    @property
    def image_captions(self) -> list[str]:
        if self._image_captions is None:
            started_at = time.perf_counter()
            self._image_captions = extract_image_captions_from_html(self.html)
            self.timings["caption_parse_seconds"] = time.perf_counter() - started_at
        return self._image_captions


@dataclass(frozen=True)
class PartialBestAttempt:
    method: str
    status: str
    seconds: float
    report: str
    error: str = ""


@dataclass(frozen=True)
class PartialBestResult:
    text: str
    report: str
    method: str
    attempts: tuple[PartialBestAttempt, ...]
    total_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract partial Wikipedia text with shared dmp_raw_html -> hybrid -> token_raw_html fallback."
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
        default="output/partial_best_text.txt",
        help="Output root/file used for the topic/language folder layout",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled",
    )
    parser.add_argument(
        "--references",
        choices=PARTIAL_REFERENCE_MODES,
        default="none",
        help="Reference output policy for the hybrid fallback; raw-HTML methods output body text only",
    )
    parser.add_argument("--hybrid-threshold", type=float, default=0.84)
    parser.add_argument("--raw-dmp-min-coverage", type=float, default=0.50)
    parser.add_argument("--raw-dmp-anchor-chars", type=int, default=600)
    parser.add_argument("--raw-dmp-chunk-size", type=int, default=32)
    parser.add_argument("--raw-dmp-max-chunks", type=int, default=24)
    parser.add_argument("--raw-dmp-match-threshold", type=float, default=0.45)
    parser.add_argument("--raw-dmp-timeout", type=float, default=1.0)
    parser.add_argument("--raw-dmp-locator-window-tokens", type=int, default=60)
    parser.add_argument("--raw-dmp-locator-refine-tokens", type=int, default=20)
    parser.add_argument("--raw-dmp-locator-min-score", type=float, default=0.72)
    parser.add_argument("--raw-dmp-locator-max-candidates", type=int, default=240)
    parser.add_argument("--raw-token-window", type=int, default=60)
    parser.add_argument("--raw-token-min-score", type=float, default=0.72)
    parser.add_argument("--raw-token-max-candidates", type=int, default=240)
    return parser


def partial_best_output_path(output: str, page: PageRequest) -> Path:
    return output_directory(output, page) / "partial_best_text.txt"


def partial_best_report_path(output: str, page: PageRequest) -> Path:
    return output_directory(output, page) / "partial_best_match_report.txt"


def run_attempt(method: str, func: Callable[[], tuple[str, str]]) -> tuple[str | None, PartialBestAttempt]:
    started_at = time.perf_counter()
    try:
        text, report = func()
    except PartialExtractionError as exc:
        return None, PartialBestAttempt(
            method=method,
            status="error",
            seconds=time.perf_counter() - started_at,
            report=exc.report_text,
            error=str(exc),
        )
    return text, PartialBestAttempt(
        method=method,
        status="ok",
        seconds=time.perf_counter() - started_at,
        report=report,
    )


def extract_partial_best_text(
    ctx: PartialBestContext,
    *,
    references_mode: str = "none",
    hybrid_threshold: float = 0.84,
    raw_dmp_min_coverage: float = 0.50,
    raw_dmp_anchor_chars: int = 600,
    raw_dmp_chunk_size: int = 32,
    raw_dmp_max_chunks: int = 24,
    raw_dmp_match_threshold: float = 0.45,
    raw_dmp_timeout: float = 1.0,
    raw_dmp_locator_window_tokens: int = 60,
    raw_dmp_locator_refine_tokens: int = 20,
    raw_dmp_locator_min_score: float = 0.72,
    raw_dmp_locator_max_candidates: int = 240,
    raw_token_window: int = 60,
    raw_token_min_score: float = 0.72,
    raw_token_max_candidates: int = 240,
) -> PartialBestResult:
    started_at = time.perf_counter()
    attempts: list[PartialBestAttempt] = []

    def dmp_raw_html() -> tuple[str, str]:
        return raw_dmp_partial_match(
            ctx.html,
            ctx.pasted_text,
            math_mode=ctx.math_mode,
            min_coverage=raw_dmp_min_coverage,
            anchor_chars=raw_dmp_anchor_chars,
            chunk_size=raw_dmp_chunk_size,
            max_chunks=raw_dmp_max_chunks,
            match_threshold=raw_dmp_match_threshold,
            timeout=raw_dmp_timeout,
            locator_window_tokens=raw_dmp_locator_window_tokens,
            locator_refine_tokens=raw_dmp_locator_refine_tokens,
            locator_min_score=raw_dmp_locator_min_score,
            locator_max_candidates=raw_dmp_locator_max_candidates,
            raw_visible_data=ctx.raw_visible_data,
        )

    def hybrid() -> tuple[str, str]:
        result = extract_partial_hybrid_text(
            ctx.full_clean_text,
            ctx.pasted_text,
            threshold=hybrid_threshold,
            references_mode=references_mode,
            reference_entries=ctx.reference_entries,
            copied_image_captions=ctx.image_captions,
        )
        return result.text, result.report

    def token_raw_html() -> tuple[str, str]:
        return raw_token_partial_match(
            ctx.html,
            ctx.pasted_text,
            math_mode=ctx.math_mode,
            window_tokens=raw_token_window,
            min_score=raw_token_min_score,
            max_candidates=raw_token_max_candidates,
            raw_visible_data=ctx.raw_visible_data,
        )

    for method, func in (
        ("dmp_raw_html", dmp_raw_html),
        ("hybrid", hybrid),
        ("token_raw_html", token_raw_html),
    ):
        text, attempt = run_attempt(method, func)
        attempts.append(attempt)
        if text is not None:
            total_seconds = time.perf_counter() - started_at
            return PartialBestResult(
                text=text,
                report=build_best_report(ctx, method, attempts, total_seconds),
                method=method,
                attempts=tuple(attempts),
                total_seconds=total_seconds,
            )

    total_seconds = time.perf_counter() - started_at
    report = build_best_report(ctx, "", attempts, total_seconds)
    message = "failed: all partial best fallback methods failed"
    raise PartialExtractionError(message, report)


def build_best_report(
    ctx: PartialBestContext,
    final_method: str,
    attempts: list[PartialBestAttempt],
    total_seconds: float,
) -> str:
    lines = [
        "Wikipedia Partial Best Extraction Report",
        "",
        f"Fallback order: {' -> '.join(FALLBACK_ORDER)}",
        f"Final method: {final_method or 'none'}",
        f"Total runtime: {total_seconds:.3f} seconds",
        "",
        "Shared timings:",
    ]
    for key in (
        "input_read_seconds",
        "fetch_seconds",
        "raw_visible_seconds",
        "full_clean_seconds",
        "reference_parse_seconds",
        "caption_parse_seconds",
    ):
        if key in ctx.timings:
            lines.append(f"- {key}: {ctx.timings[key]:.3f} seconds")
    lines.extend(["", "Attempts:"])
    for attempt in attempts:
        line = f"- {attempt.method}: {attempt.status}, {attempt.seconds:.3f} seconds"
        if attempt.error:
            line += f", {attempt.error}"
        lines.append(line)
    for attempt in attempts:
        lines.extend(["", f"--- {attempt.method} report ---", attempt.report.strip()])
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    output_path = partial_best_output_path(args.output, page)
    report_path = partial_best_report_path(args.output, page)
    runtime_path = runtime_output_path(args.output, page)
    ctx = PartialBestContext(page=page, input_path=input_path, math_mode=args.math)

    try:
        result = extract_partial_best_text(
            ctx,
            references_mode=args.references,
            hybrid_threshold=args.hybrid_threshold,
            raw_dmp_min_coverage=args.raw_dmp_min_coverage,
            raw_dmp_anchor_chars=args.raw_dmp_anchor_chars,
            raw_dmp_chunk_size=args.raw_dmp_chunk_size,
            raw_dmp_max_chunks=args.raw_dmp_max_chunks,
            raw_dmp_match_threshold=args.raw_dmp_match_threshold,
            raw_dmp_timeout=args.raw_dmp_timeout,
            raw_dmp_locator_window_tokens=args.raw_dmp_locator_window_tokens,
            raw_dmp_locator_refine_tokens=args.raw_dmp_locator_refine_tokens,
            raw_dmp_locator_min_score=args.raw_dmp_locator_min_score,
            raw_dmp_locator_max_candidates=args.raw_dmp_locator_max_candidates,
            raw_token_window=args.raw_token_window,
            raw_token_min_score=args.raw_token_min_score,
            raw_token_max_candidates=args.raw_token_max_candidates,
        )
        write_text_file(output_path, result.text)
        write_text_file(report_path, result.report)
        update_runtime_report(
            runtime_path,
            page,
            {runtime_label("Partial best runtime", args.math, page): result.total_seconds},
        )
    except PartialExtractionError as exc:
        write_text_file(report_path, exc.report_text)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Updated partial best match report: {report_path}")
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved partial best text: {output_path}")
    print(f"Updated partial best match report: {report_path}")
    print(f"Updated runtime report: {runtime_path}")
    print(f"Final method: {result.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
