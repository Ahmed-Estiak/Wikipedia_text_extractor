"""Benchmark partial extraction methods and append results to CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from extract_partial_dmp import dmp_partial_match
from wiki_text_extractor import (
    PageRequest,
    PartialExtractionError,
    clean_wikipedia_html_with_references,
    extract_image_captions_from_html,
    extract_partial_hybrid_text,
    extract_partial_token_text,
    extract_reference_entries_from_html,
    fetch_page_html,
    output_directory,
    page_request_from_url,
    topic_file_stem,
    write_text_file,
)


DEFAULT_INPUT_PATH = Path("input_text") / "partial_input.txt"
CSV_FIELDS = [
    "run_id",
    "timestamp_utc",
    "topic",
    "lang",
    "input_file",
    "input_sha256",
    "input_chars",
    "math_mode",
    "method",
    "status",
    "fetch_seconds",
    "clean_seconds",
    "match_seconds",
    "estimated_total_seconds",
    "output_chars",
    "output_sha256",
    "equals_hybrid",
    "start_offset",
    "end_offset",
    "score_summary",
    "settings",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    # Defines one command that benchmarks hybrid, token, and DMP on the same input.
    parser = argparse.ArgumentParser(
        description="Run partial hybrid/token/DMP matchers and append timing rows to CSV."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Pasted source text file used by all three matchers",
    )
    parser.add_argument(
        "--csv",
        default="output/partial_method_benchmark.csv",
        help="CSV output root/file; rows are appended under the topic/language folder",
    )
    parser.add_argument(
        "--math",
        choices=("remove", "latex", "keep"),
        default="latex",
        help="How math equations should be handled before matching",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.84,
        help="Hybrid sentence-window threshold",
    )
    parser.add_argument(
        "--token-window",
        type=int,
        default=60,
        help="Token matcher start/end anchor token count",
    )
    parser.add_argument(
        "--token-min-score",
        type=float,
        default=0.72,
        help="Token matcher minimum boundary score",
    )
    parser.add_argument(
        "--token-max-candidates",
        type=int,
        default=240,
        help="Token matcher maximum candidate windows per boundary",
    )
    parser.add_argument(
        "--dmp-min-coverage",
        type=float,
        default=0.72,
        help="DMP matcher minimum local boundary coverage",
    )
    parser.add_argument(
        "--dmp-anchor-chars",
        type=int,
        default=600,
        help="DMP matcher normalized copied characters used per boundary",
    )
    parser.add_argument(
        "--dmp-timeout",
        type=float,
        default=1.0,
        help="DMP local diff timeout in seconds",
    )
    return parser


def read_pasted_text(path: Path) -> str:
    # Reads the copied input once so every matcher sees exactly the same text.
    if not path.exists():
        raise FileNotFoundError(
            f"Input text file not found: {path}. Create it and paste the Wikipedia text there."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input text file is empty: {path}")
    return text


def benchmark_csv_path(output: str, page: PageRequest) -> Path:
    # Stores benchmark CSVs beside the other topic/language outputs.
    suffix = Path(output).suffix or ".csv"
    return output_directory(output, page) / f"{topic_file_stem(page)}_partial_benchmark{suffix}"


def benchmark_text_output_path(output: str, page: PageRequest, method: str) -> Path:
    # Stores the latest benchmark clean text for one method; each run overwrites it.
    return output_directory(output, page) / f"partial_benchmark_{method}_text.txt"


def clear_benchmark_text_outputs(output: str, page: PageRequest) -> dict[str, Path]:
    # Removes stale benchmark text files before a new three-method run starts.
    paths = {
        method: benchmark_text_output_path(output, page, method)
        for method in ("hybrid", "token", "dmp")
    }
    for path in paths.values():
        path.unlink(missing_ok=True)
    return paths


def text_hash(text: str) -> str:
    # Short hash keeps CSV readable while still flagging output/input changes.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def report_value(report: str, label: str) -> str:
    # Pulls numeric diagnostics out of human-readable method reports.
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", report, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def append_benchmark_rows(path: Path, rows: list[dict[str, str]]) -> None:
    # Appends rows and writes the CSV header only when creating a fresh file.
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def benchmark_failure_text(method: str, error: str, report: str) -> str:
    # Writes a visible placeholder when a benchmark method cannot produce clean text.
    lines = [
        "Benchmark method failed",
        "",
        f"Method: {method}",
        f"Error: {error}",
    ]
    if report.strip():
        lines.extend(["", "Report:", report.strip()])
    return "\n".join(lines)


def method_row(
    base: dict[str, str],
    method: str,
    status: str,
    match_seconds: float,
    fetch_seconds: float,
    clean_seconds: float,
    text: str = "",
    hybrid_text: str | None = None,
    start_offset: str = "",
    end_offset: str = "",
    score_summary: str = "",
    settings: str = "",
    error: str = "",
) -> dict[str, str]:
    # Builds one CSV row for one method run.
    row = dict(base)
    row.update(
        {
            "method": method,
            "status": status,
            "fetch_seconds": f"{fetch_seconds:.6f}",
            "clean_seconds": f"{clean_seconds:.6f}",
            "match_seconds": f"{match_seconds:.6f}",
            "estimated_total_seconds": f"{(fetch_seconds + clean_seconds + match_seconds):.6f}",
            "output_chars": str(len(text)) if text else "0",
            "output_sha256": text_hash(text) if text else "",
            "equals_hybrid": "" if hybrid_text is None or method == "hybrid" else str(text == hybrid_text),
            "start_offset": start_offset,
            "end_offset": end_offset,
            "score_summary": score_summary,
            "settings": settings,
            "error": error,
        }
    )
    return row


def run_method(
    method: str,
    runner: Callable[[], tuple[str, str, str, str, str]],
    base: dict[str, str],
    fetch_seconds: float,
    clean_seconds: float,
    hybrid_text: str | None,
    settings: str,
    output_path: Path,
) -> tuple[dict[str, str], str | None]:
    # Times one matcher and converts success/failure into a CSV row.
    started_at = time.perf_counter()
    try:
        text, start_offset, end_offset, score_summary, _report = runner()
        write_text_file(output_path, text)
        match_seconds = time.perf_counter() - started_at
        row = method_row(
            base,
            method,
            "ok",
            match_seconds,
            fetch_seconds,
            clean_seconds,
            text=text,
            hybrid_text=hybrid_text,
            start_offset=start_offset,
            end_offset=end_offset,
            score_summary=score_summary,
            settings=settings,
        )
        return row, text
    except (PartialExtractionError, RuntimeError, ValueError) as exc:
        match_seconds = time.perf_counter() - started_at
        report = getattr(exc, "report_text", "")
        write_text_file(output_path, benchmark_failure_text(method, str(exc), report))
        report_lines = report.splitlines()
        row = method_row(
            base,
            method,
            "error",
            match_seconds,
            fetch_seconds,
            clean_seconds,
            hybrid_text=hybrid_text,
            score_summary=report_lines[2] if len(report_lines) > 2 else "",
            settings=settings,
            error=str(exc),
        )
        return row, None


def main(argv: Iterable[str] | None = None) -> int:
    # Runs shared fetch/clean once, then benchmarks the three matching methods.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    csv_path = benchmark_csv_path(args.csv, page)
    text_output_paths = clear_benchmark_text_outputs(args.csv, page)

    try:
        pasted_text = read_pasted_text(input_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        base = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "topic": page.title,
            "lang": page.lang,
            "input_file": str(input_path),
            "input_sha256": text_hash(pasted_text),
            "input_chars": str(len(pasted_text)),
            "math_mode": args.math,
        }

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
        reference_entries = extract_reference_entries_from_html(html)
        clean_seconds = time.perf_counter() - clean_started_at

        def run_hybrid() -> tuple[str, str, str, str, str]:
            result = extract_partial_hybrid_text(
                full_text,
                pasted_text,
                threshold=args.hybrid_threshold,
                references_mode="none",
                reference_entries=reference_entries,
                copied_image_captions=captions,
            )
            return (
                result.text,
                str(result.start),
                str(result.end),
                f"confidence={result.confidence}",
                result.report,
            )

        hybrid_row, hybrid_text = run_method(
            "hybrid",
            run_hybrid,
            base,
            fetch_seconds,
            clean_seconds,
            None,
            f"threshold={args.hybrid_threshold:.3f};references=none",
            text_output_paths["hybrid"],
        )

        def run_token() -> tuple[str, str, str, str, str]:
            result = extract_partial_token_text(
                full_text,
                pasted_text,
                window_tokens=args.token_window,
                min_score=args.token_min_score,
                confirm_mode="none",
                max_candidates=args.token_max_candidates,
                copied_image_captions=captions,
            )
            return (
                result.text,
                str(result.start),
                str(result.end),
                f"start={result.start_match.score:.3f};end={result.end_match.score:.3f}",
                result.report,
            )

        token_row, _token_text = run_method(
            "token",
            run_token,
            base,
            fetch_seconds,
            clean_seconds,
            hybrid_text,
            (
                f"window={args.token_window};min_score={args.token_min_score:.3f};"
                f"max_candidates={args.token_max_candidates};confirm=none"
            ),
            text_output_paths["token"],
        )

        def run_dmp() -> tuple[str, str, str, str, str]:
            text, report = dmp_partial_match(
                full_text,
                pasted_text,
                min_coverage=args.dmp_min_coverage,
                anchor_chars=args.dmp_anchor_chars,
                timeout=args.dmp_timeout,
                copied_image_captions=captions,
            )
            return (
                text,
                report_value(report, "Start offset"),
                report_value(report, "End offset"),
                f"start={report_value(report, 'Start coverage')};end={report_value(report, 'End coverage')}",
                report,
            )

        dmp_row, _dmp_text = run_method(
            "dmp",
            run_dmp,
            base,
            fetch_seconds,
            clean_seconds,
            hybrid_text,
            (
                f"min_coverage={args.dmp_min_coverage:.3f};"
                f"anchor_chars={args.dmp_anchor_chars};timeout={args.dmp_timeout:.3f}"
            ),
            text_output_paths["dmp"],
        )

        rows = [hybrid_row, token_row, dmp_row]
        append_benchmark_rows(csv_path, rows)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Appended benchmark rows: {csv_path}")
    print("Updated benchmark text files:")
    for method, path in text_output_paths.items():
        if path.exists():
            print(f"{method}: {path}")
    print("method,status,match_seconds,estimated_total_seconds,equals_hybrid")
    for row in rows:
        print(
            ",".join(
                [
                    row["method"],
                    row["status"],
                    row["match_seconds"],
                    row["estimated_total_seconds"],
                    row["equals_hybrid"],
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
