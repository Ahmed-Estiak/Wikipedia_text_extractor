"""Benchmark partial extraction methods and append 5-way results to CSV."""

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
from extract_partial_dmp_raw_html import raw_dmp_partial_match
from extract_partial_token_raw_html import raw_token_partial_match
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
BENCHMARK_SCHEMA_VERSION = "2"
BENCHMARK_METHODS = ("hybrid", "token", "dmp", "token_raw_html", "dmp_raw_html")
METHOD_META = {
    "hybrid": ("hybrid", "clean_text", "clean_text_chars", True),
    "token": ("token", "clean_text", "clean_text_chars", True),
    "dmp": ("dmp", "clean_text", "clean_text_chars", True),
    "token_raw_html": ("token", "raw_html", "raw_html_chars", False),
    "dmp_raw_html": ("dmp", "raw_html", "raw_html_chars", False),
}
CSV_FIELDS = [
    "schema_version",
    "run_id",
    "timestamp_utc",
    "topic",
    "lang",
    "input_file",
    "input_sha256",
    "input_chars",
    "math_mode",
    "method",
    "method_family",
    "match_surface",
    "offset_basis",
    "uses_full_clean",
    "status",
    "fetch_seconds",
    "full_clean_seconds",
    "match_seconds",
    "estimated_total_seconds",
    "output_file",
    "output_chars",
    "output_sha256",
    "equals_hybrid",
    "equals_token",
    "start_offset",
    "end_offset",
    "start_score",
    "end_score",
    "score_summary",
    "settings",
    "error",
]


MethodRunner = Callable[[], tuple[str, str, str, str, str, str, str]]


def build_parser() -> argparse.ArgumentParser:
    # Defines one command that benchmarks all five partial matching methods.
    parser = argparse.ArgumentParser(
        description="Run partial hybrid/token/DMP/raw-HTML matchers and append 5-way timing rows to CSV."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Wikipedia page URL")
    source.add_argument("--title", help="Wikipedia page title")
    parser.add_argument("--lang", default="en", help="Wikipedia language code")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Pasted source text file used by all five matchers",
    )
    parser.add_argument(
        "--csv",
        default="output/partial_method_benchmark_5way.csv",
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
        help="Clean-token matcher start/end anchor token count",
    )
    parser.add_argument(
        "--token-refine-tokens",
        type=int,
        default=20,
        help="Clean-token matcher backward/forward refinement chunk token count",
    )
    parser.add_argument(
        "--token-min-score",
        type=float,
        default=0.72,
        help="Clean-token matcher minimum boundary score",
    )
    parser.add_argument(
        "--token-max-candidates",
        type=int,
        default=240,
        help="Clean-token matcher maximum candidate windows per boundary",
    )
    parser.add_argument(
        "--dmp-min-coverage",
        type=float,
        default=0.72,
        help="Clean-DMP matcher minimum local boundary coverage",
    )
    parser.add_argument(
        "--dmp-anchor-chars",
        type=int,
        default=300,
        help="Clean-DMP matcher normalized copied characters used per boundary window",
    )
    parser.add_argument(
        "--dmp-refine-chars",
        type=int,
        default=100,
        help="Clean-DMP matcher normalized copied characters used per refinement window",
    )
    parser.add_argument(
        "--dmp-timeout",
        type=float,
        default=1.0,
        help="Clean-DMP local diff timeout in seconds",
    )
    parser.add_argument(
        "--raw-token-window",
        type=int,
        default=60,
        help="Raw-HTML token matcher one-shot start/end anchor token count",
    )
    parser.add_argument(
        "--raw-token-min-score",
        type=float,
        default=0.72,
        help="Raw-HTML token matcher minimum boundary score",
    )
    parser.add_argument(
        "--raw-token-max-candidates",
        type=int,
        default=240,
        help="Raw-HTML token matcher maximum candidate windows per boundary",
    )
    parser.add_argument(
        "--raw-dmp-min-coverage",
        type=float,
        default=0.50,
        help="Raw-HTML DMP matcher minimum local boundary coverage",
    )
    parser.add_argument(
        "--raw-dmp-anchor-chars",
        type=int,
        default=600,
        help="Raw-HTML DMP matcher normalized copied characters used per boundary",
    )
    parser.add_argument(
        "--raw-dmp-timeout",
        type=float,
        default=1.0,
        help="Raw-HTML DMP local diff timeout in seconds",
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
    # Stores the clean 5-way benchmark CSV beside other topic/language outputs.
    suffix = Path(output).suffix or ".csv"
    return output_directory(output, page) / f"{topic_file_stem(page)}_partial_benchmark_5way{suffix}"


def benchmark_text_output_path(output: str, page: PageRequest, method: str) -> Path:
    # Stores the latest benchmark clean text for one method; each run overwrites it.
    return output_directory(output, page) / f"partial_benchmark_{method}_text.txt"


def clear_benchmark_text_outputs(output: str, page: PageRequest) -> dict[str, Path]:
    # Removes stale benchmark text files before a new five-method run starts.
    paths = {
        method: benchmark_text_output_path(output, page, method)
        for method in BENCHMARK_METHODS
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
    # Appends rows and writes the v2 CSV header only when creating a fresh file.
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
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
    full_clean_seconds: float,
    uses_full_clean: bool,
    output_path: Path,
    text: str = "",
    start_offset: str = "",
    end_offset: str = "",
    start_score: str = "",
    end_score: str = "",
    score_summary: str = "",
    settings: str = "",
    error: str = "",
) -> dict[str, str]:
    # Builds one v2 CSV row for one method run.
    method_family, match_surface, offset_basis, _meta_uses_full_clean = METHOD_META[method]
    estimated_total = fetch_seconds + match_seconds
    if uses_full_clean:
        estimated_total += full_clean_seconds
    row = dict(base)
    row.update(
        {
            "method": method,
            "method_family": method_family,
            "match_surface": match_surface,
            "offset_basis": offset_basis,
            "uses_full_clean": str(uses_full_clean),
            "status": status,
            "fetch_seconds": f"{fetch_seconds:.6f}",
            "full_clean_seconds": f"{full_clean_seconds:.6f}" if uses_full_clean else "",
            "match_seconds": f"{match_seconds:.6f}",
            "estimated_total_seconds": f"{estimated_total:.6f}",
            "output_file": str(output_path),
            "output_chars": str(len(text)) if text else "0",
            "output_sha256": text_hash(text) if text else "",
            "equals_hybrid": "",
            "equals_token": "",
            "start_offset": start_offset,
            "end_offset": end_offset,
            "start_score": start_score,
            "end_score": end_score,
            "score_summary": score_summary,
            "settings": settings,
            "error": error,
        }
    )
    return row


def run_method(
    method: str,
    runner: MethodRunner,
    base: dict[str, str],
    fetch_seconds: float,
    full_clean_seconds: float,
    settings: str,
    output_path: Path,
) -> tuple[dict[str, str], str | None]:
    # Times one matcher and converts success/failure into a CSV row.
    started_at = time.perf_counter()
    uses_full_clean = METHOD_META[method][3]
    try:
        text, start_offset, end_offset, start_score, end_score, score_summary, _report = runner()
        write_text_file(output_path, text)
        match_seconds = time.perf_counter() - started_at
        row = method_row(
            base,
            method,
            "ok",
            match_seconds,
            fetch_seconds,
            full_clean_seconds,
            uses_full_clean,
            output_path,
            text=text,
            start_offset=start_offset,
            end_offset=end_offset,
            start_score=start_score,
            end_score=end_score,
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
            full_clean_seconds,
            uses_full_clean,
            output_path,
            score_summary=report_lines[2] if len(report_lines) > 2 else "",
            settings=settings,
            error=str(exc),
        )
        return row, None


def update_baseline_comparisons(rows: list[dict[str, str]], texts: dict[str, str]) -> None:
    # Fills output equality columns after all methods have had a chance to run.
    hybrid_text = texts.get("hybrid")
    token_text = texts.get("token")
    for row in rows:
        method = row["method"]
        text = texts.get(method)
        if not text:
            continue
        if hybrid_text is not None and method != "hybrid":
            row["equals_hybrid"] = str(text == hybrid_text)
        if token_text is not None and method != "token":
            row["equals_token"] = str(text == token_text)


def main(argv: Iterable[str] | None = None) -> int:
    # Runs shared fetch/full-clean once, then benchmarks five partial matching methods.
    args = build_parser().parse_args(argv)
    page = page_request_from_url(args.url) if args.url else PageRequest(args.title, args.lang)
    input_path = Path(args.input)
    csv_path = benchmark_csv_path(args.csv, page)
    text_output_paths = clear_benchmark_text_outputs(args.csv, page)

    try:
        pasted_text = read_pasted_text(input_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        base = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
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
        full_clean_seconds = time.perf_counter() - clean_started_at

        def run_hybrid() -> tuple[str, str, str, str, str, str, str]:
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
                "",
                "",
                f"confidence={result.confidence}",
                result.report,
            )

        def run_token() -> tuple[str, str, str, str, str, str, str]:
            result = extract_partial_token_text(
                full_text,
                pasted_text,
                window_tokens=args.token_window,
                refine_tokens=args.token_refine_tokens,
                min_score=args.token_min_score,
                confirm_mode="none",
                max_candidates=args.token_max_candidates,
                copied_image_captions=captions,
            )
            return (
                result.text,
                str(result.start),
                str(result.end),
                f"{result.start_match.score:.3f}",
                f"{result.end_match.score:.3f}",
                f"start={result.start_match.score:.3f};end={result.end_match.score:.3f}",
                result.report,
            )

        def run_dmp() -> tuple[str, str, str, str, str, str, str]:
            text, report = dmp_partial_match(
                full_text,
                pasted_text,
                min_coverage=args.dmp_min_coverage,
                anchor_chars=args.dmp_anchor_chars,
                refine_chars=args.dmp_refine_chars,
                timeout=args.dmp_timeout,
                copied_image_captions=captions,
            )
            start_score = report_value(report, "Start coverage")
            end_score = report_value(report, "End coverage")
            return (
                text,
                report_value(report, "Start offset"),
                report_value(report, "End offset"),
                start_score,
                end_score,
                f"start={start_score};end={end_score}",
                report,
            )

        def run_token_raw_html() -> tuple[str, str, str, str, str, str, str]:
            text, report = raw_token_partial_match(
                html,
                pasted_text,
                math_mode=args.math,
                window_tokens=args.raw_token_window,
                min_score=args.raw_token_min_score,
                max_candidates=args.raw_token_max_candidates,
                confirm_mode="none",
            )
            start_score = report_value(report, "Start token score")
            end_score = report_value(report, "End token score")
            return (
                text,
                report_value(report, "Raw start offset"),
                report_value(report, "Raw end offset"),
                start_score,
                end_score,
                f"start={start_score};end={end_score}",
                report,
            )

        def run_dmp_raw_html() -> tuple[str, str, str, str, str, str, str]:
            text, report = raw_dmp_partial_match(
                html,
                pasted_text,
                math_mode=args.math,
                min_coverage=args.raw_dmp_min_coverage,
                anchor_chars=args.raw_dmp_anchor_chars,
                timeout=args.raw_dmp_timeout,
            )
            start_score = report_value(report, "Start coverage")
            end_score = report_value(report, "End coverage")
            return (
                text,
                report_value(report, "Raw start offset"),
                report_value(report, "Raw end offset"),
                start_score,
                end_score,
                f"start={start_score};end={end_score}",
                report,
            )

        runners: list[tuple[str, MethodRunner, str]] = [
            ("hybrid", run_hybrid, f"threshold={args.hybrid_threshold:.3f};references=none"),
            (
                "token",
                run_token,
                (
                    f"window={args.token_window};refine={args.token_refine_tokens};"
                    f"min_score={args.token_min_score:.3f};max_candidates={args.token_max_candidates};confirm=none"
                ),
            ),
            (
                "dmp",
                run_dmp,
                (
                    f"min_coverage={args.dmp_min_coverage:.3f};anchor_chars={args.dmp_anchor_chars};"
                    f"refine_chars={args.dmp_refine_chars};timeout={args.dmp_timeout:.3f}"
                ),
            ),
            (
                "token_raw_html",
                run_token_raw_html,
                (
                    f"window={args.raw_token_window};min_score={args.raw_token_min_score:.3f};"
                    f"max_candidates={args.raw_token_max_candidates};confirm=none"
                ),
            ),
            (
                "dmp_raw_html",
                run_dmp_raw_html,
                (
                    f"min_coverage={args.raw_dmp_min_coverage:.3f};"
                    f"anchor_chars={args.raw_dmp_anchor_chars};timeout={args.raw_dmp_timeout:.3f}"
                ),
            ),
        ]

        rows: list[dict[str, str]] = []
        texts: dict[str, str] = {}
        for method, runner, settings in runners:
            row, text = run_method(
                method,
                runner,
                base,
                fetch_seconds,
                full_clean_seconds,
                settings,
                text_output_paths[method],
            )
            rows.append(row)
            if text is not None:
                texts[method] = text
        update_baseline_comparisons(rows, texts)
        append_benchmark_rows(csv_path, rows)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Appended 5-way benchmark rows: {csv_path}")
    print("Updated benchmark text files:")
    for method in BENCHMARK_METHODS:
        path = text_output_paths[method]
        if path.exists():
            print(f"{method}: {path}")
    print("method,status,match_seconds,estimated_total_seconds,equals_hybrid,equals_token")
    for row in rows:
        print(
            ",".join(
                [
                    row["method"],
                    row["status"],
                    row["match_seconds"],
                    row["estimated_total_seconds"],
                    row["equals_hybrid"],
                    row["equals_token"],
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
