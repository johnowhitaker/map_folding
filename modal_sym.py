import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).parent
TOTAL_8X8_DEPTH36_CASES = 240_192_829

app = modal.App("map-folding-symmetry")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "clang")
    .add_local_dir(ROOT, "/root/map_folding", copy=True)
    .run_commands(
        "clang++ -O3 -std=c++20 -pthread "
        "-o /root/map_folding/sym_fold_remote /root/map_folding/sym_fold.cpp"
    )
)


def parse_result(stdout: str) -> str:
    for line in stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            return text
    return ""


def summarize_jsonl(path: Path, total_cases: int, chunk_size: int) -> dict:
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                row["_line_no"] = line_no
                rows.append(row)

    expected_starts = list(range(0, total_cases, chunk_size))
    seen = {}
    duplicate_starts = []
    for row in rows:
        start = int(row["start_case"])
        if start in seen:
            duplicate_starts.append(start)
        else:
            seen[start] = row

    missing_starts = [start for start in expected_starts if start not in seen]
    extra_starts = sorted(start for start in seen if start not in set(expected_starts))
    unique_sum = sum(int(row["count"]) for row in seen.values())
    all_rows_sum = sum(int(row["count"]) for row in rows)

    return {
        "path": str(path),
        "rows": len(rows),
        "unique_chunks": len(seen),
        "expected_chunks": len(expected_starts),
        "duplicate_count": len(duplicate_starts),
        "duplicate_starts": duplicate_starts[:20],
        "missing_count": len(missing_starts),
        "missing_starts": missing_starts[:20],
        "extra_count": len(extra_starts),
        "extra_starts": extra_starts[:20],
        "unique_sum": str(unique_sum),
        "all_rows_sum": str(all_rows_sum),
        "complete": len(seen) == len(expected_starts) and not duplicate_starts and not extra_starts,
    }


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=24 * 60 * 60,
    max_containers=100,
    scaledown_window=300,
)
def run_chunk(
    n: int,
    threads: int,
    expanded_n: int,
    start_case: int,
    max_cases: int,
) -> dict:
    workdir = Path("/root/map_folding")
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "MF_STREAM_CASES": "1",
        "MF_EXPANDED_N": str(expanded_n),
    }
    run_cmd = [
        "./sym_fold_remote",
        str(n),
        str(threads),
        str(max_cases),
        str(start_case),
    ]
    start = time.perf_counter()
    result = subprocess.run(
        run_cmd,
        cwd=workdir,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    wall_s = time.perf_counter() - start
    count = parse_result(result.stdout)
    if result.returncode != 0 or not count:
        raise RuntimeError(
            f"chunk failed start={start_case} max={max_cases} rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    progress_matches = re.findall(r"completed=(\d+)", result.stderr)
    completed = int(progress_matches[-1]) if progress_matches else max_cases
    return {
        "n": n,
        "threads": threads,
        "expanded_n": expanded_n,
        "start_case": start_case,
        "max_cases": max_cases,
        "count": count,
        "wall_s": wall_s,
        "completed_hint": completed,
        "stderr_tail": "\n".join(result.stderr.splitlines()[-6:]),
    }


@app.local_entrypoint()
def main(
    n: int = 8,
    threads: int = 16,
    expanded_n: int = 36,
    start_case: int = 0,
    max_cases: int = 100000,
):
    result = run_chunk.remote(n, threads, expanded_n, start_case, max_cases)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def sweep(
    n: int = 8,
    threads: int = 16,
    expanded_n: int = 36,
    total_cases: int = TOTAL_8X8_DEPTH36_CASES,
    chunk_size: int = 100000,
    start_case: int = 0,
    chunk_count: int = 0,
    out: str = "",
    resume_existing: bool = False,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    end_case = total_cases
    starts = list(range(start_case, end_case, chunk_size))
    if chunk_count > 0:
        starts = starts[:chunk_count]
    inputs = [
        (n, threads, expanded_n, start, min(chunk_size, end_case - start))
        for start in starts
    ]
    if not inputs:
        raise ValueError("No chunks selected")

    if not out:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = f"results/8x8-depth{expanded_n}-{stamp}.jsonl"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if resume_existing:
        summary = summarize_jsonl(out_path, total_cases, chunk_size)
        existing = set()
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        existing.add(int(json.loads(text)["start_case"]))
        before = len(inputs)
        inputs = [item for item in inputs if item[3] not in existing]
        print(
            f"resume_existing=1 skipped={before - len(inputs)} "
            f"remaining={len(inputs)} existing_rows={summary['rows']} "
            f"unique_chunks={summary['unique_chunks']}",
            flush=True,
        )
        if not inputs:
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            return

    total = 0
    completed = 0
    started = time.perf_counter()
    print(f"chunks={len(inputs)} chunk_size={chunk_size} out={out_path}", flush=True)

    with out_path.open("a", encoding="utf-8") as f:
        for result in run_chunk.starmap(inputs, order_outputs=False):
            completed += 1
            total += int(result["count"])
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            elapsed = time.perf_counter() - started
            print(
                f"completed={completed}/{len(inputs)} "
                f"start={result['start_case']} wall_s={result['wall_s']:.2f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    summary = {
        "chunks": len(inputs),
        "chunk_size": chunk_size,
        "start_case": start_case,
        "total_cases": total_cases,
        "expanded_n": expanded_n,
        "partial_sum": str(total),
        "elapsed_s": elapsed,
        "out": str(out_path),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


@app.local_entrypoint()
def validate(
    out: str,
    total_cases: int = TOTAL_8X8_DEPTH36_CASES,
    chunk_size: int = 50000,
    expected: str = "",
):
    summary = summarize_jsonl(Path(out), total_cases, chunk_size)
    if expected:
        summary["expected"] = expected
        summary["matches_expected"] = summary["unique_sum"] == expected
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
