#!/usr/bin/env python3
"""Run small 9x9 seed/depth profiling sweeps for sym_fold9."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_RE = re.compile(
    r"(?P<label>[a-z_]+)_count=(?P<count>\d+) "
    r"min=(?P<min>\d+) p50=(?P<p50>\d+) p90=(?P<p90>\d+) "
    r"p99=(?P<p99>\d+) max=(?P<max>\d+) mean=(?P<mean>\d+) total=(?P<total>\d+)"
)


def parse_summary(text: str, label: str) -> dict[str, int]:
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match and match.group("label") == label:
            return {
                key: int(value)
                for key, value in match.groupdict().items()
                if key != "label"
            }
    raise RuntimeError(f"missing {label!r} summary in output:\n{text}")


def run_solver(
    exe: Path,
    n: int,
    threads: int,
    env_extra: dict[str, str],
    args: list[str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_extra)
    return subprocess.run(
        [str(exe), str(n), str(threads), *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path(__file__).with_name("sym_fold9"))
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--depths", type=int, nargs="+", default=[40, 42])
    parser.add_argument("--seed-starts", type=int, nargs="+", default=[0, 1000, 10000, 100000, 400000])
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--profile-cases", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    if not args.exe.exists():
        print(f"solver not found: {args.exe}", file=sys.stderr)
        return 2

    rows: list[dict[str, int | str]] = []
    for depth in args.depths:
        for seed_start in args.seed_starts:
            base_env = {
                "MF_CENTER_N": "5",
                "MF_CENTER_STREAM_DEDUP": "1",
                "MF_SEED_START": str(seed_start),
                "MF_SEED_COUNT": str(args.seed_count),
                "MF_STREAM_CASES": "1",
                "MF_EXPANDED_N": str(depth),
            }
            setup = run_solver(
                args.exe,
                args.n,
                args.threads,
                {**base_env, "MF_SETUP_ONLY": "1", "MF_SEED_COUNTS_SUMMARY": "1"},
                [],
                args.timeout,
            )
            expanded = parse_summary(setup.stderr, "seed_expanded")
            row: dict[str, int | str] = {
                "depth": depth,
                "seed_start": seed_start,
                "expanded_mean": expanded["mean"],
                "expanded_p90": expanded["p90"],
                "expanded_max": expanded["max"],
                "expanded_total": expanded["total"],
            }

            if not args.skip_profile:
                profile = run_solver(
                    args.exe,
                    args.n,
                    args.threads,
                    {**base_env, "MF_PROFILE_WORK": "1"},
                    [str(args.profile_cases), "0"],
                    args.timeout,
                )
                nodes = parse_summary(profile.stderr, "profile_nodes")
                leaves = parse_summary(profile.stderr, "profile_leaves")
                row.update(
                    {
                        "profile_cases": nodes["count"],
                        "nodes_mean": nodes["mean"],
                        "nodes_p90": nodes["p90"],
                        "nodes_max": nodes["max"],
                        "leaves_mean": leaves["mean"],
                    }
                )
            rows.append(row)
            print(row, flush=True)

    print()
    headers = list(rows[0].keys()) if rows else []
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
