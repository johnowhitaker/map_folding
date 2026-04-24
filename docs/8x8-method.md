# Calculating the 8x8 Map Folding Count

This note summarizes the method used to replicate the 8x8 square map folding count:

```
162403827553180928
```

The run completed on 2026-04-23 Pacific time. It used a symmetry-reduced CPU solver distributed across Modal workers, not the earlier GPU prototypes.

## Problem Overview

The map folding, or stamp folding, problem asks how many valid ways a grid of connected unit squares can be folded down into a one-square stack. A folding is valid when the adjacency constraints of the original grid can be realized by successive folds without forcing two connected edges to pass through each other.

The search space grows very quickly. The original implementation in this repo follows Fred Lunnon's depth-first search formulation, translated through Sean Irvine's Java implementation. It is correct, but the raw search is too large for 8x8 without stronger structure.

## Symmetry-Reduced Solver

The fastest path in this repo is `sym_fold.cpp`, an attributed C++ port of gsitcia's Code Golf Stack Exchange solver. The important change is not a micro-optimization of the original DFS. It changes the search space for square grids.

The solver:

1. Numbers cells in a center-out spiral.
2. Enumerates the center cycle, using a 3x3 center for odd `n` and a 4x4 center for even `n`.
3. Deduplicates center cycles under the eight D4 symmetries of the square and stack reversal.
4. Carries an orbit multiplier for each remaining representative.
5. Expands those representatives outward to a chosen depth.
6. Counts each expanded partial case independently to completion.

For 8x8, the useful expansion depth was 36. At that depth the solver produces 240,192,829 independent expanded cases. Materializing that whole set is unnecessary and inconvenient, so the production run used streaming mode:

```
MF_STREAM_CASES=1 MF_EXPANDED_N=36 ./sym_fold 8 16 50000 0
```

The arguments after `8 16` select a deterministic slice of expanded-case indices: `max_cases` and `start_case`.

Observed 8x8 expanded-case counts:

| Expanded Depth | Cases |
| ---: | ---: |
| 24 | 414,992 |
| 26 | 2,512,120 |
| 28 | 5,158,056 |
| 30 | 9,447,816 |
| 32 | 66,351,430 |
| 34 | 125,889,268 |
| 36 | 240,192,829 |

The same solver verified smaller cases before the 8x8 run:

| Grid | Count | Time |
| --- | ---: | ---: |
| 6x6 | 123,912,532,224 | 2.53s local |
| 7x7 | 129,950,723,279,272 | 1,034.94s local |

## Modal Split

`modal_sym.py` wraps the solver in a Modal app. Each remote function call runs one independent chunk:

```
./sym_fold_remote 8 16 <max_cases> <start_case>
```

The full replication used:

```
modal run modal_sym.py::sweep --n 8 --threads 16 --expanded-n 36 \
  --total-cases 240192829 --chunk-size 50000 \
  --out results/8x8-depth36-full.jsonl
```

Run configuration:

| Field | Value |
| --- | ---: |
| Expanded cases | 240,192,829 |
| Chunk size | 50,000 |
| Chunks | 4,804 |
| Last chunk size | 42,829 |
| CPUs per chunk | 16 |
| Modal max containers | 100 |
| Main Modal app cost | $266.75 |
| Wall time | 9,992.25s |

Every completed chunk appends one JSON row with its `start_case`, `max_cases`, partial `count`, wall time, and stderr tail. If the client or worker is interrupted, the same sweep can be resumed with `--resume-existing` to skip chunk starts already present in the JSONL file.

## Chunk Statistics

The final output file was `results/8x8-depth36-full.jsonl`. It is ignored by git, but these are the validation stats from that file:

| Metric | Value |
| --- | ---: |
| Rows | 4,804 |
| Unique chunks | 4,804 |
| Expected chunks | 4,804 |
| Duplicate starts | 0 |
| Missing starts | 0 |
| Sum | 162,403,827,553,180,928 |

Per-chunk wall times were highly uneven:

| Metric | Seconds |
| --- | ---: |
| Min | 31.84 |
| Median | 144.64 |
| Mean | 189.10 |
| p90 | 349.29 |
| p95 | 460.47 |
| p99 | 814.25 |
| Max | 1,934.29 |

Chunk wall-time buckets:

| Range | Chunks |
| --- | ---: |
| <60s | 172 |
| 60-120s | 1,602 |
| 120-180s | 1,309 |
| 180-300s | 1,070 |
| 300-600s | 524 |
| 600-900s | 100 |
| 900-1200s | 21 |
| >=1200s | 6 |

Total summed chunk wall time was 908,441.41s, or 252.34 container-hours. With 9,992.25s elapsed wall time, the effective average active concurrency was about 90.9 chunks, close to the 100-container cap.

The slowest chunk started at expanded-case index 50,000,000 and took 1,934.29s. The final straggler started at 237,050,000 and took 1,178.88s. This long tail is why finer chunking or cost-weighted scheduling is likely the most immediate wall-clock improvement.

## Notes

The earlier CUDA versions were useful correctness and scaling experiments, but the symmetry-reduced CPU approach was the decisive jump. For this problem size, reducing the tree beat pushing the larger tree onto GPUs.

The current chunking scheme splits by deterministic expanded-case index ranges. That makes the run easy to resume and validate, but it does not balance equal work. A better production scheduler would either use smaller chunks near heavy bands or run a cheap lookahead estimator to schedule similarly expensive buckets.

The next large algorithmic targets are residual symmetry beyond the center cycle and a possible meet-in-the-middle or memoized frontier representation. Both need careful proof because the folding constraints depend on the stack cycle, not just local grid adjacency.
