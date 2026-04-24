# Map Folding Experiments

Date: 2026-04-23

## Baselines

Local machine: Apple silicon macOS, no local CUDA. Modal GPU tests used `nvidia/cuda:12.8.0-devel-ubuntu22.04` on `A10G:4`.

| Solver | Case | Settings | Count | Time |
| --- | ---: | --- | ---: | ---: |
| `mf.c` | 5x5 | single process | 186086600 | 0.59s real |
| `mf.c` | 6x6 | single process | 123912532224 | 288.49s real |
| `run_parallel.sh` + `mf.c` | 6x6 | `MOD=24` | 123912532224 | 107.34s real |
| `mf_fast.cpp` | 5x5 | depth 12, 12 threads | 186086600 | 0.08s real |
| `mf_fast.cpp` | 6x6 | depth 12, 12 threads | 123912532224 | 34.79s real |
| `mf_fast.cpp` | 6x6 | depth 14, 12 threads | 123912532224 | 34.23s real |
| `mf_fast.cpp` | 6x6 | depth 16, 12 threads | 123912532224 | 33.92s real |
| `mf_fast.cpp` | 6x6 | depth 18, 12 threads | 123912532224 | 33.92s real |
| `mf_fast.cpp` | 6x6 | depth 20, 12 threads | 123912532224 | 34.36s real |
| `gpu_folding.cu` | 6x6 | depth 18, 1 A10G effectively | 123912532224 | 135.75s Modal wall |
| `gpu_folding_multigpu.cu` | 6x6 | depth 18, 4x A10G | 123912532224 | 20.57s Modal wall |
| `gpu_folding_v2.cu` | 5x5 | depth 12, 4x A10G | 186086600 | 0.98s solve |
| `gpu_folding_v2.cu` | 6x6 | depth 18, batch 100000, 4x A10G | 123912532224 | 40.18s solve |
| `gpu_folding_v2.cu` | 6x6 | depth 18, batch 5000, 4x A10G | 123912532224 | 20.42s solve |
| `gpu_folding_v2.cu` | 6x6 | depth 20, batch 5000, 4x A10G | 123912532224 | 21.30s solve |
| `gpu_folding_v2.cu` | 6x6 | depth 18, batch 5000, 1x A10G | 123912532224 | 76.70s solve |
| `sym_fold.cpp` | 6x6 | symmetry reduction, 12 local threads | 123912532224 | 2.53s real |
| `sym_fold.cpp` | 7x7 | symmetry reduction, 12 local threads | 129950723279272 | 1034.94s real |
| `modal_sym.py` + `sym_fold.cpp` | 8x8 | depth 36, 4,804 Modal chunks, 16 CPUs/chunk | 162403827553180928 | 9992.25s Modal wall, $266.75 main app cost |

## Prefix Counts

These counts are prefix states at the given Lunnon DFS depth after the normal-form filter.

| Case | Depth | States | Notes |
| --- | ---: | ---: | --- |
| 7x7 | 18 | 293428 | Materialized collector |
| 7x7 | 20 | 751142 | Materialized collector |
| 7x7 | 22 | 5917082 | Materialized collector |
| 7x7 | 23 | 8783910 | Materialized collector |
| 8x8 | 20 | 1428966 | Materialized collector |
| 8x8 | 22 | 3658148 | Materialized collector |
| 8x8 | 24 | 8377532 | Materialized collector |
| 8x8 | 26 | 72578586 | Count-only collector |
| 8x8 | 28 | 153872392 | Count-only collector |
| 8x8 | 30 | 319976090 | Count-only collector |
| 8x8 | 32 | 632660280 | Count-only collector |

The materialized collector can consume multiple GB for 8x8 depth 24+ and is not suitable for deeper probes. The next path is streaming batches.

## Symmetry Solver

`sym_fold.cpp` is an attributed C++ port of gsitcia's Code Golf Stack Exchange answer. It changes the search space instead of only speeding up the old DFS:

1. Number cells in a center-out spiral.
2. Enumerate the center `3x3` or `4x4` cycle.
3. Deduplicate center cycles under D4 square symmetries and stack reversal.
4. Expand representatives to a configurable depth, preserving each representative's orbit multiplier.
5. Count each expanded case independently.

Observed symmetry-expanded case counts for 8x8:

| Expanded Depth | Cases | Setup Time |
| ---: | ---: | ---: |
| 24 | 414992 | 0.04s |
| 26 | 2512120 | 0.11s |
| 28 | 5158056 | 0.28s |
| 30 | 9447816 | 0.58s |
| 32 | 66351430 | 3.06s |
| 34 | 125889268 | 9.04s |
| 36 | 240192829 | 17.25s |

8x8 sampling:

| Solver | Expanded Depth | Range | Partial Sum | Time |
| --- | ---: | --- | ---: | ---: |
| `sym_fold.cpp` | 24 | case 100000 only | 961512726016 | 67.25s |
| `sym_fold.cpp` | 24 | case 400000 only | 39987229696 | 1.80s |
| `sym_fold.cpp` | 32 | first 1000 cases | 4354325657600 | 21.83s |
| `sym_fold.cpp` | 36 | first 100000 cases, materialized | 35390073426944 | 164.41s |
| `sym_fold.cpp` | 36 | first 100000 cases, streaming | 35390073426944 | 154.04s |
| `modal_sym.py` | 36 | first 100000 cases, 16 Modal CPUs | 35390073426944 | 144.67s chunk wall |

Full 8x8 replication:

| Solver | Expanded Depth | Chunking | Result | Wall Time | Cost |
| --- | ---: | --- | ---: | ---: | ---: |
| `modal_sym.py` + `sym_fold.cpp` | 36 | 4,804 chunks x 50,000 cases, 16 CPUs/chunk, max 100 containers | 162403827553180928 | 9992.25s | $266.75 main app cost |

Validation for `results/8x8-depth36-full.jsonl`:

- rows: 4804
- unique chunks: 4804
- expected chunks: 4804
- duplicate starts: 0
- missing starts: 0
- summed count: `162403827553180928`
- summary file: `results/8x8-depth36-full.jsonl.summary.json`

Streaming mode (`MF_STREAM_CASES=1`) avoids materializing all expanded cases. This is required for depth 36+ and for chunked distributed runs.

## Efficiency Leaps To Test

1. Larger CPU allocations per Modal container: the current run was capped by 100 containers x 16 CPUs. If Modal allows 32-64 CPUs/container, this should reduce wall time at roughly similar CPU-second cost, subject to per-chunk parallel scaling.
2. Finer or cost-weighted chunking: 50k-case chunks worked, but the final tail had 900-1179s stragglers. Smaller chunks or pilot-cost buckets should reduce tail latency.
3. Streaming GPU batches: avoid holding all prefixes in RAM; this is required for 8x8 depth sweeps and checkpointable runs.
4. Distributed prefix ranges: give Modal many independent jobs over deterministic prefix-index intervals, with durable per-range partial sums. This is a scaling leap rather than a kernel micro-optimization.
5. Cost-weighted scheduling: prefix state count is a weak load-balance proxy. Use a shallow pilot estimate per prefix or per prefix group to schedule heavy ranges earlier.
6. Additional symmetry reduction: the center-cycle D4/reversal reduction is validated against 6x6, 7x7, and the published 8x8 value. Residual stabilizer handling for expanded cases may still be a route to smaller per-ring searches, but it needs proof before use.
7. Meet-in-the-middle or memoization by frontier/canonical stack state: potentially the largest algorithmic leap, but state identity must preserve the folding constraints. Worth exploring separately from CUDA tuning.
