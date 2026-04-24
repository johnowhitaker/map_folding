# 9x9 Feasibility Notes

This directory is for speculative 9x9 work. The proven 8x8 replication path remains in the repo root as `sym_fold.cpp` plus `modal_sym.py`.

Current status: the existing symmetry-reduced CPU method is strong enough to replicate 8x8, but a direct 9x9 extension still looks one to two orders of magnitude beyond a small Modal budget. The experiments below are aimed at finding the next algorithmic reduction.

[OEIS A001418](https://oeis.org/A001418) currently lists terms through 7x7. The 8x8 value replicated in this repo is:

```
162403827553180928
```

## Experimental Solver

`sym_fold9.cpp` starts from `../../sym_fold.cpp` and adds:

- `MAX_N=100`, enough for 9x9.
- 128-bit masks for valid insertion positions past cell 64.
- Optional `MF_CENTER_N=5` for a 5x5 center on 9x9.
- Optional `MF_CENTER_STREAM_DEDUP=1` to deduplicate center cases during generation instead of storing all raw center cycles.
- Optional `MF_SEED_START` / `MF_SEED_COUNT` to run a slice of center representatives.
- Optional `MF_SEED_COUNTS_SUMMARY=1` to summarize per-seed expansion counts.
- Optional `MF_PROFILE_WORK=1` to measure recursive node counts for a bounded generated-prefix sample.

`profile_sweep.py` wraps those profiling modes for repeatable seed/depth sweeps.

Build:

```
clang++ -O3 -std=c++20 -pthread -o experiments/9x9/sym_fold9 experiments/9x9/sym_fold9.cpp
```

Known-value checks:

| Command | Result |
| --- | ---: |
| `./experiments/9x9/sym_fold9 5 12` | 186086600 |
| `./experiments/9x9/sym_fold9 6 12` | 123912532224 |

The 128-bit path is slower on small cases than `sym_fold.cpp`, but it keeps the 9x9 search mechanically valid.

## Center Size

For 9x9, a 5x5 center is feasible to enumerate:

| Center | Raw Center Cycles | Reduced Cases | Setup Time | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 3x3 | 76 | 11 | 0.04s | ~136MB |
| 5x5 | 3,721,732 | 465,883 | 3.90s | ~1.26GB |
| 5x5 streamed dedup | 3,721,732 | 465,883 | 4.73s | ~0.68GB |

The 5x5 center is not automatically smaller at shallow depths. It starts with many more representatives than a 3x3 center expanded to the mid-20s. It begins to win after enough outer cells have been expanded.

Symmetry-expanded case counts for 9x9:

| Center | Expanded Depth | Cases | Count Time |
| ---: | ---: | ---: | ---: |
| 3x3 | 24 | 433,131 | 0.04s |
| 3x3 | 25 | 582,735 | 0.04s |
| 3x3 | 28 | 5,563,637 | 0.21s |
| 3x3 | 31 | 54,606,399 | 0.71s |
| 3x3 | 34 | 145,435,372 | 5.32s |
| 3x3 | 36 | 272,367,378 | 11.86s |
| 3x3 | 38 | 1,940,249,690 | 43.24s |
| 3x3 | 40 | 3,557,736,484 | 131.84s |
| 3x3 | 42 | 6,519,969,787 | 292.58s |
| 5x5 | 25 | 465,883 | 3.91s |
| 5x5 | 27 | 3,157,223 | 3.88s |
| 5x5 | 29 | 6,590,480 | 4.00s |
| 5x5 | 31 | 43,115,223 | 4.30s |
| 5x5 | 33 | 82,910,931 | 6.31s |
| 5x5 | 34 | 113,588,290 | 7.80s |
| 5x5 | 36 | 211,178,331 | 13.02s |
| 5x5 | 38 | 1,527,566,062 | 37.53s |
| 5x5 | 40 | 2,747,090,186 | 104.37s |
| 5x5 | 42 | 5,059,911,673 | 228.60s |

At depth 40, the 5x5 center has about 23% fewer cases than the 3x3-center route.

Center orbit sizes are almost always full for 9x9's 5x5 center:

| Orbit Size | Representatives |
| ---: | ---: |
| 8 | 1,333 |
| 16 | 464,550 |

That makes residual stabilizer symmetry a small opportunity, not a likely breakthrough.

## Ring Order Experiments

The experimental solver supports simple odd-square ring order variants through `MF_ORDER`, while keeping a locked center square in default order with `MF_LOCK_CENTER_N`.

These variants all preserve the 5x5 known value when the 3x3 center is locked:

| Order | 5x5 Result | Local Time |
| --- | ---: | ---: |
| `spiral` | 186,086,600 | 0.187s |
| `reverse` | 186,086,600 | 0.176s |
| `opposite` | 186,086,600 | 0.172s |
| `axes_first` | 186,086,600 | 0.168s |
| `corners_first` | 186,086,600 | 0.169s |

On 7x7 prefix counts, the default spiral order is clearly best among these simple variants:

| Expanded Depth | Order | Cases |
| ---: | --- | ---: |
| 24 | `spiral` | 433,131 |
| 24 | `reverse` | 1,312,883 |
| 24 | `opposite` | 433,131 |
| 24 | `axes_first` | 433,131 |
| 24 | `corners_first` | 1,519,456 |
| 28 | `spiral` | 5,563,637 |
| 28 | `reverse` | 21,693,370 |
| 28 | `opposite` | 19,551,708 |
| 28 | `axes_first` | 77,441,004 |
| 28 | `corners_first` | 10,226,999,250 |
| 32 | `spiral` | 75,020,101 |
| 32 | `reverse` | 255,347,442 |
| 32 | `opposite` | 79,696,867 |
| 32 | `axes_first` | 1,288,577,220 |

The default spiral seems strong because it grows each ring as a connected path. Scattering additions around the perimeter delays constraints and greatly increases the intermediate state count.

## 7x7 Larger-Center Check

Using a 5x5 center on 7x7 does not produce an obvious smaller-case speedup. A 50,000-case local sample on 12 threads was roughly comparable to the default 3x3-center route:

| Route | Split | Cases in Full Split | Sample Cases | Wall Time |
| --- | ---: | ---: | ---: | ---: |
| 3x3 center, stream to depth 24 | 24 | 433,131 | 50,000 | 298.55s |
| 5x5 center | 25 | 465,883 | 50,000 | 281.61s |

The 5x5-center sample was slightly faster per sampled case, but it has more cases overall and uses the slower 128-bit experimental path. This does not look like a clean win for 7x7. For 9x9, the value of the 5x5 center is mainly that it gives a more useful seed/scheduling unit.

## Complete Sample Timings

These are tiny complete 9x9 samples using `MF_CENTER_N=5`, streaming mode, and 12 local threads. They include local setup and any streaming skip cost.

| Depth | Start Case | Cases | Wall Time | Notes |
| ---: | ---: | ---: | ---: | --- |
| 40 | 0 | 1 | 4.91s | dominated by setup |
| 40 | 0 | 10 | 5.21s | |
| 40 | 0 | 100 | 26.97s | about 4.6 completed cases/s late in the run |
| 40 | 0 | 1,000 | timed out at 240s | 864 completed after 235s |
| 40 | 1,000,000 | 100 | 11.02s | lighter range |
| 40 | 100,000,000 | 100 | 13.89s | lighter range |
| 40 | 2,000,000,000 | 100 | 90.77s | mostly stream-skip cost |
| 42 | 0 | 100 | 9.50s | |
| 42 | 0 | 500 | 79.28s | about 6.6 completed cases/s late in the run |

Depth 42 has smaller residual subtrees, but many more subtrees. In these samples it does not obviously reduce total work versus depth 40.

## Split-Depth Work Profiling

`MF_PROFILE_WORK=1` instruments a bounded sample of generated prefixes and counts recursive calls below each prefix. This is a better work proxy than result size. Example:

```
MF_CENTER_N=5 MF_CENTER_STREAM_DEDUP=1 MF_STREAM_CASES=1 \
  MF_EXPANDED_N=42 MF_PROFILE_WORK=1 \
  ./experiments/9x9/sym_fold9 9 12 20 0
```

A small sweep over 20 center seeds per band and the first 20 generated prefixes in each band:

| Depth | Seed Start | Mean Expanded Cases/Seed | Mean Nodes/Prefix | Product Proxy |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 0 | 39,311 | 58,371,404 | 2.295e12 |
| 40 | 1,000 | 6,967 | 281,034,179 | 1.958e12 |
| 40 | 10,000 | 3,226 | 33,067,695 | 0.107e12 |
| 40 | 100,000 | 2,093 | 85,470,490 | 0.179e12 |
| 40 | 400,000 | 133,714 | 117,874,620 | 15.761e12 |
| 42 | 0 | 70,306 | 21,190,884 | 1.490e12 |
| 42 | 1,000 | 11,945 | 80,510,546 | 0.962e12 |
| 42 | 10,000 | 3,805 | 14,607,765 | 0.056e12 |
| 42 | 100,000 | 4,230 | 51,188,192 | 0.217e12 |
| 42 | 400,000 | 334,645 | 63,715,951 | 21.322e12 |
| 44 | 0 | 383,362 | 5,067,633 | 1.943e12 |
| 44 | 1,000 | 93,773 | 3,688,572 | 0.346e12 |
| 44 | 10,000 | 42,861 | 476,219 | 0.020e12 |
| 44 | 100,000 | 26,378 | 7,187,276 | 0.190e12 |
| 44 | 400,000 | 2,802,946 | 28,123,526 | 78.829e12 |

This is a biased first-prefix sample, not a final estimate. It still shows two useful things:

- Seed ranges differ by orders of magnitude in both expansion count and residual work.
- A single global split depth is unlikely to be ideal. Some bands look better with deeper splits, while the heavy high-index band gets much worse by depth 44.

The next scheduler should create adaptive work units: expand each seed until a cheap lookahead score reaches a target range, then dispatch those prefixes. Equal-sized global prefix chunks are the wrong abstraction for 9x9.

## Four-Stack Formulation

`stack_dp.cpp` tests a different model. Each of the four parity classes of grid edges is a matching. If the final stack is scanned from top to bottom, each matching must be properly nested: the first endpoint of an edge pushes onto that parity stack, and the second endpoint must pop the same edge.

This counts valid linear stack orders directly. Fixing the top cell to `0` and multiplying by `n*n` matches the known square counts:

| Command | Result | Notes |
| --- | ---: | --- |
| `./experiments/9x9/stack_dp 2` | 8 | exact |
| `./experiments/9x9/stack_dp 3` | 1,368 | exact |
| `./experiments/9x9/stack_dp 4` | 300,608 | exact, ~0.21s |
| `./experiments/9x9/stack_dp 5` | aborted | hit 5,000,000 memo states before finishing |

The naive memo key is `placed set + four open-edge stacks`. That is too large as a direct DFS state, but the formulation is valuable because it gives a cleaner target for a frontier DP: preserve the four page stacks only where they cross the frontier, not for every arbitrary placed subset.

## Rough Cost Projection

The 8x8 Modal run cost about $266.75. Its summed 16-CPU chunk wall time was 908,441s, or about 4,037 CPU-hours. That implies roughly $0.066 per CPU-hour for this Modal setup.

A direct 9x9 run with the current depth-40/42 approach appears to require roughly 1-3 million CPU-hours from the tiny complete samples. That suggests:

| Estimate | CPU-hours | Modal Cost at $0.066/CPU-hour | Wall at 1,600 CPUs |
| --- | ---: | ---: | ---: |
| Optimistic direct extension | 1,000,000 | ~$66,000 | ~26 days |
| Middle estimate | 2,000,000 | ~$132,000 | ~52 days |
| Pessimistic direct extension | 3,000,000 | ~$198,000 | ~78 days |

This is not a precision forecast. It is enough to say that the current algorithm is not a $500 run for 9x9. A publishable 9x9 result likely needs at least a 100x algorithmic improvement, or a much larger compute budget.

## Promising Directions

### Seed-Indexed Scheduling

The current streaming chunker indexes global expanded cases. For 9x9, skipping to high `start_case` values can itself cost tens of seconds locally and would become worse at deeper split depths. A better scheduler should partition by 5x5 center representative, or by a precomputed table of per-representative expansion counts, so each worker starts near its assigned subtree.

The experimental solver has `MF_SEED_START` and `MF_SEED_COUNT` for this. For example:

```
MF_CENTER_N=5 MF_CENTER_STREAM_DEDUP=1 MF_SEED_START=1000 MF_SEED_COUNT=10 \
  MF_STREAM_CASES=1 MF_EXPANDED_N=40 MF_SETUP_ONLY=1 \
  ./experiments/9x9/sym_fold9 9 12
```

### Cost-Weighted Chunks

The 8x8 run had a long tail: median chunk time was 144.64s, but the slowest chunk took 1,934.29s. For 9x9, equal-size chunks are a poor unit. A shallow lookahead score per seed or per generated prefix should let us bucket chunks by expected cost and reduce tail latency.

The split-depth profiling strengthens this: prefix count is not enough. Some seed slices have fewer generated prefixes but much larger residual trees. A work queue should score prefixes by a shallow recursive node estimate, not by prefix cardinality alone.

### Ring Frontier Dynamic Programming

After completing a square ring, future cells only attach to the current outer boundary, but the existing stack cycle still encodes interior constraints. The most promising algorithmic leap would be a compressed frontier state: keep the boundary order plus enough jump/link information to reproduce `get_good` for future insertions, then merge identical states with multiplicities.

If this works, the search becomes a transfer between rings rather than a flat enumeration of billions of stack cycles. The hard part is proving the compressed signature preserves all four edge-parity constraints.

The four-stack formulation may be a better way to define that signature. A completed or partially completed region should only need to expose the open edge-stack obligations that cross the boundary, plus enough ordering information to merge the four stack pages consistently.

### Meet-in-the-Middle

Another possible route is to build inward from the outside and outward from the center, then join compatible boundary states. This has the same core difficulty as ring DP: defining a compact, canonical boundary state that preserves the cycle and edge-jump constraints.

### Residual Symmetry

The current method removes D4 and stack-reversal symmetry at the center cycle. Most center orbits have size 16, so there may not be much residual symmetry left, but stabilizer-aware expansion could still help special center cases. This is probably smaller than the frontier-DP opportunity.
