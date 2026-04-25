# Map Folding At Home Path To Real 9x9 Work

Status, 2026-04-24:

- Public demo is intentionally bounded 7x7: `n=7`, prefix depth `14`, stop depth `28`, `64` prefixes per unit.
- The demo checks the coordinator, leases, browser worker, visualization, and result aggregation. It is not meant to compute the 7x7 answer.
- Local full-demo smoke test through the WASM worker completed all `65` units / `4127` prefixes in about `2s` wall time and totaled `35044546` expanded prefixes.
- Public deploy smoke test completed one unit with `1390581` expanded prefixes, `3436669` visited nodes, and the worker reported `kernel=wasm`.

The useful boundary for real 8x8/9x9 volunteer work is now:

```json
{
  "n": 9,
  "depth": 32,
  "stopDepth": 81,
  "cases": [
    {"prefix": "base64 cycle bytes", "multiplier": "16"}
  ]
}
```

The browser does not need to know how the symmetry quotient was produced. It only needs cycle bytes in the center-out index order plus the orbit multiplier. The checked-in `mfah/static/sym_kernel.wasm` consumes that shape already; today's raw `prefixes` payload is treated as `cases` with multiplier `1`.

## Next server-side work

1. Add an exporter beside `experiments/9x9/sym_fold9.cpp` that writes JSONL work units instead of solving them locally.
2. Generate symmetry-reduced seed cases exactly as `sym_fold9.cpp` does:
   - default 9x9 center is `5x5`,
   - quotient by D4 square symmetries and stack reversal,
   - retain each representative cycle with its orbit multiplier.
3. Expand seed cases to a chosen `MF_EXPANDED_N` on the server, then pack the expanded weighted cycles into `cases` work units.
4. Use the existing profiling helpers (`MF_PROFILE_WORK`, `MF_SEED_COUNTS_SUMMARY`, `MF_COST_DEPTH`) to make work units approximately equal cost before publishing them.
5. Add redundant assignment or spot checks. Public clients cannot be trusted for final arithmetic without either quorum, randomized verification, or both.

## Open research knobs

- `MF_CENTER_N=5` is the obvious 9x9 analogue of the 8x8 `4x4` center reduction, but it may produce a very uneven seed distribution. We need measured seed-expanded histograms before choosing unit boundaries.
- For browser work, units should target seconds to low minutes, not hours. Long units make cancellation, stale leases, and dishonest clients harder to manage.
- The WASM kernel currently reports progress only between cases. If a single expanded case is too large, split earlier or add a deeper prefix expansion before publishing.
- The client kernel is scalar CPU WASM. GPU/WebGPU only helps if we find a much flatter, wider frontier; the current recursive tree is branch-heavy and irregular.
- The same `cases` payload can feed Modal workers, native C++ workers, and browser workers, which gives us a way to cross-check volunteer results against trusted compute.
