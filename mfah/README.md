# Map Folding At Home

Volunteer-compute prototype for the map folding search.

The Flask coordinator leases deterministic pieces of a raw symmetry-solver search. Browser clients run the matching JavaScript worker, submit the raw count for each work unit, and the server aggregates the global answer contribution.

Default campaign:

- Grid: `5x5`
- Prefix depth: `14`
- Work-unit size: `64` prefixes
- Known answer check: `186086600`

Run locally:

```
cd /Users/johno/projects/map_folding
python3 -m flask --app mfah.app run --host 127.0.0.1 --port 5050
```

Then open `http://127.0.0.1:5050`.

Useful environment variables:

```
MFAH_DB=/path/to/mfah.sqlite3
MFAH_N=5
MFAH_PREFIX_DEPTH=14
MFAH_STOP_DEPTH=25
MFAH_PREFIXES_PER_UNIT=64
MFAH_LEASE_SECONDS=300
MFAH_VERIFY_RESULTS=0
```

If `MFAH_STOP_DEPTH` is omitted, each worker completes the full raw search for the configured `n`. For 9x9, set a bounded stop depth; completing raw 9x9 from browser clients is not practical with this first worker.

`MFAH_VERIFY_RESULTS=1` makes the server recompute submitted units before accepting them. That is useful for local testing, but defeats the purpose of donated compute. A public deployment should add redundant assignment or spot checks before trusting arbitrary clients.

Current client kernel:

- Runs in Web Workers so the UI stays responsive.
- Uses the same center-out raw cycle insertion logic as the C++ solver for the default campaign.
- Uses fast 32-bit masks for `n <= 5` and a BigInt fallback for larger browser-side experiments.

This is intentionally shaped as a coordinator plus replaceable client kernel. The next step for real 8x8/9x9 volunteer work is to replace the default 5x5 campaign generator with the symmetry-reduced 8x8/9x9 prefix format and swap the worker for a WASM build of that kernel.
