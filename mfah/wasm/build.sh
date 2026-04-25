#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
OUT="$ROOT/mfah/static/sym_kernel.wasm"

if command -v clang >/dev/null 2>&1 && clang --target=wasm32 -v >/dev/null 2>&1; then
  CLANG="${CLANG:-clang}"
elif [ -x /opt/homebrew/opt/llvm/bin/clang ]; then
  CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang}"
else
  echo "No clang with wasm32 support found. On macOS: brew install llvm lld" >&2
  exit 1
fi

export PATH="/opt/homebrew/opt/lld/bin:/opt/homebrew/opt/llvm/bin:$PATH"
mkdir -p "$(dirname "$OUT")"

"$CLANG" --target=wasm32 -O3 -Wall -Wextra -nostdlib \
  -Wl,--no-entry \
  -Wl,--export-memory \
  -Wl,--export=input_ptr \
  -Wl,--export=output_ptr \
  -Wl,--export=case_stride \
  -Wl,--export=max_cases \
  -Wl,--export=solve_cases \
  -o "$OUT" \
  "$ROOT/mfah/wasm/sym_kernel.c"

ls -lh "$OUT"
