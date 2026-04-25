let cancelled = false;
let wasmKernelPromise = null;

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "cancel") {
    cancelled = true;
    return;
  }
  if (message.type === "work") {
    cancelled = false;
    try {
      const result = await solvePayload(message.payload);
      self.postMessage({ type: "result", ...result });
    } catch (error) {
      self.postMessage({ type: "error", error: String(error && error.message ? error.message : error) });
    }
  }
};

async function solvePayload(payload) {
  const kernel = payload.forceJs ? null : await loadWasmKernel();
  if (kernel && payload.n <= 9) {
    return solvePayloadWasm(payload, kernel);
  }
  return solvePayloadJs(payload);
}

async function loadWasmKernel() {
  if (!wasmKernelPromise) {
    wasmKernelPromise = (async () => {
      try {
        const response = await fetch("/static/sym_kernel.wasm");
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
        const exp = instance.exports;
        const required = ["memory", "input_ptr", "output_ptr", "case_stride", "max_cases", "solve_cases"];
        if (!required.every((name) => exp[name])) {
          throw new Error("sym_kernel.wasm is missing required exports");
        }
        return {
          memory: exp.memory,
          input: exp.input_ptr(),
          output: exp.output_ptr(),
          stride: exp.case_stride(),
          maxCases: exp.max_cases(),
          solveCases: exp.solve_cases,
        };
      } catch {
        return null;
      }
    })();
  }
  return wasmKernelPromise;
}

function solvePayloadWasm(payload, kernel) {
  const started = performance.now();
  const n = payload.n;
  const n2 = n * n;
  const depth = payload.depth;
  const stopDepth = payload.stopDepth || n2;
  const cases = payloadCases(payload);
  const total = cases.length;
  const requestedBatchSize = Number(payload.wasmBatchSize || 16);
  const batchSize = Math.max(
    1,
    Math.min(kernel.maxCases, Number.isFinite(requestedBatchSize) ? requestedBatchSize : 16)
  );
  let nodes = 0;
  let raw = 0n;

  for (let start = 0; start < total; start += batchSize) {
    if (cancelled) {
      break;
    }

    const end = Math.min(total, start + batchSize);
    writeWasmCases(kernel, cases, start, end, n2);
    const status = kernel.solveCases(n, depth, stopDepth, end - start);
    const result = readWasmOutput(kernel);
    if (status !== 0 || result.status !== 0) {
      throw new Error(`WASM solver rejected work unit with status ${status}/${result.status}`);
    }
    raw += result.count;
    nodes += result.nodes;

    self.postMessage({
      type: "progress",
      done: end,
      total,
      nodes,
      nodesDelta: result.nodes,
      rawCount: raw.toString(),
      currentPrefix: Array.from(decodePrefix(cases[start].prefix, n2).slice(0, depth)),
      kernel: "wasm",
    });
  }

  return {
    rawCount: raw.toString(),
    nodes,
    elapsedMs: Math.round(performance.now() - started),
    cancelled,
    kernel: "wasm",
  };
}

function payloadCases(payload) {
  if (Array.isArray(payload.cases)) {
    return payload.cases.map((item) => {
      if (typeof item === "string") {
        return { prefix: item, multiplier: 1n };
      }
      return {
        prefix: item.prefix || item.cycle || item.encoded,
        multiplier: BigInt(item.multiplier || 1),
      };
    });
  }
  return payload.prefixes.map((prefix) => ({ prefix, multiplier: 1n }));
}

function writeWasmCases(kernel, cases, start, end, n2) {
  const memory = new Uint8Array(kernel.memory.buffer);
  const view = new DataView(kernel.memory.buffer);
  const count = end - start;
  memory.fill(0, kernel.input, kernel.input + kernel.stride * count);
  for (let local = 0; local < count; local += 1) {
    const item = cases[start + local];
    const offset = kernel.input + local * kernel.stride;
    memory.set(decodePrefix(item.prefix, n2), offset);
    view.setBigUint64(offset + 100, item.multiplier, true);
  }
}

function readWasmOutput(kernel) {
  const view = new DataView(kernel.memory.buffer, kernel.output, 32);
  const count =
    BigInt(view.getUint32(0, true)) |
    (BigInt(view.getUint32(4, true)) << 32n) |
    (BigInt(view.getUint32(8, true)) << 64n) |
    (BigInt(view.getUint32(12, true)) << 96n);
  const nodeCount =
    BigInt(view.getUint32(16, true)) |
    (BigInt(view.getUint32(20, true)) << 32n);
  return {
    count,
    nodes: Number(nodeCount),
    status: view.getUint32(24, true),
  };
}

function solvePayloadJs(payload) {
  const started = performance.now();
  const n = payload.n;
  const n2 = n * n;
  const depth = payload.depth;
  const stopDepth = payload.stopDepth || n2;
  const edges = makeEdges(n);
  const useNumberMasks = stopDepth <= 30;
  let nodes = 0;
  let raw = 0n;
  const total = payload.prefixes.length;
  const progressEvery = Math.max(1, Math.floor(total / 24));

  for (let index = 0; index < total; index += 1) {
    if (cancelled) {
      break;
    }
    const cycle = decodePrefix(payload.prefixes[index], n2);
    const before = nodes;
    if (useNumberMasks) {
      const counted = countNumber(cycle, depth, edges, stopDepth, () => {
        nodes += 1;
      });
      raw += BigInt(counted);
    } else {
      const counted = countBigInt(cycle, depth, edges, stopDepth, () => {
        nodes += 1;
      });
      raw += counted;
    }

    if (index === 0 || index + 1 === total || (index + 1) % progressEvery === 0) {
      self.postMessage({
        type: "progress",
        done: index + 1,
        total,
        nodes,
        nodesDelta: nodes - before,
        rawCount: raw.toString(),
        currentPrefix: Array.from(cycle.slice(0, depth)),
      });
    }
  }

  return {
    rawCount: raw.toString(),
    nodes,
    elapsedMs: Math.round(performance.now() - started),
    cancelled,
    kernel: "js",
  };
}

function decodePrefix(encoded, n2) {
  const text = atob(encoded);
  const cycle = new Uint8Array(n2);
  for (let i = 0; i < text.length; i += 1) {
    cycle[i] = text.charCodeAt(i);
  }
  return cycle;
}

function countNumber(cycle, i, edges, stopDepth, touch) {
  touch();
  if (i === stopDepth) {
    return 1;
  }
  let good = goodMaskNumber(cycle, i, edges);
  if (good === 0) {
    return 0;
  }

  let total = 0;
  while (good !== 0) {
    const bit = good & -good;
    const j = 31 - Math.clz32(bit);
    good -= bit;
    const next = cycle[j];
    cycle[i] = next;
    cycle[j] = i;
    total += countNumber(cycle, i + 1, edges, stopDepth, touch);
    cycle[j] = next;
  }
  return total;
}

function goodMaskNumber(cycle, i, edges) {
  let good = (1 << i) - 1;
  for (const edge of edges) {
    const other = edge[i];
    if (other >= i) {
      continue;
    }

    let allowed = 1 << other;
    let j = cycle[other];
    while (j !== other) {
      const j1 = edge[j];
      if (j1 < i) {
        j = j1;
      }
      allowed |= 1 << j;
      j = cycle[j];
    }

    good &= allowed;
    if (good === 0) {
      return 0;
    }
  }
  return good;
}

function popcountNumber(value) {
  let count = 0;
  while (value !== 0) {
    value &= value - 1;
    count += 1;
  }
  return count;
}

function countBigInt(cycle, i, edges, stopDepth, touch) {
  touch();
  if (i === stopDepth) {
    return 1n;
  }
  let good = goodMaskBigInt(cycle, i, edges);
  if (good === 0n) {
    return 0n;
  }

  let total = 0n;
  while (good !== 0n) {
    const bit = good & -good;
    const j = bitLength(bit) - 1;
    good -= bit;
    const next = cycle[j];
    cycle[i] = next;
    cycle[j] = i;
    total += countBigInt(cycle, i + 1, edges, stopDepth, touch);
    cycle[j] = next;
  }
  return total;
}

function goodMaskBigInt(cycle, i, edges) {
  let good = (1n << BigInt(i)) - 1n;
  for (const edge of edges) {
    const other = edge[i];
    if (other >= i) {
      continue;
    }

    let allowed = 1n << BigInt(other);
    let j = cycle[other];
    while (j !== other) {
      const j1 = edge[j];
      if (j1 < i) {
        j = j1;
      }
      allowed |= 1n << BigInt(j);
      j = cycle[j];
    }

    good &= allowed;
    if (good === 0n) {
      return 0n;
    }
  }
  return good;
}

function bitLength(value) {
  return value.toString(2).length;
}

function popcountBigInt(value) {
  let count = 0;
  while (value !== 0n) {
    value &= value - 1n;
    count += 1;
  }
  return count;
}

function buildSpiralMap(width, height) {
  const grid = Array.from({ length: width }, () => Array(height).fill(width * height));
  let x = 0;
  let y = 0;
  let dx = 1;
  let dy = 0;
  for (let i = width * height - 1; i >= 0; i -= 1) {
    grid[x][y] = i;
    const x1 = x + dx;
    const y1 = y + dy;
    if (
      x1 < 0 ||
      x1 >= width ||
      y1 < 0 ||
      y1 >= height ||
      grid[x1][y1] < width * height
    ) {
      const oldDy = dy;
      dy = dx;
      dx = -oldDy;
      x += dx;
      y += dy;
    } else {
      x = x1;
      y = y1;
    }
  }
  return grid;
}

function makeEdges(n) {
  const n2 = n * n;
  const edges = Array.from({ length: 4 }, () => new Int16Array(n2).fill(n2));
  const map = buildSpiralMap(n, n);
  for (let x = 0; x < n; x += 1) {
    for (let y = 0; y < n; y += 1) {
      const idx = map[x][y];
      if (x > 0) {
        const edge = edges[x % 2];
        const other = map[x - 1][y];
        edge[idx] = other;
        edge[other] = idx;
      }
      if (y > 0) {
        const edge = edges[2 + (y % 2)];
        const other = map[x][y - 1];
        edge[idx] = other;
        edge[other] = idx;
      }
    }
  }
  return edges;
}
