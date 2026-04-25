let cancelled = false;

self.onmessage = (event) => {
  const message = event.data;
  if (message.type === "cancel") {
    cancelled = true;
    return;
  }
  if (message.type === "work") {
    cancelled = false;
    try {
      const result = solvePayload(message.payload);
      self.postMessage({ type: "result", ...result });
    } catch (error) {
      self.postMessage({ type: "error", error: String(error && error.message ? error.message : error) });
    }
  }
};

function solvePayload(payload) {
  const started = performance.now();
  const n = payload.n;
  const n2 = n * n;
  const depth = payload.depth;
  const edges = makeEdges(n);
  const useNumberMasks = n2 <= 30;
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
      const counted = countNumber(cycle, depth, edges, n2, () => {
        nodes += 1;
      });
      raw += BigInt(counted);
    } else {
      const counted = countBigInt(cycle, depth, edges, n2, () => {
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

function countNumber(cycle, i, edges, n2, touch) {
  touch();
  let good = goodMaskNumber(cycle, i, edges);
  if (good === 0) {
    return 0;
  }
  if (i === n2 - 1) {
    return popcountNumber(good);
  }

  let total = 0;
  while (good !== 0) {
    const bit = good & -good;
    const j = 31 - Math.clz32(bit);
    good -= bit;
    const next = cycle[j];
    cycle[i] = next;
    cycle[j] = i;
    total += countNumber(cycle, i + 1, edges, n2, touch);
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

function countBigInt(cycle, i, edges, n2, touch) {
  touch();
  let good = goodMaskBigInt(cycle, i, edges);
  if (good === 0n) {
    return 0n;
  }
  if (i === n2 - 1) {
    return BigInt(popcountBigInt(good));
  }

  let total = 0n;
  while (good !== 0n) {
    const bit = good & -good;
    const j = bitLength(bit) - 1;
    good -= bit;
    const next = cycle[j];
    cycle[i] = next;
    cycle[j] = i;
    total += countBigInt(cycle, i + 1, edges, n2, touch);
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
