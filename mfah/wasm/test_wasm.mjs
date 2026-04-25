import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const wasmPath = new URL("../static/sym_kernel.wasm", import.meta.url);
const bytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const exp = instance.exports;
const input = exp.input_ptr();
const output = exp.output_ptr();
const stride = exp.case_stride();
const maxCases = exp.max_cases();

function makePayload(n, depth, stopDepth, count) {
  const code = [
    "import json",
    "from mfah.folding import generate_prefixes, count_payload_raw",
    `n=${n}; d=${depth}; s=${stopDepth}; c=${count}`,
    "prefixes=generate_prefixes(n,d)[:c]",
    'payload={"n":n,"depth":d,"stopDepth":s,"prefixes":prefixes}',
    'print(json.dumps({"payload":payload,"expected":count_payload_raw(payload)}))',
  ].join("\n");
  const result = spawnSync("python3", ["-c", code], { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout);
  }
  return JSON.parse(result.stdout);
}

function solve(payload) {
  if (payload.prefixes.length > maxCases) {
    throw new Error(`too many cases: ${payload.prefixes.length} > ${maxCases}`);
  }
  const memory = new Uint8Array(exp.memory.buffer);
  const view = new DataView(exp.memory.buffer);
  const n2 = payload.n * payload.n;
  memory.fill(0, input, input + stride * payload.prefixes.length);
  for (let i = 0; i < payload.prefixes.length; i += 1) {
    memory.set(Buffer.from(payload.prefixes[i], "base64"), input + i * stride);
    view.setBigUint64(input + i * stride + 100, 1n, true);
  }

  const status = exp.solve_cases(payload.n, payload.depth, payload.stopDepth || n2, payload.prefixes.length);
  const out = new DataView(exp.memory.buffer, output, 32);
  const count =
    BigInt(out.getUint32(0, true)) |
    (BigInt(out.getUint32(4, true)) << 32n) |
    (BigInt(out.getUint32(8, true)) << 64n) |
    (BigInt(out.getUint32(12, true)) << 96n);
  const nodes =
    BigInt(out.getUint32(16, true)) |
    (BigInt(out.getUint32(20, true)) << 32n);

  return {
    status,
    outStatus: out.getUint32(24, true),
    count,
    nodes,
  };
}

for (const [n, depth, stopDepth, count] of [
  [5, 14, 20, 16],
  [6, 14, 20, 16],
  [7, 14, 28, 64],
]) {
  const { payload, expected } = makePayload(n, depth, stopDepth, count);
  const got = solve(payload);
  console.log(
    `${n}x${n} d${depth}->${stopDepth} prefixes=${count} expected=${expected} got=${got.count} nodes=${got.nodes}`
  );
  if (got.count !== BigInt(expected) || got.status !== 0 || got.outStatus !== 0) {
    process.exit(1);
  }
}
