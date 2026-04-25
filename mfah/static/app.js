const els = {
  displayName: document.getElementById("displayName"),
  globalProgress: document.getElementById("globalProgress"),
  globalBar: document.getElementById("globalBar"),
  unitCount: document.getElementById("unitCount"),
  activeClients: document.getElementById("activeClients"),
  resultMetricLabel: document.getElementById("resultMetricLabel"),
  answerCompleted: document.getElementById("answerCompleted"),
  connectionState: document.getElementById("connectionState"),
  startButton: document.getElementById("startButton"),
  pauseButton: document.getElementById("pauseButton"),
  threadCount: document.getElementById("threadCount"),
  personalUnits: document.getElementById("personalUnits"),
  personalTime: document.getElementById("personalTime"),
  personalNodes: document.getElementById("personalNodes"),
  sessionRate: document.getElementById("sessionRate"),
  currentUnit: document.getElementById("currentUnit"),
  prefixProgress: document.getElementById("prefixProgress"),
  campaignLabel: document.getElementById("campaignLabel"),
  gridSize: document.getElementById("gridSize"),
  prefixDepth: document.getElementById("prefixDepth"),
  stopDepth: document.getElementById("stopDepth"),
  totalPrefixes: document.getElementById("totalPrefixes"),
  knownAnswer: document.getElementById("knownAnswer"),
  recentLog: document.getElementById("recentLog"),
  canvas: document.getElementById("foldCanvas"),
};

const state = {
  config: null,
  clientId: localStorage.getItem("mfah.clientId") || crypto.randomUUID(),
  displayName: localStorage.getItem("mfah.displayName") || `folder-${Math.random().toString(16).slice(2, 6)}`,
  running: false,
  slots: [],
  sessionStartedAt: 0,
  sessionUnits: 0,
  sessionNodes: 0,
  latestPrefix: null,
  latestPrefixN: 5,
  latestPrefixDepth: 0,
  completedLogged: false,
};

localStorage.setItem("mfah.clientId", state.clientId);
els.displayName.value = state.displayName;

init();

async function init() {
  fillWorkerOptions();
  attachEvents();
  await registerClient();
  state.config = await getJson("/api/config");
  updateCampaign(state.config);
  drawFold();
  await refreshStats();
  setInterval(refreshStats, 3000);
  setInterval(sendHeartbeat, 15000);
}

function attachEvents() {
  els.startButton.addEventListener("click", startComputing);
  els.pauseButton.addEventListener("click", pauseComputing);
  els.displayName.addEventListener("change", async () => {
    state.displayName = els.displayName.value.trim() || "anonymous";
    localStorage.setItem("mfah.displayName", state.displayName);
    await registerClient();
  });
  window.addEventListener("resize", drawFold);
}

function fillWorkerOptions() {
  const max = Math.max(1, Math.min(8, navigator.hardwareConcurrency || 4));
  for (let i = 1; i <= max; i += 1) {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = String(i);
    els.threadCount.append(option);
  }
  els.threadCount.value = String(Math.min(2, max));
}

async function registerClient() {
  const response = await postJson("/api/client", {
    clientId: state.clientId,
    displayName: state.displayName,
  });
  if (response.clientId && response.clientId !== state.clientId) {
    state.clientId = response.clientId;
    localStorage.setItem("mfah.clientId", state.clientId);
  }
}

function updateCampaign(config) {
  els.campaignLabel.textContent = `${config.n}x${config.n}`;
  els.gridSize.textContent = `${config.n} x ${config.n}`;
  els.prefixDepth.textContent = String(config.prefixDepth);
  els.stopDepth.textContent = String(config.stopDepth || config.n2);
  els.resultMetricLabel.textContent = config.resultLabel || "Current answer sum";
  els.knownAnswer.textContent = config.knownAnswer ? formatInteger(config.knownAnswer) : "-";
  state.latestPrefixN = config.n;
}

async function startComputing() {
  if (state.running) {
    return;
  }
  state.running = true;
  state.sessionStartedAt = performance.now();
  state.sessionUnits = 0;
  state.sessionNodes = 0;
  state.completedLogged = false;
  setConnection("running", "running");
  els.startButton.disabled = true;
  els.pauseButton.disabled = false;
  els.threadCount.disabled = true;

  const count = Number(els.threadCount.value);
  state.slots = Array.from({ length: count }, (_, index) => new WorkerSlot(index));
  for (const slot of state.slots) {
    slot.run();
  }
}

function pauseComputing() {
  state.running = false;
  els.pauseButton.disabled = true;
  setConnection("idle", "finishing");
}

function finishIfStopped() {
  if (state.running) {
    return;
  }
  const active = state.slots.some((slot) => slot.busy);
  if (!active) {
    els.startButton.disabled = false;
    els.pauseButton.disabled = true;
    els.threadCount.disabled = false;
    setConnection("idle", "idle");
  }
}

class WorkerSlot {
  constructor(index) {
    this.index = index;
    this.worker = new Worker("/static/worker.js");
    this.busy = false;
    this.current = null;
    this.worker.onmessage = (event) => this.handleMessage(event.data);
  }

  async run() {
    while (state.running) {
      let lease;
      try {
        lease = await postJson("/api/work", { clientId: state.clientId });
      } catch (error) {
        logLine("coordinator", `request failed: ${error.message || error}`);
        setConnection("error", "network issue");
        await delay(3000);
        continue;
      }

      if (!lease.work) {
        if (lease.stats?.isComplete) {
          state.running = false;
          if (!state.completedLogged) {
            logLine("campaign", "complete");
            state.completedLogged = true;
          }
          setConnection("idle", "complete");
          applyStats(lease.stats);
          break;
        }
        setConnection("idle", "waiting");
        await refreshStats();
        await delay(5000);
        continue;
      }

      this.busy = true;
      this.current = lease.work;
      els.currentUnit.textContent = `#${lease.work.workUnitId}`;
      els.prefixProgress.textContent = `0 / ${lease.work.payload.prefixes.length}`;
      setConnection("running", "running");
      await this.compute(lease.work);
      this.busy = false;
      this.current = null;
      finishIfStopped();
    }
    this.busy = false;
    finishIfStopped();
  }

  compute(work) {
    return new Promise((resolve) => {
      this.resolve = resolve;
      this.worker.postMessage({ type: "work", payload: work.payload });
    });
  }

  async handleMessage(message) {
    if (message.type === "progress") {
      const work = this.current;
      if (!work) {
        return;
      }
      els.currentUnit.textContent = `#${work.workUnitId}`;
      els.prefixProgress.textContent = `${message.done} / ${message.total}`;
      state.latestPrefix = message.currentPrefix;
      state.latestPrefixN = work.payload.n;
      state.latestPrefixDepth = work.payload.depth;
      drawFold();
      return;
    }

    if (message.type === "error") {
      logLine("worker", message.error);
      setConnection("error", "worker error");
      this.resolve?.();
      return;
    }

    if (message.type === "result") {
      const work = this.current;
      if (!work) {
        this.resolve?.();
        return;
      }
      try {
        const submitted = await postJson("/api/result", {
          clientId: state.clientId,
          leaseId: work.leaseId,
          workUnitId: work.workUnitId,
          rawCount: message.rawCount,
          elapsedMs: message.elapsedMs,
          nodes: message.nodes,
        });
        if (submitted.ok) {
          state.sessionUnits += 1;
          state.sessionNodes += message.nodes;
          const contributionLabel = state.config?.isFullSearch ? "answer contribution" : "expanded prefixes";
          logLine(`#${work.workUnitId}`, `${formatInteger(submitted.answerContribution || "0")} ${contributionLabel}`);
          applyStats(submitted.stats);
        } else {
          logLine(`#${work.workUnitId}`, submitted.error || "rejected");
        }
      } catch (error) {
        logLine(`#${work.workUnitId}`, `submit failed: ${error.message || error}`);
      }
      updateSessionRate();
      this.resolve?.();
    }
  }
}

async function refreshStats() {
  try {
    const stats = await getJson(`/api/stats?client_id=${encodeURIComponent(state.clientId)}`);
    applyStats(stats);
  } catch {
    setConnection("error", "offline");
  }
}

function applyStats(stats) {
  const complete = stats.status.complete || { units: 0, prefixes: 0 };
  const leased = stats.status.leased || { units: 0, prefixes: 0 };
  const totalUnits = stats.totalUnits || 0;
  const doneUnits = complete.units || 0;
  const pct = totalUnits ? (doneUnits / totalUnits) * 100 : 0;

  els.globalProgress.textContent = `${pct.toFixed(2)}%`;
  els.globalBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  els.unitCount.textContent = `${formatInteger(doneUnits)} / ${formatInteger(totalUnits)}`;
  els.activeClients.textContent = formatInteger(stats.activeClients || 0);
  els.resultMetricLabel.textContent = stats.resultLabel || "Current answer sum";
  els.answerCompleted.textContent = formatInteger(stats.answerCompleted || "0");
  els.totalPrefixes.textContent = formatInteger(stats.totalPrefixes || 0);
  els.campaignLabel.textContent = `${stats.n}x${stats.n} | ${formatInteger(leased.units || 0)} active`;
  els.stopDepth.textContent = String(stats.stopDepth || state.config?.stopDepth || "-");

  if (stats.personal) {
    els.personalUnits.textContent = formatInteger(stats.personal.units);
    els.personalTime.textContent = formatDuration(stats.personal.elapsedMs || 0);
    els.personalNodes.textContent = formatInteger(stats.personal.nodes || 0);
  }
  updateSessionRate();
}

function updateSessionRate() {
  if (!state.sessionStartedAt || state.sessionUnits === 0) {
    els.sessionRate.textContent = "0 units/min";
    return;
  }
  const minutes = Math.max(1 / 60, (performance.now() - state.sessionStartedAt) / 60000);
  els.sessionRate.textContent = `${(state.sessionUnits / minutes).toFixed(1)} units/min`;
}

function drawFold() {
  const canvas = els.canvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * dpr));
  const height = Math.max(240, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8f3e8";
  ctx.fillRect(0, 0, width, height);

  const n = state.latestPrefixN || state.config?.n || 5;
  const coords = coordsFromMap(buildSpiralMap(n, n));
  const placedOrder = state.latestPrefix
    ? cycleOrder(state.latestPrefix, state.latestPrefixDepth || state.latestPrefix.length)
    : [];
  const rankByCell = new Map(placedOrder.map((cell, rank) => [cell, rank]));
  const pad = 28 * dpr;
  const size = Math.min(width - pad * 2, height - pad * 2);
  const cell = size / n;
  const left = (width - cell * n) / 2;
  const top = (height - cell * n) / 2;

  ctx.lineWidth = Math.max(1, dpr);
  ctx.strokeStyle = "#d5cbb9";
  for (let id = 0; id < n * n; id += 1) {
    const [x, y] = coords[id];
    const px = left + x * cell;
    const py = top + y * cell;
    if (rankByCell.has(id)) {
      const rank = rankByCell.get(id);
      ctx.fillStyle = colorForRank(rank, Math.max(1, placedOrder.length - 1));
    } else {
      ctx.fillStyle = "#eee7da";
    }
    ctx.fillRect(px + 2 * dpr, py + 2 * dpr, cell - 4 * dpr, cell - 4 * dpr);
    ctx.strokeRect(px + 2 * dpr, py + 2 * dpr, cell - 4 * dpr, cell - 4 * dpr);
  }

  if (placedOrder.length > 1) {
    ctx.beginPath();
    placedOrder.forEach((id, index) => {
      const [x, y] = coords[id];
      const px = left + x * cell + cell / 2;
      const py = top + y * cell + cell / 2;
      if (index === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    });
    ctx.strokeStyle = "rgba(31, 38, 39, 0.72)";
    ctx.lineWidth = Math.max(2, 3 * dpr);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();
  }

  ctx.fillStyle = "#1f2627";
  ctx.font = `${Math.max(11, 12 * dpr)}px ui-sans-serif, system-ui`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let id = 0; id < n * n; id += 1) {
    const [x, y] = coords[id];
    const px = left + x * cell + cell / 2;
    const py = top + y * cell + cell / 2;
    ctx.fillText(String(id), px, py);
  }

  if (placedOrder.length === 0) {
    ctx.fillStyle = "#697271";
    ctx.font = `${Math.max(14, 16 * dpr)}px ui-sans-serif, system-ui`;
    ctx.fillText("waiting for a prefix", width / 2, height - 24 * dpr);
  }
}

function cycleOrder(prefix, depth) {
  if (!prefix.length) {
    return [];
  }
  const order = [];
  const seen = new Set();
  let cell = 0;
  for (let i = 0; i < depth; i += 1) {
    if (seen.has(cell) || cell >= depth) {
      break;
    }
    order.push(cell);
    seen.add(cell);
    cell = prefix[cell];
    if (cell === 0) {
      break;
    }
  }
  return order;
}

function colorForRank(rank, maxRank) {
  const t = maxRank ? rank / maxRank : 0;
  const stops = [
    [15, 118, 110],
    [37, 99, 235],
    [180, 83, 9],
  ];
  const segment = t < 0.5 ? 0 : 1;
  const local = segment === 0 ? t * 2 : (t - 0.5) * 2;
  const a = stops[segment];
  const b = stops[segment + 1];
  const rgb = a.map((value, index) => Math.round(value + (b[index] - value) * local));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
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

function coordsFromMap(map) {
  const coords = Array(map.length * map[0].length);
  for (let x = 0; x < map.length; x += 1) {
    for (let y = 0; y < map[0].length; y += 1) {
      coords[map[x][y]] = [x, y];
    }
  }
  return coords;
}

function logLine(label, detail) {
  const item = document.createElement("li");
  const left = document.createElement("strong");
  const right = document.createElement("span");
  left.textContent = label;
  right.textContent = detail;
  item.append(left, right);
  els.recentLog.prepend(item);
  while (els.recentLog.children.length > 8) {
    els.recentLog.lastElementChild.remove();
  }
}

function setConnection(kind, text) {
  els.connectionState.className = `pill ${kind}`;
  els.connectionState.textContent = text;
}

async function sendHeartbeat() {
  try {
    await postJson("/api/heartbeat", { clientId: state.clientId });
  } catch {
    // The next stats refresh will surface coordinator issues.
  }
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function formatInteger(value) {
  const text = String(value ?? "0");
  if (!/^\d+$/.test(text)) {
    return text || "0";
  }
  return text.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatDuration(ms) {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes < 60) {
    return `${minutes}m ${rest}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
