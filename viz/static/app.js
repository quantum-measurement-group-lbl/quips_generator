"use strict";

const els = {
  omega: document.getElementById("omega_khz"),
  mass: document.getElementById("mass_fg"),
  gamma: document.getElementById("gamma_khz"),
  eta: document.getElementById("eta"),
  form: document.getElementById("params"),
  runBtn: document.getElementById("run-btn"),
  status: document.getElementById("status"),
  slider: document.getElementById("slider"),
  playBtn: document.getElementById("play-btn"),
  timeLabel: document.getElementById("time-label"),
};

let runId = null;
let nFrames = 0;
let frameTimes = [];
let frameCache = {};
let pending = {};
let currentFrame = 0;
let playing = false;
let playTimer = null;

function setStatus(msg) {
  els.status.textContent = msg || "";
}

function renderTimeSeries(payload) {
  const xTraces = [
    { x: payload.times_ms, y: payload.xc_pm, name: "⟨x⟩", line: { width: 1, color: "#1f77b4" } },
    { x: payload.times_ms, y: payload.xc_kalman_pm, name: "Kalman", line: { width: 1.2, color: "#d62728" } },
  ];
  const pTraces = [
    { x: payload.times_ms, y: payload.pc, name: "⟨p⟩", line: { width: 1, color: "#1f77b4" } },
    { x: payload.times_ms, y: payload.pc_kalman, name: "Kalman", line: { width: 1.2, color: "#d62728" } },
  ];
  const baseLayout = (yTitle) => ({
    margin: { l: 60, r: 20, t: 10, b: 30 },
    xaxis: { title: "t [ms]" },
    yaxis: { title: yTitle },
    showlegend: true,
    legend: { x: 1, xanchor: "right", y: 1 },
    shapes: [{
      type: "line", xref: "x", yref: "paper",
      x0: payload.frame_time_ms[0], x1: payload.frame_time_ms[0],
      y0: 0, y1: 1, line: { color: "#888", width: 1, dash: "dot" },
    }],
  });
  Plotly.react("plot-x", xTraces, baseLayout("⟨x⟩ [pm]"), { displayModeBar: false });
  Plotly.react("plot-p", pTraces, baseLayout("⟨p⟩ [kg·m/s]"), { displayModeBar: false });
}

function renderWigner(payload, W) {
  const trace = {
    type: "heatmap",
    x: payload.x_grid_pm,
    y: payload.p_grid,
    z: W,
    zmin: payload.wigner_min,
    zmax: payload.wigner_max,
    colorscale: "Viridis",
    colorbar: { title: "W" },
  };
  const layout = {
    margin: { l: 70, r: 20, t: 10, b: 50 },
    xaxis: { title: "x [pm]" },
    yaxis: { title: "p [kg·m/s]", scaleanchor: null },
  };
  Plotly.react("plot-wigner", [trace], layout, { displayModeBar: false });
}

let payloadRef = null;

async function fetchFrame(idx) {
  if (frameCache[idx]) return frameCache[idx];
  if (pending[idx]) return pending[idx];
  const p = fetch(`/api/wigner/${idx}?run_id=${encodeURIComponent(runId)}`)
    .then(async (r) => {
      if (r.status === 409) {
        // Cache invalidated; reload everything.
        await loadInitial();
        throw new Error("stale run");
      }
      if (!r.ok) throw new Error(`frame ${idx}: ${r.status}`);
      const j = await r.json();
      frameCache[idx] = j.W;
      delete pending[idx];
      return j.W;
    })
    .catch((e) => { delete pending[idx]; throw e; });
  pending[idx] = p;
  return p;
}

async function setFrame(idx) {
  if (idx < 0 || idx >= nFrames) return;
  currentFrame = idx;
  els.slider.value = String(idx);
  const t = frameTimes[idx];
  els.timeLabel.textContent = `t = ${t.toFixed(3)} ms`;
  // Update cursor lines on the time-series plots.
  const cursorShape = [{
    type: "line", xref: "x", yref: "paper",
    x0: t, x1: t, y0: 0, y1: 1,
    line: { color: "#888", width: 1, dash: "dot" },
  }];
  Plotly.relayout("plot-x", { shapes: cursorShape });
  Plotly.relayout("plot-p", { shapes: cursorShape });

  // Lazy-prefetch a couple ahead.
  for (let k = 1; k <= 3; k++) {
    const j = idx + k;
    if (j < nFrames && !frameCache[j] && !pending[j]) fetchFrame(j).catch(() => {});
  }

  try {
    const W = await fetchFrame(idx);
    if (currentFrame === idx) {
      Plotly.restyle("plot-wigner", { z: [W] });
    }
  } catch (e) {
    setStatus(String(e.message || e));
  }
}

function stopPlayback() {
  playing = false;
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  els.playBtn.textContent = "Play";
}

function startPlayback() {
  if (nFrames === 0) return;
  playing = true;
  els.playBtn.textContent = "Pause";
  playTimer = setInterval(() => {
    setFrame((currentFrame + 1) % nFrames);
  }, 80);
}

function applyPayload(payload) {
  payloadRef = payload;
  runId = payload.run_id;
  nFrames = payload.n_frames;
  frameTimes = payload.frame_time_ms;
  frameCache = {};
  pending = {};
  currentFrame = 0;

  els.omega.value = (payload.params.omega_hz / 1e3).toString();
  els.mass.value = (payload.params.mass_kg * 1e18).toString();
  els.gamma.value = (payload.params.gamma_ba_hz / 1e3).toString();
  els.eta.value = payload.params.eta.toString();

  els.slider.min = "0";
  els.slider.max = String(nFrames - 1);
  els.slider.value = "0";

  renderTimeSeries(payload);
  // Seed wigner with a placeholder of zeros so layout renders before frame 0 arrives.
  const ny = payload.p_grid.length, nx = payload.x_grid_pm.length;
  const seed = Array.from({ length: ny }, () => new Array(nx).fill(payload.wigner_min));
  renderWigner(payload, seed);

  setFrame(0);
}

async function loadInitial() {
  setStatus("loading…");
  const r = await fetch("/api/data");
  if (!r.ok) { setStatus(`load failed: ${r.status}`); return; }
  const payload = await r.json();
  applyPayload(payload);
  setStatus(`run ${payload.run_id.slice(0, 8)} · ${payload.n_frames} frames`);
}

async function runSimulation(ev) {
  ev.preventDefault();
  stopPlayback();
  setStatus("simulating…");
  els.runBtn.disabled = true;
  try {
    const body = {
      omega_hz: parseFloat(els.omega.value) * 1e3,
      mass_kg: parseFloat(els.mass.value) * 1e-18,
      gamma_ba_hz: parseFloat(els.gamma.value) * 1e3,
      eta: parseFloat(els.eta.value),
    };
    const r = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      setStatus(`simulate failed: ${r.status}`);
      return;
    }
    const payload = await r.json();
    applyPayload(payload);
    setStatus(`run ${payload.run_id.slice(0, 8)} · ${payload.n_frames} frames`);
  } finally {
    els.runBtn.disabled = false;
  }
}

els.slider.addEventListener("input", (e) => {
  stopPlayback();
  setFrame(parseInt(e.target.value, 10));
});
els.playBtn.addEventListener("click", () => {
  if (playing) stopPlayback(); else startPlayback();
});
els.form.addEventListener("submit", runSimulation);

loadInitial();
