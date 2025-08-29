// ===== Elements =====
const els = {
  target: document.getElementById("target"),
  activation: document.getElementById("activation"),
  lam: document.getElementById("lam"),
  seed: document.getElementById("seed"),
  noise: document.getElementById("noise"),

  layersPanel: document.getElementById("layersPanel"),
  addLayer: document.getElementById("addLayer"),
  clearLayers: document.getElementById("clearLayers"),
  depthBadge: document.getElementById("depthBadge"),

  lamVal: document.getElementById("lamVal"),
  mseVal: document.getElementById("mseVal"),
  paramsText: document.getElementById("paramsText"),
  chart: document.getElementById("chart"),
  residual: document.getElementById("residual"),

  randomize: document.getElementById("randomize"),
  presetWide: document.getElementById("presetWide"),
  presetPyramid: document.getElementById("presetPyramid"),
  presetDeepThin: document.getElementById("presetDeepThin"),
};

// ===== State =====
let layerWidths = [64]; // one hidden layer

// ===== Helpers =====
function lamFromSlider(v) { return Math.pow(10, parseFloat(v)); }
function updateLamLabel() { els.lamVal.textContent = `1e${els.lam.value}`; }
function setDepthBadge() { els.depthBadge.textContent = layerWidths.length.toString(); }

// Render one layer row (dark card)
function layerRow(i, width) {
  const id = `layer-${i}`;
  return `
    <div class="rounded-xl border border-slate-700 bg-slate-900 p-3">
      <div class="flex items-center justify-between">
        <div class="text-sm font-semibold text-slate-200">Layer ${i + 1}</div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-slate-400">width:
            <span id="${id}-val" class="font-semibold text-slate-100">${width}</span>
          </span>
          <button data-remove="${i}" type="button"
            class="text-xs px-2 py-1 rounded-lg bg-rose-900/40 text-rose-200 hover:bg-rose-800/60">
            Remove
          </button>
        </div>
      </div>
      <input id="${id}" type="range" min="1" max="2048" value="${width}" class="w-full mt-2 accent-indigo-500">
    </div>
  `;
}

// Re-render the entire layers panel
function renderLayers() {
  els.layersPanel.innerHTML = layerWidths.map((w, i) => layerRow(i, w)).join("");
  setDepthBadge();

  // Bind per-layer slider + remove buttons
  layerWidths.forEach((_, i) => {
    const slider = document.getElementById(`layer-${i}`);
    const val = document.getElementById(`layer-${i}-val`);
    slider.addEventListener("input", () => {
      layerWidths[i] = parseInt(slider.value, 10);
      val.textContent = slider.value;
      scheduleFetch();
    });
  });

  els.layersPanel.querySelectorAll("button[data-remove]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.currentTarget.getAttribute("data-remove"), 10);
      if (layerWidths.length <= 1) return; // keep at least one layer
      layerWidths.splice(idx, 1);
      renderLayers();
      fetchApprox();
    });
  });

  // Enable/disable add button at cap 10
  els.addLayer.disabled = layerWidths.length >= 10;
}

// Presets
function applyPreset(widths) {
  layerWidths = widths.slice(0, 10);
  renderLayers();
  fetchApprox();
}

// ===== Fetch & Plot =====
async function fetchApprox() {
  updateLamLabel();

  const params = new URLSearchParams({
    target: els.target.value,
    activation: els.activation.value,
    layers: layerWidths.join(","),           // backend expects CSV
    lam: lamFromSlider(els.lam.value),
    seed: els.seed.value,
    noise: els.noise.value,
  });

  const res = await fetch(`/approximate?${params.toString()}`);
  const data = await res.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  els.mseVal.textContent = data.mse.toFixed(6);
  const L = data.params.layers.join(" · ");
  els.paramsText.textContent =
    `${data.params.target} • ${data.params.activation} • layers=[${L}] (depth=${data.params.depth}) ` +
    `• λ=${Number(data.params.lambda).toExponential(1)} • seed=${data.params.seed} • noise=${data.params.noise}`;

  const x = data.x;
  const yTrue = data.y_true;
  const yPred = data.y_pred;

  // Plotly dark palette
  const axisCommon = {
    tickcolor: "#94a3b8",
    tickfont: { color: "#cbd5e1" },
    gridcolor: "#334155",
    zerolinecolor: "#475569",
    linecolor: "#475569",
  };
  const layoutMain = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb" },
    margin: { t: 20, r: 10, b: 40, l: 50 },
    xaxis: { title: "x ∈ [0, 1]", ...axisCommon },
    yaxis: { title: "f(x)", ...axisCommon },
    legend: { orientation: "h", x: 0, y: 1.15, font: { color: "#e5e7eb" } },
  };

  const truthTrace = { x, y: yTrue, name: "Target", mode: "lines", line: { width: 3 } };
  const approxTrace = { x, y: yPred, name: "Approximation", mode: "lines", line: { dash: "dot", width: 3 } };
  Plotly.react(els.chart, [truthTrace, approxTrace], layoutMain, { responsive: true });

  // Residuals
  const residuals = yTrue.map((yt, i) => yt - yPred[i]);
  const zeroLine = new Array(residuals.length).fill(0);
  const layoutRes = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb" },
    margin: { t: 10, r: 10, b: 40, l: 50 },
    xaxis: { title: "x", ...axisCommon },
    yaxis: { title: "f(x) − ŷ(x)", ...axisCommon },
    showlegend: false,
  };
  const residualTrace = { x, y: residuals, name: "Residual", mode: "lines", line: { width: 2 } };
  const zeroTrace = { x, y: zeroLine, name: "0", mode: "lines", line: { width: 1, dash: "dot" } };
  Plotly.react(els.residual, [residualTrace, zeroTrace], layoutRes, { responsive: true });
}

// Debounce
let t;
function scheduleFetch() {
  clearTimeout(t);
  t = setTimeout(fetchApprox, 80);
}

// Bindings & bootstrap
function bindGlobal() {
  ["change", "input"].forEach(evt => {
    els.target.addEventListener(evt, scheduleFetch);
    els.activation.addEventListener(evt, scheduleFetch);
    els.lam.addEventListener(evt, scheduleFetch);
    els.seed.addEventListener(evt, scheduleFetch);
    els.noise.addEventListener(evt, scheduleFetch);
  });

  els.addLayer.addEventListener("click", () => {
    if (layerWidths.length >= 10) return;
    layerWidths.push(64);
    renderLayers();
    fetchApprox();
  });

  els.clearLayers.addEventListener("click", () => {
    layerWidths = [64];
    renderLayers();
    fetchApprox();
  });

  els.randomize.addEventListener("click", () => {
    els.seed.value = Math.floor(Math.random() * 100000);
    fetchApprox();
  });

  els.presetWide.addEventListener("click", () => applyPreset([256, 256]));
  els.presetPyramid.addEventListener("click", () => applyPreset([256, 128, 64, 32]));
  els.presetDeepThin.addEventListener("click", () => applyPreset([64, 64, 64, 64, 64, 64]));
}

function bootstrap() {
  updateLamLabel();
  renderLayers();
  bindGlobal();
  fetchApprox();
}

bootstrap();
