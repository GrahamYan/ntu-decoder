const state = {
  family: "surface",
  instanceId: "surface-7",
  view: "syndrome",
  errorCount: 3,
  seed: 11,
  selected: { row: 3, col: 3 },
};

const codeFamilies = {
  surface: {
    name: "Rotated surface code",
    instances: [
      { id: "surface-7", n: 49, k: 1, d: 7, rows: 7, cols: 7, rounds: 7 },
      { id: "surface-11", n: 121, k: 1, d: 11, rows: 11, cols: 11, rounds: 11 },
      { id: "surface-19", n: 361, k: 1, d: 19, rows: 19, cols: 19, rounds: 19 },
      { id: "surface-25", n: 625, k: 1, d: 25, rows: 25, cols: 25, rounds: 25 },
    ],
    motif: "planar local stabilizer neighborhood",
  },
  bb: {
    name: "Bivariate bicycle code",
    instances: [
      { id: "bb-72", n: 72, k: 12, d: 6, rows: 6, cols: 12, rounds: 6 },
      { id: "bb-144", n: 144, k: 12, d: 12, rows: 12, cols: 12, rounds: 12 },
    ],
    motif: "cyclic polynomial Tanner neighborhood",
  },
};

const els = {
  codeInstance: document.querySelector("#code-instance"),
  errorSlider: document.querySelector("#error-slider"),
  layoutLabel: document.querySelector("#layout-label"),
  detectorLabel: document.querySelector("#detector-label"),
  viewLabel: document.querySelector("#view-label"),
  codeLattice: document.querySelector("#code-lattice"),
  detectorLattice: document.querySelector("#detector-lattice"),
  perceptionMap: document.querySelector("#perception-map"),
  codeReadout: document.querySelector("#code-readout"),
  nReadout: document.querySelector("#n-readout"),
  kReadout: document.querySelector("#k-readout"),
  dReadout: document.querySelector("#d-readout"),
  inputReadout: document.querySelector("#input-readout"),
  syndromeReadout: document.querySelector("#syndrome-readout"),
  resampleBtn: document.querySelector("#resample-btn"),
  centerBtn: document.querySelector("#center-btn"),
  copyBibtex: document.querySelector("#copy-bibtex"),
  bibtex: document.querySelector("#bibtex"),
};

function labelFor(instance) {
  return `[[${instance.n},${instance.k},${instance.d}]]`;
}

function currentInstance() {
  return codeFamilies[state.family].instances.find((item) => item.id === state.instanceId);
}

function centerSelection(instance) {
  state.selected = {
    row: Math.floor(instance.rows / 2),
    col: Math.floor(instance.cols / 2),
  };
}

function makeSvg(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function rng(seed) {
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function positionKey(row, col) {
  return `${row},${col}`;
}

function sampleErrors(instance) {
  const random = rng(state.seed + instance.n * 13 + state.errorCount * 97);
  const errors = new Map();
  const selectedKey = positionKey(
    Math.min(instance.rows - 1, Math.max(0, state.selected.row)),
    Math.min(instance.cols - 1, Math.max(0, state.selected.col)),
  );
  errors.set(selectedKey, {
    row: Math.min(instance.rows - 1, Math.max(0, state.selected.row)),
    col: Math.min(instance.cols - 1, Math.max(0, state.selected.col)),
  });

  const target = Math.min(state.errorCount, instance.rows * instance.cols);
  while (errors.size < target) {
    const row = Math.floor(random() * instance.rows);
    const col = Math.floor(random() * instance.cols);
    errors.set(positionKey(row, col), { row, col });
  }

  return Array.from(errors.values());
}

function detectorDimensions(instance) {
  if (state.family === "surface") {
    return { rows: instance.rows + 1, cols: instance.cols + 1 };
  }
  return { rows: instance.rows, cols: instance.cols };
}

function detectorOffsets(instance, row, col) {
  if (state.family === "surface") {
    return [
      { row, col },
      { row: row + 1, col },
      { row, col: col + 1 },
      { row: row + 1, col: col + 1 },
    ];
  }

  const wrap = (value, size) => ((value % size) + size) % size;
  const offsets = [
    { dr: 0, dc: 0 },
    { dr: 1, dc: 2 },
    { dr: 2, dc: 1 },
    { dr: 3, dc: 3 },
  ];
  return offsets.map(({ dr, dc }) => ({
    row: wrap(row + dr, instance.rows),
    col: wrap(col + dc, instance.cols),
  }));
}

function detectorEvents(instance, errors) {
  const dims = detectorDimensions(instance);
  const parity = new Map();

  errors.forEach((error) => {
    detectorOffsets(instance, error.row, error.col).forEach((detector) => {
      if (
        detector.row < 0 ||
        detector.col < 0 ||
        detector.row >= dims.rows ||
        detector.col >= dims.cols
      ) {
        return;
      }
      const key = positionKey(detector.row, detector.col);
      parity.set(key, !parity.get(key));
    });
  });

  return Array.from(parity.entries())
    .filter(([, active]) => active)
    .map(([key]) => {
      const [row, col] = key.split(",").map(Number);
      return { row, col };
    });
}

function geometry(rows, cols, width = 520, height = 360) {
  const pad = 42;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const step = Math.min(plotW / Math.max(1, cols - 1), plotH / Math.max(1, rows - 1));
  const offsetX = (width - step * Math.max(1, cols - 1)) / 2;
  const offsetY = (height - step * Math.max(1, rows - 1)) / 2;
  return {
    step,
    offsetX,
    offsetY,
    radius: Math.max(3.1, Math.min(10, step * 0.22)),
    x: (col) => offsetX + col * step,
    y: (row) => offsetY + row * step,
  };
}

function localDataMotif(instance) {
  const { row, col } = state.selected;
  const raw = [
    { row, col },
    { row: row - 1, col },
    { row: row + 1, col },
    { row, col: col - 1 },
    { row, col: col + 1 },
  ];

  if (state.family === "bb") {
    const wrap = (value, size) => ((value % size) + size) % size;
    return raw.map((item) => ({
      row: wrap(item.row, instance.rows),
      col: wrap(item.col, instance.cols),
    }));
  }

  return raw.filter(
    (item) => item.row >= 0 && item.col >= 0 && item.row < instance.rows && item.col < instance.cols,
  );
}

function drawGridLines(svg, rows, cols, geom, className = "grid-line") {
  for (let row = 0; row < rows; row += 1) {
    svg.appendChild(
      makeSvg("line", {
        x1: geom.x(0),
        y1: geom.y(row),
        x2: geom.x(cols - 1),
        y2: geom.y(row),
        class: className,
      }),
    );
  }

  for (let col = 0; col < cols; col += 1) {
    svg.appendChild(
      makeSvg("line", {
        x1: geom.x(col),
        y1: geom.y(0),
        x2: geom.x(col),
        y2: geom.y(rows - 1),
        class: className,
      }),
    );
  }
}

function drawDataLattice(svg, instance, errors) {
  svg.innerHTML = "";
  const geom = geometry(instance.rows, instance.cols);
  const errorKeys = new Set(errors.map((item) => positionKey(item.row, item.col)));
  const motifKeys = new Set(localDataMotif(instance).map((item) => positionKey(item.row, item.col)));

  if (state.family === "bb") {
    svg.appendChild(
      makeSvg("rect", {
        x: geom.x(0) - 18,
        y: geom.y(0) - 18,
        width: geom.step * (instance.cols - 1) + 36,
        height: geom.step * (instance.rows - 1) + 36,
        rx: 20,
        class: "bb-plane",
      }),
    );
  }

  drawGridLines(svg, instance.rows, instance.cols, geom);

  motifKeys.forEach((key) => {
    const [row, col] = key.split(",").map(Number);
    svg.appendChild(
      makeSvg("rect", {
        x: geom.x(col) - geom.step * 0.36,
        y: geom.y(row) - geom.step * 0.36,
        width: geom.step * 0.72,
        height: geom.step * 0.72,
        rx: 6,
        class: "motif",
      }),
    );
  });

  for (let row = 0; row < instance.rows; row += 1) {
    for (let col = 0; col < instance.cols; col += 1) {
      const key = positionKey(row, col);
      const point = makeSvg("circle", {
        cx: geom.x(col),
        cy: geom.y(row),
        r: motifKeys.has(key) ? geom.radius * 1.22 : geom.radius,
        class: [
          "qubit",
          errorKeys.has(key) ? "error" : "",
          state.selected.row === row && state.selected.col === col ? "selected" : "",
        ]
          .filter(Boolean)
          .join(" "),
      });
      point.addEventListener("click", () => {
        state.selected = { row, col };
        render();
      });
      svg.appendChild(point);
    }
  }

  const label = makeSvg("text", { x: 18, y: 330, class: "chart-label" });
  label.textContent = `${labelFor(instance)} data-qubit layout, n = ${instance.n}`;
  svg.appendChild(label);
}

function drawDetectorLattice(svg, instance, events) {
  svg.innerHTML = "";
  const dims = detectorDimensions(instance);
  const geom = geometry(dims.rows, dims.cols);
  const eventKeys = new Set(events.map((item) => positionKey(item.row, item.col)));
  const localKeys = new Set(
    detectorOffsets(instance, state.selected.row, state.selected.col)
      .filter((item) => item.row >= 0 && item.col >= 0 && item.row < dims.rows && item.col < dims.cols)
      .map((item) => positionKey(item.row, item.col)),
  );

  if (state.family === "bb") {
    svg.appendChild(
      makeSvg("rect", {
        x: geom.x(0) - 18,
        y: geom.y(0) - 18,
        width: geom.step * (dims.cols - 1) + 36,
        height: geom.step * (dims.rows - 1) + 36,
        rx: 20,
        class: "bb-plane",
      }),
    );
  }

  drawGridLines(svg, dims.rows, dims.cols, geom);

  localKeys.forEach((key) => {
    const [row, col] = key.split(",").map(Number);
    svg.appendChild(
      makeSvg("rect", {
        x: geom.x(col) - geom.step * 0.34,
        y: geom.y(row) - geom.step * 0.34,
        width: geom.step * 0.68,
        height: geom.step * 0.68,
        rx: 6,
        class: "detector-window",
      }),
    );
  });

  for (let row = 0; row < dims.rows; row += 1) {
    for (let col = 0; col < dims.cols; col += 1) {
      const key = positionKey(row, col);
      svg.appendChild(
        makeSvg("circle", {
          cx: geom.x(col),
          cy: geom.y(row),
          r: eventKeys.has(key) ? geom.radius * 1.28 : geom.radius * 0.86,
          class: ["check", eventKeys.has(key) ? "event" : "", localKeys.has(key) ? "local" : ""]
            .filter(Boolean)
            .join(" "),
        }),
      );
    }
  }

  const label = makeSvg("text", { x: 18, y: 330, class: "chart-label" });
  label.textContent =
    state.family === "surface"
      ? "Surface checks: adjacent plaquette/star events toggle by parity"
      : "BB checks: cyclic Tanner-neighborhood events toggle by parity";
  svg.appendChild(label);
}

function drawMiniGrid(svg, x0, y0, size, activeCells = [], label = "") {
  const step = size / 4;
  for (let i = 0; i < 5; i += 1) {
    svg.appendChild(makeSvg("line", { x1: x0, y1: y0 + i * step, x2: x0 + size, y2: y0 + i * step, class: "mini-grid-line" }));
    svg.appendChild(makeSvg("line", { x1: x0 + i * step, y1: y0, x2: x0 + i * step, y2: y0 + size, class: "mini-grid-line" }));
  }
  activeCells.forEach(([row, col]) => {
    svg.appendChild(
      makeSvg("rect", {
        x: x0 + col * step + step * 0.16,
        y: y0 + row * step + step * 0.16,
        width: step * 0.68,
        height: step * 0.68,
        rx: 5,
        class: "mini-motif",
      }),
    );
  });
  const text = makeSvg("text", { x: x0, y: y0 + size + 28, class: "perception-label" });
  text.textContent = label;
  svg.appendChild(text);
}

function drawArrow(svg, x1, y1, x2, y2) {
  svg.appendChild(makeSvg("line", { x1, y1, x2, y2, class: "perception-arrow" }));
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = [
    [x2, y2],
    [x2 - 10 * Math.cos(angle - 0.45), y2 - 10 * Math.sin(angle - 0.45)],
    [x2 - 10 * Math.cos(angle + 0.45), y2 - 10 * Math.sin(angle + 0.45)],
  ]
    .map((point) => point.join(","))
    .join(" ");
  svg.appendChild(makeSvg("polygon", { points: head, class: "perception-arrow-head" }));
}

function drawNode(svg, x, y, w, h, title, body, className = "") {
  svg.appendChild(makeSvg("rect", { x, y, width: w, height: h, rx: 8, class: `perception-node ${className}` }));
  const titleNode = makeSvg("text", { x: x + 16, y: y + 28, class: "perception-title" });
  titleNode.textContent = title;
  svg.appendChild(titleNode);
  const bodyNode = makeSvg("text", { x: x + 16, y: y + 54, class: "perception-body" });
  bodyNode.textContent = body;
  svg.appendChild(bodyNode);
}

function drawSyndromeStory(svg, instance, errors, events) {
  svg.innerHTML = "";
  drawNode(svg, 30, 44, 230, 92, "Data-qubit error", `${errors.length} physical error(s) sampled`, "hot");
  drawNode(svg, 392, 44, 240, 92, "Detector events", `${events.length} parity changes observed`, "warm");
  drawNode(svg, 760, 28, 240, 64, "Physical route", "predict errored data qubits");
  drawNode(svg, 760, 124, 240, 64, "Logical route", `predict [0,1]^${instance.k}`);
  drawArrow(svg, 260, 90, 392, 90);
  drawArrow(svg, 632, 82, 760, 60);
  drawArrow(svg, 632, 98, 760, 156);

  const note = makeSvg("text", { x: 30, y: 245, class: "perception-note" });
  note.textContent =
    "A decoder sees detector events, not the hidden error directly. The model must infer either a correction or the final logical status.";
  svg.appendChild(note);
}

function drawInvarianceStory(svg, instance) {
  svg.innerHTML = "";
  const motif = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
  ];
  drawMiniGrid(svg, 55, 54, 130, motif, "spatial shift");
  drawMiniGrid(svg, 330, 54, 130, motif, "temporal repeat");
  drawMiniGrid(svg, 610, 38, 170, motif, "scale expansion");
  drawArrow(svg, 205, 118, 305, 118);
  drawArrow(svg, 480, 118, 585, 118);
  drawNode(svg, 820, 72, 190, 96, "Same motif M", codeFamilies[state.family].motif);

  const equation = makeSvg("text", { x: 55, y: 265, class: "perception-equation" });
  equation.textContent = `N(v) = v M(x,y,t), preserved for ${labelFor(instance)}`;
  svg.appendChild(equation);
}

function drawNetworkStory(svg, instance) {
  svg.innerHTML = "";
  drawNode(svg, 28, 72, 170, 86, "Stem", "local detector roles");
  drawNode(svg, 248, 72, 180, 86, "Spatial transformer", "within each round");
  drawNode(svg, 478, 72, 180, 86, "Temporal recurrent", "along t axis");
  drawNode(svg, 708, 72, 180, 86, "Cross attention", "global readout");
  drawNode(svg, 920, 72, 94, 86, "Output", `[0,1]^${instance.k}`, "cool");
  drawArrow(svg, 198, 115, 248, 115);
  drawArrow(svg, 428, 115, 478, 115);
  drawArrow(svg, 658, 115, 708, 115);
  drawArrow(svg, 888, 115, 920, 115);

  const note = makeSvg("text", { x: 28, y: 250, class: "perception-note" });
  note.textContent =
    "Figure 1b intuition: map code locality into network locality, then aggregate across time and logical readout tokens.";
  svg.appendChild(note);
}

function drawPerception(instance, errors, events) {
  if (state.view === "syndrome") {
    els.viewLabel.textContent = "Errors -> detectors";
    drawSyndromeStory(els.perceptionMap, instance, errors, events);
    return;
  }

  if (state.view === "invariance") {
    els.viewLabel.textContent = "Spatial, temporal, scale invariance";
    drawInvarianceStory(els.perceptionMap, instance);
    return;
  }

  els.viewLabel.textContent = "Local perception -> logical readout";
  drawNetworkStory(els.perceptionMap, instance);
}

function populateInstances() {
  const family = codeFamilies[state.family];
  els.codeInstance.innerHTML = "";
  family.instances.forEach((instance) => {
    const option = document.createElement("option");
    option.value = instance.id;
    option.textContent = `${labelFor(instance)} ${family.name}`;
    els.codeInstance.appendChild(option);
  });

  state.instanceId = family.instances[0].id;
  els.codeInstance.value = state.instanceId;
  centerSelection(currentInstance());
}

function updateReadouts(instance, events) {
  const checks = instance.n - instance.k;
  const label = labelFor(instance);
  els.layoutLabel.textContent = label;
  els.detectorLabel.textContent = `${events.length} events`;
  els.codeReadout.textContent = label;
  els.nReadout.textContent = `n = ${instance.n}`;
  els.kReadout.textContent = `k = ${instance.k}`;
  els.dReadout.textContent = `d = ${instance.d}`;
  els.inputReadout.textContent = `${instance.rounds} x ${checks}`;
  els.syndromeReadout.textContent = `${events.length} detector events`;
}

function render() {
  const instance = currentInstance();
  state.selected.row = Math.min(instance.rows - 1, Math.max(0, state.selected.row));
  state.selected.col = Math.min(instance.cols - 1, Math.max(0, state.selected.col));

  const errors = sampleErrors(instance);
  const events = detectorEvents(instance, errors);

  drawDataLattice(els.codeLattice, instance, errors);
  drawDetectorLattice(els.detectorLattice, instance, events);
  drawPerception(instance, errors, events);
  updateReadouts(instance, events);
}

document.querySelectorAll("[data-family]").forEach((button) => {
  button.addEventListener("click", () => {
    state.family = button.dataset.family;
    document.querySelectorAll("[data-family]").forEach((item) => item.classList.toggle("active", item === button));
    populateInstances();
    render();
  });
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    document.querySelectorAll("[data-view]").forEach((item) => item.classList.toggle("active", item === button));
    render();
  });
});

els.codeInstance.addEventListener("change", (event) => {
  state.instanceId = event.target.value;
  centerSelection(currentInstance());
  render();
});

els.errorSlider.addEventListener("input", (event) => {
  state.errorCount = Number(event.target.value);
  render();
});

els.resampleBtn.addEventListener("click", () => {
  state.seed += 17;
  render();
});

els.centerBtn.addEventListener("click", () => {
  centerSelection(currentInstance());
  render();
});

els.copyBibtex.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(els.bibtex.textContent);
    els.copyBibtex.textContent = "Copied";
    window.setTimeout(() => {
      els.copyBibtex.textContent = "Copy BibTeX";
    }, 1200);
  } catch {
    els.copyBibtex.textContent = "Select text";
  }
});

populateInstances();
render();

const assistantWidget = document.querySelector(".assistant-widget");
const assistantToggle = document.querySelector("#assistant-toggle");
const assistantHeroButton = document.querySelector("#open-assistant-hero");
const assistantClose = document.querySelector("#assistant-close");
const assistantForm = document.querySelector("#assistant-form");
const assistantInput = document.querySelector("#assistant-input");
const assistantMessages = document.querySelector("#assistant-messages");
const assistantStatus = document.querySelector("#assistant-status");
const assistantEndpoint = assistantWidget?.dataset.endpoint || "";
const assistantHistory = [];
const assistantConfigured =
  assistantEndpoint &&
  !assistantEndpoint.includes("YOUR_WORKERS_SUBDOMAIN") &&
  /^https:\/\/.+\/chat$/.test(assistantEndpoint);

function setAssistantOpen(open) {
  if (!assistantWidget) return;
  assistantWidget.classList.toggle("open", open);
  assistantToggle?.setAttribute("aria-expanded", String(open));
  if (open) {
    window.setTimeout(() => assistantInput?.focus(), 80);
  }
}

function setAssistantStatus(text, type = "") {
  if (!assistantStatus) return;
  assistantStatus.textContent = text;
  assistantStatus.className = ["assistant-status", type].filter(Boolean).join(" ");
}

function addAssistantMessage(text, role) {
  const message = document.createElement("div");
  message.className = `assistant-message assistant-message-${role === "user" ? "user" : "bot"}`;
  message.textContent = text;
  assistantMessages.appendChild(message);
  assistantMessages.scrollTop = assistantMessages.scrollHeight;
  return message;
}

if (assistantWidget) {
  if (assistantConfigured) {
    setAssistantStatus("Connected to the paper and code assistant backend.", "ready");
  } else {
    setAssistantStatus(
      "Backend not connected yet. Deploy the Cloudflare Worker, then replace the endpoint in index.html.",
      "error",
    );
    assistantInput.disabled = true;
    assistantForm.querySelector("button").disabled = true;
  }
}

assistantToggle?.addEventListener("click", () => {
  setAssistantOpen(!assistantWidget.classList.contains("open"));
});

assistantHeroButton?.addEventListener("click", () => {
  setAssistantOpen(true);
});

assistantClose?.addEventListener("click", () => {
  setAssistantOpen(false);
});

assistantForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!assistantConfigured) return;

  const question = assistantInput.value.trim();
  if (!question) return;

  assistantInput.value = "";
  assistantInput.disabled = true;
  assistantForm.querySelector("button").disabled = true;
  addAssistantMessage(question, "user");
  const pending = addAssistantMessage("Thinking...", "bot");
  setAssistantStatus("Retrieving relevant paper context...", "ready");

  try {
    const response = await fetch(assistantEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: question,
        history: assistantHistory.slice(-6),
      }),
    });

    if (!response.ok) {
      throw new Error(`Assistant request failed (${response.status})`);
    }

    const data = await response.json();
    const answer = data.answer || "The assistant did not return an answer.";
    pending.textContent = answer;
    assistantHistory.push({ role: "user", content: question });
    assistantHistory.push({ role: "assistant", content: answer });
    setAssistantStatus(
      data.sources?.length ? `Sources: ${data.sources.join(", ")}` : "Answered from paper and code context.",
      "ready",
    );
  } catch (error) {
    pending.textContent =
      "I could not reach the assistant backend. Check the Worker URL and Cloudflare deployment.";
    setAssistantStatus(error.message, "error");
  } finally {
    assistantInput.disabled = false;
    assistantForm.querySelector("button").disabled = false;
    assistantInput.focus();
  }
});
