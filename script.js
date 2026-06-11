const state = {
  family: "surface",
  instanceId: "surface-7",
  errorCount: 2,
  seed: Math.floor(Math.random() * 10000),
  anchoredErrors: null,
};

const codeFamilies = {
  surface: {
    name: "Rotated surface code",
    instances: [
      { id: "surface-7", n: 49, k: 1, d: 7, rows: 7, cols: 7 },
      { id: "surface-11", n: 121, k: 1, d: 11, rows: 11, cols: 11 },
      { id: "surface-15", n: 225, k: 1, d: 15, rows: 15, cols: 15 },
      { id: "surface-19", n: 361, k: 1, d: 19, rows: 19, cols: 19 },
    ],
  },
  bb: {
    name: "Bivariate bicycle code",
    instances: [
      { id: "bb-72", n: 72, k: 12, d: 6, rows: 6, cols: 6 },
      { id: "bb-144", n: 144, k: 12, d: 12, rows: 6, cols: 12 },
    ],
  },
};

const bbGenerators = {
  // Example BB family used for the visual: A = x^3 + y + y^2, B = y^3 + x + x^2.
  A: [
    { dr: 0, dc: 3 },
    { dr: 1, dc: 0 },
    { dr: 2, dc: 0 },
  ],
  B: [
    { dr: 3, dc: 0 },
    { dr: 0, dc: 1 },
    { dr: 0, dc: 2 },
  ],
};

const els = {
  codeInstance: document.querySelector("#code-instance"),
  errorSlider: document.querySelector("#error-slider"),
  layoutLabel: document.querySelector("#layout-label"),
  codeLattice: document.querySelector("#code-lattice"),
  codeReadout: document.querySelector("#code-readout"),
  nReadout: document.querySelector("#n-readout"),
  kReadout: document.querySelector("#k-readout"),
  dReadout: document.querySelector("#d-readout"),
  syndromeReadout: document.querySelector("#syndrome-readout"),
  resampleBtn: document.querySelector("#resample-btn"),
};

function currentInstance() {
  return codeFamilies[state.family].instances.find((item) => item.id === state.instanceId);
}

function makeSvg(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function rng(seed) {
  let value = seed || 1;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

function wrap(value, size) {
  return ((value % size) + size) % size;
}

function codeLabel(instance) {
  return `[[${instance.n},${instance.k},${instance.d}]]`;
}

function optionLabel(instance, family) {
  if (family === "surface") return `${codeLabel(instance)} d=${instance.d}`;
  return `${codeLabel(instance)} ${instance.rows}x${instance.cols} torus`;
}

function generateAnchoredErrors() {
  const random = rng(state.seed);
  const firstType = random() > 0.5 ? "X" : "Z";
  const errors = [
    {
      dRow: 0,
      dCol: 0,
      block: "L",
      type: firstType,
    },
  ];

  if (state.errorCount > 1) {
    errors.push({
      dRow: firstType === "Z" ? -1 : 0,
      dCol: 1,
      block: "R",
      type: firstType,
    });
  }

  while (errors.length < state.errorCount) {
    const dRow = Math.floor(random() * 3) - 1;
    const dCol = Math.floor(random() * 3) - 1;
    const block = random() > 0.5 ? "L" : "R";
    const type = random() > 0.5 ? "X" : "Z";
    const key = `${dRow},${dCol},${block}`;
    if (!errors.some((item) => `${item.dRow},${item.dCol},${item.block}` === key)) {
      errors.push({ dRow, dCol, block, type });
    }
  }

  state.anchoredErrors = errors;
}

function getPhysicalErrors(instance) {
  if (!state.anchoredErrors || state.anchoredErrors.length !== state.errorCount) {
    generateAnchoredErrors();
  }

  const centerRow = Math.floor(instance.rows / 2);
  const centerCol = Math.floor(instance.cols / 2);

  return state.anchoredErrors.map((error) => {
    if (state.family === "surface") {
      return {
        row: Math.min(instance.rows - 1, Math.max(0, centerRow + error.dRow)),
        col: Math.min(instance.cols - 1, Math.max(0, centerCol + error.dCol)),
        type: error.type,
      };
    }

    return {
      row: wrap(centerRow + error.dRow, instance.rows),
      col: wrap(centerCol + error.dCol, instance.cols),
      block: error.block,
      type: error.type,
    };
  });
}

function toggleEvent(events, key) {
  events.set(key, !events.get(key));
}

function surfaceChecksForError(error) {
  return [
    { row: error.row - 0.5, col: error.col - 0.5 },
    { row: error.row - 0.5, col: error.col + 0.5 },
    { row: error.row + 0.5, col: error.col - 0.5 },
    { row: error.row + 0.5, col: error.col + 0.5 },
  ];
}

function bbConnectionsForError(instance, error) {
  const connections = [];
  const addTargets = (offsets, checkType, layer, sign) => {
    offsets.forEach((offset) => {
      const target = {
        row: wrap(error.row + sign * offset.dr, instance.rows),
        col: wrap(error.col + sign * offset.dc, instance.cols),
        type: checkType,
      };
      connections.push({
        source: error,
        target,
        layer,
        wrap:
          Math.abs(sign * offset.dr) > instance.rows / 2 ||
          Math.abs(sign * offset.dc) > instance.cols / 2 ||
          target.row !== error.row + sign * offset.dr ||
          target.col !== error.col + sign * offset.dc,
      });
    });
  };

  if (error.type === "Z") {
    addTargets(error.block === "L" ? bbGenerators.A : bbGenerators.B, "X", error.block === "L" ? "A" : "B", -1);
  } else {
    addTargets(error.block === "L" ? bbGenerators.B : bbGenerators.A, "Z", error.block === "L" ? "B" : "A", 1);
  }

  return connections;
}

function computeSyndrome(instance, errors) {
  const events = new Map();

  errors.forEach((error) => {
    if (state.family === "surface") {
      surfaceChecksForError(error).forEach((check) => {
        if (
          check.row < -0.5 ||
          check.col < -0.5 ||
          check.row > instance.rows - 0.5 ||
          check.col > instance.cols - 0.5
        ) {
          return;
        }

        const parity = Math.floor(check.row + 0.5) + Math.floor(check.col + 0.5);
        const checkType = parity % 2 === 0 ? "X" : "Z";
        if ((error.type === "X" && checkType === "Z") || (error.type === "Z" && checkType === "X")) {
          toggleEvent(events, `surface_${check.row}_${check.col}`);
        }
      });
      return;
    }

    bbConnectionsForError(instance, error).forEach((connection) => {
      toggleEvent(events, `bb_${connection.target.type}_${connection.target.row}_${connection.target.col}`);
    });
  });

  return Array.from(events.entries())
    .filter(([, active]) => active)
    .map(([key]) => key);
}

function drawText(svg, text, attrs) {
  const element = makeSvg("text", attrs);
  element.textContent = text;
  svg.appendChild(element);
  return element;
}

function drawSurface(svg, instance, errors, activeEvents) {
  const width = 800;
  const height = 500;
  const step = Math.min((width - 170) / Math.max(1, instance.cols - 1), (height - 110) / Math.max(1, instance.rows - 1));
  const offsetX = (width - step * (instance.cols - 1)) / 2;
  const offsetY = (height - step * (instance.rows - 1)) / 2;
  const eventSet = new Set(activeEvents);
  const errorSet = new Map(errors.map((error) => [`${error.row}_${error.col}`, error]));

  for (let row = 0; row < instance.rows; row += 1) {
    svg.appendChild(makeSvg("line", {
      x1: offsetX,
      y1: offsetY + row * step,
      x2: offsetX + (instance.cols - 1) * step,
      y2: offsetY + row * step,
      class: "qec-grid-line",
    }));
  }

  for (let col = 0; col < instance.cols; col += 1) {
    svg.appendChild(makeSvg("line", {
      x1: offsetX + col * step,
      y1: offsetY,
      x2: offsetX + col * step,
      y2: offsetY + (instance.rows - 1) * step,
      class: "qec-grid-line",
    }));
  }

  for (let row = 0; row <= instance.rows; row += 1) {
    for (let col = 0; col <= instance.cols; col += 1) {
      if (
        (row === 0 && col === 0) ||
        (row === instance.rows && col === 0) ||
        (row === 0 && col === instance.cols) ||
        (row === instance.rows && col === instance.cols)
      ) {
        continue;
      }

      const checkRow = row - 0.5;
      const checkCol = col - 0.5;
      const parity = row + col;
      const type = parity % 2 === 0 ? "X" : "Z";
      const active = eventSet.has(`surface_${checkRow}_${checkCol}`);
      svg.appendChild(makeSvg("circle", {
        cx: offsetX + checkCol * step,
        cy: offsetY + checkRow * step,
        r: active ? step * 0.26 : step * 0.12,
        class: [
          "qec-check",
          type === "X" ? "x-check" : "z-check",
          active ? "active" : "",
        ].filter(Boolean).join(" "),
      }));
    }
  }

  for (let row = 0; row < instance.rows; row += 1) {
    for (let col = 0; col < instance.cols; col += 1) {
      const error = errorSet.get(`${row}_${col}`);
      const cx = offsetX + col * step;
      const cy = offsetY + row * step;
      svg.appendChild(makeSvg("circle", {
        cx,
        cy,
        r: error ? step * 0.18 : step * 0.11,
        class: ["qec-data", error ? "error" : ""].filter(Boolean).join(" "),
      }));
      if (error) {
        drawText(svg, error.type, {
          x: cx,
          y: cy + 1,
          class: "qec-error-label",
          "text-anchor": "middle",
          "dominant-baseline": "middle",
        });
      }
    }
  }

  drawText(svg, "Surface code: local plaquette/star checks on a planar lattice", {
    x: 24,
    y: 470,
    class: "qec-svg-caption",
  });
}

function bbPointIndex(item) {
  if (item.kind === "data") {
    return item.block === "L"
      ? { row: 2 * item.row, col: 2 * item.col + 1 }
      : { row: 2 * item.row + 1, col: 2 * item.col };
  }

  return item.type === "X"
    ? { row: 2 * item.row + 1, col: 2 * item.col + 1 }
    : { row: 2 * item.row, col: 2 * item.col };
}

function bbPosition(step, offsetX, offsetY, item) {
  const point = bbPointIndex(item);
  return {
    x: offsetX + point.col * step,
    y: offsetY + point.row * step,
  };
}

function drawBBBoundaryArrows(svg, left, top, width, height) {
  const defs = makeSvg("defs");
  const marker = makeSvg("marker", {
    id: "bb-arrow-head",
    viewBox: "0 0 10 10",
    refX: "8",
    refY: "5",
    markerWidth: "6",
    markerHeight: "6",
    orient: "auto-start-reverse",
  });
  marker.appendChild(makeSvg("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "bb-arrow-head" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  [
    `M ${left + width * 0.70} ${top - 15} L ${left + width * 0.36} ${top - 15}`,
    `M ${left + width * 0.28} ${top + height + 15} L ${left + width * 0.62} ${top + height + 15}`,
    `M ${left - 15} ${top + height * 0.30} L ${left - 15} ${top + height * 0.68}`,
    `M ${left + width + 15} ${top + height * 0.70} L ${left + width + 15} ${top + height * 0.34}`,
  ].forEach((d) => {
    svg.appendChild(makeSvg("path", {
      d,
      class: "bb-boundary-arrow",
      "marker-end": "url(#bb-arrow-head)",
    }));
  });
}

function drawTorusEdge(svg, instance, step, offsetX, offsetY, connection) {
  const source = bbPosition(step, offsetX, offsetY, {
    kind: "data",
    row: connection.source.row,
    col: connection.source.col,
    block: connection.source.block,
  });
  const target = bbPosition(step, offsetX, offsetY, {
    kind: "check",
    row: connection.target.row,
    col: connection.target.col,
    type: connection.target.type,
  });
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const bend = (connection.layer === "A" ? -1 : 1) * Math.min(34, 10 + length * 0.05);
  const midX = (source.x + target.x) / 2 - (dy / length) * bend;
  const midY = (source.y + target.y) / 2 + (dx / length) * bend;
  const path = `M ${source.x.toFixed(1)} ${source.y.toFixed(1)} Q ${midX.toFixed(1)} ${midY.toFixed(1)} ${target.x.toFixed(1)} ${target.y.toFixed(1)}`;

  svg.appendChild(makeSvg("path", {
    d: path,
    class: [
      "bb-edge",
      connection.layer === "A" ? "a-edge" : "b-edge",
      connection.wrap ? "wrap-edge" : "",
    ].filter(Boolean).join(" "),
  }));
}

function drawBB(svg, instance, errors, activeEvents) {
  const width = 800;
  const height = 500;
  const latticeRows = instance.rows * 2;
  const latticeCols = instance.cols * 2;
  const step = Math.min((width - 180) / Math.max(1, latticeCols - 1), (height - 150) / Math.max(1, latticeRows - 1));
  const offsetX = (width - step * (latticeCols - 1)) / 2;
  const offsetY = (height - step * (latticeRows - 1)) / 2 + 2;
  const frameX = offsetX - step * 0.5;
  const frameY = offsetY - step * 0.5;
  const frameW = step * latticeCols;
  const frameH = step * latticeRows;
  const eventSet = new Set(activeEvents);
  const errorSet = new Map(errors.map((error) => [`${error.block}_${error.row}_${error.col}`, error]));
  const connections = errors.flatMap((error) => bbConnectionsForError(instance, error));

  svg.appendChild(makeSvg("rect", {
    x: frameX,
    y: frameY,
    width: frameW,
    height: frameH,
    rx: 0,
    class: "bb-torus-frame",
  }));
  drawBBBoundaryArrows(svg, frameX, frameY, frameW, frameH);

  connections.forEach((connection) => drawTorusEdge(svg, instance, step, offsetX, offsetY, connection));

  for (let row = 0; row < instance.rows; row += 1) {
    for (let col = 0; col < instance.cols; col += 1) {
      ["L", "R"].forEach((block) => {
        const error = errorSet.get(`${block}_${row}_${col}`);
        const pos = bbPosition(step, offsetX, offsetY, { kind: "data", row, col, block });
        svg.appendChild(makeSvg("circle", {
          cx: pos.x,
          cy: pos.y,
          r: error ? step * 0.28 : step * 0.145,
          class: ["qec-data", "bb-data", block === "L" ? "left-register" : "right-register", error ? "error" : ""].filter(Boolean).join(" "),
        }));
        if (error) {
          drawText(svg, error.type, {
            x: pos.x,
            y: pos.y + 1,
            class: "qec-error-label",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
          });
        }
      });

      ["X", "Z"].forEach((type) => {
        const pos = bbPosition(step, offsetX, offsetY, { kind: "check", row, col, type });
        const active = eventSet.has(`bb_${type}_${row}_${col}`);
        svg.appendChild(makeSvg("circle", {
          cx: pos.x,
          cy: pos.y,
          r: active ? step * 0.28 : step * 0.145,
          class: ["qec-check", "bb-check", type === "X" ? "x-check" : "z-check", active ? "active" : ""].filter(Boolean).join(" "),
        }));
      });
    }
  }

  drawText(svg, "BB code: data and checks interleave on one periodic lattice; curves show cyclic shifts", {
    x: 24,
    y: 470,
    class: "qec-svg-caption",
  });
}

function drawLattice(svg, instance, errors, activeEvents) {
  svg.innerHTML = "";
  if (state.family === "surface") {
    drawSurface(svg, instance, errors, activeEvents);
  } else {
    drawBB(svg, instance, errors, activeEvents);
  }
}

function populateInstances() {
  const family = codeFamilies[state.family];
  els.codeInstance.innerHTML = "";
  family.instances.forEach((instance) => {
    const option = document.createElement("option");
    option.value = instance.id;
    option.textContent = optionLabel(instance, state.family);
    els.codeInstance.appendChild(option);
  });
  state.instanceId = family.instances[0].id;
  els.codeInstance.value = state.instanceId;
}

function updateUI() {
  const instance = currentInstance();
  const errors = getPhysicalErrors(instance);
  const events = computeSyndrome(instance, errors);

  els.layoutLabel.textContent = `${codeLabel(instance)} Target`;
  els.codeReadout.textContent = codeLabel(instance);
  els.nReadout.textContent = `n = ${instance.n}`;
  els.kReadout.textContent = `k = ${instance.k}`;
  els.dReadout.textContent = `d = ${instance.d}`;
  els.syndromeReadout.textContent = `${events.length}`;
  document.querySelectorAll(".bb-only").forEach((item) => {
    item.hidden = state.family !== "bb";
  });

  drawLattice(els.codeLattice, instance, errors, events);
}

document.querySelectorAll("[data-family]").forEach((button) => {
  button.addEventListener("click", () => {
    state.family = button.dataset.family;
    document.querySelectorAll("[data-family]").forEach((item) => item.classList.toggle("active", item === button));
    populateInstances();
    state.anchoredErrors = null;
    updateUI();
  });
});

els.codeInstance.addEventListener("change", (event) => {
  state.instanceId = event.target.value;
  updateUI();
});

els.errorSlider.addEventListener("input", (event) => {
  state.errorCount = Number(event.target.value);
  state.anchoredErrors = null;
  updateUI();
});

els.resampleBtn.addEventListener("click", () => {
  state.seed = Math.floor(Math.random() * 10000);
  state.anchoredErrors = null;
  updateUI();
});

populateInstances();
updateUI();

// =====================================================
// AI Assistant Widget
// =====================================================
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
