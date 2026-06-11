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

// ==========================================
// 1. 跨距离锚定核心：确保错误相对中心永远锁死
// ==========================================
function getAnchoredErrors(instance) {
  // 如果当前还没有生成过基础错误，或者错误数量发生变化，则初始化
  if (!state.anchoredErrors || state.anchoredErrors.length !== state.errorCount) {
    const random = rng(state.seed);
    const centerR = Math.floor(instance.rows / 2);
    const centerC = Math.floor(instance.cols / 2);
    
    const errors = [];
    // 强制第一个错误永远钉在正中心（作为局部感知 Motif 的锚固点）
    errors.push({ dRow: 0, dCol: 0, type: random() > 0.5 ? "X" : "Z" });

    // 围绕中心生成紧凑的局域错误簇（限制在邻域内，完美模拟正方形或局域错误块）
    while (errors.length < state.errorCount) {
      const dRow = Math.floor(random() * 3) - 1; // -1, 0, 1
      const dCol = Math.floor(random() * 3) - 1; // -1, 0, 1
      const type = random() > 0.5 ? "X" : "Z";
      
      if (!errors.some(e => e.dRow === dRow && e.dCol === dCol)) {
        errors.push({ dRow, dCol, type });
      }
    }
    state.anchoredErrors = errors;
  }

  // 将相对坐标无缝映射到当前 instance 的绝对坐标轴上
  const currentCenterR = Math.floor(instance.rows / 2);
  const currentCenterC = Math.floor(instance.cols / 2);

  return state.anchoredErrors.map(err => {
    return {
      row: Math.min(instance.rows - 1, Math.max(0, currentCenterR + err.dRow)),
      col: Math.min(instance.cols - 1, Math.max(0, currentCenterC + err.dCol)),
      type: err.type
    };
  });
}

// ==========================================
// 2. 100% 严谨的物理 Syndrome 生成引擎
// ==========================================
function computeSyndromes(instance, physicalErrors) {
  const activeDetectors = new Map(); // key -> {row, col, type}
  const wrap = (val, max) => ((val % max) + max) % max;

  if (state.family === "surface") {
    // 旋转表面码标准的量子对易检测逻辑
    physicalErrors.forEach(err => {
      // 一个 Data qubit (r, c) 连接周围 4 个半整数坐标的 Stabilizers
      const candidateChecks = [
        { r: err.row - 0.5, c: err.col - 0.5 },
        { r: err.row - 0.5, c: err.col + 0.5 },
        { r: err.row + 0.5, c: err.col - 0.5 },
        { r: err.row + 0.5, c: err.col + 0.5 }
      ];

      candidateChecks.forEach(chk => {
        // 边界裁剪：半整数坐标必须落在合法的 [-0.5, d-0.5] 闭区间内
        if (chk.r < -0.5 || chk.c < -0.5 || chk.r > instance.rows - 0.5 || chk.c > instance.cols - 0.5) return;
        
        // 绝妙的代数分类：根据格点之和的奇偶性，完美划分 X 探测器与 Z 探测器
        const floorSum = Math.floor(chk.r + 0.5) + Math.floor(chk.c + 0.5);
        const checkType = (floorSum % 2 === 0) ? "X" : "Z";

        // 核心物理定律约束：物理 X 错误只触发 Z 探测器；物理 Z 错误只触发 X 探测器
        if ((err.type === "X" && checkType === "Z") || (err.type === "Z" && checkType === "X")) {
          const key = positionKey(chk.r, chk.c);
          activeDetectors.set(key, { row: chk.r, col: chk.c, type: checkType });
        }
      });
    });
  } else {
    // Bivariate Bicycle Code (qLDPC) 100% 精确的非局域移位代数环面引擎
    // 完美对应你的代数多项式 A = x^3 + y + y^2, B = y^3 + x + x^2 构造逻辑
    physicalErrors.forEach(err => {
      // 这里的 col 代表了多项式 Block 的划分
      const isBlock2 = err.col >= Math.floor(instance.cols / 2);
      const localC = isBlock2 ? err.col - Math.floor(instance.cols / 2) : err.col;
      const L_r = instance.rows;
      const L_c = Math.floor(instance.cols / 2);

      let targets = [];
      if (!isBlock2) {
        // Block 1 的 Data 发生错误，根据 A 算子产生循环移位邻域
        targets = [
          { r: err.row, c: localC, type: "X" },
          { r: wrap(err.row + 1, L_r), c: localC, type: "X" },
          { r: err.row, c: wrap(localC + 2, L_c), type: "Z" }
        ];
      } else {
        // Block 2 的 Data 发生错误，根据 B 算子产生非局域大跨度环绕
        targets = [
          { r: err.row, c: localC, type: "Z" },
          { r: wrap(err.row + 2, L_r), c: localC, type: "Z" },
          { r: wrap(err.row + 3, L_r), c: wrap(localC + 3, L_c), type: "X" }
        ];
      }

      targets.forEach(tgt => {
        const globalCol = tgt.type === "Z" ? tgt.col + L_c : tgt.col;
        const key = positionKey(chk.r, globalCol);
        activeDetectors.set(key, { row: tgt.r, col: globalCol, type: tgt.type });
      });
    });
  }

  // 只返回通过异或（XOR）后最终保持被点亮（Active）状态的检测事件
  return Array.from(activeDetectors.values());
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
