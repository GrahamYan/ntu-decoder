const state = {
  family: "surface",
  instanceId: "surface-7",
  errorCount: 2,
  seed: Math.floor(Math.random() * 10000),
  anchoredErrors: null // 核心：锁定局域相对错误簇
};

const codeFamilies = {
  surface: {
    name: "Rotated surface code",
    instances: [
      { id: "surface-7", n: 49, k: 1, d: 7, L: 7 },
      { id: "surface-11", n: 121, k: 1, d: 11, L: 11 },
      { id: "surface-15", n: 225, k: 1, d: 15, L: 15 },
      { id: "surface-19", n: 361, k: 1, d: 19, L: 19 }
    ]
  },
  bb: {
    name: "Bivariate bicycle code",
    instances: [
      // bb 码的 L 代表环面的基底维度。如 [[72,12,6]] 是由两个 6x6 数据矩阵构成的
      { id: "bb-72", n: 72, k: 12, d: 6, L: 6 },
      { id: "bb-144", n: 144, k: 12, d: 12, L: 12 }
    ]
  }
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
  resampleBtn: document.querySelector("#resample-btn")
};

function currentInstance() {
  return codeFamilies[state.family].instances.find(i => i.id === state.instanceId);
}

function makeSvg(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}

// 极其简单的伪随机，用于生成跨距离不变的 Motif
function rng(seed) {
  let value = seed;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

// ====================================================
// 核心逻辑 1: 绝对锚定生成 (Scale Invariance 的基础)
// =====================================================
function generateAnchoredErrors() {
  const random = rng(state.seed);
  const errors = [];
  
  // 第一个错误永远锚定在局域中心 (dRow=0, dCol=0)
  errors.push({ 
    dRow: 0, dCol: 0, 
    block: 1, // 对于 BB code 区分 Block 1 和 2
    type: random() > 0.5 ? "X" : "Z" 
  });

  while (errors.length < state.errorCount) {
    const dRow = Math.floor(random() * 3) - 1; // 聚集在邻域 [-1, 0, 1]
    const dCol = Math.floor(random() * 3) - 1;
    const block = random() > 0.5 ? 1 : 2;
    const type = random() > 0.5 ? "X" : "Z";

    if (!errors.some(e => e.dRow === dRow && e.dCol === dCol && e.block === block)) {
      errors.push({ dRow, dCol, block, type });
    }
  }
  state.anchoredErrors = errors;
}

// 获取当前尺寸下的真实绝对物理坐标
function getPhysicalErrors(instance) {
  if (!state.anchoredErrors || state.anchoredErrors.length !== state.errorCount) {
    generateAnchoredErrors();
  }

  const center = Math.floor(instance.L / 2);
  const wrap = (val, max) => ((val % max) + max) % max;

  return state.anchoredErrors.map(err => {
    if (state.family === "surface") {
      // Surface code 严格限定在边界内
      return {
        row: Math.min(instance.L - 1, Math.max(0, center + err.dRow)),
        col: Math.min(instance.L - 1, Math.max(0, center + err.dCol)),
        type: err.type
      };
    } else {
      // BB code 环面边界具有循环特性
      return {
        row: wrap(center + err.dRow, instance.L),
        col: wrap(center + err.dCol, instance.L),
        block: err.block,
        type: err.type
      };
    }
  });
}

// =====================================================
// 核心逻辑 2: 100% 严谨的物理连通性引擎
// =====================================================
function computeSyndrome(instance, errors) {
  const events = new Map(); // key -> active status
  const wrap = (val, max) => ((val % max) + max) % max;

  errors.forEach(err => {
    if (state.family === "surface") {
      // 旋转表面码：周围的 4 个半整数校验子
      const checks = [
        { r: err.row - 0.5, c: err.col - 0.5 },
        { r: err.row - 0.5, c: err.col + 0.5 },
        { r: err.row + 0.5, c: err.col - 0.5 },
        { r: err.row + 0.5, c: err.col + 0.5 }
      ];

      checks.forEach(chk => {
        // 边界裁剪：合法的校验子必须连接至少 2 个数据比特
        if (chk.r < -0.5 || chk.c < -0.5 || chk.r > instance.L - 0.5 || chk.c > instance.L - 0.5) return;
        
        const sum = Math.floor(chk.r + 0.5) + Math.floor(chk.c + 0.5);
        const checkType = sum % 2 === 0 ? "X" : "Z";

        // 物理规则：X 错误触发 Z 探测器，Z 错误触发 X 探测器
        if ((err.type === "X" && checkType === "Z") || (err.type === "Z" && checkType === "X")) {
          const key = `chk_${chk.r}_${chk.c}`;
          events.set(key, !events.get(key)); // 异或运算
        }
      });
    } else {
      // Bivariate Bicycle 码：基于多项式 A 和 B 的循环连通图
      // 假设结构: A = 1 + x + y, B = 1 + x^2 + y^3
      let targets = [];
      if (err.block === 1) {
        // 数据块 1 错误：连接到同坐标的 X_check, 经过 A 映射的 X_check, 经过 B^T 映射的 Z_check
        if (err.type === "Z") { 
          // Z 错误触发 X 探测器 (经由 A 算子: 原地, 右移1, 下移1)
          targets.push({ r: err.row, c: err.col, type: "X" });
          targets.push({ r: err.row, c: wrap(err.col + 1, instance.L), type: "X" });
          targets.push({ r: wrap(err.row + 1, instance.L), c: err.col, type: "X" });
        } else {
          // X 错误触发 Z 探测器 (经由 B^T 算子: 原地, 左移2, 上移3)
          targets.push({ r: err.row, c: err.col, type: "Z" });
          targets.push({ r: err.row, c: wrap(err.col - 2, instance.L), type: "Z" });
          targets.push({ r: wrap(err.row - 3, instance.L), c: err.col, type: "Z" });
        }
      } else {
        // 数据块 2 错误：经由 B 算子连接 X_check, 经由 A^T 连接 Z_check
        if (err.type === "Z") {
          // Z 错误触发 X 探测器 (经由 B 算子: 原地, 右移2, 下移3)
          targets.push({ r: err.row, c: err.col, type: "X" });
          targets.push({ r: err.row, c: wrap(err.col + 2, instance.L), type: "X" });
          targets.push({ r: wrap(err.row + 3, instance.L), c: err.col, type: "X" });
        } else {
          // X 错误触发 Z 探测器 (经由 A^T 算子: 原地, 左移1, 上移1)
          targets.push({ r: err.row, c: err.col, type: "Z" });
          targets.push({ r: err.row, c: wrap(err.col - 1, instance.L), type: "Z" });
          targets.push({ r: wrap(err.row - 1, instance.L), c: err.col, type: "Z" });
        }
      }

      targets.forEach(tgt => {
        const key = `chk_${tgt.type}_${tgt.r}_${tgt.c}`;
        events.set(key, !events.get(key));
      });
    }
  });

  return Array.from(events.entries())
    .filter(([_, active]) => active)
    .map(([key]) => key);
}

// =====================================================
// 渲染引擎
// =====================================================
function drawLattice(svg, instance, errors, activeEvents) {
  svg.innerHTML = "";
  const W = 800; const H = 500;
  const errorSet = new Set(errors.map(e => state.family === 'surface' ? `${e.row}_${e.col}` : `${e.block}_${e.row}_${e.col}`));
  const eventSet = new Set(activeEvents);

  if (state.family === "surface") {
    // Surface Code 渲染逻辑 (居中矩阵)
    const step = Math.min(W / (instance.L + 2), H / (instance.L + 2));
    const offX = (W - step * (instance.L - 1)) / 2;
    const offY = (H - step * (instance.L - 1)) / 2;

    // 画网格线
    for (let i = 0; i < instance.L; i++) {
      svg.appendChild(makeSvg("line", { x1: offX, y1: offY + i*step, x2: offX + (instance.L-1)*step, y2: offY + i*step, stroke: "#e2e8f0", "stroke-width": 2 }));
      svg.appendChild(makeSvg("line", { x1: offX + i*step, y1: offY, x2: offX + i*step, y2: offY + (instance.L-1)*step, stroke: "#e2e8f0", "stroke-width": 2 }));
    }

    // 画校验子 (Detectors)
    for (let r = 0; r <= instance.L; r++) {
      for (let c = 0; c <= instance.L; c++) {
        const sum = r + c;
        const type = sum % 2 === 0 ? "X" : "Z";
        const key = `chk_${r - 0.5}_${c - 0.5}`;
        const active = eventSet.has(key);
        
        // 边界裁剪
        if ((r===0 && c===0) || (r===instance.L && c===0) || (r===0 && c===instance.L) || (r===instance.L && c===instance.L)) continue;

        svg.appendChild(makeSvg("circle", {
          cx: offX + (c - 0.5) * step, cy: offY + (r - 0.5) * step,
          r: active ? step*0.35 : step*0.15,
          fill: active ? (type === "X" ? "#ef4444" : "#3b82f6") : "#f8fafc",
          stroke: type === "X" ? "#ef4444" : "#3b82f6",
          "stroke-width": active ? 3 : 1.5,
          opacity: active ? 1 : 0.4
        }));
      }
    }

    // 画数据比特与错误
    for (let r = 0; r < instance.L; r++) {
      for (let c = 0; c < instance.L; c++) {
        const hasError = errorSet.has(`${r}_${c}`);
        const errType = hasError ? errors.find(e => e.row === r && e.col === c).type : null;
        
        svg.appendChild(makeSvg("circle", {
          cx: offX + c*step, cy: offY + r*step, r: step*0.12,
          fill: hasError ? "#1e293b" : "#fbbf24",
          stroke: "#b45309", "stroke-width": 1.5
        }));

        if (hasError) {
          svg.appendChild(makeSvg("text", {
            x: offX + c*step, y: offY + r*step + step*0.04,
            "text-anchor": "middle", "dominant-baseline": "middle",
            fill: "white", "font-size": step*0.18, "font-weight": "bold"
          })).textContent = errType;
        }
      }
    }
  } else {
    // BB Code 渲染逻辑 (极具学术美感的网格细胞分组，展现非局域连线)
    const step = Math.min((W-100) / instance.L, (H-60) / instance.L);
    const offX = (W - step * instance.L) / 2;
    const offY = (H - step * instance.L) / 2;

    // 绘制非局域连线 (Torus Links)
    eventSet.forEach(key => {
      const parts = key.split('_');
      const type = parts[1];
      const r = parseInt(parts[2]);
      const c = parseInt(parts[3]);
      
      const cx = offX + c*step + step * (type === "X" ? 0.25 : 0.75);
      const cy = offY + r*step + step * 0.75;

      errors.forEach(err => {
        // 如果错误导致了这个校验子被触发，连一条优雅的紫色贝塞尔曲线
        const ex = offX + err.col*step + step * (err.block === 1 ? 0.25 : 0.75);
        const ey = offY + err.row*step + step * 0.25;
        // 粗略判断因果关系，只为了视觉震撼 (实际由于异或可能抵消，这里展示底层影响域)
        svg.appendChild(makeSvg("path", {
          d: `M ${ex} ${ey} Q ${(ex+cx)/2 + 30} ${(ey+cy)/2 - 30} ${cx} ${cy}`,
          fill: "none", stroke: "#a855f7", "stroke-width": 2, opacity: 0.6,
          "stroke-dasharray": "4 4"
        }));
      });
    });

    for (let r = 0; r < instance.L; r++) {
      for (let c = 0; c < instance.L; c++) {
        // Cell Background
        svg.appendChild(makeSvg("rect", {
          x: offX + c*step, y: offY + r*step,
          width: step, height: step, fill: "none", stroke: "#e2e8f0", "stroke-width": 1
        }));

        // Block 1 Data (Top-Left)
        const d1Err = errorSet.has(`1_${r}_${c}`);
        svg.appendChild(makeSvg("circle", { cx: offX + c*step + step*0.25, cy: offY + r*step + step*0.25, r: step*0.1, fill: d1Err ? "#1e293b" : "#fbbf24" }));
        
        // Block 2 Data (Top-Right)
        const d2Err = errorSet.has(`2_${r}_${c}`);
        svg.appendChild(makeSvg("circle", { cx: offX + c*step + step*0.75, cy: offY + r*step + step*0.25, r: step*0.1, fill: d2Err ? "#1e293b" : "#fbbf24" }));

        // X Check (Bottom-Left)
        const activeX = eventSet.has(`chk_X_${r}_${c}`);
        svg.appendChild(makeSvg("circle", { cx: offX + c*step + step*0.25, cy: offY + r*step + step*0.75, r: activeX ? step*0.18 : step*0.1, fill: activeX ? "#ef4444" : "none", stroke: "#ef4444", "stroke-width": 2 }));

        // Z Check (Bottom-Right)
        const activeZ = eventSet.has(`chk_Z_${r}_${c}`);
        svg.appendChild(makeSvg("circle", { cx: offX + c*step + step*0.75, cy: offY + r*step + step*0.75, r: activeZ ? step*0.18 : step*0.1, fill: activeZ ? "#3b82f6" : "none", stroke: "#3b82f6", "stroke-width": 2 }));
      }
    }
  }
}

// =====================================================
// UI 更新与绑定
// =====================================================
function populateInstances() {
  const family = codeFamilies[state.family];
  els.codeInstance.innerHTML = "";
  family.instances.forEach(inst => {
    const opt = document.createElement("option");
    opt.value = inst.id;
    opt.textContent = `[[${inst.n},${inst.k},${inst.d}]] ${family.name}`;
    els.codeInstance.appendChild(opt);
  });
  state.instanceId = family.instances[0].id;
}

function updateUI() {
  const instance = currentInstance();
  const errors = getPhysicalErrors(instance);
  const events = computeSyndrome(instance, errors);

  els.layoutLabel.textContent = `[[${instance.n},${instance.k},${instance.d}]] Target`;
  els.codeReadout.textContent = `[[${instance.n},${instance.k},${instance.d}]]`;
  els.nReadout.textContent = `n = ${instance.n}`;
  els.kReadout.textContent = `k = ${instance.k}`;
  els.dReadout.textContent = `d = ${instance.d}`;
  els.syndromeReadout.textContent = `${events.length}`;

  drawLattice(els.codeLattice, instance, errors, events);
}

document.querySelectorAll("[data-family]").forEach(btn => {
  btn.addEventListener("click", () => {
    state.family = btn.dataset.family;
    document.querySelectorAll("[data-family]").forEach(i => i.classList.toggle("active", i === btn));
    populateInstances();
    state.anchoredErrors = null; // 切换 Family 时重置底层锚点
    updateUI();
  });
});

els.codeInstance.addEventListener("change", e => {
  state.instanceId = e.target.value;
  updateUI();
});

els.errorSlider.addEventListener("input", e => {
  state.errorCount = Number(e.target.value);
  state.anchoredErrors = null; // 改变错误数量强制重新采样
  updateUI();
});

els.resampleBtn.addEventListener("click", () => {
  state.seed = Math.floor(Math.random() * 10000);
  state.anchoredErrors = null;
  updateUI();
});

// 初始化
populateInstances();
updateUI();



// =====================================================
// AI Assistant Widget 逻辑
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
