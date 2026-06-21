const DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const MAX_MESSAGE_CHARS = 900;
const MAX_HISTORY_ITEMS = 6;

const knowledgeChunks = [
  {
    id: "Abstract",
    keywords: [
      "abstract",
      "summary",
      "overview",
      "main contribution",
      "主要贡献",
      "摘要",
      "总结",
      "ntu",
      "foundation decoder",
    ],
    text:
      "Foundation decoders are high-capacity neural decoders for fault-tolerant quantum computing. " +
      "Their construction faces a steep scaling barrier because larger code distances rapidly increase " +
      "the cost of syndrome generation and neural optimization. Neural Transfer Unification (NTU) aligns " +
      "decoding tasks across code distances through algebraic structures shared by scalable code families. " +
      "This lets knowledge learned on smaller codes accelerate large-scale decoder training. The paper " +
      "instantiates NTU as NTU-Transformer for planar surface codes and bivariate bicycle (BB) codes.",
  },
  {
    id: "Motivation",
    keywords: [
      "motivation",
      "why",
      "problem",
      "scaling barrier",
      "cold start",
      "为什么",
      "问题",
      "瓶颈",
      "冷启动",
      "训练成本",
    ],
    text:
      "Fault-tolerant quantum computation requires accurate and efficient decoders. Conventional decoders " +
      "such as matching-based methods and belief propagation are important but can be limited by scalability " +
      "or degeneracy at large code distances. Neural foundation decoders can be competitive, but training " +
      "large-distance models from scratch is expensive and can suffer from a cold-start optimization plateau. " +
      "NTU addresses this by replacing distance-specific training with reusable cross-distance transfer.",
  },
  {
    id: "Structural isomorphism",
    keywords: [
      "structural isomorphism",
      "scale invariance",
      "algebraic",
      "polynomial",
      "motif",
      "local neighborhood",
      "结构同构",
      "尺度不变",
      "代数",
      "局部",
      "motif",
      "邻域",
    ],
    text:
      "The central physical principle is structural isomorphism across code distances. Syndrome detectors " +
      "are represented by spatiotemporal polynomial coordinates v = x^i y^j t^k. The localized topological " +
      "neighborhood of a detector is written as N(v) = v · M(x,y,t), where M encodes an invariant local " +
      "correlation motif. When a structured code family scales, its global boundary changes but the local " +
      "generator polynomials remain conserved, allowing local error features learned on small codes to " +
      "remain meaningful on larger codes.",
  },
  {
    id: "Surface codes",
    keywords: [
      "surface",
      "surface code",
      "rotated surface code",
      "planar",
      "d=19",
      "d=25",
      "PyMatching",
      "correlated matching",
      "表面码",
      "码距",
      "匹配",
    ],
    text:
      "For planar rotated surface codes under circuit-level depolarizing noise, the paper evaluates a " +
      "standard Z-basis memory experiment with d syndrome extraction rounds. NTU-Transformer is trained " +
      "from scratch on d = 7 and then sequentially transferred to larger distances. At p = 0.3%, the " +
      "transferred model significantly outperforms standard PyMatching and closely approaches correlated " +
      "PyMatching. It also reaches [[625,1,25]] with only 27% additional training overhead relative to the " +
      "transferred baseline.",
  },
  {
    id: "Cold-start dynamics",
    keywords: [
      "cold start",
      "plateau",
      "training dynamics",
      "collapse",
      "scratch",
      "transfer",
      "lift off",
      "冷启动",
      "平台期",
      "塌缩",
      "从零训练",
      "迁移",
    ],
    text:
      "A key empirical advantage of NTU is the removal of the cold-start plateau. Scratch-initialized " +
      "large-distance NTU-Transformer models can remain close to random guessing for thousands of steps, " +
      "and the paper diagnoses this as a constant-output collapse mode. Transfer initialization supplies " +
      "a pre-aligned local perception backbone, so target-side training can immediately enter a useful " +
      "fine-tuning trajectory instead of first escaping a trivial predictor.",
  },
  {
    id: "BB codes",
    keywords: [
      "BB",
      "bivariate bicycle",
      "qLDPC",
      "RelayBP",
      "BP+OSD",
      "[[72,12,6]]",
      "[[144,12,12]]",
      "bicycle",
      "非局部",
      "量子LDPC",
    ],
    text:
      "The paper also evaluates bivariate bicycle (BB) codes, a quasi-cyclic qLDPC family with non-local " +
      "hypergraph structure. NTU-Transformer is evaluated on [[72,12,6]] and transferred to [[144,12,12]]. " +
      "On [[72,12,6]], NTU-Transformer outperforms BP+OSD and advanced Relay-BP variants across tested " +
      "physical error rates, especially in the low-physical-error regime. Transfer to [[144,12,12]] avoids " +
      "the scratch-training plateau for both Transformer and NTU-NeuralBP variants.",
  },
  {
    id: "NTU-Transformer architecture",
    keywords: [
      "architecture",
      "NTU-Transformer",
      "STEM",
      "RoPE",
      "embedding",
      "attention",
      "transformer",
      "架构",
      "位置编码",
      "嵌入",
      "注意力",
    ],
    text:
      "NTU-Transformer uses two targeted designs. The scalable transformer embedding model (STEM) builds " +
      "local inductive bias by aggregating explicit topological neighbors and grouping them into same-type, " +
      "cross-type, and temporal predecessor detectors. Geometry-aware RoPE encodes unnormalized relative " +
      "algebraic shifts instead of normalizing by global code size, preserving local attention phase patterns " +
      "when the code distance grows.",
  },
  {
    id: "Ablations",
    keywords: [
      "ablation",
      "RoPE ablation",
      "stem ablation",
      "embedding ablation",
      "full model",
      "消融",
      "对比实验",
      "RoPE",
      "stem",
    ],
    text:
      "The ablation studies test whether transfer depends on structural alignment. On surface-code transfer " +
      "from d = 7 to d = 11, the full model with physical-coordinate RoPE and a local one-ring stem lifts " +
      "off fastest; removing either component delays transfer, while removing both leaves the model close " +
      "to random guessing. On BB-code transfer from [[72,12,6]] to [[144,12,12]], breaking either the RoPE " +
      "alignment or polynomial-induced embedding alignment removes the measurable early transfer advantage.",
  },
  {
    id: "Discussion",
    keywords: [
      "discussion",
      "impact",
      "real time",
      "deployment",
      "future",
      "latency",
      "讨论",
      "意义",
      "实时",
      "部署",
      "未来",
    ],
    text:
      "The paper argues that generating high-quality full-precision foundation decoders is a major prerequisite " +
      "for real-time neural decoding. Hardware deployment may require compression, distillation, quantization, " +
      "and pruning. By reducing the training barrier for large-distance base models, NTU provides a practical " +
      "starting point for future hardware-integrated, low-latency neural decoders.",
  },
  {
    id: "Code: repository layout",
    keywords: [
      "code",
      "source",
      "file",
      "layout",
      "implementation",
      "where",
      "代码",
      "文件",
      "源码",
      "实现",
      "在哪",
      "目录",
    ],
    text:
      "The local code directory contains six main files: BP.py, MWPM.py, Transformer_0319 (3).py, " +
      "ldpc_gen.py, run_all (2).sh, and utils.py. BP.py evaluates BP/BP+OSD on detector-error models. " +
      "MWPM.py generates rotated surface-code samples with Stim and evaluates correlated PyMatching. " +
      "Transformer_0319 (3).py contains the main surface-code NTU-Transformer training implementation. " +
      "ldpc_gen.py constructs bivariate bicycle CSS codes and Stim circuits. run_all (2).sh launches " +
      "multi-GPU scaling-ladder training. utils.py contains shared DEM conversion and loss utilities.",
  },
  {
    id: "Code: Transformer main model",
    keywords: [
      "Transformer_0319",
      "AlphaQubitV2",
      "FullMapper",
      "main model",
      "NTU-Transformer",
      "transformer code",
      "模型",
      "主模型",
      "训练代码",
      "FullMapper",
      "AlphaQubitV2",
    ],
    text:
      "Transformer_0319 (3).py implements the main surface-code neural decoder. FullMapper builds a " +
      "joint X/Z stabilizer mapping from Stim detector coordinates: gather_z/gather_x map internal layout " +
      "positions to Stim detector indices, valid_z/valid_x mark boundary-valid detectors, same-type " +
      "diagonal neighbors support local spatial embeddings, and cross-type hint neighbors connect nearby " +
      "opposite stabilizer types. AlphaQubitV2 is the main model: it embeds X and Z stabilizers, concatenates " +
      "them into one spatial sequence, applies per-stabilizer recurrent blocks and spatial Transformer blocks, " +
      "then uses cross-attention readout to predict the final logical observable.",
  },
  {
    id: "Code: CoordinateRoPE and attention",
    keywords: [
      "CoordinateRoPE",
      "RoPE",
      "apply_rope_2d",
      "SpatialTransformerBlock",
      "attention",
      "position",
      "positional",
      "坐标",
      "位置编码",
      "注意力",
    ],
    text:
      "CoordinateRoPE in Transformer_0319 (3).py implements 2D rotary positional encoding. The head dimension " +
      "must be divisible by 4, so half of each attention head is assigned to x and half to y. get_freqs builds " +
      "coordinate-dependent frequencies from physical detector coordinates. apply_rope_2d splits q and k into " +
      "x/y halves, applies rotate-half sinusoidal phases separately, and concatenates them. SpatialTransformerBlock " +
      "uses RMSNorm, qkv projection, 2D RoPE, scaled_dot_product_attention, output projection, and a SwiGLU MLP.",
  },
  {
    id: "Code: online surface-code dataset",
    keywords: [
      "OnlineSurfaceCodeDataset",
      "dataset",
      "Stim",
      "sampler",
      "MWPM teacher",
      "FakeEnd",
      "process supervision",
      "数据集",
      "采样",
      "伪标签",
      "蒸馏",
    ],
    text:
      "OnlineSurfaceCodeDataset in Transformer_0319 (3).py generates rotated surface-code memory-z circuits " +
      "online with Stim. During training it samples detector events and observables, then uses correlated " +
      "PyMatching as a teacher to produce per-time-step fake-end labels by masking future detectors. During " +
      "evaluation, is_eval=True skips the expensive MWPM teacher and returns only detector events and true " +
      "observable labels. Random seeds combine time, os.urandom, worker id, and DDP rank.",
  },
  {
    id: "Code: training loop and transfer",
    keywords: [
      "run_training",
      "training loop",
      "resume",
      "transfer",
      "checkpoint",
      "DDP",
      "loss",
      "beta",
      "mask_prob",
      "drop_ratio",
      "训练循环",
      "迁移",
      "断点",
      "检查点",
      "损失",
    ],
    text:
      "run_training in Transformer_0319 (3).py sets up DDP, creates online training and validation loaders, " +
      "builds AlphaQubitV2 with d_model=512 and n_heads=8, and supports checkpoint transfer through --resume. " +
      "When the checkpoint distance or rounds differ from the target, mapper/coordinate/gather/valid/neighbor " +
      "buffers are dropped before loading weights, preserving transferable learnable parameters while rebuilding " +
      "distance-specific buffers. Training uses AdamW, warmup plus cosine decay, gradient accumulation, bfloat16 " +
      "autocast, gradient clipping, stabilizer dropout that decays over training, and a combined loss: final true " +
      "label BCE plus beta-weighted process-supervision BCE against MWPM fake labels. beta decays from 8.0 toward 0.2.",
  },
  {
    id: "Code: Transformer CLI",
    keywords: [
      "CLI",
      "arguments",
      "command",
      "torchrun",
      "parser",
      "--d",
      "--train_p",
      "--eval_p",
      "--resume",
      "--output",
      "命令",
      "参数",
      "运行",
    ],
    text:
      "Transformer_0319 (3).py accepts command-line arguments --d, --train_p, --eval_p, --target_high, " +
      "--target_low, --batch_size, --max_steps, --lr, --resume, --output, --rounds, and --eval_samples. " +
      "If --rounds is -1, it is set to d. The shell script run_all (2).sh launches torchrun with 8 GPUs, " +
      "a scaling ladder currently including d=15 and d=19, and transfers checkpoints through --resume. " +
      "One implementation note: run_all (2).sh calls Transformer_0319.py, while the local file name shown " +
      "in the provided directory is Transformer_0319 (3).py, so the filename may need to be aligned before running.",
  },
  {
    id: "Code: MWPM baseline",
    keywords: [
      "MWPM",
      "PyMatching",
      "correlated",
      "baseline",
      "generate_surface_code_data",
      "run_pymatching_dem",
      "LER",
      "匹配",
      "基线",
      "评估",
    ],
    text:
      "MWPM.py implements a correlated PyMatching baseline for rotated surface codes. generate_surface_code_data " +
      "creates a Stim generated surface_code:rotated_memory_z circuit with depolarizing and measurement noise, " +
      "samples detector events and observable flips, and extracts a detector error model with decompose_errors=True. " +
      "run_pymatching_dem builds pymatching.Matching.from_detector_error_model(dem), decodes a batch, flattens " +
      "predictions and labels to avoid broadcasting mistakes, and reports total errors plus correlated logical " +
      "error rate. The main block currently tests distance_to_test=[23] with r=d.",
  },
  {
    id: "Code: BP and BPOSD baseline",
    keywords: [
      "BP.py",
      "BPDecoder",
      "BPOSD",
      "BP+OSD",
      "ldpc",
      "dem_to_check_matrix",
      "baseline",
      "belief propagation",
      "置信传播",
      "基线",
    ],
    text:
      "BP.py defines dem_to_check_matrix and BPDecoder. dem_to_check_matrix converts a flattened Stim " +
      "DetectorErrorModel into a sparse parity-check matrix H, a logical matrix L, and channel probabilities. " +
      "BPDecoder loads a DEM from file, initializes ldpc.bp_decoder and ldpc.bposd_decoder with product_sum BP, " +
      "channel_probs from the DEM, max_iter, and BPOSD osd_cs with osd_order=10. evaluate() decodes each syndrome, " +
      "computes predicted_logical_flip = prediction @ L.T mod 2, compares it with the true logical flip, and " +
      "returns accuracy. The script currently points to data/ldpc/test_72_0.005.npz and data/ldpc/72_12_6_0.005.dem.",
  },
  {
    id: "Code: LDPC and BB circuit generation",
    keywords: [
      "ldpc_gen",
      "BB code",
      "bivariate bicycle",
      "css_code",
      "qcc_circuit",
      "build_circuit",
      "row_echelon",
      "kernel",
      "DEM",
      "LDPC",
      "BB码",
      "电路",
      "生成",
    ],
    text:
      "ldpc_gen.py constructs bivariate bicycle CSS codes and Stim circuits. row_echelon and kernel implement " +
      "binary linear algebra; css_code stores hx, hz, computes ranks, LDPC parameters, and logical operators. " +
      "create_circulant_matrix builds cyclic shift matrices, and create_bivariate_bicycle_codes constructs " +
      "hx=[A,B] and hz=[B.T,A.T] from bivariate polynomial powers. build_circuit lays out X-check ancillas, " +
      "left/right data qubits, and Z-check ancillas, then appends repeated noisy CNOT/measurement blocks, " +
      "detectors, and OBSERVABLE_INCLUDE instructions. qcc_circuit wraps this for BB-code circuits; the main " +
      "block generates a [[72,12,6]]-style DEM at p=0.005 and saves data/ldpc/72_12_6_0.005.dem.",
  },
  {
    id: "Code: shared utilities",
    keywords: [
      "utils.py",
      "FocalLoss",
      "SyndromeConsistencyLoss",
      "torch_scatter",
      "dem_to_check_matrix",
      "loss",
      "工具",
      "损失",
      "综合征一致性",
    ],
    text:
      "utils.py contains shared utilities. Its dem_to_check_matrix mirrors the DEM-to-H/L/probability conversion " +
      "used for BP. FocalLoss wraps binary_cross_entropy_with_logits and applies alpha*(1-pt)^gamma to emphasize " +
      "hard examples near the decision boundary. SyndromeConsistencyLoss is a physical constraint loss for a " +
      "flattened PyG-style graph: it maps sigmoid logits to parity-domain values v=1-2p, gathers variable-node " +
      "values along variable-to-check edges, scatter-multiplies them into predicted syndrome values, maps true " +
      "syndrome bits to ±1, and applies MSE.",
  },
];

function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...headers,
    },
  });
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = (env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const allowOrigin = allowed.includes(origin) ? origin : allowed[0] || "*";
  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9+\-.%[\]]{2,}/g) || []).filter(
    (token) => !["the", "and", "for", "with", "from", "that", "this", "are"].includes(token),
  );
}

function scoreChunk(query, chunk) {
  const q = query.toLowerCase();
  const id = chunk.id.toLowerCase();
  const text = chunk.text.toLowerCase();
  let score = 0;

  for (const token of tokenize(q)) {
    if (text.includes(token)) score += token.length > 4 ? 2 : 1;
    if (id.includes(token)) score += token.length > 4 ? 4 : 2;
  }

  for (const keyword of chunk.keywords) {
    const key = keyword.toLowerCase();
    if (q.includes(key)) score += 5;
  }

  return score;
}

function retrieveContext(query) {
  return knowledgeChunks
    .map((chunk) => ({ ...chunk, score: scoreChunk(query, chunk) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

function formatContext(chunks) {
  return chunks.map((chunk) => `[${chunk.id}]\n${chunk.text}`).join("\n\n");
}

function cleanHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .slice(-MAX_HISTORY_ITEMS)
    .filter((item) => item && ["user", "assistant"].includes(item.role) && item.content)
    .map((item) => ({
      role: item.role,
      content: String(item.content).slice(0, 800),
    }));
}

function extractAnswer(result) {
  if (!result) return "";
  if (typeof result.response === "string") return result.response;
  if (typeof result.answer === "string") return result.answer;
  if (result.result && typeof result.result.response === "string") return result.result.response;
  if (Array.isArray(result.choices)) {
    return result.choices[0]?.message?.content || result.choices[0]?.text || "";
  }
  return "";
}

function normalizeSymbolKinds(answer) {
  return answer
    .replace(/两个主要类[:：]\s*`?dem_to_check_matrix`?\s*和\s*`?BPDecoder`?/g, "一个函数 `dem_to_check_matrix` 和一个类 `BPDecoder`")
    .replace(/`dem_to_check_matrix`类/g, "`dem_to_check_matrix` 函数")
    .replace(/dem_to_check_matrix类/g, "dem_to_check_matrix 函数")
    .replace(/`dem_to_check_matrix` 的类/g, "`dem_to_check_matrix` 函数")
    .replace(/dem_to_check_matrix 的类/g, "dem_to_check_matrix 函数");
}

async function handleChat(request, env) {
  const headers = corsHeaders(request, env);
  let body;

  try {
    body = await request.json();
  } catch {
    return json({ error: "Invalid JSON body." }, 400, headers);
  }

  const message = String(body.message || "").trim();
  if (!message) return json({ error: "Message is required." }, 400, headers);
  if (message.length > MAX_MESSAGE_CHARS) {
    return json({ error: `Message is too long. Keep it under ${MAX_MESSAGE_CHARS} characters.` }, 400, headers);
  }

  const chunks = retrieveContext(message);
  const context = formatContext(chunks);
  const history = cleanHistory(body.history);
  const model = env.MODEL || DEFAULT_MODEL;

  const systemPrompt =
    "You are the NTU Decoder paper and code assistant. Answer questions about the paper " +
    "\"Efficient foundation decoders for fault-tolerant quantum computing\" and the provided implementation notes. " +
    "Use only the provided context. If the context does not specify the answer, say that it is not specified in the provided context. " +
    "Answer in the same language as the user's question when possible. Be concise but technically precise. " +
    "When answering code questions, mention the relevant file or class/function names from the context. " +
    "Preserve the kind of each symbol exactly: do not call a function a class, and do not call a class a function. " +
    "Do not output code blocks unless the provided context includes a complete code snippet; prefer implementation summaries and command names. " +
    "Do not invent repository links, model links, numerical results, or implementation details that are not in the context.";

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `Paper and code context:\n${context}` },
    ...history,
    { role: "user", content: message },
  ];

  try {
    const result = await env.AI.run(model, { messages, max_tokens: 700 });
    const answer = normalizeSymbolKinds(extractAnswer(result).trim());
    if (!answer) throw new Error("Empty model response.");

    return json(
      {
        answer,
        sources: chunks.map((chunk) => chunk.id),
        model,
      },
      200,
      headers,
    );
  } catch (error) {
    return json(
      {
        error: "The model request failed.",
        detail: error.message,
      },
      502,
      headers,
    );
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const headers = corsHeaders(request, env);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers });
    }

    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ok: true, service: "ntu-paper-assistant" }, 200, headers);
    }

    if (request.method === "POST" && url.pathname === "/chat") {
      return handleChat(request, env);
    }

    return json({ error: "Not found. Use POST /chat." }, 404, headers);
  },
};
