const DEFAULT_MODEL = "@cf/zai-org/glm-4.7-flash";
const MAX_MESSAGE_CHARS = 900;
const MAX_HISTORY_ITEMS = 6;
const MAX_CONTEXT_CHUNKS = 5;

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
      "repository",
      "repo",
      "readme",
      "requirements",
      "environment",
    ],
    text:
      "The current GitHub repository is organized around a top-level inference launcher and two code-family " +
      "directories. inference.sh is the unified evaluation entry point for surface-code and BB-code checkpoints. " +
      "codes/Surface contains transformer.py for the surface NTU-Transformer, inference.py for evaluation, " +
      "baseline.py for PyMatching baselines, and train.sh/inference.sh/baseline.sh launchers. codes/BB contains " +
      "transformer.py for the BB NTU-Transformer, neural_bp.py for NTU-Neural-BP, baseline.py for BP-OSD and " +
      "Relay BP baselines, plus shell launchers for BB72 training, BB144 transfer, Neural-BP training, and BB72 " +
      "baseline sweeps. requirements.txt and environment.yml define the Python environment. webpage contains the " +
      "GitHub Pages project page and webpage/paper-assistant-worker contains this Cloudflare Worker assistant.",
  },
  {
    id: "Code: install and checkpoints",
    keywords: [
      "install",
      "installation",
      "environment.yml",
      "requirements.txt",
      "conda",
      "pip",
      "hugging face",
      "checkpoint",
      "ckpt",
      "model weights",
      "安装",
      "环境",
      "依赖",
      "权重",
      "检查点",
    ],
    text:
      "The README recommends Python >= 3.10 and a CUDA-capable GPU. The conda path is conda env create -f " +
      "environment.yml followed by conda activate tennis; the pip path is installing a suitable PyTorch build " +
      "then pip install -r requirements.txt. Optional baseline dependencies include ldpc for BP-OSD and relay_bp " +
      "for Relay BP. Checkpoints can be loaded locally with --ckpt or --ckpt_path, or downloaded through " +
      "download_from_hf / hf_hub_download from Dreamworldsmile/ntu-surface-code-decoder. The README lists surface " +
      "d=7,11,15,19,23,25 checkpoints plus BB72 Transformer and Neural-BP checkpoints.",
  },
  {
    id: "Code: unified inference",
    keywords: [
      "inference.sh",
      "inference",
      "eval",
      "evaluate",
      "shots",
      "--code",
      "--model",
      "--block_size",
      "--hf_repo",
      "--ckpt",
      "推理",
      "评估",
      "运行",
    ],
    text:
      "The top-level inference.sh script dispatches evaluation. For surface codes it requires --code surface, " +
      "--d, and --shots, defaults --eval_p to 0.003 and batch size to 256, then calls codes/Surface/inference.py " +
      "with either --ckpt_path from --ckpt or --hf_repo. For BB codes it requires --code bb, --block_size 72 or 144, " +
      "and --shots; --model defaults to transformer but may be neural_bp. BB72 maps to torus_l=6, torus_m=6, " +
      "rounds=6; BB144 maps to torus_l=12, torus_m=6, rounds=12. Transformer eval calls codes/BB/transformer.py " +
      "eval with A_x 3, A_y 1 2, B_x 1 2, B_y 3, d_model=512, n_heads=8, and logical_anchor_mode representative. " +
      "Neural-BP eval calls codes/BB/neural_bp.py eval with block_size, p, rounds, hidden_dim=64, and num_iter.",
  },
  {
    id: "Code: surface transformer",
    keywords: [
      "codes/Surface/transformer.py",
      "surface transformer",
      "AlphaQubitV2",
      "FullMapper",
      "FullMappingInfo",
      "CoordinateRoPE",
      "SpatialTransformerBlock",
      "AQCrossAttentionLayer",
      "OnlineSurfaceCodeDataset",
      "surface model",
      "表面码",
      "模型",
      "架构",
    ],
    text:
      "codes/Surface/transformer.py implements the surface-code NTU-Transformer. FullMapper builds the joint " +
      "X/Z stabilizer mapping from Stim rotated_memory_z detector coordinates. It creates gather_z/gather_x " +
      "indices, valid_z/valid_x masks, same-type neighbor tables, and cross-type hint neighbors. AlphaQubitV2 " +
      "uses shared discrete embeddings for X and Z stabilizers, adds temporal encodings and coordinate RoPE, " +
      "processes the concatenated X+Z stabilizer sequence with 5 RecurrentBlock layers and 6 SpatialTransformerBlock " +
      "layers, and uses AQCrossAttentionLayer readout to predict the logical observable. OnlineSurfaceCodeDataset " +
      "generates Stim samples online; training can use correlated PyMatching pseudo-labels for process supervision.",
  },
  {
    id: "Code: surface training and baselines",
    keywords: [
      "codes/Surface/train.sh",
      "codes/Surface/inference.py",
      "codes/Surface/baseline.py",
      "codes/Surface/baseline.sh",
      "run_training",
      "train.sh",
      "resume",
      "transfer",
      "PyMatching",
      "correlated",
      "standard",
      "baseline",
      "训练循环",
      "迁移",
      "基线",
    ],
    text:
      "codes/Surface/train.sh launches distributed surface training with torchrun on 8 processes per node. It " +
      "supports --mode scratch and --mode transfer. Transfer requires either --ckpt or --hf_ckpt and passes " +
      "--resume or --hf_resume into transformer.py. Required training arguments include --d, --train_p, --eval_p, " +
      "--target_high, --target_low, --batch_size, --lr, --max_steps, and --output_dir. run_training in " +
      "codes/Surface/transformer.py supports checkpoint transfer by rebuilding distance-specific mapper buffers " +
      "while loading transferable learned weights. codes/Surface/inference.py evaluates a checkpoint. " +
      "codes/Surface/baseline.py implements evaluate_ler_pymatching for standard or correlated PyMatching baselines; " +
      "baseline.sh wraps it with --d, --p, --shots, and --mode.",
  },
  {
    id: "Code: BB transformer construction",
    keywords: [
      "codes/BB/transformer.py",
      "BB transformer",
      "AlphaQubitV2_BB",
      "BBMapper",
      "BBMappingInfo",
      "CartesianRoPE",
      "signed wrap",
      "logical readout bias",
      "bivariate bicycle",
      "BB码",
      "循环码",
      "环面",
    ],
    text:
      "codes/BB/transformer.py contains the BB NTU-Transformer. It constructs bivariate bicycle CSS codes with " +
      "create_bivariate_bicycle_codes, using hx=[A,B] and hz=[B.T,A.T] from cyclic shift matrices. build_circuit " +
      "creates the noisy Stim circuit with X-check ancillas, left/right data registers, and Z-check ancillas. " +
      "BBMapper builds detector mappings for a torus with l*m sites, rounds+1 time slices, Z/X gather buffers, " +
      "same-type neighbor offsets, cross-type hint offsets, and spatial coordinates. The default polynomial " +
      "parameters used by the launchers are A_x=3, A_y=1,2, B_x=1,2, B_y=3.",
  },
  {
    id: "Code: BB transformer architecture",
    keywords: [
      "AlphaQubitV2_BB",
      "SignedWrapRoPE",
      "folded relative phases",
      "rope_delta_mode",
      "logical_anchor_mode",
      "LogicalBasis",
      "build_default_observables",
      "representative",
      "readout",
      "注意力",
      "逻辑读出",
    ],
    text:
      "AlphaQubitV2_BB in codes/BB/transformer.py mirrors the surface model but is torus-aware. It embeds Z and X " +
      "detectors with high-cardinality local syndrome-pattern embeddings, temporal embeddings, cross-type hint " +
      "embeddings, and a learned type embedding. It uses 5 recurrent layers and 6 spatial Transformer layers. " +
      "Instead of ordinary absolute RoPE, it computes folded relative phases over the torus; rope_delta_mode can " +
      "be signed_neg_half, signed_pos_half, or raw_mod. The readout has K logical query embeddings and a static " +
      "logical_readout_bias derived from logical/stabilizer overlap. build_default_observables and related " +
      "LogicalBasis utilities define the output logical basis for the K=12 BB codes.",
  },
  {
    id: "Code: BB transformer train and transfer",
    keywords: [
      "train_transformer_bb72.sh",
      "transfer_transformer_bb144.sh",
      "main_train_transformer",
      "main_eval_transformer",
      "BB72",
      "BB144",
      "transfer",
      "torchrun",
      "skip_oom_probe",
      "训练",
      "迁移",
    ],
    text:
      "codes/BB/train_transformer_bb72.sh trains the BB72 Transformer with torchrun --standalone, default " +
      "NPROC_PER_NODE=8, torus_l=6, torus_m=6, rounds=6, p=0.005, d_model=512, n_heads=8, logical_anchor_mode " +
      "representative, BATCH_SIZE=128, TARGET_BS=2048, LR=5e-4, WARMUP=200, and output under experiments. " +
      "codes/BB/transfer_transformer_bb144.sh transfers to BB144 using torus_l=12, torus_m=6, rounds=12, and " +
      "a resume checkpoint. In transformer.py, train and eval are subcommands selected by the first positional " +
      "argument; eval accepts --ckpt_path or --hf_repo/--hf_filename plus --shots and --batch_size.",
  },
  {
    id: "Code: Neural-BP",
    keywords: [
      "codes/BB/neural_bp.py",
      "Neural-BP",
      "NeuralBPDecoder",
      "NeuralBPLayer",
      "forward",
      "MessagePassing",
      "OnlineBBPDataset",
      "FocalLoss",
      "SyndromeConsistencyLoss",
      "dem_to_check_matrix",
      "置信传播",
      "GNN",
    ],
    text:
      "codes/BB/neural_bp.py implements NTU-Neural-BP. dem_to_check_matrix converts a Stim detector error model " +
      "into parity-check matrix H, logical matrix L, and channel probabilities. OnlineBBPDataset samples error " +
      "vectors e from those probabilities. It computes the syndrome s = H e mod 2 as check-node input x_c, uses " +
      "initial log-likelihood ratios as variable-node input x_v, sets the physical-error target to y=e, and also " +
      "stores logical_flip = L e mod 2 for validation. The dataset builds PyTorch Geometric BipartiteData with " +
      "variable-to-check and check-to-variable edges. This matrix conversion and dataset sampling are preprocessing, " +
      "not operations inside each NeuralBPLayer iteration. NeuralBPLayer is a MessagePassing layer: forward starts " +
      "from h_v and h_c, propagates variable-to-check messages, combines them with h_c_initial through the v2c MLP " +
      "and check GRU, then propagates check-to-variable messages through the c2v MLP and variable GRU. " +
      "NeuralBPDecoder repeats the same NeuralBPLayer num_iterations times and predicts physical error logits with " +
      "readout(h_v). In training mode it returns " +
      "loss = FocalLoss(output, y) + lambda_syn * SyndromeConsistencyLoss(output_logits, x_c, edge_index_v2c), " +
      "where lambda_syn is 0.2 in the code.",
  },
  {
    id: "Code: Neural-BP commands",
    keywords: [
      "train_neural_bp.sh",
      "generate_dems",
      "neural_bp.py train",
      "neural_bp.py eval",
      "BLOCK_SIZE",
      "hidden_dim",
      "num_iter",
      "dem_path",
      "Neural-BP training",
      "Neural-BP eval",
      "训练neural bp",
    ],
    text:
      "codes/BB/train_neural_bp.sh launches Neural-BP training. It is controlled by environment variables such " +
      "as BLOCK_SIZE, NPROC_PER_NODE, BATCH_SIZE, TARGET_BS, MAX_STEPS, LR, HIDDEN_DIM, NUM_ITER, NUM_WORKERS, " +
      "and VAL_WORKERS. neural_bp.py has train, generate_dems, and eval subcommands. train requires --block_size, " +
      "--p, --hidden_dim, --num_iter, --dem_path, and --output. generate_dems writes detector error models under " +
      "data/ldpc by default. eval requires --block_size and --shots, uses --ckpt_path or --hf_repo/--hf_filename, " +
      "and can accept --dem_path; otherwise it can build/load the default DEM for the requested BB size.",
  },
  {
    id: "Code: BB baselines",
    keywords: [
      "codes/BB/baseline.py",
      "run_baseline_bb72.sh",
      "BP-OSD",
      "BPOSD",
      "Relay BP",
      "RelayBpObservableDecoder",
      "BposdLogicalDecoder",
      "generate_dem",
      "baseline",
      "基线",
      "BP+OSD",
    ],
    text:
      "codes/BB/baseline.py implements BB72 baseline experiments. It can generate DEMs, run BP-OSD, and run " +
      "Relay BP. CSSCode, create_bivariate_bicycle_code, and build_circuit construct the BB72 circuit. " +
      "load_dem_matrices extracts H, L, and probabilities from a DEM. BposdLogicalDecoder wraps ldpc BP-OSD " +
      "with product_sum BP and osd_cs; RelayBpObservableDecoder wraps relay_bp presets. run_baseline_bb72.sh " +
      "selects the method through METHOD=bposd or METHOD=relaybp.",
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

function isCodeQuery(query) {
  return /code|repo|repository|github|file|class|function|script|run|train|eval|inference|baseline|checkpoint|ckpt|代码|仓库|文件|类|函数|脚本|运行|训练|推理|评估|基线|检查点/i.test(query);
}

function scoreChunk(query, chunk) {
  const q = query.toLowerCase();
  const id = chunk.id.toLowerCase();
  const text = chunk.text.toLowerCase();
  let score = 0;

  if (isCodeQuery(q) && id.startsWith("code:")) score += 3;
  if (q.includes("surface") && id.includes("surface")) score += 4;
  if ((q.includes("bb") || q.includes("bivariate") || q.includes("bicycle")) && id.includes("bb")) score += 4;
  if ((q.includes("neural-bp") || q.includes("neural bp") || q.includes("gnn") || q.includes("置信传播")) && id.includes("neural-bp")) score += 6;

  for (const token of tokenize(q)) {
    if (text.includes(token)) score += token.length > 4 ? 2 : 1;
    if (id.includes(token)) score += token.length > 4 ? 4 : 2;
    if (/^[a-z0-9_/-]{8,}$/.test(token) && text.includes(token)) score += 10;
  }

  for (const keyword of chunk.keywords) {
    const key = keyword.toLowerCase();
    if (q.includes(key)) score += 5;
  }

  return score;
}

function retrieveContext(query) {
  const pool = isCodeQuery(query)
    ? knowledgeChunks.filter((chunk) => chunk.id.startsWith("Code:"))
    : knowledgeChunks;

  const ranked = pool
    .map((chunk) => ({ ...chunk, score: scoreChunk(query, chunk) }))
    .sort((a, b) => b.score - a.score)
    .filter((chunk) => chunk.score > 0);

  return (ranked.length ? ranked : pool)
    .slice(0, MAX_CONTEXT_CHUNKS);
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

function hasChinese(text) {
  return /[\u3400-\u9fff]/.test(text);
}

function directCodeAnswer(message) {
  const q = message.toLowerCase();
  const asksNeuralBp =
    q.includes("neuralbpdecoder") ||
    q.includes("neural-bp") ||
    q.includes("neural bp") ||
    q.includes("neural_bp");

  const asksMechanism =
    /work|forward|architecture|mechanism|implement|how|怎么|如何|工作|实现|机制|架构/.test(q);

  if (!asksNeuralBp || !asksMechanism) return null;

  const sources = ["Code: Neural-BP", "Code: Neural-BP commands"];
  if (hasChinese(message)) {
    return {
      answer:
        "`NeuralBPDecoder` 在 `codes/BB/neural_bp.py` 中实现，是 BB code 的 Neural-BP physical-error decoder。\n\n" +
        "前处理阶段：`dem_to_check_matrix` 从 Stim detector error model 得到 parity-check matrix `H`、logical matrix `L` 和 channel probabilities。`OnlineBBPDataset` 采样 error vector `e`，计算 syndrome `s = H e mod 2` 作为 check-node input `x_c`，使用 initial LLR 作为 variable-node input `x_v`，训练目标是 physical-error vector `y=e`，并生成带 v2c/c2v edges 的 `BipartiteData`。\n\n" +
        "模型阶段：`NeuralBPDecoder` 编码 `x_c` 和 `x_v`，重复调用 `NeuralBPLayer` 共 `num_iterations` 次。每层先做 variable-to-check message passing，经 v2c MLP 和 check GRU 更新 check states；再做 check-to-variable message passing，经 c2v MLP 和 variable GRU 更新 variable states。最后 `readout(h_v)` 输出 physical error logits。训练损失是 `FocalLoss(output, y) + 0.2 * SyndromeConsistencyLoss(...)`。",
      sources,
    };
  }

  return {
    answer:
      "`NeuralBPDecoder` is implemented in `codes/BB/neural_bp.py` as the BB-code Neural-BP physical-error decoder.\n\n" +
      "Preprocessing: `dem_to_check_matrix` converts a Stim detector error model into parity-check matrix `H`, logical matrix `L`, and channel probabilities. `OnlineBBPDataset` samples an error vector `e`, computes the syndrome `s = H e mod 2` as check-node input `x_c`, uses initial LLRs as variable-node input `x_v`, sets the target to the physical-error vector `y=e`, and yields `BipartiteData` with v2c/c2v edges.\n\n" +
      "Model forward: `NeuralBPDecoder` encodes `x_c` and `x_v`, repeats `NeuralBPLayer` for `num_iterations`, then applies `readout(h_v)` to predict physical-error logits. Each layer runs variable-to-check message passing through the v2c MLP and check GRU, followed by check-to-variable message passing through the c2v MLP and variable GRU. Training uses `FocalLoss(output, y) + 0.2 * SyndromeConsistencyLoss(...)`.",
    sources,
  };
}

function extractAnswer(result) {
  if (!result) return "";
  if (typeof result.response === "string") return result.response;
  if (typeof result.answer === "string") return result.answer;
  if (typeof result.output_text === "string") return result.output_text;
  if (typeof result.text === "string") return result.text;
  if (result.result && typeof result.result.response === "string") return result.result.response;
  if (result.result && typeof result.result.answer === "string") return result.result.answer;
  if (result.result && typeof result.result.output_text === "string") return result.result.output_text;
  if (result.result && Array.isArray(result.result.choices)) {
    return extractAnswer({ choices: result.result.choices });
  }
  if (Array.isArray(result.choices)) {
    const choice = result.choices[0] || {};
    const content = choice.message?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((part) => part?.text || part?.content || "")
        .filter(Boolean)
        .join("\n");
    }
    return choice.text || choice.message?.reasoning_content || choice.message?.reasoning || "";
  }
  return "";
}

function normalizeSymbolKinds(answer) {
  return answer
    .replace(/神经平移统一/g, "Neural Transfer Unification (NTU)")
    .replace(/双变量自行车码/g, "bivariate bicycle (BB) codes")
    .replace(/拔除冷启动穿过的平缓区/g, "避免冷启动平台期")
    .replace(/两个主要类[:：]\s*`?dem_to_check_matrix`?\s*和\s*`?BPDecoder`?/g, "一个函数 `dem_to_check_matrix` 和一个类 `BPDecoder`")
    .replace(/`dem_to_check_matrix`类/g, "`dem_to_check_matrix` 函数")
    .replace(/dem_to_check_matrix类/g, "dem_to_check_matrix 函数")
    .replace(/`dem_to_check_matrix` 的类/g, "`dem_to_check_matrix` 函数")
    .replace(/dem_to_check_matrix 的类/g, "dem_to_check_matrix 函数")
    .replace(/`?NeuralBPDecoder`? 是序列化过程。/g, "`NeuralBPDecoder` 是 `codes/BB/neural_bp.py` 中的模型类。")
    .replace(/构建带边的云端数据集/g, "构建带有 variable-to-check 和 check-to-variable 边的 `BipartiteData` 图数据")
    .replace(/使用 `readout` 和 `FocalLoss` 进行预测与训练/g, "用 `readout(h_v)` 预测 physical error logits；训练损失是 `FocalLoss(output, y) + lambda_syn * SyndromeConsistencyLoss(...)`");
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

  const direct = directCodeAnswer(message);
  if (direct) {
    return json(
      {
        ...direct,
        model: "grounded-code-context",
      },
      200,
      headers,
    );
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
    "Preserve code abbreviations and names exactly; for example, keep BP as belief propagation and do not invent renamed losses. " +
    "Keep implementation object names in English when translation would be ambiguous, such as BipartiteData, MessagePassing, logits, and readout. " +
    "Keep paper-specific names in English when possible, especially Neural Transfer Unification (NTU), NTU-Transformer, and bivariate bicycle (BB) codes. " +
    "For architecture questions, separate preprocessing/data loading from the model forward pass and per-layer iteration. " +
    "If multiple implementation details are close, state only the one explicitly present in the context. " +
    "Keep answers under 180 words unless the user explicitly asks for more detail. " +
    "Do not output code blocks unless the provided context includes a complete code snippet; prefer implementation summaries and command names. " +
    "Do not invent repository links, model links, numerical results, or implementation details that are not in the context.";

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `Paper and code context:\n${context}` },
    ...history,
    { role: "user", content: message },
  ];

  try {
    const result = await env.AI.run(model, {
      messages,
      max_completion_tokens: 520,
      chat_template_kwargs: { enable_thinking: false },
    });
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
