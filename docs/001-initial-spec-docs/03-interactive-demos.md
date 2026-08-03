# Spec 03 — Interactive demos: the Python-to-web bridge

**Companion to:** `01-architecture-and-design-system.md`, `02-content-and-publishing.md`
**Answers:** how a model you trained in PyTorch ends up as something a stranger can click on.

---

## 1. The decision that actually matters

There isn't one bridge between Python and a web UI — there are three, and picking the wrong one is the main way personal-site demos die. They die by costing money quietly, by breaking six months later when a free tier changes, or by never shipping because the author tried to build a production service for a portfolio piece.

The decision rule, in order. **Stop at the first tier that works.**

```
Can the interesting result be precomputed offline?
│
├── YES ──► TIER 0: ship the artefact. Explorer in JS over a JSON file.
│           Cost £0. Never breaks. Loads instantly.
│
└── NO ──► Does it need a GPU, >100MB of weights, or private data?
           │
           ├── NO ──► TIER 1: run it in the browser.
           │          PyTorch → ONNX → onnxruntime-web. Cost £0.
           │          Pay once in download size.
           │
           └── YES ─► TIER 2: hosted Python API.
                      Real infrastructure. Real cost. Real attack surface.
                      Justify it per demo.
```

Most demos worth building are Tier 0. Your molecule generator is a Tier 1. You may never need a Tier 2, and that is a good outcome, not a limitation.

The instinct from a Dash background is to reach straight for Tier 2, because Dash *is* Tier 2 — a Python process holding state with the browser as a thin client. Resist it. On a personal site, a server that must stay warm to show someone a demo is a liability that will eventually embarrass you.

---

## 2. Tier 0 — precomputed artefacts

**What it is:** run the expensive thing offline, ship the results as static data, build a small interactive explorer over them.

This is what Distill does, and it's why those articles still work years later. The interactivity people actually value is usually *exploring a result space*, not *triggering a computation*.

**When it applies — more often than you'd think:**

| Instead of | Ship |
|---|---|
| Live sampling from a generative model | 5,000 pre-sampled molecules with descriptors and embeddings, filterable |
| A live property predictor | Predictions over a benchmark set, with a residual plot you can brush |
| Live MD | A precomputed trajectory as compressed frames, with a scrubber |
| A live latent-space walk | A precomputed grid of interpolations, indexed by two sliders |
| "Try your own input" | 30 hand-picked inputs that show the interesting failure modes |

That last row is worth sitting with. A curated set of inputs demonstrating where a model succeeds *and where it breaks* is more informative than a free-text box, and it's the version that shows scientific judgement.

**Implementation.** A Python script in `scripts/` writes `public/demos/<slug>/data.json`. An island (React or Svelte, `client:visible`) fetches it and renders. Keep the payload under ~2MB; use `.json.gz` above that (Cloudflare serves it with the right encoding automatically).

**Cost:** zero, forever. **Failure mode:** none worth planning for.

---

## 3. Tier 1 — run the model in the browser

**What it is:** export the trained model to ONNX, run inference client-side with WebAssembly. No server, no cost, no cold start, and it keeps working whether one person visits or ten thousand do.

<cite index="14-1">ONNX Runtime Web offers WebAssembly for CPU and WebGL, WebGPU or WebNN for GPU execution</cite>, so the same artefact degrades gracefully across devices. <cite index="12-1">PyTorch models export to ONNX, which the runtime then serves in the browser.</cite>

### 3.1 The export

```python
# scripts/export_onnx.py
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

model = load_checkpoint("checkpoints/molgen.pt").eval()

# One forward step. The sampling loop lives in JavaScript.
dummy = torch.zeros(1, 16, dtype=torch.long)

torch.onnx.export(
    model, dummy, "build/molgen.onnx",
    input_names=["tokens"], output_names=["logits"],
    dynamic_axes={"tokens": {0: "batch", 1: "seq"},
                  "logits": {0: "batch", 1: "seq"}},
    opset_version=17,
)

# int8 typically cuts size ~4x with negligible quality loss for a model this size.
quantize_dynamic("build/molgen.onnx",
                 "public/demos/molgen/molgen.int8.onnx",
                 weight_type=QuantType.QUInt8)
```

**Export the single forward pass, not the sampling loop.** Control flow exports badly and debugging an ONNX graph containing a loop is miserable. Drive the loop from JavaScript instead.

**Skip the KV cache for v1.** Re-running the full prefix at each step is O(n²), but SMILES are short (~40–80 tokens) and the model is small — it will be fast enough, and it avoids threading `past_key_values` through the ONNX signature. Add caching only if you measure a problem.

**Verify parity before shipping.** Run the same 200 prompts through PyTorch and through `onnxruntime` in Python, and assert the logits match to a sensible tolerance. Almost every "the demo gives different results" bug is caught here, cheaply.

### 3.2 Running it

Load `onnxruntime-web` **inside a Web Worker**. Inference on the main thread will freeze the page during sampling and it will look broken.

```js
// src/demos/molgen/worker.ts
import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "/demos/_ort/";   // self-host; don't hotlink a CDN
let session;

async function init() {
  session = await ort.InferenceSession.create("/demos/molgen/molgen.int8.onnx", {
    executionProviders: ["webgpu", "wasm"],   // WebGPU when available, WASM otherwise
    graphOptimizationLevel: "all",
  });
}

async function step(tokens) {
  const input = new ort.Tensor("int64", BigInt64Array.from(tokens.map(BigInt)),
                               [1, tokens.length]);
  const { logits } = await session.run({ tokens: input });
  return logits;   // caller applies temperature / top-k and samples
}
```

Self-host the runtime's `.wasm` files under `public/demos/_ort/`. Hotlinking a CDN means your demo breaks when someone else's CDN does, and it complicates the COEP headers from doc 01 §7.

### 3.3 RDKit.js — the piece that makes chemistry demos work

<cite index="21-1">RDKit.js is the official JavaScript distribution of the RDKit, compiled to WebAssembly</cite>. It gives you SMILES parsing, canonicalisation, descriptors and 2D depiction client-side — which means a molecular demo needs no server at all.

```js
const RDKit = await window.initRDKitModule({
  locateFile: () => "/demos/_rdkit/RDKit_minimal.wasm",
});

const mol = RDKit.get_mol(smiles);
if (!mol) return { valid: false };          // invalid SMILES — expect plenty

const svg  = mol.get_svg(260, 200);          // 2D depiction, already an SVG
const desc = JSON.parse(mol.get_descriptors());
mol.delete();                                // ← REQUIRED. See below.
```

Three things to get right:

1. **`mol.delete()` is not optional.** Objects live in the WASM heap and are not garbage collected. Generating a few hundred molecules without freeing them will exhaust memory and crash the tab. Wrap every `get_mol` in a `try/finally`.
2. **Pin the version.** <cite index="21-1">As of April 2026 the rdkit-js repository and its npm releases were in a maintenance transition, with the original maintainer stepping back and the RDKit core team looking to transfer npm release maintenance.</cite> The library works fine; just don't float on `latest`. Vendor `RDKit_minimal.js` and `RDKit_minimal.wasm` into `public/demos/_rdkit/` and commit them.
3. **The descriptor JSON is rich** — <cite index="24-1">it includes exact and average molecular weight, Lipinski HBA/HBD counts, rotatable bonds, heavy atom count, heteroatoms, fraction sp3 carbon, and ring counts including aromatic, aliphatic and saturated rings</cite>, among others. You rarely need to compute anything yourself.

### 3.4 When you need actual Python in the browser

<cite index="17-1">Pyodide is a port of CPython to WebAssembly, plus a layer letting JavaScript and Python call into each other; it loads packages as precompiled WASM wheels.</cite> NumPy, SciPy, scikit-learn and pandas all work.

Use it when you genuinely need Python semantics — an existing scikit-learn pipeline you'd rather not reimplement, or a demo where the *point* is running user-supplied Python.

Do not use it as a general route to running your models. **PyTorch does not run in Pyodide.** And as one recent assessment puts it plainly, <cite index="17-1">for compute-bound model inference, moving the code into Pyodide frequently relocates the engineering cost without moving the bottleneck.</cite> ONNX Runtime Web is the right tool for inference; Pyodide is the right tool for Python.

### 3.5 The honest cost of Tier 1

Download size, paid once per visitor and then cached:

| Asset | Rough size | Notes |
|---|---|---|
| ONNX Runtime Web (WASM) | 2–10 MB | Varies a lot by build; use the minimal one |
| `RDKit_minimal.wasm` | ~8 MB | Only load it on chemistry demos |
| Small int8 transformer | 5–20 MB | Depends entirely on parameter count |
| Pyodide core, if used | 6–10 MB | Plus wheels per package |

**Measure these for your actual build; treat the table as order-of-magnitude.**

That total is far too much to load on page view — which is exactly why §5 makes click-to-load mandatory. With an explicit "Run demo" button and immutable cache headers, it's a perfectly reasonable trade: one deliberate 25MB download for a thing that then runs instantly, offline, forever, at zero marginal cost.

---

## 4. Tier 2 — a hosted Python API

Only when Tier 1 genuinely can't work: a GPU is required, weights exceed a few hundred MB, the model can't be shipped to clients, or the demo needs a database.

### 4.1 Options

| Platform | Fit | Notes |
|---|---|---|
| **Modal** | Best fit for you | <cite index="28-1">Python-first serverless compute for ML inference, with infrastructure defined in code, automatic GPU scaling and per-second billing.</cite> Decorator-based, scales to zero, and the deployment story is a Python file — no Dockerfile, no YAML. |
| **Hugging Face Spaces** | Lowest effort | <cite index="33-1">Gradio/Streamlit/Docker SDKs, a free CPU tier and paid GPU hardware.</cite> If you already have a Gradio app, this is ~20 minutes' work. Iframe it into a project page. |
| Fly.io / Render | General-purpose | Fine, but you're managing a container to serve one endpoint. |
| Replicate | Model-as-API | Good for published models, opinionated packaging via Cog. |

**Recommendation: Modal for anything custom, HF Spaces for anything already Gradio-shaped.**

### 4.2 The shape

```python
# demos/protein_embed/app.py
import modal

app = modal.App("prowe-demos")
image = modal.Image.debian_slim().pip_install("torch", "fastapi[standard]", "transformers")

@app.function(image=image, gpu="T4", scaledown_window=120,
              secrets=[modal.Secret.from_name("demo-config")])
@modal.fastapi_endpoint(method="POST")
def embed(item: dict):
    seq = item["sequence"][:1024]          # hard cap. Always.
    ...
    return {"embedding": vec, "model": "esm2-t12", "version": "2026.08"}
```

Contract rules:

- **Stateless.** Every request carries everything it needs. No sessions.
- **Versioned response.** Include the model version so a cached client can detect staleness.
- **Hard input caps** enforced server-side, not just in the UI.
- **Explicit timeout**, and a client that shows something useful when it fires.

### 4.3 Guardrails — do not skip these

A public endpoint that costs money per invocation is a standing invitation.

- [ ] **CORS allowlist** to your domain only. Not `*`.
- [ ] **Rate limit per IP** — 10/min is generous for a demo.
- [ ] **Hard spend cap** on the platform, with an email alert well below it.
- [ ] **Input size caps** enforced server-side.
- [ ] **Request timeout** on both ends.
- [ ] **No secrets in client code.** If a demo needs a key, it goes in a Cloudflare Worker that proxies and rate-limits; the key never reaches the browser.
- [ ] **Log nothing user-submitted.** No retention, and say so in the demo footer.
- [ ] **Cold start is a UX problem, not an infrastructure one.** Scale-to-zero means the first request after idle takes seconds. Show a real progress state that says the model is starting, not a spinner.

---

## 5. The demo contract

The point of this section: **make the second demo cheap to add.** Without a shared interface you'll hand-build each one, and after two you'll stop.

### 5.1 Structure

```
src/demos/<slug>/
├── demo.json         # manifest
├── Island.tsx        # the UI; default-exports a component
├── worker.ts         # inference, off the main thread (Tier 1 only)
└── fallback.svg      # static image shown if the runtime fails  ← required

public/demos/<slug>/  # weights, wasm, precomputed data
public/demos/_ort/    # shared ONNX Runtime wasm
public/demos/_rdkit/  # shared RDKit wasm
```

### 5.2 Manifest

```json
{
  "slug": "molgen",
  "title": "Generate small molecules",
  "tier": 1,
  "blurb": "Sample from a transformer trained on drug-like molecules, then filter by property.",
  "downloadMB": 24,
  "requires": ["wasm"],
  "fallback": "A grid of 5,000 pre-generated molecules.",
  "source": "https://github.com/patrickwrowe/Language-Models-for-AI-Molecule-Generation"
}
```

### 5.3 `Demo.astro`

One component, used identically from any project page or MDX post:

```mdx
<Demo slug="molgen" />
```

It renders, before anything loads, a plate-styled frame containing: the title, the blurb, the download size, a **Run demo** button, and a link to the source. Nothing heavy is fetched until the button is pressed.

On press it dynamically imports `Island.tsx`, shows real progress (bytes, not a spinner), and mounts. On failure it shows the fallback image and the source link, with a plain sentence saying what went wrong.

### 5.4 The rules

1. **Nothing loads until asked.** No demo asset is fetched on page view. Non-negotiable — it's what makes a 24MB demo compatible with the performance budget in doc 01 §8.
2. **Every demo has a static fallback.** A figure that conveys the same point. The page must still make its argument on a locked-down browser or a bad connection.
3. **Every demo shows its provenance.** Model, training data, and a link to the code, in the frame. This is a scientist's site; an unexplained black box undercuts it.
4. **Every demo is honest about failure.** If 30% of generated SMILES are invalid, show the validity rate. That's the interesting number, and hiding it is the tell of a demo built to impress rather than inform.
5. **Demos degrade, they don't block.** No demo is load-bearing for understanding a page.

---

## 6. Worked example — the molecular generation demo

Your `Language-Models-for-AI-Molecule-Generation` repo, end to end. This is the Phase 5 deliverable and the flagship interactive piece.

### 6.1 What the visitor does

1. Lands on `/work/molecular-generation/`, reads the case study, sees a static grid of generated molecules.
2. Presses **Run demo** (frame says: 24 MB, runs entirely in your browser).
3. Progress bar; model and RDKit load.
4. Adjusts **temperature** and optionally types a **scaffold prefix**.
5. Presses **Generate 12**.
6. A grid appears: each molecule drawn as 2D SVG, captioned with MW, logP and ring count — typeset with the same mono labels as the wall labels elsewhere on the site, so the demo looks like part of the site rather than an embedded app.
7. Above the grid: **"14 of 20 samples were valid SMILES (70%)"**. This is the honest number and the most scientifically interesting thing on the page.
8. Sliders filter the grid by property.

### 6.2 The pipeline

```
PyTorch checkpoint
   │  scripts/export_onnx.py
   ├─► molgen.onnx ──► quantize_dynamic ──► molgen.int8.onnx   (~6 MB)
   │
   │  scripts/export_tokenizer.py
   ├─► tokenizer.json                                          (~4 KB)
   │
   │  scripts/presample.py   ← Tier 0 fallback AND initial view
   └─► sample.json: 5,000 molecules + descriptors               (~1.5 MB)

Browser:
   worker.ts    onnxruntime-web  → logits → temperature/top-k → token
   Island.tsx   RDKit.js         → validate, canonicalise, descriptors, SVG
```

### 6.3 Build order

**Step 1 — ship Tier 0 first.** `presample.py` generates 5,000 molecules offline with descriptors. The island browses and filters them. This alone is a good demo, it ships in a day, and it becomes the fallback and the pre-run view for the live version. **Do not skip this step to get to the interesting one.**

**Step 2 — export and verify.** ONNX export, int8 quantisation, parity check against PyTorch in Python. Measure the actual file size.

**Step 3 — the worker.** Sampling loop in JS: forward pass, temperature scale, top-k, sample, append, repeat until stop token or max length. Post each completed SMILES back to the main thread as it finishes, so molecules appear progressively rather than in one batch at the end.

**Step 4 — RDKit integration.** Validate, canonicalise, deduplicate against the training set (ship a Bloom filter of training SMILES — a few hundred KB — so you can honestly label which outputs are novel), draw, describe. Free every mol.

**Step 5 — the UI.** Deliberately plain: two controls, a button, a grid. The design work is in the typography of the molecule captions, not in the chrome.

### 6.4 Details that will bite

- **Invalid SMILES are normal.** Character-level models produce plenty. Don't filter silently — count them and display the rate.
- **Duplicates and memorisation.** Some samples will be training molecules. The Bloom filter makes novelty a number you can show rather than a claim you have to make.
- **`mol.delete()`.** Again. Generating 20 molecules and forgetting is survivable; a user pressing Generate forty times is not.
- **Mobile memory.** A 24MB WASM working set is tight on older phones. Feature-detect, and show the Tier 0 browser instead of crashing.
- **Cold cache on first visit is the entire cost.** Immutable cache headers (doc 01 §7) mean it's paid once.

---

## 7. Summary table

| | Tier 0 | Tier 1 | Tier 2 |
|---|---|---|---|
| Where it runs | Nowhere — precomputed | Browser (WASM/WebGPU) | Hosted Python |
| Marginal cost | £0 | £0 | Per request |
| First load | Instant | 5–30 MB, once | Instant |
| Latency | None | Fast after load | Cold start seconds |
| Breaks when | Never | Browser is very old | Free tier changes, quota hit, service down |
| Effort | Low | Medium | Medium + ongoing ops |
| Use for | Most demos | Small models, chemistry | GPU, big weights, private data |

**Default to Tier 0. Reach for Tier 1 when interaction genuinely requires computation. Justify Tier 2 individually, every time.**

---

## 8. First three demos, recommended

1. **Molecule browser (Tier 0).** 5,000 pre-generated molecules, filterable by property, each drawn with RDKit.js. Ships in a day. Doubles as the fallback for demo 2.
2. **Live molecular generation (Tier 1).** §6. The flagship.
3. **Interatomic potential explorer (Tier 0).** A precomputed energy surface for a carbon system — drag two atoms, read the GAP-20 energy off a precomputed grid, with the DFT reference overlaid. It makes a fifteen-year-old idea legible in about four seconds, and no one else has it.

That third one is the most distinctive thing on this list. It's the demo that comes from your actual research rather than from a general ML repertoire, and it fits Tier 0 exactly.
