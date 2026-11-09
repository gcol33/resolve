// One inference worker of the page's pool: its own ONNX Runtime, one model at
// a time. A job carries the model's bytes, the settings predict() reads and,
// the first time this worker sees it, the run's encoding of the plots.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.bundle.min.mjs";
import { predict } from "./resolve.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1;

// Encodings of the current run only, by signature; a new run drops the others.
let run = null;
const encodings = new Map();

self.onmessage = async ({ data }) => {
  const { id, runId, sig, encoded, model, meta } = data;
  try {
    if (runId !== run) { run = runId; encodings.clear(); }
    if (encoded) encodings.set(sig, encoded);
    const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    const out = await predict(ort, session, meta, encodings.get(sig),
                              { onBatch: (fraction) => self.postMessage({ id, fraction }) });
    await session.release();
    self.postMessage({ id, out }, Object.values(out).map((a) => a.buffer));
  } catch (e) {
    self.postMessage({ id, error: e.message ?? String(e) });
  }
};
