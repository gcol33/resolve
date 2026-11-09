// Gap fitting off the page's thread: one fit per message, so a large dataset
// leaves the page responsive while the complete plots are searched. Where the
// page may share memory (cross-origin isolated), the searches run on
// `threads` workers of their own.

import { fitGaps } from "./gaps.js";
import { RowPool } from "./plot_neighbours.js";

let pool = null;
function poolOf(threads) {
  const size = typeof SharedArrayBuffer === "undefined" ? 0 : Math.max(0, threads | 0);
  if (pool && pool.workers.length === size) return pool;
  pool?.terminate();
  pool = new RowPool(Array.from({ length: size },
                                () => new Worker(new URL("neighbours_worker.js", import.meta.url), { type: "module" })));
  return pool;
}

self.onmessage = async ({ data: { id, target, raw, recorded, plots, threads } }) => {
  try {
    const { fitted, method, report } = await fitGaps(target, raw, recorded, plots, { pool: poolOf(threads) });
    self.postMessage({ id, fitted, method, report }, [fitted.buffer, method.buffer]);
  } catch (e) {
    self.postMessage({ id, error: e.message });
  }
};
