// Neighbour searches over plots, the engine's plot_neighbours.cpp (RESOLVE
// 1.1.0): by species composition (cosine similarity of presence sets) through
// nearest-neighbour descent, by position through an exact k-d tree, each with
// per-plot hiding rules.
//
// The descent (Dong, Charikar & Li 2011, WWW) compares each plot with its
// neighbours' neighbours, round after round, until the lists settle; its
// cost grows close to linearly in the number of plots. Each round reads the
// lists as they stood when it began, a plot's list is the top k of every plot
// proposed to it, and samples are drawn by hashing, so the lists do not
// depend on the order plots are processed in: on one thread or on many, the
// page returns the engine's lists exactly.
//
// Every step is a kernel over a range of rows (KERNELS). A RowPool runs a
// kernel over all rows, on the calling thread or shared out among workers
// (neighbours_worker.js) that read and write the same SharedArrayBuffers; a
// list is changed under its own lock, as in the engine.

import { greatCircleKm } from "./geo.js";

const EARTH_RADIUS_KM = 6371;
const RAD = Math.PI / 180;
const BLOCK = 256;

// An array the pool's workers can share, where the page may share memory.
export function allocate(Type, length) {
  const bytes = length * Type.BYTES_PER_ELEMENT;
  return new Type(typeof SharedArrayBuffer === "undefined" ? new ArrayBuffer(bytes) : new SharedArrayBuffer(bytes));
}
const shared = (Type, values) => { const out = allocate(Type, values.length); out.set(values); return out; };

// murmur3's finaliser and a hash of three words built from it.
function fmix32(h) {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}
const hash3 = (a, b, c) => fmix32((a ^ fmix32((b ^ fmix32((c ^ 0x9e3779b9) >>> 0)) >>> 0)) >>> 0);

const SALT_START = 0, SALT_NEW = 1, SALT_REVERSE_NEW = 2, SALT_REVERSE_OLD = 3;
const salt = (what, round) => (round * 4 + what) >>> 0;

function lock(locks, i) { while (Atomics.compareExchange(locks, i, 0, 1) !== 0); }
const unlock = (locks, i) => Atomics.store(locks, i, 0);

// Sorts v[0, length) ascending and drops repeats; returns the new length.
function sortUnique(v, length) {
  v.subarray(0, length).sort();
  let w = 0;
  for (let i = 0; i < length; i++) if (w === 0 || v[i] !== v[w - 1]) v[w++] = v[i];
  return w;
}

// Whether `value` is among the sorted v[begin, begin + length).
function sortedHas(v, begin, length, value) {
  let lo = begin, hi = begin + length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (v[mid] < value) lo = mid + 1; else hi = mid; }
  return lo < begin + length && v[lo] === value;
}

// ---------------------------------------------------------------------------
// Presence sets
// ---------------------------------------------------------------------------

// The species of each plot as a set of ranks, rarest first among these plots
// (ties by species id). `species`: per plot its species ids.
export class PresenceSets {
  constructor(species) {
    const n = species.length;
    let width = 0;
    for (const row of species) for (const s of row) width = Math.max(width, s + 1);
    const frequency = new Float64Array(width);
    const rows = species.map((row) => Array.from(new Set(row)));
    for (const row of rows) for (const s of row) frequency[s]++;
    const byRarity = Array.from({ length: width }, (_, i) => i).sort((a, b) => frequency[a] - frequency[b] || a - b);
    const rankOf = new Int32Array(width);
    byRarity.forEach((s, r) => { rankOf[s] = r; });
    this.offsets = allocate(Int32Array, n + 1);
    this.sizes = allocate(Int32Array, n);
    rows.forEach((row, i) => { this.offsets[i + 1] = this.offsets[i] + row.length; this.sizes[i] = row.length; });
    this.ranks = allocate(Int32Array, this.offsets[n]);
    rows.forEach((row, i) => {
      const r = Int32Array.from(row, (s) => rankOf[s]).sort();
      this.ranks.set(r, this.offsets[i]);
    });
    this.postingOffsets = allocate(Int32Array, width + 1);
    for (const r of this.ranks) this.postingOffsets[r + 1]++;
    for (let i = 0; i < width; i++) this.postingOffsets[i + 1] += this.postingOffsets[i];
    this.postings = allocate(Int32Array, this.ranks.length);
    const fill = this.postingOffsets.slice(0, width);
    for (let i = 0; i < n; i++) {
      for (let e = this.offsets[i]; e < this.offsets[i + 1]; e++) this.postings[fill[this.ranks[e]]++] = i;
    }
  }

  get size() { return this.sizes.length; }
  count(plot) { return this.sizes[plot]; }
  overlap(a, b) { return overlap(this, a, b); }
  cosine(a, b) { return cosine(this, a, b); }
}

// Sets reach a worker as plain objects, so the measures read their arrays.
function overlap(sets, a, b) {
  const r = sets.ranks;
  let x = sets.offsets[a], y = sets.offsets[b], same = 0;
  const xEnd = sets.offsets[a + 1], yEnd = sets.offsets[b + 1];
  while (x < xEnd && y < yEnd) {
    if (r[x] < r[y]) x++;
    else if (r[y] < r[x]) y++;
    else { same++; x++; y++; }
  }
  return same;
}
const similarity = (sets, a, b, same) => same / Math.sqrt(sets.sizes[a] * sets.sizes[b]);
function cosine(sets, a, b) {
  const same = overlap(sets, a, b);
  return same === 0 ? 0 : similarity(sets, a, b, same);
}

// The species `plot` shares with every plot of [0, nCandidates) other than
// itself that shares any: [candidate, overlap] pairs by increasing candidate,
// walked through the species' postings.
export function overlaps(sets, plot, nCandidates) {
  const counts = new Int32Array(nCandidates), touched = [];
  for (let e = sets.offsets[plot]; e < sets.offsets[plot + 1]; e++) {
    const s = sets.ranks[e];
    for (let p = sets.postingOffsets[s]; p < sets.postingOffsets[s + 1]; p++) {
      const other = sets.postings[p];
      if (other >= nCandidates || other === plot) continue;
      if (counts[other]++ === 0) touched.push(other);
    }
  }
  touched.sort((a, b) => a - b);
  return touched.map((other) => [other, counts[other]]);
}

// ---------------------------------------------------------------------------
// Fold rules
// ---------------------------------------------------------------------------

// Plot v is hidden from plot u when it lies in u's own fold or in the fold
// `held` (-1 for none), or within u's buffer: nearer than bufferKm[u] km or
// fewer than bufferYears[u] survey years apart (NaN: no buffer). `folds` and
// the buffers hold one value per plot the rules apply to; `coordinates`
// (latitude / longitude pairs) and `years` are those plots' own.
export class FoldRules {
  constructor(folds, held = -1, { bufferKm = null, bufferYears = null, coordinates = null, years = null } = {}) {
    this.folds = shared(Int32Array, folds);
    this.held = held;
    this.bufferKm = bufferKm ? shared(Float64Array, bufferKm) : null;
    this.bufferYears = bufferYears ? shared(Float64Array, bufferYears) : null;
    this.coordinates = bufferKm ? shared(Float64Array, coordinates) : null;
    this.years = bufferYears ? shared(Float64Array, years) : null;
  }
}

// Whether `rules` hide plot v from plot u; a search passes the km it computed
// itself, NaN when it did not.
export function foldHiding(rules) {
  const { folds, held, bufferKm, bufferYears, coordinates: c, years } = rules;
  return (u, v, _similarity, km) => {
    const f = folds[v];
    if (f === folds[u] || f === held) return true;
    if (bufferKm && !Number.isNaN(bufferKm[u])) {
      if (km === undefined || Number.isNaN(km)) km = greatCircleKm(c[2 * u], c[2 * u + 1], c[2 * v], c[2 * v + 1]);
      if (km < bufferKm[u]) return true;
    }
    return !!bufferYears && Math.abs(years[u] - years[v]) < bufferYears[u];
  };
}

// ---------------------------------------------------------------------------
// Running kernels
// ---------------------------------------------------------------------------

// This thread's scratch arrays, kept between jobs.
let scratch = { width: 0, k: 0 };
function scratchFor(width, k) {
  if (scratch.width < width || scratch.k < k) {
    width = Math.max(width, scratch.width);
    k = Math.max(k, scratch.k);
    scratch = { width, k, marks: new Uint8Array(width), a: new Int32Array(4 * k + 8), b: new Int32Array(4 * k + 8),
                keys: new Uint32Array(4 * k + 8), km: new Float64Array(k), rows: new Int32Array(k) };
  }
  return scratch;
}

function runRows(name, state, nRows, args, counter) {
  const kernel = KERNELS[name];
  const sets = state.sets ?? args.sets ?? null;
  const rules = state.rules ?? args.rules ?? null;
  const ctx = { hides: rules ? foldHiding(rules) : null,
                scratch: scratchFor(sets ? sets.postingOffsets.length - 1 : 0, state.k ?? args.k ?? 1) };
  let total = 0;
  for (;;) {
    const lo = Atomics.add(counter, 0, BLOCK);
    if (lo >= nRows) return total;
    total += kernel(state, lo, Math.min(nRows, lo + BLOCK), args, ctx);
  }
}

// One job of a worker: blocks of rows taken from a shared counter until none
// are left.
export function runJob({ id, kernel, state, nRows, args, counter }) {
  try {
    return { id, total: runRows(kernel, state, nRows, args, counter) };
  } catch (e) {
    return { id, error: e.message };
  }
}

// Runs a kernel over rows [0, nRows) and returns the sum of what it returns:
// on this thread without workers or shared memory, else on every worker.
// `workers`: objects with postMessage and an onmessage property, as a Worker.
export class RowPool {
  constructor(workers = []) {
    this.workers = typeof SharedArrayBuffer === "undefined" ? [] : workers;
    this.pending = new Map();
    this.jobs = 0;
    for (const w of this.workers) {
      w.onmessage = ({ data }) => {
        const job = this.pending.get(data.id);
        this.pending.delete(data.id);
        if (data.error) job.reject(new Error(data.error)); else job.resolve(data.total);
      };
    }
  }

  get threads() { return Math.max(1, this.workers.length); }

  async run(kernel, state, nRows, args = {}) {
    if (nRows <= 0) return 0;
    if (!this.workers.length) return runRows(kernel, state, nRows, args, new Int32Array(1));
    const counter = allocate(Int32Array, 1);
    const totals = await Promise.all(this.workers.map((w) => new Promise((resolve, reject) => {
      const id = ++this.jobs;
      this.pending.set(id, { resolve, reject });
      w.postMessage({ id, kernel, state, nRows, args, counter });
    })));
    return totals.reduce((a, b) => a + b, 0);
  }

  terminate() { for (const w of this.workers) w.terminate(); }
}

// ---------------------------------------------------------------------------
// Neighbour descent
// ---------------------------------------------------------------------------

export const SET_GRAPH_DEFAULTS = { k: 50, sampleRate: 0.5, stopFraction: 0.001, maxRounds: 30 };

// Slot u holds plot u's open list, slot n + u its hidden list (u < nHidden).
// An entry is a candidate plot and the number of species it shares with the
// slot's plot; similarity is ranked exactly by overlap^2 / count.
const plotOf = (st, slot) => (slot < st.n ? slot : slot - st.n);

// Whether entry a (plot pa, overlap oa) precedes entry b in one slot.
function better(sizes, pa, oa, pb, ob) {
  const lhs = oa * oa * sizes[pb], rhs = ob * ob * sizes[pa];
  return lhs > rhs || (lhs === rhs && pa < pb);
}

// The slot keeps the entry if it ranks within the top k, is not there yet
// and, given a rule, is not hidden from plot u by it; the rank is checked
// first, as most proposals fall below a full list's last entry.
function insert(st, slot, plot, same, u, hides) {
  const { k, plots, overlaps, counts, locks } = st, sizes = st.sets.sizes;
  const base = slot * k;
  lock(locks, slot);
  const c = counts[slot];
  let keep = c < k || better(sizes, plot, same, plots[base + c - 1], overlaps[base + c - 1]);
  for (let m = 0; keep && m < c; m++) keep = plots[base + m] !== plot;
  if (keep && hides) keep = !hides(u, plot, similarity(st.sets, u, plot, same), NaN);
  if (keep) {
    let pos = c < k ? c : k - 1;
    while (pos > 0 && better(sizes, plot, same, plots[base + pos - 1], overlaps[base + pos - 1])) {
      plots[base + pos] = plots[base + pos - 1];
      overlaps[base + pos] = overlaps[base + pos - 1];
      pos--;
    }
    plots[base + pos] = plot;
    overlaps[base + pos] = same;
    if (c < k) counts[slot] = c + 1;
  }
  unlock(locks, slot);
}

// Plot v proposed to plot u, sharing `same` species: to u's open list, and to
// its hidden list unless the rules hide v from u.
function propose(st, hides, u, v, same) {
  if (u === v || v >= st.nCandidates || same === 0) return;
  insert(st, u, v, same, u, null);
  if (u < st.nHidden) insert(st, st.n + u, v, same, u, hides);
}

// Whether owner a precedes owner b in plot v's reverse sample: smaller
// (hash, owner) first.
function reverseBefore(v, s, a, b) {
  const ha = hash3(v, a, s), hb = hash3(v, b, s);
  return ha < hb || (ha === hb && a < b);
}

// Plot v's reverse list keeps the `sample` owners that come first, the
// hashed sample of all owners offered to it.
function reverseInsert(st, list, counts, v, owner, s) {
  const sample = st.sample, base = v * sample;
  lock(st.locks, v);
  const c = counts[v];
  let keep = c < sample || reverseBefore(v, s, owner, list[base + c - 1]);
  for (let m = 0; keep && m < c; m++) keep = list[base + m] !== owner;
  if (keep) {
    let pos = c < sample ? c : sample - 1;
    while (pos > 0 && reverseBefore(v, s, owner, list[base + pos - 1])) { list[base + pos] = list[base + pos - 1]; pos--; }
    list[base + pos] = owner;
    if (c < sample) counts[v] = c + 1;
  }
  unlock(st.locks, v);
}

export const KERNELS = {
  // Initial lists: candidates sharing the plot's rarest species, walked from
  // a hashed start through each species' candidate postings.
  start(st, lo, hi, args, ctx) {
    const { k, nCandidates, sets } = st, { offsets, ranks, postingOffsets, postings } = sets;
    for (let q = lo; q < hi; q++) {
      let taken = 0;
      for (let j = offsets[q]; j < offsets[q + 1] && taken < 2 * k; j++) {
        const s = ranks[j];
        const begin = postingOffsets[s];
        let a = begin, b = postingOffsets[s + 1];
        while (a < b) { const mid = (a + b) >>> 1; if (postings[mid] < nCandidates) a = mid + 1; else b = mid; }
        const length = a - begin;
        if (length === 0) continue;
        const first = hash3(q, s, salt(SALT_START, 0)) % length;
        for (let m = 0; m < Math.min(length, k) && taken < 2 * k; m++) {
          const v = postings[begin + ((first + m) % length)];
          if (v === q) continue;
          propose(st, ctx.hides, q, v, overlap(sets, q, v));
          taken++;
        }
      }
    }
    return 0;
  },

  // Each slot's new entries, a hashed sample of `sample`, sorted by plot.
  sampleNew(st, lo, hi, args, ctx) {
    const { k, sample, startPlots, startCounts, startFresh, newPlots, newCounts } = st;
    const { a: fresh, keys } = ctx.scratch;
    for (let slot = lo; slot < hi; slot++) {
      let c = 0;
      for (let m = 0; m < startCounts[slot]; m++) if (startFresh[slot * k + m]) fresh[c++] = startPlots[slot * k + m];
      if (c > sample) {
        const owner = plotOf(st, slot), s = salt(SALT_NEW, st.round);
        for (let m = 0; m < c; m++) keys[m] = hash3(owner, fresh[m], s);
        for (let m = 1; m < c; m++) {
          const x = fresh[m], h = keys[m];
          let j = m;
          while (j > 0 && (keys[j - 1] > h || (keys[j - 1] === h && fresh[j - 1] > x))) {
            keys[j] = keys[j - 1]; fresh[j] = fresh[j - 1]; j--;
          }
          keys[j] = h; fresh[j] = x;
        }
        c = sample;
      }
      fresh.subarray(0, c).sort();
      newPlots.set(fresh.subarray(0, c), slot * sample);
      newCounts[slot] = c;
    }
    return 0;
  },

  // Reverse entries: each slot's owner offered to the plots its new and old
  // entries name.
  reverse(st, lo, hi) {
    const { k, sample, startPlots, startCounts, startFresh, newPlots, newCounts } = st;
    const sNew = salt(SALT_REVERSE_NEW, st.round), sOld = salt(SALT_REVERSE_OLD, st.round);
    for (let slot = lo; slot < hi; slot++) {
      const owner = plotOf(st, slot);
      for (let m = 0; m < newCounts[slot]; m++) {
        reverseInsert(st, st.reverseNew, st.reverseNewCounts, newPlots[slot * sample + m], owner, sNew);
      }
      for (let m = 0; m < startCounts[slot]; m++) {
        if (!startFresh[slot * k + m]) reverseInsert(st, st.reverseOld, st.reverseOldCounts, startPlots[slot * k + m], owner, sOld);
      }
    }
    return 0;
  },

  // Local join: every pair of a plot's new entries, and each new entry with
  // each old one, proposed to each other.
  join(st, lo, hi, args, ctx) {
    const { n, k, sample, nHidden, nCandidates, sets, startPlots, startCounts, startFresh, newPlots, newCounts } = st;
    const { offsets, ranks } = sets;
    const { marks, a: fresh, b: old } = ctx.scratch;
    const hides = ctx.hides;
    const join = (u, v) => {
      if (u >= nCandidates && v >= nCandidates) return;
      let same = 0;
      for (let e = offsets[v]; e < offsets[v + 1]; e++) same += marks[ranks[e]];
      propose(st, hides, u, v, same);
      propose(st, hides, v, u, same);
    };
    for (let i = lo; i < hi; i++) {
      let nf = 0, no = 0;
      for (let slot = i; slot === i || (slot === n + i && i < nHidden); slot += n) {
        for (let m = 0; m < newCounts[slot]; m++) fresh[nf++] = newPlots[slot * sample + m];
        for (let m = 0; m < startCounts[slot]; m++) if (!startFresh[slot * k + m]) old[no++] = startPlots[slot * k + m];
      }
      for (let m = 0; m < st.reverseNewCounts[i]; m++) fresh[nf++] = st.reverseNew[i * sample + m];
      for (let m = 0; m < st.reverseOldCounts[i]; m++) old[no++] = st.reverseOld[i * sample + m];
      nf = sortUnique(fresh, nf);
      no = sortUnique(old, no);
      let w = 0;
      for (let m = 0, f = 0; m < no; m++) {
        while (f < nf && fresh[f] < old[m]) f++;
        if (f >= nf || fresh[f] !== old[m]) old[w++] = old[m];
      }
      no = w;
      // The species of u marked once, each partner's shared species counted
      // against the marks.
      for (let x = 0; x < nf; x++) {
        const u = fresh[x];
        for (let e = offsets[u]; e < offsets[u + 1]; e++) marks[ranks[e]] = 1;
        for (let y = x + 1; y < nf; y++) join(u, fresh[y]);
        for (let y = 0; y < no; y++) join(u, old[y]);
        for (let e = offsets[u]; e < offsets[u + 1]; e++) marks[ranks[e]] = 0;
      }
    }
    return 0;
  },

  // An entry is new when the round brought it, or it was new and not joined;
  // returns the count of entries the round brought.
  flags(st, lo, hi) {
    const { k, sample, plots, counts, fresh, startPlots, startCounts, startFresh, newPlots, newCounts } = st;
    let brought = 0;
    for (let slot = lo; slot < hi; slot++) {
      const base = slot * k;
      for (let m = 0; m < counts[slot]; m++) {
        const plot = plots[base + m];
        let was = -1;
        for (let t = 0; t < startCounts[slot]; t++) if (startPlots[base + t] === plot) { was = t; break; }
        const isNew = was < 0 || (startFresh[base + was] === 1 && !sortedHas(newPlots, slot * sample, newCounts[slot], plot));
        fresh[base + m] = isNew ? 1 : 0;
        if (was < 0) brought++;
      }
    }
    return brought;
  },

  // A hidden list starts as the open list without the plots the rules hide.
  seedHidden(st, lo, hi, args, ctx) {
    const { n, k, plots, overlaps, counts, fresh, sets } = st;
    for (let u = lo; u < hi; u++) {
      let c = 0;
      const base = (n + u) * k;
      for (let m = 0; m < counts[u]; m++) {
        const plot = plots[u * k + m], same = overlaps[u * k + m];
        if (ctx.hides(u, plot, similarity(sets, u, plot, same), NaN)) continue;
        plots[base + c] = plot;
        overlaps[base + c] = same;
        fresh[base + c] = 1;
        c++;
      }
      counts[n + u] = c;
    }
    return 0;
  },

  // Rows of lists; a slot short of k candidates sharing a species is filled
  // with the lowest-row candidates sharing none, at similarity 0.
  lists(st, lo, hi, { firstSlot, indices, similarities }, ctx) {
    const { n, k, nCandidates, plots, overlaps, counts, sets } = st;
    for (let row = lo; row < hi; row++) {
      const slot = firstSlot + row, plot = plotOf(st, slot);
      let c = counts[slot];
      for (let m = 0; m < c; m++) {
        indices[row * k + m] = plots[slot * k + m];
        similarities[row * k + m] = similarity(sets, plot, plots[slot * k + m], overlaps[slot * k + m]);
      }
      for (let v = 0; v < nCandidates && c < k; v++) {
        if (v === plot || overlap(sets, plot, v) > 0) continue;
        if (slot >= n && ctx.hides(plot, v, 0, NaN)) continue;
        indices[row * k + c] = v;
        similarities[row * k + c] = 0;
        c++;
      }
      for (; c < k; c++) { indices[row * k + c] = -1; similarities[row * k + c] = NaN; }
    }
    return 0;
  },

  nearest: (index, lo, hi, args, ctx) => placeNearest(index, lo, hi, args, ctx),
};

// Neighbour lists of every plot of a PresenceSets among its first
// `nCandidates` plots, built by SetNeighbourGraph.build. A plot is never its
// own neighbour; ties go to the lower row; a plot with fewer than k
// candidates sharing a species lists the lowest-row candidates sharing none
// after them, at similarity 0.
export class SetNeighbourGraph {
  constructor(sets, nCandidates, config, pool) {
    this.config = { ...SET_GRAPH_DEFAULTS, ...config };
    this.pool = pool;
    this.rounds = 0;
    const k = this.config.k;
    this.st = { sets, n: sets.size, k, nCandidates, nHidden: 0, sample: Math.max(1, Math.trunc(this.config.sampleRate * k)),
                round: 0, rules: null, plots: null, overlaps: null, fresh: null, counts: null, locks: null };
    this.reserve(0);
  }

  static async build(sets, nCandidates, config = {}, pool = new RowPool()) {
    const graph = new SetNeighbourGraph(sets, nCandidates, config, pool);
    await pool.run("start", graph.st, graph.st.n);
    graph.st.fresh.fill(1);
    await graph.descend();
    return graph;
  }

  // Room for the open lists and `hiddenPlots` hidden ones, the open kept.
  reserve(hiddenPlots) {
    const st = this.st, slots = st.n + hiddenPlots;
    if (st.counts && st.counts.length >= slots) return;
    const grow = (old, Type, width) => {
      const out = allocate(Type, slots * width);
      if (old) out.set(old.subarray(0, st.n * width));
      return out;
    };
    st.plots = grow(st.plots, Int32Array, st.k);
    st.overlaps = grow(st.overlaps, Int32Array, st.k);
    st.fresh = grow(st.fresh, Uint8Array, st.k);
    st.counts = grow(st.counts, Int32Array, 1);
    st.locks = allocate(Int32Array, slots);
  }

  async descend() {
    const st = this.st, k = st.k, n = st.n, pool = this.pool;
    const slots = n + st.nHidden;
    const round = {
      startPlots: allocate(Int32Array, slots * k), startCounts: allocate(Int32Array, slots),
      startFresh: allocate(Uint8Array, slots * k),
      newPlots: allocate(Int32Array, slots * st.sample), newCounts: allocate(Int32Array, slots),
      reverseNew: allocate(Int32Array, n * st.sample), reverseNewCounts: allocate(Int32Array, n),
      reverseOld: allocate(Int32Array, n * st.sample), reverseOldCounts: allocate(Int32Array, n),
    };
    Object.assign(st, round);
    for (let r = 0; r < this.config.maxRounds; r++, this.rounds++) {
      st.round = this.rounds;
      st.startPlots.set(st.plots.subarray(0, slots * k));
      st.startCounts.set(st.counts.subarray(0, slots));
      st.startFresh.set(st.fresh.subarray(0, slots * k));
      st.reverseNewCounts.fill(0);
      st.reverseOldCounts.fill(0);
      await pool.run("sampleNew", st, slots);
      await pool.run("reverse", st, slots);
      await pool.run("join", st, n);
      const changed = await pool.run("flags", st, slots);
      if (changed < this.config.stopFraction * slots * k) { this.rounds++; break; }
    }
    for (const name of Object.keys(round)) st[name] = null;
  }

  // Slots [firstSlot, firstSlot + nRows) as { indices: Int32Array,
  // similarities: Float64Array, k }, (nRows, k) row-major, -1 and NaN after
  // the last.
  async lists(firstSlot, nRows) {
    const k = this.st.k;
    const indices = allocate(Int32Array, nRows * k), similarities = allocate(Float64Array, nRows * k);
    await this.pool.run("lists", this.st, nRows, { firstSlot, indices, similarities });
    return { indices, similarities, k };
  }

  // The open lists of plots [first, first + count).
  open(first = 0, count = this.st.n - first) { return this.lists(first, count); }

  // The neighbours of plots [0, nQueries) that `rules` (FoldRules) leave
  // visible, by continuing the descent beside the open lists.
  async hidden(nQueries, rules) {
    const st = this.st;
    this.reserve(nQueries);
    st.nHidden = nQueries;
    st.rules = rules;
    await this.pool.run("seedHidden", st, nQueries);
    await this.descend();
    const out = await this.lists(st.n, nQueries);
    st.rules = null;
    st.nHidden = 0;
    return out;
  }
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

const LEAF_PLACES = 16;
const BOUND_SLACK_KM = 1e-7;
const chordToKm = (chord) => 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, chord / 2));

// Exact nearest positions among the first `nCandidates` rows of a Float64Array
// of latitude / longitude pairs, by the haversine distance; a row with a NaN
// coordinate is never a neighbour and has none.
export class PlaceIndex {
  constructor(coordinates, nCandidates, pool = new RowPool()) {
    const n = coordinates.length / 2;
    this.pool = pool;
    const lat = allocate(Float64Array, n), lon = allocate(Float64Array, n), xyz = allocate(Float64Array, 3 * n);
    const order = [];
    for (let i = 0; i < n; i++) {
      const la = coordinates[2 * i], lo = coordinates[2 * i + 1];
      lat[i] = la; lon[i] = lo;
      const phi = la * RAD, lambda = lo * RAD;
      xyz[3 * i] = Math.cos(phi) * Math.cos(lambda);
      xyz[3 * i + 1] = Math.cos(phi) * Math.sin(lambda);
      xyz[3 * i + 2] = Math.sin(phi);
      if (i < nCandidates && Number.isFinite(la) && Number.isFinite(lo)) order.push(i);
    }
    this.index = { lat, lon, xyz, order: shared(Int32Array, order) };
    const nodes = [];
    if (order.length) this.build(nodes, 0, order.length);
    const m = nodes.length, index = this.index;
    Object.assign(index, { nodes: m, lo: allocate(Float64Array, 3 * m), hi: allocate(Float64Array, 3 * m),
                           begin: allocate(Int32Array, m), end: allocate(Int32Array, m),
                           left: allocate(Int32Array, m), right: allocate(Int32Array, m) });
    nodes.forEach((node, id) => {
      index.lo.set(node.lo, 3 * id);
      index.hi.set(node.hi, 3 * id);
      index.begin[id] = node.begin; index.end[id] = node.end;
      index.left[id] = node.left; index.right[id] = node.right;
    });
  }

  build(nodes, begin, end) {
    const { xyz, order } = this.index;
    const node = { lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity], begin, end, left: -1, right: -1 };
    for (let i = begin; i < end; i++) {
      for (let d = 0; d < 3; d++) {
        const x = xyz[3 * order[i] + d];
        if (x < node.lo[d]) node.lo[d] = x;
        if (x > node.hi[d]) node.hi[d] = x;
      }
    }
    const id = nodes.length;
    nodes.push(node);
    if (end - begin > LEAF_PLACES) {
      let axis = 0;
      for (let d = 1; d < 3; d++) if (node.hi[d] - node.lo[d] > node.hi[axis] - node.lo[axis]) axis = d;
      const part = Array.from(order.subarray(begin, end));
      part.sort((p, q) => xyz[3 * p + axis] - xyz[3 * q + axis] || p - q);
      order.set(part, begin);
      const mid = begin + ((end - begin) >> 1);
      node.left = this.build(nodes, begin, mid);
      node.right = this.build(nodes, mid, end);
    }
    return id;
  }

  // For every row of `queries`, its k nearest candidates, nearest first, ties
  // to the lower row, never itself. Given `rules` (FoldRules over these rows),
  // the candidates they hide from a query are skipped.
  // Returns { indices: Int32Array, km: Float64Array, k }, (queries, k) row-major.
  async nearest(queries, k, rules = null) {
    const nQueries = queries.length;
    const indices = allocate(Int32Array, nQueries * k).fill(-1), km = allocate(Float64Array, nQueries * k).fill(NaN);
    if (k < 1 || !this.index.nodes) return { indices, km, k };
    await this.pool.run("nearest", this.index, nQueries, { queries: shared(Int32Array, queries), k, indices, km, rules });
    return { indices, km, k };
  }
}

function placeNearest(index, lo, hi, { queries, k, indices, km: kmOut, rules }, ctx) {
  const { lat, lon, xyz, order, lo: boxLo, hi: boxHi, left, right, begin, end } = index;
  const hides = ctx.hides;
  // A query's km buffer hides every candidate within it: boxes wholly inside
  // are passed over.
  const bufferKm = rules ? rules.bufferKm : null;
  let hiddenWithin = NaN;
  const foundKm = ctx.scratch.km, foundRow = ctx.scratch.rows;
  let q = 0, px = 0, py = 0, pz = 0, found = 0;
  const boxKm = (id, far) => {
    let s = 0;
    for (let d = 0; d < 3; d++) {
      const p = d === 0 ? px : d === 1 ? py : pz, a = boxLo[3 * id + d], b = boxHi[3 * id + d];
      const gap = far ? Math.max(Math.abs(p - a), Math.abs(p - b)) : p < a ? a - p : p > b ? p - b : 0;
      s += gap * gap;
    }
    return chordToKm(Math.sqrt(s));
  };
  const visit = (id) => {
    if (boxKm(id, false) - BOUND_SLACK_KM > (found === k ? foundKm[k - 1] : Infinity)) return;
    if (!Number.isNaN(hiddenWithin) && boxKm(id, true) + BOUND_SLACK_KM < hiddenWithin) return;
    if (left[id] < 0) {
      for (let i = begin[id]; i < end[id]; i++) {
        const v = order[i];
        if (v === q) continue;
        const f = greatCircleKm(lat[q], lon[q], lat[v], lon[v]);
        if (found === k && !(f < foundKm[k - 1] || (f === foundKm[k - 1] && v < foundRow[k - 1]))) continue;
        if (hides && hides(q, v, NaN, f)) continue;
        let j = found < k ? found++ : k - 1;
        while (j > 0 && (f < foundKm[j - 1] || (f === foundKm[j - 1] && v < foundRow[j - 1]))) {
          foundKm[j] = foundKm[j - 1]; foundRow[j] = foundRow[j - 1]; j--;
        }
        foundKm[j] = f; foundRow[j] = v;
      }
      return;
    }
    const leftFirst = boxKm(left[id], false) <= boxKm(right[id], false);
    visit(leftFirst ? left[id] : right[id]);
    visit(leftFirst ? right[id] : left[id]);
  };
  for (let qi = lo; qi < hi; qi++) {
    q = queries[qi];
    if (!Number.isFinite(lat[q]) || !Number.isFinite(lon[q])) continue;
    px = xyz[3 * q]; py = xyz[3 * q + 1]; pz = xyz[3 * q + 2];
    hiddenWithin = bufferKm ? bufferKm[q] : NaN;
    found = 0;
    visit(0);
    for (let m = 0; m < found; m++) { indices[qi * k + m] = foundRow[m]; kmOut[qi * k + m] = foundKm[m]; }
  }
  return 0;
}
