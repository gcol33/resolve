// Gap fitting, the engine's gap_fitting.cpp (RESOLVE 1.1.0), its searches in
// plot_neighbours.js and its geography in geo.js.
//
// A dataset usually records a variable (survey year, coordinates, altitude)
// on most of its plots and lacks it on some. The plots that carry it are
// labelled examples from the same surveys as the plots that lack it (the
// gaps). After prediction, gap fitting chooses on the complete plots how to
// combine the raw prediction with the dataset's own neighbours, and applies
// that choice to the gaps. The candidates, in the order a tie is broken:
//
//   local_constant         the complete plots' centre;
//   resolve                the raw prediction;
//   offset_slope           y = mean(y) + b (raw - mean(raw)), b by least squares;
//   knn_species            the median recorded value of the k complete plots
//                          most similar in species composition (Ochiai);
//   stack_species          raw and that median combined by least squares;
//   knn_spatial            the median over the k nearest complete plots;
//   stack_species_spatial  raw and both medians by least squares.
//
// The choice is judged by cross-validation over the complete plots, with folds
// laid out to hold plots out the way the gaps are held out (k-fold
// nearest-neighbour distance matching, Linnenbrink et al. 2024, Geoscientific
// Model Development): squares of the equal-area plane, survey-year periods,
// both, or plot by plot, at a range of sizes, the layout whose held-out plots
// lie from the other folds as the gaps lie from the complete plots (relative
// Wasserstein distance of the nearest-neighbour distances in species
// dissimilarity, km and years). Every fitted part is fitted on the other folds;
// the candidate with the lowest out-of-fold error is refitted on every
// complete plot and predicts the gaps, and the whole procedure, repeated with
// each fold held out, gives the error it is reported with (nested
// cross-validation). Gaps with and without coordinates choose apart.

import { geographicMedian, greatCircleKm, Laea } from "./geo.js";
import { FoldRules, PlaceIndex, PresenceSets, RowPool, SetNeighbourGraph, foldHiding, overlaps } from "./plot_neighbours.js";

export const GAP_METHODS = ["local_constant", "resolve", "offset_slope", "knn_species",
                            "stack_species", "knn_spatial", "stack_species_spatial"];
export const GAP_RECORDED = -1;
export const GAP_SEED = 20261106;
const K_GRID = [1, 3, 5, 10, 20, 50];
const TWO_PI = 2 * Math.PI;

const SPECIES = 0, SPATIAL = 1;
const KIND_NAMES = ["species", "spatial"];
const CANDIDATES = [
  { method: 0, combination: "constant", reads: [false, false] },
  { method: 1, combination: "raw", reads: [false, false] },
  { method: 2, combination: "least_squares", reads: [false, false] },
  { method: 3, combination: "neighbours", reads: [true, false] },
  { method: 4, combination: "least_squares", reads: [true, false] },
  { method: 5, combination: "neighbours", reads: [false, true] },
  { method: 6, combination: "least_squares", reads: [true, true] },
];

// Where a plot's calibrated value comes from: its own record or the fit.
export const gapSource = (code) => (code === GAP_RECORDED ? "recorded" : "imputed");

// The chosen method in words, with the k it chose per neighbour kind
// ([kind, k] pairs, as in the report) and the centre the scale takes.
export function describeGapMethod(method, k, scale) {
  const kOf = (kind) => k.find(([name]) => name === kind)?.[1];
  const similar = `the ${kOf("species")} plots most similar in species composition`;
  const nearest = `the ${kOf("spatial")} nearest plots`;
  const centre = { log: "median", geographic: "geographic median", circular: "mean direction" }[scale] ?? "mean";
  switch (GAP_METHODS[method]) {
    case "local_constant": return `the ${centre} of the plots recording it`;
    case "resolve": return "the model's value";
    case "offset_slope": return "the model's value rescaled to the plots recording it";
    case "knn_species": return `the median of ${similar}`;
    case "knn_spatial": return `the median of ${nearest}`;
    case "stack_species": return `the model's value combined with ${similar}`;
    case "stack_species_spatial":
      return `the model's value combined with the ${kOf("species")} most similar and the ${kOf("spatial")} nearest plots`;
    default: throw new Error(`gap fitting: unknown method ${method}`);
  }
}

// ---------------------------------------------------------------------------
// Small numerics
// ---------------------------------------------------------------------------

// Rows of d values held in one Float64Array.
const rowsOf = (n, d, fill = NaN) => ({ n, d, v: new Float64Array(n * d).fill(fill) });
const rowFinite = (m, i) => { for (let j = 0; j < m.d; j++) if (!Number.isFinite(m.v[i * m.d + j])) return false; return true; };
const pick = (m, index) => {
  const out = rowsOf(index.length, m.d);
  index.forEach((i, r) => { for (let j = 0; j < m.d; j++) out.v[r * m.d + j] = m.v[i * m.d + j]; });
  return out;
};

// numpy's median: the middle value, or the mean of the two middle values.
function median(values) {
  const v = Float64Array.from(values).sort();
  const n = v.length;
  if (n === 0) return NaN;
  return n % 2 === 1 ? v[(n - 1) / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// numpy's (and torch's) linear-interpolation quantile.
function quantile(sorted, q) {
  const pos = q * (sorted.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// The first `width` columns of (rows, k) neighbour lists.
function narrowed(lists, width) {
  const rows = lists.indices.length / lists.k, out = new Int32Array(rows * width);
  for (let r = 0; r < rows; r++) out.set(lists.indices.subarray(r * lists.k, r * lists.k + width), r * width);
  return out;
}

const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN);
const remainder = (x, p) => x - p * Math.floor(x / p);

// ---------------------------------------------------------------------------
// The plane values are fitted in
// ---------------------------------------------------------------------------

class Plane {
  constructor(target, completeValues) {
    this.scale = target.scale;
    this.period = target.period ?? 360;
    this.d = this.scale === "circular" || this.scale === "geographic" ? 2 : 1;
    if (this.scale === "geographic") {
      const lats = [], lons = [];
      for (let i = 0; i < completeValues.n; i++) { lats.push(completeValues.v[2 * i]); lons.push(completeValues.v[2 * i + 1]); }
      this.projection = new Laea(...geographicMedian(lats, lons));
    }
  }

  // Values in the target's units (rows of 1, or latitude / longitude) to plane rows.
  to(values) {
    const out = rowsOf(values.n, this.d);
    for (let i = 0; i < values.n; i++) {
      const v = values.v[i * values.d];
      if (this.scale === "linear") out.v[i] = v;
      else if (this.scale === "log") out.v[i] = Math.log1p(Math.max(v, 0));
      else if (this.scale === "circular") {
        const a = v * (TWO_PI / this.period);
        out.v[2 * i] = Math.sin(a); out.v[2 * i + 1] = Math.cos(a);
      } else {
        const [e, nn] = this.projection.forward(values.v[2 * i], values.v[2 * i + 1]);
        out.v[2 * i] = e / 1000; out.v[2 * i + 1] = nn / 1000;
      }
    }
    return out;
  }

  // One plane row back to the target's units: a number, or [lat, lon].
  backRow(m, i) {
    if (this.scale === "linear") return m.v[i];
    if (this.scale === "log") return Math.expm1(m.v[i]);
    if (this.scale === "circular") {
      const a = Math.atan2(m.v[2 * i], m.v[2 * i + 1]);
      return remainder(a * (this.period / TWO_PI), this.period);
    }
    if (!rowFinite(m, i)) return [NaN, NaN];
    return this.projection.inverse(m.v[2 * i] * 1000, m.v[2 * i + 1] * 1000);
  }

  // Each plot's error in the quantity the target's headline aggregates.
  errors(predicted, truth) {
    const out = new Float64Array(predicted.n);
    for (let i = 0; i < predicted.n; i++) {
      const p = this.backRow(predicted, i), t = this.backRow(truth, i);
      if (this.scale === "linear") out[i] = Math.abs(p - t);
      else if (this.scale === "log") out[i] = Math.abs(Math.log1p(Math.max(p, 0)) - Math.log1p(t));
      else if (this.scale === "circular") {
        const d = remainder(Math.abs(p - t), this.period);
        out[i] = Math.min(d, this.period - d);
      } else out[i] = greatCircleKm(p[0], p[1], t[0], t[1]);
    }
    return out;
  }

  // The headline over plots: the median km of a position, the mean otherwise.
  aggregate(errors) { return this.scale === "geographic" ? median(errors) : mean(Array.from(errors)); }

  // The local constant of some plane rows: their geographic median, the median
  // on a log scale, the mean otherwise (of sine and cosine for a bearing).
  centre(truth) {
    if (this.scale === "geographic") {
      const lats = [], lons = [];
      for (let i = 0; i < truth.n; i++) { const [la, lo] = this.backRow(truth, i); lats.push(la); lons.push(lo); }
      const [e, nn] = this.projection.forward(...geographicMedian(lats, lons));
      return Float64Array.of(e / 1000, nn / 1000);
    }
    const col = (j) => Array.from({ length: truth.n }, (_, i) => truth.v[i * truth.d + j]);
    if (this.scale === "log") return Float64Array.of(median(col(0)));
    return Float64Array.from({ length: this.d }, (_, j) => mean(col(j)));
  }
}

// ---------------------------------------------------------------------------
// Neighbour medians
// ---------------------------------------------------------------------------

// The first k neighbours' plane values summarised per row: the coordinate-wise
// median, or for a bearing their mean direction; NaN without any neighbour.
function neighbourMedian(values, neighbours, width, rows, k, circular) {
  const d = values.d, out = rowsOf(rows.length, d);
  rows.forEach((row, r) => {
    const found = [];
    for (let j = 0; j < k; j++) { const nb = neighbours[row * width + j]; if (nb >= 0) found.push(nb); }
    if (!found.length) return;
    if (circular) {
      const m = [0, 0];
      for (const nb of found) for (let c = 0; c < 2; c++) m[c] += values.v[nb * d + c];
      const norm = Math.max(Math.hypot(m[0] / found.length, m[1] / found.length), 1e-12);
      for (let c = 0; c < 2; c++) out.v[r * d + c] = m[c] / found.length / norm;
    } else {
      for (let c = 0; c < d; c++) out.v[r * d + c] = median(found.map((nb) => values.v[nb * d + c]));
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------

// Eigenvalues and eigenvectors of a small symmetric matrix (cyclic Jacobi).
function symmetricEigen(a, n) {
  const m = Float64Array.from(a), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p * n + q] ** 2;
    if (off < 1e-300) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = m[p * n + q];
        if (apq === 0) continue;
        const theta = (m[q * n + q] - m[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const mkp = m[k * n + p], mkq = m[k * n + q];
          m[k * n + p] = c * mkp - s * mkq; m[k * n + q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p * n + k], mqk = m[q * n + k];
          m[p * n + k] = c * mpk - s * mqk; m[q * n + k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: Array.from({ length: n }, (_, i) => m[i * n + i]), vectors: v };
}

// The minimum-norm least-squares solution of D c = t (LAPACK gelsd's answer):
// through the pseudo-inverse of the normal matrix, singular values below
// machine precision times the largest treated as zero. Real columns, or
// complex ones given as [re, im] pairs and solved in their real embedding.
function lstsq(columns, target, complex) {
  const p = columns.length, rows = target.length / (complex ? 2 : 1);
  const n = complex ? 2 * p : p;
  // The real design: a complex column z gives the real columns [Re, -Im; Im, Re].
  const design = [];
  if (!complex) for (const c of columns) design.push(c);
  else {
    for (const c of columns) design.push(Float64Array.from({ length: 2 * rows }, (_, i) => (i < rows ? c[2 * i] : c[2 * (i - rows) + 1])));
    for (const c of columns) design.push(Float64Array.from({ length: 2 * rows }, (_, i) => (i < rows ? -c[2 * i + 1] : c[2 * (i - rows)])));
  }
  const y = complex ? Float64Array.from({ length: 2 * rows }, (_, i) => (i < rows ? target[2 * i] : target[2 * (i - rows) + 1])) : target;
  const g = new Float64Array(n * n), b = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let r = 0; r < y.length; r++) s += design[i][r] * design[j][r];
      g[i * n + j] = g[j * n + i] = s;
    }
    let s = 0;
    for (let r = 0; r < y.length; r++) s += design[i][r] * y[r];
    b[i] = s;
  }
  const { values, vectors } = symmetricEigen(g, n);
  const largest = Math.max(...values.map(Math.abs));
  const cutoff = (Number.EPSILON * Math.max(y.length, n)) ** 2 * largest;
  const x = new Float64Array(n);
  for (let e = 0; e < n; e++) {
    if (!(values[e] > cutoff)) continue;
    let proj = 0;
    for (let i = 0; i < n; i++) proj += vectors[i * n + e] * b[i];
    for (let i = 0; i < n; i++) x[i] += (vectors[i * n + e] * proj) / values[e];
  }
  if (!complex) return Array.from(x);
  return Array.from({ length: p }, (_, j) => [x[j], x[p + j]]);
}

// y = mean(y) + sum_j c_j (x_j - mean(x_j)) on the complete plots where every
// term is defined, applied to the complete plots and the gaps. One coefficient
// per term for every axis of the plane: real, or complex for a bearing (a turn
// and a shrink of the term's direction).
function leastSquares(truth, cal, test, circular) {
  const defined = (xs, i) => xs.every((x) => rowFinite(x, i));
  const d = truth.d, nTerms = cal.length;
  const rows = [];
  for (let i = 0; i < truth.n; i++) if (defined(cal, i)) rows.push(i);
  const undefinedRows = (n) => rowsOf(n, d);
  if (!rows.length) {
    return { cal: undefinedRows(cal[0].n), test: undefinedRows(test[0].n),
             coefficients: Array(nTerms).fill(NaN), nFitted: 0 };
  }
  // A bearing row (sin, cos) lifted to the complex number cos + i sin.
  const lift = (m, i) => (circular ? [Number.isFinite(m.v[2 * i + 1]) ? m.v[2 * i + 1] : 0,
                                      Number.isFinite(m.v[2 * i]) ? m.v[2 * i] : 0]
                                   : Array.from({ length: d }, (_, j) => m.v[i * d + j]));
  const width = circular ? 2 : d;
  const meanOf = (m) => {
    const s = new Float64Array(width);
    for (const i of rows) lift(m, i).forEach((x, j) => { s[j] += x; });
    return s.map((x) => x / rows.length);
  };
  const yBar = meanOf(truth), xBar = cal.map(meanOf);
  const columns = cal.map((x, t) => {
    const out = new Float64Array(rows.length * width);
    rows.forEach((i, r) => lift(x, i).forEach((v, j) => { out[r * width + j] = v - xBar[t][j]; }));
    return out;
  });
  const target = new Float64Array(rows.length * width);
  rows.forEach((i, r) => lift(truth, i).forEach((v, j) => { target[r * width + j] = v - yBar[j]; }));
  const coefficients = lstsq(columns, target, circular);

  const apply = (xs) => {
    const n = xs[0].n, out = rowsOf(n, d);
    for (let i = 0; i < n; i++) {
      if (!defined(xs, i)) continue;
      const value = Array.from(yBar);
      xs.forEach((x, t) => {
        const z = lift(x, i).map((v, j) => v - xBar[t][j]);
        if (circular) {
          const [cr, ci] = coefficients[t];
          value[0] += cr * z[0] - ci * z[1];
          value[1] += cr * z[1] + ci * z[0];
        } else for (let j = 0; j < d; j++) value[j] += coefficients[t] * z[j];
      });
      if (circular) { out.v[2 * i] = value[1]; out.v[2 * i + 1] = value[0]; }
      else for (let j = 0; j < d; j++) out.v[i * d + j] = value[j];
    }
    return out;
  };
  return { cal: apply(cal), test: apply(test), coefficients, nFitted: rows.length };
}

function describeCoefficient(term, c) {
  if (Array.isArray(c)) {
    return { term, complex: true, modulus: Math.hypot(c[0], c[1]), angle_deg: Math.atan2(c[1], c[0]) * 180 / Math.PI };
  }
  return { term, complex: false, value: c };
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

// Each year's distance to the nearest reference year; NaN where missing.
function nearestYearGap(years, reference) {
  const ref = Float64Array.from(reference.filter(Number.isFinite)).sort();
  return years.map((y) => {
    if (!Number.isFinite(y) || !ref.length) return NaN;
    let lo = 0, hi = ref.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ref[mid] < y) lo = mid + 1; else hi = mid; }
    const below = ref[Math.max(0, lo - 1)], above = ref[Math.min(ref.length - 1, lo)];
    return Math.min(Math.abs(y - below), Math.abs(y - above));
  });
}

function summariseAxis(axis, distances, matched) {
  const out = { axis, matched, n: 0, quantiles: [NaN, NaN, NaN, NaN, NaN] };
  if (!distances) return out;
  const known = Float64Array.from(distances.filter(Number.isFinite)).sort();
  out.n = known.length;
  if (known.length) out.quantiles = [0, 0.25, 0.5, 0.75, 1].map((q) => quantile(known, q));
  return out;
}

const range = (n, first = 0) => Array.from({ length: n }, (_, i) => first + i);

// ---------------------------------------------------------------------------
// The fold layout
// ---------------------------------------------------------------------------

// murmur3's finaliser over the words in turn, as the engine hashes them.
function mix32(h) {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}
function hashWords(seed, words) {
  let h = mix32(seed >>> 0);
  for (const w of words) h = mix32((((h ^ w) >>> 0) + 0x9e3779b9) >>> 0);
  return h;
}
const lowWord = (v) => v >>> 0;
const highWord = (v) => Math.floor(v / 4294967296) >>> 0;

const GROUP_STREAM = 1, SAMPLE_STREAM = 2, BUFFER_STREAM = 3;
const BLOCK_SIZES = 10, PERIOD_LENGTHS = 8, SIMILAR_LIST = 256;

// The area between two samples' empirical distribution functions, divided by
// the sum of their means; 0 when both are all zero, NaN when either is empty.
function relativeWasserstein(a, b) {
  if (!a.length || !b.length) return NaN;
  a = Float64Array.from(a).sort();
  b = Float64Array.from(b).sort();
  let i = 0, j = 0, area = 0, x = Math.min(a[0], b[0]);
  while (i < a.length || j < b.length) {
    const next = j >= b.length || (i < a.length && a[i] <= b[j]) ? a[i] : b[j];
    area += Math.abs(i / a.length - j / b.length) * (next - x);
    x = next;
    while (i < a.length && a[i] === x) i++;
    while (j < b.length && b[j] === x) j++;
  }
  const means = a.reduce((s, v) => s + v, 0) / a.length + b.reduce((s, v) => s + v, 0) / b.length;
  return means > 0 ? area / means : 0;
}

const moreSimilar = (x, y) => (x[0] !== y[0] ? y[0] - x[0] : x[1] - y[1]);

class LayoutMatcher {
  constructor(split, search, gaps, { seed, matchingSample }) {
    this.split = split;
    this.search = search;
    this.n = split.nCal;
    this.seed = seed >>> 0;
    this.positions();
    this.surveyYears();
    this.sample(matchingSample);
    this.gapDissimilarity = Array.from(gaps.similarity, (s) => 1 - s);
    const nGaps = gaps.similarity.length;
    this.gapKmAll = split.kmAxis ? Array.from(gaps.km) : new Array(nGaps).fill(NaN);
    this.gapYearsAll = split.timeAxis ? Array.from(gaps.years) : new Array(nGaps).fill(NaN);
    this.gapKm = this.gapKmAll.filter(Number.isFinite);
    this.gapYears = this.gapYearsAll.filter(Number.isFinite);
  }

  specs() {
    const spec = (blockKm, blockYears, buffer = "none") => ({ blockKm, blockYears, buffer });
    const out = [spec(NaN, NaN)];
    for (const s of this.blockSizes) out.push(spec(s, NaN));
    for (const w of this.periodLengths) out.push(spec(NaN, w));
    for (const s of this.blockSizes) for (const w of this.periodLengths) out.push(spec(s, w));
    if (this.gapKm.length) out.push(spec(NaN, NaN, "km"));
    if (this.gapYears.length) out.push(spec(NaN, NaN, "years"));
    if (this.gapKm.length && this.gapYears.length) out.push(spec(NaN, NaN, "both"));
    return out;
  }

  // The buffers of spec added to its folds: each complete plot takes the
  // distances of one gap drawn by the hash of its row, among the gaps with a
  // distance on the buffer's axes (with both, the gap's two together).
  withBuffers(folds, spec) {
    const layout = { folds, bufferKm: null, bufferYears: null };
    if (spec.buffer === "none") return layout;
    const km = spec.buffer !== "years", years = spec.buffer !== "km";
    const pool = range(this.gapKmAll.length).filter((g) =>
      (km && Number.isFinite(this.gapKmAll[g])) || (years && Number.isFinite(this.gapYearsAll[g])));
    if (km) layout.bufferKm = new Float64Array(this.n).fill(NaN);
    if (years) layout.bufferYears = new Float64Array(this.n).fill(NaN);
    for (let i = 0; i < this.n; i++) {
      const g = pool[hashWords(this.seed, [BUFFER_STREAM, lowWord(i), highWord(i)]) % pool.length];
      if (km) layout.bufferKm[i] = this.gapKmAll[g];
      if (years) layout.bufferYears[i] = this.gapYearsAll[g];
    }
    return layout;
  }

  // The groups under `spec` dealt to nFolds folds: largest first (ties by
  // hash, then first plot), each to the fold holding fewest plots (ties to the
  // lower); empty folds dropped and the rest numbered in order.
  deal(spec, nFolds) {
    const space = !Number.isNaN(spec.blockKm), time = !Number.isNaN(spec.blockYears);
    const index = new Map(), groups = [], groupOf = new Int32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      let key = [0, 0, 0, -1], own = !space && !time;
      if (space) {
        if (Number.isFinite(this.east[i]) && Number.isFinite(this.north[i])) {
          key[0] = Math.floor(this.east[i] / spec.blockKm);
          key[1] = Math.floor(this.north[i] / spec.blockKm);
        } else own = true;
      }
      if (time) {
        if (Number.isFinite(this.years[i])) key[2] = Math.floor((this.years[i] - this.yearOrigin) / spec.blockYears);
        else own = true;
      }
      if (own) key = [0, 0, 0, i];
      const id = key.join(",");
      let g = index.get(id);
      if (g === undefined) { g = groups.length; index.set(id, g); groups.push({ key, size: 0, first: i, hash: 0 }); }
      groups[g].size++;
      groupOf[i] = g;
    }
    for (const g of groups) {
      g.hash = hashWords(this.seed, [GROUP_STREAM, ...g.key.flatMap((v) => [lowWord(v), highWord(v)])]);
    }
    const order = range(groups.length).sort((x, y) => {
      const gx = groups[x], gy = groups[y];
      if (gx.size !== gy.size) return gy.size - gx.size;
      if (gx.hash !== gy.hash) return gx.hash - gy.hash;
      return gx.first - gy.first;
    });
    const counts = new Array(nFolds).fill(0), foldOfGroup = new Int32Array(groups.length);
    for (const g of order) {
      let f = 0;
      for (let c = 1; c < nFolds; c++) if (counts[c] < counts[f]) f = c;
      foldOfGroup[g] = f;
      counts[f] += groups[g].size;
    }
    const renumber = new Int32Array(nFolds).fill(-1);
    let next = 0;
    for (let f = 0; f < nFolds; f++) if (counts[f] > 0) renumber[f] = next++;
    const folds = Int32Array.from(groupOf, (g) => renumber[foldOfGroup[g]]);
    return { folds, largest: Math.max(...counts) };
  }

  async mismatch(layout) {
    const rules = foldRules(this.split, layout, -1), hides = foldHiding(rules);
    const out = [["similarity", relativeWasserstein(this.heldDissimilarity(hides), this.gapDissimilarity)]];
    if (this.split.kmAxis) out.push(["km", relativeWasserstein(await this.heldKm(rules), this.gapKm)]);
    if (this.split.timeAxis) out.push(["years", relativeWasserstein(this.heldYears(hides), this.gapYears)]);
    return out;
  }

  positions() {
    const n = this.n, c = this.split.coordsCal;
    this.east = new Float64Array(n).fill(NaN);
    this.north = new Float64Array(n).fill(NaN);
    this.blockSizes = [];
    if (!c) return;
    const lats = [], lons = [];
    for (let i = 0; i < n; i++) if (rowFinite(c, i)) { lats.push(c.v[2 * i]); lons.push(c.v[2 * i + 1]); }
    if (lats.length < 2) return;
    const projection = new Laea(...geographicMedian(lats, lons));
    let loE = Infinity, hiE = -Infinity, loN = Infinity, hiN = -Infinity;
    for (let i = 0; i < n; i++) {
      if (!rowFinite(c, i)) continue;
      const [e, nn] = projection.forward(c.v[2 * i], c.v[2 * i + 1]);
      this.east[i] = e / 1000;
      this.north[i] = nn / 1000;
      if (!Number.isFinite(this.east[i]) || !Number.isFinite(this.north[i])) continue;
      loE = Math.min(loE, this.east[i]); hiE = Math.max(hiE, this.east[i]);
      loN = Math.min(loN, this.north[i]); hiN = Math.max(hiN, this.north[i]);
    }
    const extent = Math.max(hiE - loE, hiN - loN);
    if (!(extent > 0)) return;
    let side = extent;
    for (let j = 0; j < BLOCK_SIZES; j++) this.blockSizes.push(side /= 2);
  }

  surveyYears() {
    const y = this.split.yearsCal;
    this.years = y ? Float64Array.from(y) : new Float64Array(this.n).fill(NaN);
    this.yearOrigin = 0;
    this.periodLengths = [];
    if (!y) return;
    let lo = Infinity, hi = -Infinity, known = 0;
    for (const v of this.years) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); known++; }
    if (known < 2) return;
    this.yearOrigin = lo;
    let length = hi - lo;
    for (let j = 0; j < PERIOD_LENGTHS; j++) {
      length /= 2;
      if (length < 1) break;
      this.periodLengths.push(length);
    }
  }

  // The complete plots the layouts' distances are measured on: all, or the
  // `size` with the lowest hash of their row.
  sample(size) {
    let rows = range(this.n);
    if (this.n > size) {
      const h = Uint32Array.from(rows, (i) => hashWords(this.seed, [SAMPLE_STREAM, lowWord(i), highWord(i)]));
      rows.sort((a, b) => (h[a] !== h[b] ? h[a] - h[b] : a - b));
      rows = rows.slice(0, size).sort((a, b) => a - b);
    }
    this.sampleRows = rows;
    this.similar = [];
    this.whole = [];
    for (const plot of rows) {
      const list = this.similarities(plot).sort(moreSimilar);
      this.whole.push(list.length <= SIMILAR_LIST);
      this.similar.push(list.length <= SIMILAR_LIST ? list : list.slice(0, SIMILAR_LIST));
    }
    this.byYear = range(this.n).filter((i) => Number.isFinite(this.years[i]))
      .sort((a, b) => (this.years[a] !== this.years[b] ? this.years[a] - this.years[b] : a - b));
    this.yearRank = new Int32Array(this.n).fill(-1);
    this.byYear.forEach((plot, r) => { this.yearRank[plot] = r; });
  }

  similarities(plot) {
    const sets = this.search.sets, own = sets.count(plot);
    return overlaps(sets, plot, this.n).map(([other, shared]) => [shared / Math.sqrt(own * sets.count(other)), other]);
  }

  // Each matching plot's dissimilarity to the most similar plot the rule
  // leaves it (1 when it shares no species with any).
  heldDissimilarity(hides) {
    return this.sampleRows.map((plot, s) => {
      let best = NaN;
      for (const [similarity, other] of this.similar[s]) if (!hides(plot, other, NaN, NaN)) { best = similarity; break; }
      if (Number.isNaN(best)) {
        best = 0;
        if (!this.whole[s]) {
          for (const [similarity, other] of this.similarities(plot)) if (!hides(plot, other, NaN, NaN)) best = Math.max(best, similarity);
        }
      }
      return 1 - best;
    });
  }

  // Each located matching plot's km to the nearest plot the rules leave it.
  async heldKm(rules) {
    const rows = this.sampleRows.filter((plot) => Number.isFinite(this.east[plot]) && Number.isFinite(this.north[plot]));
    if (!rows.length) return [];
    const found = await this.search.places.nearest(rows, 1, rules);
    return Array.from(found.km).filter(Number.isFinite);
  }

  // Each dated matching plot's years to the nearest plot in time the rule
  // leaves it.
  heldYears(hides) {
    const out = [];
    for (const plot of this.sampleRows) {
      const r = this.yearRank[plot];
      if (r < 0) continue;
      const year = this.years[plot];
      let best = Infinity;
      for (let q = r - 1; q >= 0; q--) {
        const other = this.byYear[q];
        if (!hides(plot, other, NaN, NaN)) { best = Math.min(best, year - this.years[other]); break; }
      }
      for (let q = r + 1; q < this.byYear.length; q++) {
        const other = this.byYear[q];
        if (!hides(plot, other, NaN, NaN)) { best = Math.min(best, this.years[other] - year); break; }
      }
      if (Number.isFinite(best)) out.push(best);
    }
    return out;
  }
}

// The matched layout: the candidate with the smallest total mismatch among
// those whose largest fold holds at most maxFoldShare of the complete plots
// (the plot-by-plot layout always qualifies).
async function matchLayout(split, search, gaps, config) {
  const matcher = new LayoutMatcher(split, search, gaps, config);
  const n = split.nCal, nFolds = Math.min(config.nFolds, n);
  const report = { n_folds: 0, block_km: NaN, block_years: NaN, buffer_km: NaN, buffer_years: NaN, injected: false,
                   mismatch: [], total_mismatch: NaN, n_layouts: 0, n_matching_plots: matcher.sampleRows.length };
  const medianOf = (v) => (v ? median(Array.from(v).filter((x) => !Number.isNaN(x))) : NaN);
  let best = null;
  for (const spec of matcher.specs()) {
    const { folds, largest } = matcher.deal(spec, nFolds);
    const used = folds.reduce((m, f) => Math.max(m, f), -1) + 1;
    const plotByPlot = Number.isNaN(spec.blockKm) && Number.isNaN(spec.blockYears);
    if (!plotByPlot && (used < 2 || largest > config.maxFoldShare * n)) continue;
    report.n_layouts++;
    const layout = matcher.withBuffers(folds, spec);
    const mismatch = await matcher.mismatch(layout);
    const total = mismatch.reduce((sum, [, w]) => (Number.isFinite(w) ? sum + w : sum), 0);
    if (!best || total < report.total_mismatch) {
      best = layout;
      Object.assign(report, { n_folds: used, block_km: spec.blockKm, block_years: spec.blockYears,
                              buffer_km: medianOf(layout.bufferKm), buffer_years: medianOf(layout.bufferYears),
                              mismatch, total_mismatch: total });
    }
  }
  return { layout: best, report };
}

// A layout's hiding rules over the complete plots, `held` hidden too.
function foldRules(split, layout, held) {
  return new FoldRules(layout.folds, held, { bufferKm: layout.bufferKm, bufferYears: layout.bufferYears,
                                            coordinates: split.coordsCal && split.coordsCal.v, years: split.yearsCal });
}

// Injected folds as the layout: one per complete plot, every fold from 0 to
// the largest holding a plot.
function injectedLayout(folds, n) {
  if (!folds || folds.length !== n) throw new Error(`gap fitting: the injected folds must hold one fold per complete plot (${n})`);
  const out = Int32Array.from(folds);
  const counts = new Array(out.reduce((m, f) => Math.max(m, f), 0) + 1).fill(0);
  for (const v of out) {
    if (v < 0) throw new Error("gap fitting: an injected fold is negative");
    counts[v]++;
  }
  if (counts.includes(0) || counts.length < 2) {
    throw new Error("gap fitting: the injected folds must number at least two, each from 0 to the largest holding a plot");
  }
  return { layout: { folds: out, bufferKm: null, bufferYears: null },
           report: { n_folds: counts.length, block_km: NaN, block_years: NaN, buffer_km: NaN, buffer_years: NaN,
                     injected: true, mismatch: [], total_mismatch: NaN, n_layouts: 0, n_matching_plots: 0 } };
}

// ---------------------------------------------------------------------------
// The searches
// ---------------------------------------------------------------------------

// The neighbour searches of one target, complete plots first and the gaps
// after them: by species over a neighbour graph, by position over a k-d tree;
// only complete plots are neighbours.
class PlotSearch {
  static async build(split, speciesCal, speciesTest, kGrid, pool) {
    const s = new PlotSearch();
    s.split = split;
    s.nCal = speciesCal.length;
    s.nTest = speciesTest.length;
    s.pool = pool;
    s.sets = new PresenceSets([...speciesCal, ...speciesTest]);
    const maxK = Math.max(...kGrid);
    s.kSpecies = Math.min(maxK, s.nCal - 1);
    s.graph = await SetNeighbourGraph.build(s.sets, s.nCal, { k: Math.max(1, s.kSpecies) }, pool);
    s.open = await s.graph.open(s.nCal, s.nTest);
    s.places = split.spatialAxis || split.kmAxis
      ? new PlaceIndex(Float64Array.from([...split.coordsCal.v, ...split.coordsTest.v]), s.nCal, pool) : null;
    let located = 0;
    if (split.spatialAxis) for (let i = 0; i < s.nCal; i++) if (rowFinite(split.coordsCal, i)) located++;
    s.kSpatial = split.spatialAxis ? Math.min(maxK, located - 1) : 0;
    return s;
  }

  gapRows() { return range(this.nTest, this.nCal); }

  // Each gap's distance to its nearest complete plot on every axis; a gap
  // sharing no species with any complete plot is at similarity 0.
  async distances(split) {
    const k = this.open.k;
    const out = { similarity: Float64Array.from({ length: this.nTest }, (_, g) => {
      const s = this.open.similarities[g * k];
      return Number.isNaN(s) ? 0 : s;
    }), km: null, years: null };
    if (split.kmAxis) out.km = (await this.places.nearest(this.gapRows(), 1)).km;
    if (split.timeAxis) out.years = nearestYearGap(split.yearsTest, split.yearsCal);
    return out;
  }

  // Each complete plot's neighbours among the complete plots the layout
  // leaves it: of the other folds, the fold `held` (or -1) hidden from every
  // plot too, outside its buffers. Per kind { indices, width } or null.
  async outOfFold(layout, held) {
    const out = [null, null];
    const rules = foldRules(this.split, layout, held);
    if (this.kSpecies >= 1) {
      const lists = await this.graph.hidden(this.nCal, rules);
      out[SPECIES] = { indices: narrowed(lists, this.kSpecies), width: this.kSpecies };
    }
    if (this.places && this.kSpatial >= 1) {
      const found = await this.places.nearest(range(this.nCal), this.kSpatial, rules);
      out[SPATIAL] = { indices: found.indices, width: this.kSpatial };
    }
    return out;
  }

  // The gaps' neighbours among the complete plots, of one kind.
  async gapNeighbours(kind, width) {
    if (kind === SPECIES) return { indices: narrowed({ indices: this.open.indices, k: this.open.k }, width), width };
    return { indices: (await this.places.nearest(this.gapRows(), width)).indices, width };
  }
}

// The rows `index` of (rows, width) neighbour lists.
function neighbourRows(lists, index) {
  const out = new Int32Array(index.length * lists.width);
  index.forEach((i, r) => out.set(lists.indices.subarray(i * lists.width, (i + 1) * lists.width), r * lists.width));
  return { indices: out, width: lists.width };
}

// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

// Element-wise: m's value where finite, else the fallback's (rows, or one row).
function orFallback(m, fallback, oneRow = false) {
  const out = rowsOf(m.n, m.d);
  for (let i = 0; i < m.v.length; i++) {
    out.v[i] = Number.isFinite(m.v[i]) ? m.v[i] : oneRow ? fallback[i % m.d] : fallback.v[i];
  }
  return out;
}

// One cross-validation over the training plots (rows of the complete plots)
// predicting a set of query plots:
//   run { train, trainFolds, trainNeighbours: [species, spatial],
//         queryNeighbours: [species, spatial], queryRaw }
class CrossValidation {
  constructor(split, run, config) {
    this.split = split;
    this.run = run;
    this.config = config;
    this.truth = pick(split.truthCal, run.train);
    this.raw = pick(split.rawCal, run.train);
    const folds = [...new Set(run.trainFolds)].sort((a, b) => a - b);
    this.held = folds.map((f) => run.train.map((_, r) => r).filter((r) => run.trainFolds[r] === f));
    this.rest = folds.map((f) => run.train.map((_, r) => r).filter((r) => run.trainFolds[r] !== f));
    const plane = split.plane;
    this.constantTrain = rowsOf(this.truth.n, this.truth.d);
    this.held.forEach((rows, g) => {
      const c = plane.centre(pick(this.truth, this.rest[g]));
      for (const r of rows) for (let j = 0; j < c.length; j++) this.constantTrain.v[r * this.truth.d + j] = c[j];
    });
    this.centre = plane.centre(this.truth);
  }

  neighbourTerm(kind) {
    const train = this.run.trainNeighbours[kind];
    if (!train) return null;
    const { width } = train, circular = this.split.circular, plane = this.split.plane;
    const has = range(this.truth.n).filter((r) => train.indices[r * width] >= 0);
    const truthHas = pick(this.truth, has);
    const choice = { kind: KIND_NAMES[kind], k: 0, cv_error: [] };
    let best = NaN, any = false;
    for (const k of this.config.kGrid) {
      if (k > width) continue;
      const error = plane.aggregate(plane.errors(neighbourMedian(this.split.truthCal, train.indices, width, has, k, circular), truthHas));
      choice.cv_error.push([k, error]);
      if (!any || error < best) { best = error; choice.k = k; }
      any = true;
    }
    if (!any) return null;
    const query = this.run.queryNeighbours[kind];
    return { choice,
             train: neighbourMedian(this.split.truthCal, train.indices, width, range(this.truth.n), choice.k, circular),
             query: neighbourMedian(this.split.truthCal, query.indices, width, range(query.indices.length / width), choice.k, circular) };
  }

  fit(spec, terms) {
    const circular = this.split.circular, nQ = this.run.queryRaw.n;
    if (spec.combination === "constant") {
      const query = rowsOf(nQ, this.centre.length);
      for (let i = 0; i < nQ; i++) query.v.set(this.centre, i * this.centre.length);
      return { train: this.constantTrain, query, k: [], coefficients: [], nFitted: 0 };
    }
    if (spec.combination === "raw") return { train: this.raw, query: this.run.queryRaw, k: [], coefficients: [], nFitted: 0 };
    if (spec.combination === "neighbours") {
      const kind = spec.reads[SPECIES] ? SPECIES : SPATIAL;
      return { train: terms[kind].train, query: terms[kind].query, k: [[KIND_NAMES[kind], terms[kind].choice.k]],
               coefficients: [], nFitted: 0 };
    }
    const train = [this.raw], query = [this.run.queryRaw], names = ["resolve"], k = [];
    for (const kind of [SPECIES, SPATIAL]) {
      if (!spec.reads[kind]) continue;
      train.push(terms[kind].train); query.push(terms[kind].query); names.push(KIND_NAMES[kind]);
      k.push([KIND_NAMES[kind], terms[kind].choice.k]);
    }
    const out = rowsOf(this.truth.n, this.truth.d);
    this.held.forEach((rows, g) => {
      const rest = this.rest[g];
      const ls = leastSquares(pick(this.truth, rest), train.map((x) => pick(x, rest)), train.map((x) => pick(x, rows)), circular);
      rows.forEach((r, i) => { for (let j = 0; j < this.truth.d; j++) out.v[r * this.truth.d + j] = ls.test.v[i * this.truth.d + j]; });
    });
    const all = leastSquares(this.truth, train, query, circular);
    return { train: out, query: all.test, k, nFitted: all.nFitted,
             coefficients: names.map((name, j) => describeCoefficient(name, all.coefficients[j])) };
  }

  // groups: [{ name, spatial, judged: train row indices, predicted: query row indices }]
  choose(groups) {
    const terms = [this.neighbourTerm(SPECIES), this.neighbourTerm(SPATIAL)];
    const neighbours = terms.filter(Boolean).map((t) => t.choice);
    const usable = [];
    for (const spec of CANDIDATES) {
      const asked = !this.config.candidates || this.config.candidates.includes(GAP_METHODS[spec.method]);
      if (!asked) continue;
      if (spec.reads[SPATIAL] && !this.split.spatialAxis) continue;
      if (spec.reads.some((r, kind) => r && !terms[kind])) continue;
      usable.push({ spec, fit: this.fit(spec, terms) });
    }
    const plane = this.split.plane, d = this.truth.d, nQ = this.run.queryRaw.n;
    const results = groups.map((group) => {
      const truth = pick(this.truth, group.judged);
      const fallback = pick(this.constantTrain, group.judged);
      const choice = { group: group.name, n_complete: truth.n, n_gaps: group.predicted.length, method: 1,
                       cv_error: [], estimated_error: NaN, n_estimated: 0, coefficients: [], n_fitted: 0, k: [] };
      let best = null, bestError = NaN;
      for (const u of usable) {
        if (u.spec.reads[SPATIAL] && !group.spatial) continue;
        const error = plane.aggregate(plane.errors(orFallback(pick(u.fit.train, group.judged), fallback), truth));
        choice.cv_error.push([u.spec.method, error]);
        if (!best || error < bestError) { best = u; bestError = error; }
      }
      if (!best) {
        throw new Error(`gap fitting: ${choice.n_gaps} ${group.name} gaps and no candidate for them` +
                        (group.spatial ? "" : "; gaps without coordinates need a candidate that reads no position"));
      }
      Object.assign(choice, { method: best.spec.method, coefficients: best.fit.coefficients, n_fitted: best.fit.nFitted,
                              k: best.fit.k });
      const filled = orFallback(best.fit.query, this.centre, true);
      const query = rowsOf(nQ, d);
      for (const q of group.predicted) for (let j = 0; j < d; j++) query.v[q * d + j] = filled.v[q * d + j];
      return { choice, query };
    });
    return { neighbours, groups: results };
  }
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

// Fit the gaps of one target.
//
//   target    { name, scale: "linear" | "log" | "circular" | "geographic", period? }
//   raw       the model's prediction per plot: Float64Array(n), or (2n)
//             latitude / longitude pairs for a geographic target
//   recorded  the dataset's value per plot, laid out as raw; NaN marks a gap
//   plots     { species: per plot the sorted ids of the species present,
//               coordinates?: Float64Array(2n) latitude / longitude, NaN
//               where unknown, years?: Float64Array(n), NaN where unknown }
//   options   { spatial = true, seed = GAP_SEED, kGrid, nFolds = 10,
//               maxFoldShare = 0.5, matchingSample = 1000, candidates?,
//               folds?, pool? } where folds (one per complete plot, numbered
//             from 0) replaces the matched layout and pool (a RowPool) runs
//             the neighbour searches, on this thread without one
//
// Resolves to { fitted, method, report, folds } as the engine's GapFit:
// fitted laid out as raw, a complete plot carrying its recorded value and a
// gap the chosen method's; method GAP_RECORDED or the index into GAP_METHODS.
export async function fitGaps(target, raw, recorded, plots, options = {}) {
  const { spatial = true, seed = GAP_SEED, kGrid = K_GRID, nFolds = 10, maxFoldShare = 0.5, matchingSample = 1000,
          candidates = null, folds: injected = null, pool = new RowPool() } = options;
  const config = { spatial, seed, kGrid, nFolds, maxFoldShare, matchingSample, candidates };
  const n = plots.species.length;
  const d = target.scale === "geographic" ? 2 : 1;
  if (raw.length !== n * d || recorded.length !== n * d) {
    throw new Error(`gap fitting: target '${target.name}' needs ${d === 2 ? "latitude / longitude pairs" : "one value"} per plot`);
  }
  const rec = { n, d, v: Float64Array.from(recorded) }, rawRows = { n, d, v: Float64Array.from(raw) };
  const cal = [], test = [];
  for (let i = 0; i < n; i++) (rowFinite(rec, i) ? cal : test).push(i);

  const report = { target: target.name, scale: target.scale, n_complete: cal.length, n_gaps: test.length,
                   skipped: "", axes: [], folds: null, neighbours: [], groups: [] };
  const fitted = Float64Array.from(recorded);
  const method = new Int32Array(n).fill(GAP_RECORDED);
  if (!test.length) return { fitted, method, report, folds: null };
  if (cal.length < 2) {
    report.skipped = "fewer than two complete plots";
    for (const i of test) { for (let j = 0; j < d; j++) fitted[i * d + j] = raw[i * d + j]; method[i] = 1; }
    return { fitted, method, report, folds: null };
  }

  // The complete plots (cal) and the gaps (test) in the plane.
  const plane = new Plane(target, pick(rec, cal));
  const coords = plots.coordinates ? { n, d: 2, v: plots.coordinates } : null;
  const years = plots.years ?? null;
  const split = {
    plane, nCal: cal.length, nTest: test.length, circular: target.scale === "circular",
    truthCal: plane.to(pick(rec, cal)), rawCal: plane.to(pick(rawRows, cal)), rawTest: plane.to(pick(rawRows, test)),
    coordsCal: coords && pick(coords, cal), coordsTest: coords && pick(coords, test),
    yearsCal: years ? cal.map((i) => years[i]) : null, yearsTest: years ? test.map((i) => years[i]) : null,
  };
  let locatedCal = 0;
  if (coords) for (let i = 0; i < split.nCal; i++) if (rowFinite(split.coordsCal, i)) locatedCal++;
  const anyLocatedTest = !!coords && range(split.nTest).some((g) => rowFinite(split.coordsTest, g));
  split.spatialAxis = spatial && target.scale !== "geographic" && locatedCal >= 2;
  split.kmAxis = target.scale !== "geographic" && locatedCal >= 2 && anyLocatedTest;
  split.timeAxis = !!years && split.yearsCal.some(Number.isFinite) && split.yearsTest.some(Number.isFinite);

  const search = await PlotSearch.build(split, cal.map((i) => plots.species[i]), test.map((i) => plots.species[i]), kGrid, pool);
  const distances = await search.distances(split);
  const { layout: foldLayout, report: layout } = injected ? injectedLayout(injected, split.nCal)
    : await matchLayout(split, search, distances, config);
  const folds = foldLayout.folds;
  const matched = !layout.injected;
  report.axes.push(summariseAxis("similarity", Array.from(distances.similarity), matched));
  report.axes.push(summariseAxis("km", distances.km && Array.from(distances.km), matched && split.kmAxis));
  report.axes.push(summariseAxis("years", distances.years, matched && split.timeAxis));
  report.folds = layout;

  const nCal = split.nCal, nTest = split.nTest;
  const located = range(nTest).map((g) => split.spatialAxis && rowFinite(split.coordsTest, g));
  const calLocated = range(nCal).map((i) => !split.spatialAxis || rowFinite(split.coordsCal, i));
  const anyLocated = located.some(Boolean), anyUnlocated = located.some((l) => !l);
  const where = (flags) => flags.flatMap((f, i) => (f ? [i] : []));
  const groupsFor = (trainLocated, queryLocated, queryUnlocated) => {
    const out = [];
    if (anyLocated) out.push({ name: "located", spatial: true, judged: where(trainLocated), predicted: where(queryLocated) });
    if (anyUnlocated) out.push({ name: "unlocated", spatial: false, judged: range(trainLocated.length), predicted: where(queryUnlocated) });
    return out;
  };

  // Every complete plot predicted from the other folds; the gaps from all.
  const topNeighbours = await search.outOfFold(foldLayout, -1);
  const top = { train: range(nCal), trainFolds: Array.from(folds), trainNeighbours: topNeighbours,
                queryNeighbours: await Promise.all(topNeighbours.map((nb, kind) => (nb ? search.gapNeighbours(kind, nb.width) : null))),
                queryRaw: split.rawTest };
  const chosen = new CrossValidation(split, top, config).choose(groupsFor(calLocated, located, located.map((l) => !l)));
  report.neighbours = chosen.neighbours;

  // The whole procedure repeated with each fold held out, scored on it.
  const estimate = chosen.groups.map(() => rowsOf(nCal, split.truthCal.d));
  if (layout.n_folds >= 3) {
    for (let f = 0; f < layout.n_folds; f++) {
      const heldRows = range(nCal).filter((i) => folds[i] === f), restRows = range(nCal).filter((i) => folds[i] !== f);
      const neighbours = await search.outOfFold(foldLayout, f);
      const run = { train: restRows, trainFolds: restRows.map((i) => folds[i]),
                    trainNeighbours: neighbours.map((nb) => nb && neighbourRows(nb, restRows)),
                    queryNeighbours: neighbours.map((nb) => nb && neighbourRows(nb, heldRows)),
                    queryRaw: pick(split.rawCal, heldRows) };
      const heldLocated = heldRows.map((i) => calLocated[i]);
      const nested = new CrossValidation(split, run, config)
        .choose(groupsFor(restRows.map((i) => calLocated[i]), heldLocated, heldLocated.map(() => true)));
      nested.groups.forEach(({ query }, g) => {
        heldRows.forEach((row, q) => {
          if (!rowFinite(query, q)) return;
          for (let j = 0; j < query.d; j++) estimate[g].v[row * query.d + j] = query.v[q * query.d + j];
        });
      });
    }
  }

  const testRows = rowsOf(nTest, split.truthCal.d);
  const testMethod = new Int32Array(nTest).fill(GAP_RECORDED);
  chosen.groups.forEach(({ choice, query }, g) => {
    const scored = range(nCal).filter((i) => rowFinite(estimate[g], i));
    choice.n_estimated = scored.length;
    if (scored.length) choice.estimated_error = plane.aggregate(plane.errors(pick(estimate[g], scored), pick(split.truthCal, scored)));
    for (let q = 0; q < nTest; q++) {
      if (!rowFinite(query, q)) continue;
      for (let j = 0; j < query.d; j++) testRows.v[q * query.d + j] = query.v[q * query.d + j];
      testMethod[q] = choice.method;
    }
    report.groups.push(choice);
  });

  test.forEach((i, g) => {
    const value = plane.backRow(testRows, g);
    if (d === 2) { fitted[2 * i] = value[0]; fitted[2 * i + 1] = value[1]; } else fitted[i] = value;
    method[i] = testMethod[g];
  });
  return { fitted, method, report, folds };
}
