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
// The choice is made at the distance the gaps are from the complete plots:
// one distance per axis (species similarity, km, survey years) is drawn per
// complete plot from the gaps' own distances, and each complete plot is
// predicted with the complete plots nearer to it than that hidden
// (nearest-neighbour distance matching, Mila et al. 2022, Methods in Ecology
// and Evolution). Each candidate is scored by the target's headline error and
// the best predicts the gaps; gaps with and without coordinates choose apart.

import { geographicMedian, greatCircleKm, Laea } from "./geo.js";
import { HidingRules, PlaceIndex, PresenceSets, RowPool, SetNeighbourGraph } from "./plot_neighbours.js";

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

export const gapMethodLabel = (code) => (code === GAP_RECORDED ? "recorded" : GAP_METHODS[code]);

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
// The random draw: torch's CPU generator (mt19937) and randint
// ---------------------------------------------------------------------------

class Mt19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.index = 624;
  }

  next() {
    if (this.index >= 624) {
      for (let i = 0; i < 624; i++) {
        const y = (this.mt[i] & 0x80000000) | (this.mt[(i + 1) % 624] & 0x7fffffff);
        this.mt[i] = this.mt[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      this.index = 0;
    }
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  // torch.randint(0, high, (1,)) for high < 2^32: one 32-bit draw modulo high.
  below(high) { return this.next() % high; }
}

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

  // The local constant: the plane's origin (geographic median), the median on
  // a log scale, the mean otherwise (of sine and cosine for a bearing).
  centre(truth) {
    if (this.scale === "geographic") return new Float64Array(2);
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

function drawn(distances, n, generator) {
  const known = distances.filter(Number.isFinite);
  if (!known.length) return new Float64Array(n);
  return Float64Array.from({ length: n }, () => known[generator.below(known.length)]);
}

function summariseAxis(axis, distances, matched) {
  const out = { axis, matched, n: 0, quantiles: [NaN, NaN, NaN, NaN, NaN] };
  if (!distances) return out;
  const known = Float64Array.from(distances.filter(Number.isFinite)).sort();
  out.n = known.length;
  if (known.length) out.quantiles = [0, 0.25, 0.5, 0.75, 1].map((q) => quantile(known, q));
  return out;
}

// Fit the gaps of one target.
//
//   target    { name, scale: "linear" | "log" | "circular" | "geographic", period? }
//   raw       the model's prediction per plot: Float64Array(n), or (2n)
//             latitude / longitude pairs for a geographic target
//   recorded  the dataset's value per plot, laid out as raw; NaN marks a gap
//   plots     { species: per plot the sorted ids of the species present,
//               coordinates?: Float64Array(2n) latitude / longitude, NaN
//               where unknown, years?: Float64Array(n), NaN where unknown }
//   options   { spatial = true, seed = GAP_SEED, kGrid, radii?, pool? } where
//             radii ({ similarity, km?, years? }, one value per complete
//             plot) replace the drawn hiding distances and pool (a RowPool)
//             runs the neighbour searches, on this thread without one
//
// Resolves to { fitted, method, report } as the engine's GapFit: fitted laid out
// as raw, a complete plot carrying its recorded value and a gap the chosen
// method's; method GAP_RECORDED or the index into GAP_METHODS.
export async function fitGaps(target, raw, recorded, plots, options = {}) {
  const { spatial = true, seed = GAP_SEED, kGrid = K_GRID, radii: injected = null, pool = new RowPool() } = options;
  const n = plots.species.length;
  const d = target.scale === "geographic" ? 2 : 1;
  if (raw.length !== n * d || recorded.length !== n * d) {
    throw new Error(`gap fitting: target '${target.name}' needs ${d === 2 ? "latitude / longitude pairs" : "one value"} per plot`);
  }
  const rec = { n, d, v: Float64Array.from(recorded) }, rawRows = { n, d, v: Float64Array.from(raw) };
  const cal = [], test = [];
  for (let i = 0; i < n; i++) (rowFinite(rec, i) ? cal : test).push(i);

  const report = { target: target.name, scale: target.scale, n_complete: cal.length, n_gaps: test.length,
                   skipped: "", axes: [], neighbours: [], groups: [] };
  const fitted = Float64Array.from(recorded);
  const method = new Int32Array(n).fill(GAP_RECORDED);
  if (!test.length) return { fitted, method, report };
  if (cal.length < 2) {
    report.skipped = "fewer than two complete plots";
    for (const i of test) { for (let j = 0; j < d; j++) fitted[i * d + j] = raw[i * d + j]; method[i] = 1; }
    return { fitted, method, report };
  }

  // The fold: complete plots (cal) and gaps (test) in the plane.
  const plane = new Plane(target, pick(rec, cal));
  const truth = plane.to(pick(rec, cal));
  const rawCal = plane.to(pick(rawRows, cal)), rawTest = plane.to(pick(rawRows, test));
  const speciesCal = cal.map((i) => plots.species[i]), speciesTest = test.map((i) => plots.species[i]);
  const coords = plots.coordinates ? { n, d: 2, v: plots.coordinates } : null;
  const coordsCal = coords && pick(coords, cal), coordsTest = coords && pick(coords, test);
  const yearsCal = plots.years ? cal.map((i) => plots.years[i]) : null;
  const yearsTest = plots.years ? test.map((i) => plots.years[i]) : null;
  const nCal = cal.length, nTest = test.length;
  const located = (m, i) => m && rowFinite(m, i);
  let nCalLocated = 0;
  for (let i = 0; i < nCal; i++) if (located(coordsCal, i)) nCalLocated++;
  const spatialAxis = spatial && target.scale !== "geographic" && !!coords && nCalLocated >= 2;
  const timeAxis = !!plots.years && yearsCal.some(Number.isFinite) && yearsTest.some(Number.isFinite);
  const circular = target.scale === "circular";

  // The fold's searches over its plots, complete plots first and the gaps
  // after them: by species over a neighbour graph, by position over a k-d
  // tree; only complete plots are neighbours.
  const maxK = Math.max(...kGrid);
  const kSpecies = Math.min(maxK, nCal - 1);
  const kSpatial = spatialAxis ? Math.min(maxK, nCalLocated - 1) : 0;
  const sets = new PresenceSets([...speciesCal, ...speciesTest]);
  const graph = await SetNeighbourGraph.build(sets, nCal, { k: Math.max(1, kSpecies) }, pool);
  const open = await graph.open(nCal, nTest);
  const places = spatialAxis ? new PlaceIndex(Float64Array.from([...coordsCal.v, ...coordsTest.v]), nCal, pool) : null;
  const calRows = Array.from({ length: nCal }, (_, i) => i);
  const gapRows = Array.from({ length: nTest }, (_, g) => nCal + g);

  // Each gap's distance to its nearest complete plot on every axis; a gap
  // sharing no species with any complete plot is at similarity 0.
  const simGap = Float64Array.from({ length: nTest }, (_, g) => {
    const s = open.similarities[g * open.k];
    return Number.isNaN(s) ? 0 : s;
  });
  const kmGap = places ? (await places.nearest(gapRows, 1)).km : null;
  const yearsGap = timeAxis ? nearestYearGap(yearsTest, yearsCal) : null;

  // One hiding distance per complete plot and axis.
  let radii;
  if (injected) {
    radii = { similarity: Float64Array.from(injected.similarity),
              km: injected.km ? Float64Array.from(injected.km) : null,
              years: injected.years ? Float64Array.from(injected.years) : null };
  } else {
    const generator = new Mt19937(seed);
    radii = { similarity: drawn(Array.from(simGap), nCal, generator),
              km: spatialAxis ? drawn(Array.from(kmGap), nCal, generator) : null,
              years: timeAxis ? drawn(yearsGap, nCal, generator) : null };
  }
  if (!spatialAxis) radii.km = null;
  if (!timeAxis) radii.years = null;

  // The rules hiding a complete plot from another at the drawn radii.
  const rules = new HidingRules(radii, coordsCal && coordsCal.v, yearsCal);

  // Each complete plot's neighbours among the complete plots the radii leave
  // visible (leave-one-out), by species and by place.
  const matched = [
    kSpecies >= 1 ? await graph.hidden(nCal, rules) : null,
    places && kSpatial >= 1 ? await places.nearest(calRows, kSpatial, rules, sets) : null,
  ];

  // The gaps' neighbours among the complete plots, of one kind: (nTest, width).
  const gapNeighbours = async (kind, width) => {
    if (kind === SPATIAL) return (await places.nearest(gapRows, width)).indices;
    const out = new Int32Array(nTest * width);
    for (let g = 0; g < nTest; g++) out.set(open.indices.subarray(g * open.k, g * open.k + width), g * width);
    return out;
  };

  // Choose k by the headline error of the leave-one-out median at the gaps'
  // distance; the first k of the grid wins a tie.
  const allCal = Array.from({ length: nCal }, (_, i) => i);
  const allTest = Array.from({ length: nTest }, (_, i) => i);
  const terms = [null, null];
  for (const kind of [SPECIES, SPATIAL]) {
    if (!matched[kind]) continue;
    const width = kind === SPECIES ? kSpecies : kSpatial;
    const nb = matched[kind].k === width ? matched[kind].indices : narrowed(matched[kind], width);
    const has = allCal.filter((i) => nb[i * width] >= 0);
    const truthHas = pick(truth, has);
    const choice = { kind: KIND_NAMES[kind], k: 0, matched_error: [] };
    let best = NaN, any = false;
    for (const k of kGrid) {
      if (k > width) continue;
      const error = plane.aggregate(plane.errors(neighbourMedian(truth, nb, width, has, k, circular), truthHas));
      choice.matched_error.push([k, error]);
      if (!any || error < best) { best = error; choice.k = k; }
      any = true;
    }
    if (!any) continue;
    terms[kind] = { choice,
                    cal: neighbourMedian(truth, nb, width, allCal, choice.k, circular),
                    test: neighbourMedian(truth, await gapNeighbours(kind, width), width, allTest, choice.k, circular) };
    report.neighbours.push(choice);
  }

  report.axes.push(summariseAxis("similarity", Array.from(simGap), true));
  report.axes.push(summariseAxis("km", kmGap && Array.from(kmGap),
                                 spatialAxis && !!radii.km && Array.from(kmGap).some(Number.isFinite)));
  report.axes.push(summariseAxis("years", yearsGap, timeAxis && !!radii.years));

  // Every candidate this fit can evaluate, in tie-break order.
  const centre = plane.centre(truth);
  const constant = (m) => { const out = rowsOf(m, centre.length); for (let i = 0; i < m; i++) out.v.set(centre, i * centre.length); return out; };
  const usable = [];
  for (const spec of CANDIDATES) {
    if (spec.reads[SPATIAL] && !spatialAxis) continue;
    if (spec.reads.some((r, kind) => r && !terms[kind])) continue;
    let fit;
    if (spec.combination === "constant") fit = { cal: constant(nCal), test: constant(nTest) };
    else if (spec.combination === "raw") fit = { cal: rawCal, test: rawTest };
    else if (spec.combination === "neighbours") {
      const kind = spec.reads[SPECIES] ? SPECIES : SPATIAL;
      fit = { cal: terms[kind].cal, test: terms[kind].test, k: [[KIND_NAMES[kind], terms[kind].choice.k]] };
    } else {
      const calX = [rawCal], testX = [rawTest], names = ["resolve"], k = [];
      for (const kind of [SPECIES, SPATIAL]) {
        if (!spec.reads[kind]) continue;
        calX.push(terms[kind].cal); testX.push(terms[kind].test); names.push(KIND_NAMES[kind]);
        k.push([KIND_NAMES[kind], terms[kind].choice.k]);
      }
      const ls = leastSquares(truth, calX, testX, circular);
      fit = { cal: ls.cal, test: ls.test, k, nFitted: ls.nFitted,
              coefficients: names.map((name, j) => describeCoefficient(name, ls.coefficients[j])) };
    }
    usable.push({ spec, fit });
  }

  // Gaps with and without coordinates choose apart; each is judged on the
  // complete plots it resembles, a complete plot without a neighbour at the
  // gaps' distance read as the local constant.
  const testRows = rowsOf(nTest, plane.d);
  const testMethod = new Int32Array(nTest).fill(GAP_RECORDED);
  const groups = [
    { name: "located", gaps: allTest.filter((g) => spatialAxis && located(coordsTest, g)),
      rows: allCal.filter((i) => !spatialAxis || located(coordsCal, i)), spatial: true },
    { name: "unlocated", gaps: allTest.filter((g) => !(spatialAxis && located(coordsTest, g))),
      rows: allCal, spatial: false },
  ];
  for (const group of groups) {
    if (!group.gaps.length) continue;
    const groupTruth = pick(truth, group.rows);
    const choice = { group: group.name, n_complete: group.rows.length, n_gaps: group.gaps.length,
                     method: 1, matched_error: [], coefficients: [], n_fitted: 0, k: [] };
    let best = null, bestError = NaN;
    for (const u of usable) {
      if (u.spec.reads[SPATIAL] && !group.spatial) continue;
      const judged = pick(u.fit.cal, group.rows);
      for (let i = 0; i < judged.v.length; i++) if (!Number.isFinite(judged.v[i])) judged.v[i] = centre[i % centre.length];
      const error = plane.aggregate(plane.errors(judged, groupTruth));
      choice.matched_error.push([u.spec.method, error]);
      if (!best || error < bestError) { best = u; bestError = error; }
    }
    if (!best) throw new Error(`gap fitting: target '${target.name}' has ${group.gaps.length} ${group.name} gaps and no candidate for them`);
    choice.method = best.spec.method;
    choice.coefficients = best.fit.coefficients ?? [];
    choice.n_fitted = best.fit.nFitted ?? 0;
    choice.k = best.fit.k ?? [];
    for (const g of group.gaps) {
      for (let j = 0; j < plane.d; j++) testRows.v[g * plane.d + j] = best.fit.test.v[g * plane.d + j];
      testMethod[g] = choice.method;
    }
    report.groups.push(choice);
  }

  test.forEach((i, g) => {
    const value = plane.backRow(testRows, g);
    if (d === 2) { fitted[2 * i] = value[0]; fitted[2 * i + 1] = value[1]; } else fitted[i] = value;
    method[i] = testMethod[g];
  });
  const widthOf = (kind, width) => (matched[kind] ? narrowed(matched[kind], width) : null);
  return { fitted, method, report, radii,
           matched: { species: widthOf(SPECIES, kSpecies), spatial: widthOf(SPATIAL, kSpatial), kSpecies, kSpatial } };
}
