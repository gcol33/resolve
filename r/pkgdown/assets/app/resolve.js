// Encoding plots for a released model and predicting with its ONNX graph, in
// the browser or in Node.
//
// A model ships as `model_fp16.onnx` (or `model_fp32.onnx`) and `model.json`,
// both written by app/port/export_onnx.py. `model.json` holds the vocabularies
// and the dataset settings; `encode` follows the engine's loader with them
// (dataset.cpp, species_encoding.cpp, continuous_block.cpp), so a plot reaches
// the graph as the tensors the engine would have built for it.

export const COLUMNS = {
  plot: "PlotObservationID",
  species: "WFO_TAXON",
  abundance: "Cover %",
  genus: "WFO_GENUS",
  family: "WFO_FAMILY",
  latitude: "Latitude",
  longitude: "Longitude",
};

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// The delimiter of a delimited text table, read from its first line: the one
// of comma, semicolon and tab that occurs most often outside quotes.
export function detectDelimiter(text) {
  const end = text.indexOf("\n");
  const line = end < 0 ? text : text.slice(0, end);
  let best = ",", bestCount = -1;
  for (const d of [",", ";", "\t"]) {
    let count = 0, quoted = false;
    for (const c of line) {
      if (c === '"') quoted = !quoted;
      else if (c === d && !quoted) count++;
    }
    if (count > bestCount) { best = d; bestCount = count; }
  }
  return best;
}

// Rows of a delimited table as objects keyed by the header, with quoted
// fields, doubled quotes inside them and CRLF or LF line ends.
export function parseTable(text, delimiter = detectDelimiter(text)) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let field = "", row = [], quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === delimiter) { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [names, ...body] = rows;
  return body.map((cells) => Object.fromEntries(names.map((n, j) => [n, cells[j]])));
}

// ---------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------

const NA = new Set(["na", "n/a", "nan", "null", "none"]);
const NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

// A numeric cell as a double, or null for an empty, NA, unparseable or
// non-finite value: what the engine reads for plot-level columns after
// prediction (plot_columns.cpp, parse_cell_at<double>).
export function parseDouble(cell) {
  if (cell === undefined || cell === null) return null;
  const s = String(cell).trim();
  if (s === "" || s === "." || s === "-" || NA.has(s.toLowerCase())) return null;
  if (!NUMBER.test(s)) return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
}

// The same cell as the float32 a model's continuous block holds
// (row_cells.hpp parse_numeric_cell).
export function parseNumber(cell) {
  const v = parseDouble(cell);
  if (v === null) return null;
  const f = Math.fround(v);
  return Number.isFinite(f) ? f : null;
}

// Strings in code point order, which is the byte order of their UTF-8 form
// that the engine sorts names by.
export function compareNames(a, b) {
  if (a === b) return 0;
  const x = [...a], y = [...b];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (d !== 0) return d;
  }
  return x.length - y.length;
}

// ---------------------------------------------------------------------------
// Spelling (the engine's FuzzyIndex, fuzzy.hpp)
// ---------------------------------------------------------------------------

// Damerau-Levenshtein distance (adjacent transpositions, each edit applied
// once) between two code point arrays, or k + 1 once it must exceed k.
function editDistance(a, b, k) {
  if (Math.abs(a.length - b.length) > k) return k + 1;
  let prev2 = null, prev = Array.from({ length: b.length + 1 }, (_, j) => j), prevBest = 0;
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let d = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost);
      if (prev2 && i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d = Math.min(d, prev2[j - 2] + 1);
      row.push(d);
      if (d < best) best = d;
    }
    // A transposition reaches back two rows, so two rows past k end the search.
    if (best > k && prevBest > k) return k + 1;
    prevBest = best;
    prev2 = prev;
    prev = row;
  }
  return prev[b.length];
}

const fold = (s) => [...s.toLowerCase()];
const genusOf = (s) => s.split(" ", 1)[0].toLowerCase();

// Code point counts in 32 classes (code point mod 32). Each edit changes the
// counts of two strings by at most one in each direction, and a transposition
// not at all, so the larger of the summed excesses either way is a lower bound
// on their distance: an entry whose bound exceeds the best distance so far
// cannot be among the nearest, and its distance need not be computed.
const BAG = 32;

function bagOf(cps) {
  const bag = new Uint8Array(BAG);
  for (const c of cps) {
    const i = c.codePointAt(0) & (BAG - 1);
    if (bag[i] < 255) bag[i] += 1;
  }
  return bag;
}

function bagBound(a, b) {
  let more = 0, less = 0;
  for (let i = 0; i < BAG; i++) {
    const d = a[i] - b[i];
    if (d > 0) more += d; else less -= d;
  }
  return Math.max(more, less);
}

// An index over names, compared without regard to case, partitioned by
// genus (the first word). A query searches its own genus first and the whole
// index only when that genus holds nothing within the distance, as the
// engine's bucket hint does.
export function spellingIndex(names) {
  const all = names.map((name) => {
    const cps = fold(name);
    return { name, cps, bag: bagOf(cps) };
  });
  const genera = new Map();
  for (const e of all) {
    const g = genusOf(e.name);
    if (!genera.has(g)) genera.set(g, []);
    genera.get(g).push(e);
  }
  return { all, genera };
}

// The entries nearest to `name` within `k` edits, with their distance.
export function nearest(index, name, k) {
  const cps = fold(name);
  const bag = bagOf(cps);
  const scan = (entries) => {
    let best = k + 1, hits = [];
    for (const e of entries) {
      if (Math.abs(e.cps.length - cps.length) > best || bagBound(bag, e.bag) > best) continue;
      const d = editDistance(cps, e.cps, best);
      if (d < best) { best = d; hits = [e.name]; } else if (d === best && d <= k) hits.push(e.name);
    }
    return { distance: best, names: hits };
  };
  const own = index.genera.get(genusOf(name));
  const inGenus = own ? scan(own) : { names: [] };
  return inGenus.names.length ? inGenus : scan(index.all);
}

// The token a species name stands for, by the engine's NameResolver
// (name_resolution.cpp), or null where the vocabulary holds the name as
// written or nothing reaches a token: `known` is the vocabulary without its
// "<UNK>" code, `aliases` the release's name table {name: [token, source]},
// `k` the edits a misspelling may be away (0: no spelling step), and `cache`
// an object the caller keeps per vocabulary for the spelling index. The
// table's route carries its source; a spelling carries the entry it was read
// via and the distance, and the source of that entry when it is a table name.
export function resolveName(name, known, aliases, k, cache) {
  if (!name || known.has(name)) return null;
  const [token, source] = aliases[name] ?? [];
  if (token && known.has(token)) return { token, route: "alias", source };
  if (!k) return null;
  cache.index ??= spellingIndex([
    ...known,
    ...Object.keys(aliases).filter((a) => !known.has(a) && known.has(aliases[a][0])),
  ]);
  const near = nearest(cache.index, name, k);
  const tokens = new Set(near.names.map((n) => (known.has(n) ? n : aliases[n][0])));
  if (tokens.size !== 1) return null;
  const [target] = tokens;
  const via = near.names[0];
  return { token: target, route: "spelling", source: known.has(via) ? "" : aliases[via][1], via, distance: near.distance };
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

function poolWeight(value, weighting) {
  switch (weighting) {
    case "binary": return 1;
    case "abundance": return value;
    case "log1p": return Math.fround(Math.log1p(value));
    default: throw new Error(`pool weighting ${weighting} is not supported`);
  }
}

// The model inputs for the plots of `header` (one row per plot; a repeated
// plot ID keeps its first row) and the species records of `species`, as typed
// arrays in row-major order, with the plot IDs in the order of the rows.
export function encode(meta, header, species) {
  if (meta.selection !== "all" && meta.species_budget > 0) {
    throw new Error(`species selection ${meta.selection} is not supported`);
  }
  const vocab = meta._vocab ??= {
    species: new Map(meta.species.map((n, i) => [n, i])),
    genera: new Map(meta.genera.map((n, i) => [n, i])),
    families: new Map(meta.families.map((n, i) => [n, i])),
  };
  const presence = meta.representation === "presence_absence";

  const plotIds = [], plotOf = new Map(), headerRows = [];
  for (const row of header) {
    const id = String(row[COLUMNS.plot]);
    if (plotOf.has(id)) continue;
    plotOf.set(id, plotIds.length);
    plotIds.push(id);
    headerRows.push(row);
  }
  const n = plotIds.length;

  // One entry per species per plot where its first record stood, with the
  // sum of its records' abundances; each species' genus and family the
  // smallest name its records give over the plots kept.
  const hasAbundance = species.length > 0 && COLUMNS.abundance in species[0];
  const entries = Array.from({ length: n }, () => new Map());
  const genusOf = new Map(), familyOf = new Map();
  const smallest = (map, name, taxon) => {
    if (!taxon) return;
    const current = map.get(name);
    if (current === undefined || compareNames(taxon, current) < 0) map.set(name, taxon);
  };
  for (const row of species) {
    const p = plotOf.get(String(row[COLUMNS.plot]));
    if (p === undefined) continue;
    const name = row[COLUMNS.species] ?? "";
    let abundance = 1;
    if (hasAbundance) {
      const parsed = parseNumber(row[COLUMNS.abundance]);
      if (parsed !== null) abundance = parsed === 0 ? meta.zero_abundance_as : parsed;
    }
    const plot = entries[p];
    plot.set(name, Math.fround((plot.get(name) ?? 0) + abundance));
    if (meta.use_taxonomy) {
      smallest(genusOf, name, row[COLUMNS.genus]);
      smallest(familyOf, name, row[COLUMNS.family]);
    }
  }

  const value = (ranking) => (presence ? 1 : ranking);
  const lengths = entries.map((e) => e.size);
  // A capped model reads exactly `pool_species_cap` slots: some exported
  // graphs fix the slot axis at the cap, and the masked padding of a
  // narrower plot leaves every output unchanged.
  const widest = Math.max(1, ...lengths);
  const width = meta.pool_species_cap > 0 ? meta.pool_species_cap : widest;

  const speciesIds = new BigInt64Array(n * width);
  const genusIds = new BigInt64Array(n * width);
  const familyIds = new BigInt64Array(n * width);
  const weights = new Float32Array(n * width);
  const mask = new Uint8Array(n * width);
  const unknownFraction = new Float32Array(n);
  const unknownCount = new Float32Array(n);
  const nSpecies = new Float32Array(n);

  for (let p = 0; p < n; p++) {
    let kept = [...entries[p]];  // [name, summed abundance] in plot order
    let total = 0, unknown = 0, nUnknown = 0;
    for (const [name, ranking] of kept) {
      total += value(ranking);
      if (!vocab.species.has(name)) { unknown += value(ranking); nUnknown++; }
    }
    unknownCount[p] = nUnknown;
    nSpecies[p] = kept.length;
    unknownFraction[p] = total > 0 ? Math.fround(unknown / total) : 0;

    if (kept.length > width) {
      const order = kept.map((_, i) => i).sort((a, b) =>
        kept[b][1] - kept[a][1] || compareNames(kept[a][0], kept[b][0]));
      const keep = new Set(order.slice(0, width));
      kept = kept.filter((_, i) => keep.has(i));
    }
    kept.forEach(([name, ranking], j) => {
      const k = p * width + j;
      speciesIds[k] = BigInt(vocab.species.get(name) ?? 0);
      const genus = genusOf.get(name), family = familyOf.get(name);
      genusIds[k] = BigInt(genus === undefined ? 0 : vocab.genera.get(genus) ?? 0);
      familyIds[k] = BigInt(family === undefined ? 0 : vocab.families.get(family) ?? 0);
      weights[k] = poolWeight(value(ranking), meta.pool_weighting);
      mask[k] = 1;
    });
  }

  // The continuous block in the engine's column order: coordinates (and one
  // flag for the pair), covariates (and one flag each), the unknown-species
  // fraction and count. Missing values stay NaN under "indicate" and are
  // filled inside the graph; under "zero" they read as 0 with no flag.
  const indicate = meta.missing_values === "indicate";
  const columns = [];
  const numbers = (key) => headerRows.map((r) => parseNumber(r[key]) ?? NaN);
  const finite = (v) => (indicate ? v : v.map((x) => (Number.isNaN(x) ? 0 : x)));
  if (meta.has_coordinates) {
    const lon = numbers(COLUMNS.longitude), lat = numbers(COLUMNS.latitude);
    columns.push(finite(lon), finite(lat));
    if (indicate) columns.push(lon.map((x, i) => (Number.isNaN(x) || Number.isNaN(lat[i]) ? 1 : 0)));
  }
  const covariates = meta.covariates.map(numbers);
  columns.push(...covariates.map(finite));
  if (indicate) columns.push(...covariates.map((c) => c.map((x) => (Number.isNaN(x) ? 1 : 0))));
  if (meta.track_unknown_fraction) columns.push([...unknownFraction]);
  if (meta.track_unknown_count) columns.push([...unknownCount]);
  if (columns.length !== meta.continuous_width) {
    throw new Error(`continuous block has ${columns.length} columns, the model reads ${meta.continuous_width}`);
  }
  const continuous = new Float32Array(n * columns.length);
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < columns.length; c++) continuous[p * columns.length + c] = columns[c][p];
  }
  const hasCover = new Float32Array(n).fill(hasAbundance && !presence ? 1 : 0);
  // A plot none of whose species the vocabulary holds is not imputed
  // (suite.cpp, withhold_unimputed): the models never saw such a plot.
  const imputed = Uint8Array.from(nSpecies, (s, p) => (s - unknownCount[p] > 0 ? 1 : 0));

  return { plotIds, n, width, continuous, speciesIds, genusIds, familyIds, weights, mask,
           hasCover, unknownFraction, unknownCount, nSpecies, imputed };
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

// Predictions for every encoded plot, in batches: per target the class
// probabilities (n x classes) or the value in original units (n). `ort` is onnxruntime-web (or onnxruntime-node) and `session` an
// InferenceSession on the model's graph.
export async function predict(ort, session, meta, encoded, { batchSize = 2048, onBatch } = {}) {
  const { n, width } = encoded;
  const k = meta.continuous_width;
  const out = Object.fromEntries(meta.targets.map((t) =>
    [t.name, new Float32Array(n * (t.task === "classification" ? t.classes.length : 1))]));
  for (let start = 0; start < n; start += batchSize) {
    const b = Math.min(batchSize, n - start);
    const rows = (array, size) => array.subarray(start * size, (start + b) * size);
    const feeds = {
      continuous: new ort.Tensor("float32", rows(encoded.continuous, k), [b, k]),
      species_ids: new ort.Tensor("int64", rows(encoded.speciesIds, width), [b, width]),
      genus_ids: new ort.Tensor("int64", rows(encoded.genusIds, width), [b, width]),
      family_ids: new ort.Tensor("int64", rows(encoded.familyIds, width), [b, width]),
      weights: new ort.Tensor("float32", rows(encoded.weights, width), [b, width]),
      mask: new ort.Tensor("bool", rows(encoded.mask, width), [b, width]),
      has_cover: new ort.Tensor("float32", rows(encoded.hasCover, 1), [b]),
    };
    const results = await session.run(feeds);
    for (const t of meta.targets) {
      const size = t.task === "classification" ? t.classes.length : 1;
      out[t.name].set(results[t.name].data, start * size);
    }
    onBatch?.((start + b) / n);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Combining a target's seeds (suite.cpp combine_vote / combine_mean /
// combine_spherical)

const EARTH_RADIUS_KM = 6371;
const CANCELLED_RESULTANT = 1e-12;
// ---------------------------------------------------------------------------

// One member's class probabilities read at a coarser level: each plot's
// probability of a coarser class is the sum over the finer classes whose
// ancestor it is (`index[c]`, the ancestor of fine class c).
export function sumByAncestor(probs, n, index, nCoarse) {
  const nFine = index.length;
  const out = new Float32Array(n * nCoarse);
  for (let p = 0; p < n; p++) {
    for (let c = 0; c < nFine; c++) out[p * nCoarse + index[c]] += probs[p * nFine + c];
  }
  return out;
}

// A combined prediction (combineVote, combineMean, combineSpherical) with
// every plot that is not imputed blanked in place, as suite.cpp does: NaN for
// a number, -1 for a class code. Each array holds n plots of equal width.
export function withholdUnimputed(combined, imputed) {
  const n = imputed.length;
  for (const values of Object.values(combined)) {
    const width = values.length / n;
    const blank = values instanceof Int32Array ? -1 : NaN;
    for (let p = 0; p < n; p++) if (!imputed[p]) values.fill(blank, p * width, (p + 1) * width);
  }
  return combined;
}

export function combineVote(members, nClasses, n) {
  const value = new Int32Array(n), agreement = new Float32Array(n), mean = new Float64Array(n * nClasses);
  for (let p = 0; p < n; p++) {
    const votes = new Array(nClasses).fill(0);
    for (const probs of members) {
      let best = 0;
      for (let c = 0; c < nClasses; c++) {
        mean[p * nClasses + c] += probs[p * nClasses + c] / members.length;
        if (probs[p * nClasses + c] > probs[p * nClasses + best]) best = c;
      }
      votes[best]++;
    }
    const most = Math.max(...votes);
    let pick = -1;
    for (let c = 0; c < nClasses; c++) {
      if (votes[c] === most && (pick < 0 || mean[p * nClasses + c] > mean[p * nClasses + pick])) pick = c;
    }
    value[p] = pick;
    agreement[p] = votes[pick] / members.length;
  }
  return { value, agreement, mean };
}

export function combineMean(members, n) {
  const value = new Float64Array(n), sd = new Float64Array(n).fill(NaN);
  for (let p = 0; p < n; p++) {
    const v = members.map((m) => m[p]);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    value[p] = mean;
    if (v.length > 1) sd[p] = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1));
  }
  return { value, sd };
}

export function combineSpherical(lats, lons, n) {
  const rad = Math.PI / 180;
  const lat = new Float64Array(n), lon = new Float64Array(n), km = new Float64Array(n).fill(NaN);
  for (let p = 0; p < n; p++) {
    const pts = lats.map((la, i) => {
      const a = la[p] * rad, b = lons[i][p] * rad;
      return [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)];
    });
    const s = pts.reduce((acc, q) => acc.map((x, j) => x + q[j]), [0, 0, 0]);
    const norm = Math.hypot(...s);
    if (norm / pts.length < CANCELLED_RESULTANT) { lat[p] = lon[p] = NaN; continue; }
    const c = s.map((x) => x / norm);
    lat[p] = Math.atan2(c[2], Math.hypot(c[0], c[1])) / rad;
    lon[p] = Math.atan2(c[1], c[0]) / rad;
    if (pts.length > 1) {
      const angles = pts.map((q) => {
        const cross = [q[1] * c[2] - q[2] * c[1], q[2] * c[0] - q[0] * c[2], q[0] * c[1] - q[1] * c[0]];
        return Math.atan2(Math.hypot(...cross), q[0] * c[0] + q[1] * c[1] + q[2] * c[2]);
      }).sort((a, b) => a - b);
      const mid = (angles.length - 1) / 2;
      km[p] = ((angles[Math.floor(mid)] + angles[Math.ceil(mid)]) / 2) * EARTH_RADIUS_KM;
    }
  }
  return { lat, lon, km };
}
