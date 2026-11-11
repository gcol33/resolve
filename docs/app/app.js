// The page: read the user's tables, fetch the released models in the order
// they are needed, predict, combine each target's seeds the way the engine's
// suite does (suite.cpp), and show and export the result. Nothing leaves the
// browser except the requests for the model files.

import { COLUMNS, combineMean, combineSpherical, combineVote, encode, parseDouble, parseNumber, parseTable,
         resolveName, sumByAncestor } from "./resolve.js";
import { describeGapMethod, gapSource } from "./gaps.js";
import { eeaCell, eeaCellCode, eeaCellOutline } from "./geo.js";
import { HttpRangeSource, HttpStatusError, TerrainCorrection, TerrainField, describePlots, fetchRange,
         withRetry } from "./terrain.js";

// ---------------------------------------------------------------------------
// Cross-origin isolation, so the gap fitting's workers can share memory
// ---------------------------------------------------------------------------

// A host that sends the isolating headers (serve.py) needs nothing. Otherwise
// the site's service worker (coi_sw.js) adds them, and the page reloads once
// so that the worker serves it. A browser without service workers, or one
// where the reload did not isolate the page, carries on unisolated: the gap
// fitting then searches on one thread, with the same results.
const ISOLATION_RELOAD = "resolve-isolation-reload";

async function isolate() {
  if (self.crossOriginIsolated || !("serviceWorker" in navigator)) return;
  try {
    if (sessionStorage.getItem(ISOLATION_RELOAD)) return;
  } catch {
    return;
  }
  const controlled = navigator.serviceWorker.controller !== null;
  try {
    await navigator.serviceWorker.register(new URL("../coi_sw.js", import.meta.url),
                                           { scope: new URL("../", import.meta.url).href });
  } catch {
    return;
  }
  if (controlled) return;
  const ready = await Promise.race([navigator.serviceWorker.ready.then(() => true),
                                    new Promise((r) => setTimeout(() => r(false), 5000))]);
  if (!ready) return;
  try {
    sessionStorage.setItem(ISOLATION_RELOAD, "1");
  } catch {
    return;
  }
  location.reload();
  await new Promise(() => {});
}

await isolate();

// Where the page's files are: the model pack and its index, the terrain field
// and the example data, named on the page's element (data-models-pack,
// data-models-index, data-terrain-field, data-examples) or, for a check
// against other files, in the address (?pack=, ?index=, ?terrain=). The pack
// and its index may each name several addresses, separated by spaces, one per
// mirror and in the same order: the page reads from the first mirror that
// answers and moves on to the next when it fails.
const PARAMS = new URLSearchParams(location.search);
const PLACES = document.getElementById("resolve-app").dataset;
const given = (param, key) => PARAMS.get(param) ?? PLACES[key];
const place = (param, key) => new URL(given(param, key), location.href).href;
const places = (param, key) => given(param, key).trim().split(/\s+/).map((u) => new URL(u, location.href).href);
const PACKS = places("pack", "modelsPack");
const PACK_INDEXES = places("index", "modelsIndex");
const MIRRORS = PACKS.map((pack, i) => ({ pack, index: PACK_INDEXES[i] }));
const TERRAIN_FIELD = place("terrain", "terrainField");
const EXAMPLES = place("examples", "examples");
const CACHE = "resolve-models";

const $ = (id) => document.getElementById(id);
const state = { suites: null, tables: {}, result: null, files: new Map(), queue: [], level: 1 };

// ---------------------------------------------------------------------------
// Model files: one queue, fetched in order, cached across visits
// ---------------------------------------------------------------------------

// A promise's value, or null once `ms` have passed: a browser that blocks
// storage can leave a Cache API call pending without ever settling it.
const within = (promise, ms) =>
  Promise.race([promise.catch(() => null), new Promise((r) => setTimeout(() => r(null), ms))]);

// The released model files are one uncompressed tar (app/pack_models.py in
// the experiment repository): its index gives each file's offset and length
// in the tar and the tar's SHA-256 as the pack's version, and each file is
// read as one range of the tar.
async function readIndex(mirror) {
  const index = await withRetry(async () => {
    const response = await fetch(mirror.index, { cache: "no-store" });
    if (!response.ok) throw new HttpStatusError(mirror.index, response.status, 200);
    return response.json();
  });
  if (index.format !== 1) throw new Error(`model pack format ${index.format} is not one this page reads`);
  return index;
}

// The index of the first mirror that answers; that mirror is the one read.
let packOpened;
let mirrorInUse = 0;
const openPack = () => (packOpened ??= (async () => {
  if (PACKS.length !== PACK_INDEXES.length) {
    throw new Error(`${PACKS.length} model pack addresses and ${PACK_INDEXES.length} index addresses; each mirror needs both`);
  }
  for (let m = 0; ; m++) {
    try {
      const index = await readIndex(MIRRORS[m]);
      mirrorInUse = m;
      return index;
    } catch (e) {
      if (m + 1 >= MIRRORS.length) throw e;
    }
  }
})().catch((e) => { packOpened = null; throw e; }));

// A read of the pack (`read(url)`) from the mirror in use. When it fails, the
// read is repeated on the next mirror whose index names the same version, so
// every range comes from the same tar; a mirror holding another version is
// skipped.
async function fromPack(read) {
  const { version } = await openPack();
  for (;;) {
    const m = mirrorInUse;
    try {
      return await read(MIRRORS[m].pack);
    } catch (e) {
      if (!(await nextMirror(m, version))) throw e;
    }
  }
}

async function nextMirror(failed, version) {
  if (mirrorInUse !== failed) return true;
  for (let m = failed + 1; m < MIRRORS.length; m++) {
    const index = await readIndex(MIRRORS[m]).catch(() => null);
    if (index?.version === version) {
      if (mirrorInUse === failed) mirrorInUse = m;
      return true;
    }
  }
  return false;
}

// Where a file lies in the pack, as [begin, end).
async function packRange(path) {
  const index = await openPack();
  const at = index.files[path];
  if (!at) throw new Error(`${path} is not in the model pack`);
  return [at[0], at[0] + at[1]];
}

// A path relative to a file of the pack, as the pack names it.
const packPath = (relative, from) => new URL(relative, `https://pack/${from}`).pathname.slice(1);

// Model files are kept in IndexedDB, which a page has over plain HTTP as well
// as HTTPS, under the pack's version, so a file read once is read from the
// browser while the pack stays the same. A file is stored under the first
// mirror's address whichever mirror it came from.
let storeOpened;
const openStore = () => (storeOpened ??= "indexedDB" in window ? within(new Promise((resolve, reject) => {
  const request = indexedDB.open(CACHE, 1);
  request.onupgradeneeded = () => request.result.createObjectStore("files");
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
}), 2000) : Promise.resolve(null));

const stored = (db, mode, op) => new Promise((resolve, reject) => {
  const request = op(db.transaction("files", mode).objectStore("files"));
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

async function cached(path, onBytes) {
  const { version } = await openPack();
  const key = `${MIRRORS[0].pack}#${path}`;
  const db = await openStore();
  const kept = db && await within(stored(db, "readonly", (s) => s.get(key)), 2000);
  if (kept?.version === version) {
    onBytes?.(kept.bytes.byteLength, kept.bytes.byteLength);
    return kept.bytes;
  }
  const [begin, end] = await packRange(path);
  const bytes = (await fromPack((url) => withRetry(() => fetchRange(fetch, url, `bytes=${begin}-${end - 1}`, { onBytes })))).buffer;
  if (bytes.byteLength !== end - begin) throw new Error(`${path}: ${bytes.byteLength} bytes read from the pack, not ${end - begin}`);
  if (db) stored(db, "readwrite", (s) => s.put({ version, bytes }, key)).catch(() => { /* quota */ });
  return bytes;
}

// Model graphs download a few at a time, in the order asked for, so the first
// members arrive whole and start predicting while the rest download; all at
// once, they would share the bandwidth and all finish at the end.
const GRAPH_DOWNLOADS = 2;
const graphQueue = { running: 0, waiting: [] };

async function queued(task) {
  if (graphQueue.running >= GRAPH_DOWNLOADS) await new Promise((r) => graphQueue.waiting.push(r));
  graphQueue.running += 1;
  try {
    return await task();
  } finally {
    graphQueue.running -= 1;
    graphQueue.waiting.shift()?.();
  }
}

// The share of each graph received, by path, and the graph each member is to
// run with, by member.
const received = new Map();
const graphPaths = new Map();
const receivedOf = (member) => received.get(graphPaths.get(member)) ?? 0;

// The bytes of a model file, fetched once; a file asked for again joins the
// first request.
function file(path) {
  if (path.endsWith(".onnx")) graphPaths.set(path.slice(0, path.lastIndexOf("/")), path);
  if (!state.files.has(path)) {
    if (path.endsWith(".onnx")) {
      const track = (got, total) => { received.set(path, got / total); progress(); showBusy(); };
      state.files.set(path, queued(() => cached(path, track)));
    } else {
      state.files.set(path, cached(path));
    }
  }
  return state.files.get(path);
}

// The graph a member is run with: the one storing its weights in float16,
// cast back to float32 where read, which halves the download and on the
// released members' check plots moves no EUNIS class and a regression value
// by far less than its error (port/export_onnx.py). For a check against the
// engine, ?precision=full in the address runs the float32 graph instead.
const FULL_PRECISION = PARAMS.get("precision") === "full";
const graphOf = (meta) => (FULL_PRECISION ? meta.graph : meta.graph_half ?? meta.graph);

// Members in the order a page needs them: the targets in the order chosen,
// EUNIS first by default. A preload stops once `wanted` says its targets are
// no longer the chosen ones.
async function preload(targets, wanted = () => true) {
  for (const t of targets) {
    for (const member of t.members) {
      if (!wanted()) return;
      const meta = JSON.parse(new TextDecoder().decode(await file(`${member}/model.json`)));
      if (!wanted()) return;
      await file(`${member}/${graphOf(meta)}`);
    }
  }
}

function progress() {
  if (!state.suites) return;
  const members = chosenTargets().flatMap((t) => t.members);
  const done = members.filter((m) => receivedOf(m) === 1).length;
  $("download-status").textContent = done < members.length ? `Downloading the models: ${done} of ${members.length}` : "";
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// The fields the page reads from each table, in the order they are shown.
// Each is recognised by the first of its names the table has, compared
// without regard to case, spaces or punctuation: the archive's own names,
// Darwin Core terms (https://dwc.tdwg.org/terms/) and common short forms.
const UNCERTAINTY = "Location uncertainty (m)";
const PLOT_NAMES = ["PlotObservationID", "eventID", "PlotID", "RELEVE_NR", "plot", "releve", "id"];
const FIELDS = {
  species: [
    { role: COLUMNS.plot, label: "Plot ID", required: true, names: PLOT_NAMES },
    { role: COLUMNS.species, label: "Species", required: true,
      names: ["WFO_TAXON", "TaxonName", "scientificName", "acceptedNameUsage", "verbatimScientificName",
              "Matched concept", "Turboveg2 concept", "species", "taxon", "speciesname"] },
    { role: COLUMNS.abundance, label: "Cover (%)",
      names: ["Cover %", "Cover_Perc", "coverpercent", "cover", "organismQuantity", "abundance"] },
    { role: COLUMNS.genus, label: "Genus", names: ["WFO_GENUS", "genus"] },
    { role: COLUMNS.family, label: "Family", names: ["WFO_FAMILY", "family"] },
  ],
  header: [
    { role: COLUMNS.plot, label: "Plot ID", required: true, names: PLOT_NAMES },
    { role: COLUMNS.latitude, label: "Latitude", names: ["Latitude", "decimalLatitude", "lat"] },
    { role: COLUMNS.longitude, label: "Longitude", names: ["Longitude", "decimalLongitude", "lon", "lng", "long"] },
    { role: "Year", label: "Year", names: ["Year", "surveyyear"] },
    { role: "Month", label: "Month", names: ["Month", "surveymonth"] },
    { role: "Date", label: "Date", names: ["eventDate", "Date", "surveydate", "relevedate"] },
    { role: "Altitude (m)", label: "Altitude (m)",
      names: ["Altitude (m)", "Altitude", "elevation", "minimumElevationInMeters"] },
    { role: "Slope (°)", label: "Slope (°)", names: ["Slope (°)", "Slope"] },
    { role: "Relevé area (m²)", label: "Plot area (m²)",
      names: ["Relevé area (m²)", "Releve area", "plotarea", "area", "sampleSizeValue"] },
    { role: UNCERTAINTY, label: "Location uncertainty (m)",
      names: [UNCERTAINTY, "coordinateUncertaintyInMeters", "locationuncertainty", "uncertainty"] },
  ],
};
const key = (s) => s.toLowerCase().replace(/[^a-z0-9%]/g, "");

// Each table as read (name, columns, rows) and the column chosen per field.
const raw = {}, chosen = {};

function recognise(which, cols) {
  const keyed = new Map();
  for (const c of cols) if (!keyed.has(key(c))) keyed.set(key(c), c);
  return Object.fromEntries(FIELDS[which].map((f) =>
    [f.role, f.names.map((n) => keyed.get(key(n))).find(Boolean) ?? ""]));
}

// The plot fields come from the species table, first record per plot, when
// no header table is given (a Darwin Core occurrence file carries both).
const headerSource = () => raw.header ?? raw.species;

function setRaw(which, name, rows) {
  if (rows) {
    const cols = rows.length ? Object.keys(rows[0]) : [];
    raw[which] = { name, cols, rows };
    chosen[which] = recognise(which, cols);
  } else delete raw[which];
  if (!raw.header && raw.species) chosen.header = recognise("header", raw.species.cols);
  renderFields("species");
  renderFields("header");
  build();
}

// Year and month of a date: ISO 8601 (2004, 2004-06, 2004-06-15, or an
// interval, from its start), or day.month.year and day/month/year.
function dateParts(value) {
  const s = String(value ?? "").trim();
  let m = s.match(/^(\d{4})(?:-(\d{1,2}))?/);
  if (m) return { year: m[1], month: m[2] ? String(Number(m[2])) : "" };
  m = s.match(/^(\d{1,2})[./](\d{1,2})[./](\d{4})/);
  if (m) return { year: m[3], month: String(Number(m[2])) };
  return { year: "", month: "" };
}

function mapped(rows, map) {
  const fields = Object.entries(map).filter(([role, col]) => col && role !== "Date");
  return rows.map((r) => Object.fromEntries(fields.map(([role, col]) => [role, r[col]])));
}

function build() {
  const s = raw.species, h = headerSource();
  state.tables = {};
  if (s) {
    const rows = mapped(s.rows, chosen.species);
    state.tables.species = { name: s.name, rows, found: Object.keys(rows[0] ?? {}) };
  }
  if (!h) return;
  const map = chosen.header;
  let source = h.rows;
  if (!raw.header) {
    const seen = new Set();
    source = source.filter((r) => { const id = r[map[COLUMNS.plot]]; return !seen.has(id) && seen.add(id); });
  }
  const rows = mapped(source, map);
  if (map.Date) {
    source.forEach((r, i) => {
      const d = dateParts(r[map.Date]);
      if (!map.Year) rows[i].Year = d.year;
      if (!map.Month) rows[i].Month = d.month;
    });
  }
  const found = Object.keys(rows[0] ?? {});
  state.tables.header = { name: h.name, rows, found, fromSpecies: !raw.header };
}

function renderFields(which) {
  const box = $(`${which}-fields`), src = which === "header" ? headerSource() : raw.species;
  if (!src || !chosen[which]) { box.innerHTML = ""; return; }
  box.innerHTML = FIELDS[which].map((f) =>
    `<label><span>${escape(f.label)}${f.role === COLUMNS.plot ? '<span class="plot-link" hidden></span>' : ""}</span>` +
    ` <select class="form-select form-select-sm" data-role="${escape(f.role)}">` +
    `<option value="">${f.required ? "(choose a column)" : "(none)"}</option>` +
    src.cols.map((c) => `<option value="${escape(c)}"${c === chosen[which][f.role] ? " selected" : ""}>${escape(c)}</option>`).join("") +
    "</select></label>").join("");
  for (const select of box.querySelectorAll("select")) {
    select.addEventListener("change", () => {
      chosen[which][select.dataset.role] = select.value;
      build();
      refreshInputs();
    });
  }
}

async function readTable(input, which) {
  const f = input.files[0];
  setRaw(which, f?.name, f ? parseTable(await f.text()) : null);
  refreshInputs();
}

async function loadTable(which, url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  setRaw(which, url.split("/").pop(), parseTable(await response.text()));
}

// ---------------------------------------------------------------------------
// Example data: put into the file inputs as File objects when the page opens,
// so the inputs show their names and a chosen file replaces one of them
// ---------------------------------------------------------------------------

const DATA_INPUTS = { species: "species-file", header: "header-file" };
const example = { files: {}, description: "", source: "" };

function setFile(input, f) {
  const dt = new DataTransfer();
  if (f) dt.items.add(f);
  input.files = dt.files;
}

const isExample = (which) => {
  const f = $(DATA_INPUTS[which]).files[0];
  return Boolean(f) && f === example.files[which];
};

function showExampleNote() {
  const all = Object.keys(DATA_INPUTS).every(isExample);
  $("reset-data").hidden = all || !example.files.species;
  const species = state.tables.species, header = state.tables.header;
  $("example-note").hidden = !all || !species;
  if (all && species) {
    const plots = header?.rows.length ?? headerRows().length;
    const note = $("example-note");
    note.textContent = `The files below are the example data: ${example.description}, ` +
      `${plots.toLocaleString()} plots with ${species.rows.length.toLocaleString()} species records` +
      (example.source ? " (" : ". ");
    if (example.source) {
      const a = Object.assign(document.createElement("a"), { href: example.source, textContent: "source", target: "_blank", rel: "noopener" });
      note.append(a, "). ");
    }
    note.append("Choose your own files to replace them.");
  }
}

async function loadExample() {
  let spec;
  try {
    const response = await fetch(new URL("example.json", EXAMPLES));
    if (!response.ok) return;
    spec = await response.json();
  } catch { return; }
  for (const which of Object.keys(DATA_INPUTS)) {
    if (!spec[which]) continue;
    const response = await fetch(new URL(spec[which], EXAMPLES));
    if (!response.ok) throw new Error(`example ${which} table: ${response.status}`);
    example.files[which] = new File([await response.blob()], spec[which].split("/").pop(), { type: "text/csv" });
  }
  example.description = spec.description ?? "";
  example.source = spec.source ?? "";
  await resetData();
}

async function resetData() {
  for (const [which, id] of Object.entries(DATA_INPUTS)) {
    setFile($(id), example.files[which]);
    await readTable($(id), which);
  }
}

// The example tables belong together: a file of one's own in one input takes
// the example out of the other, and an emptied input takes its example back
// only while the other input holds the other example.
async function onDataInput(which) {
  const input = $(DATA_INPUTS[which]), other = which === "species" ? "header" : "species";
  if (!input.files.length) {
    if (example.files[which] && isExample(other)) setFile(input, example.files[which]);
  } else if (isExample(other)) {
    setFile($(DATA_INPUTS[other]), null);
    await readTable($(DATA_INPUTS[other]), other);
  }
  await readTable(input, which);
}

const headerRows = () => state.tables.header?.rows ?? [];

// The models for a target: those of the first suite, in the release's order
// of preference, whose models read only fields the tables give, so a target
// uses coordinates and date wherever the tables hold them.
function modelFor(name) {
  const have = new Set(state.tables.header?.found ?? []);
  for (const suite of state.suites) {
    const t = suite.targets.find((x) => x.name === name);
    if (t && t.inputs.every((c) => have.has(c))) return { ...t, suite };
  }
  return null;
}

// The plots that record a target, of all plots, where the tables hold its
// fields; null where they do not. A target recorded on every plot is not
// predicted; one recorded on some has its gaps fitted (fillGaps).
function recordedCount(t) {
  const header = state.tables.header;
  if (!header || !t.outputs.every((o) => header.found.includes(o))) return null;
  const recorded = header.rows.filter((r) => t.outputs.every((o) => parseNumber(r[o]) !== null)).length;
  return { recorded, total: header.rows.length };
}

function fullyRecorded(name) {
  const t = state.suites.flatMap((s) => s.targets).find((x) => x.name === name);
  const c = recordedCount(t);
  return c !== null && c.total > 0 && c.recorded === c.total;
}

// Habitat is one choice whatever its EUNIS level; the level is chosen beside
// it. A level is predicted by the models trained at it, or else read from a
// finer level's models by summing their class probabilities (`read_at`).
const targetNames = () => [...new Set(state.suites.flatMap((s) => s.targets.filter((t) => !(t.level > 1))
                                                                      .map((t) => t.name)))]
  .filter((name) => !fullyRecorded(name));

const habitatChoices = () => {
  const habitat = state.suites.flatMap((s) => s.targets).filter((t) => t.level);
  const levels = new Map();
  for (const t of habitat) if (!levels.has(t.level)) levels.set(t.level, { level: t.level, target: t });
  for (const t of habitat) {
    for (const level of Object.keys(t.read_at ?? {}).map(Number)) {
      if (!levels.has(level)) levels.set(level, { level, target: t, readAt: level });
    }
  }
  return [...levels.values()].sort((a, b) => a.level - b.level);
};
const habitatChoice = () => habitatChoices().find((c) => c.level === state.level) ?? habitatChoices()[0];

// The models a chosen variable is predicted with; for habitat, those of the
// chosen level, marked with the level they are read at when it is coarser.
function chosenModel(name) {
  if (name !== "eunis") return modelFor(name);
  const choice = habitatChoice();
  const t = choice && modelFor(choice.target.name);
  return t && choice.readAt ? { ...t, readAt: choice.readAt } : t;
}

function chosenTargets() {
  const on = new Set([...document.querySelectorAll("input[name=target]:checked")].map((i) => i.value));
  return targetNames().filter((n) => on.has(n)).map(chosenModel).filter(Boolean);
}

// How the plot IDs of the two tables meet. The header lists the plots that
// are imputed (resolve.js, encode): a species record of a plot the header
// lacks is left out, and a header plot without records has no species
// composition. Null without a header table or a chosen plot ID column.
function plotLink() {
  const species = state.tables.species, header = state.tables.header;
  if (!species || !header || header.fromSpecies || !chosen.species[COLUMNS.plot] || !chosen.header[COLUMNS.plot]) {
    return null;
  }
  const inSpecies = new Set(species.rows.map((r) => String(r[COLUMNS.plot])));
  const inHeader = new Set(header.rows.map((r) => String(r[COLUMNS.plot])));
  let shared = 0;
  for (const id of inHeader) if (inSpecies.has(id)) shared++;
  const noRecords = inHeader.size - shared, noHeader = inSpecies.size - shared;
  const kind = shared === 0 ? "broken" : noRecords === 0 && noHeader === 0 ? "linked" : "partial";
  const notes = kind !== "partial" ? [] : [
    ...(noRecords ? [`${noRecords.toLocaleString()} without species records`] : []),
    ...(noHeader ? [`${noHeader.toLocaleString()} plots of the species table are not in the header and are left out`] : []),
  ];
  const title = kind === "linked" ? `Plot IDs link the tables: all ${shared.toLocaleString()} plots are in both`
    : kind === "broken" ? "No plot ID is in both tables"
    : `${shared.toLocaleString()} plots are in both tables; ${notes.join("; ")}`;
  return { state: kind, notes, title };
}

// Chain icons after Lucide's link and unlink (ISC licence).
const LINK_ICON = {
  linked: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>' +
          '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
  broken: '<path d="m18.84 12.25 1.72-1.71h-.02a5 5 0 0 0-.12-7.07 5 5 0 0 0-6.95 0l-1.72 1.71"/>' +
          '<path d="m5.17 11.75-1.71 1.71a5 5 0 0 0 .12 7.07 5 5 0 0 0 6.95 0l1.71-1.71"/>' +
          '<path d="M8 2v3M2 8h3M16 22v-3M22 16h-3"/>',
};
LINK_ICON.partial = LINK_ICON.linked;

// The link icon beside both tables' plot ID field; returns plotLink().
function renderPlotLink() {
  const link = plotLink();
  for (const span of document.querySelectorAll("#resolve-app .plot-link")) {
    span.hidden = !link;
    if (!link) continue;
    span.className = `plot-link ${link.state}`;
    span.title = link.title;
    span.setAttribute("role", "img");
    span.setAttribute("aria-label", link.title);
    span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ` +
      `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${LINK_ICON[link.state]}</svg>`;
  }
  return link;
}

function refreshInputs() {
  const species = state.tables.species, header = state.tables.header;
  $("species-info").textContent = species ? `${species.rows.length.toLocaleString()} records` + quantityNote() : "";
  const link = renderPlotLink();
  $("header-info").textContent = !header ? ""
    : header.fromSpecies ? `No header table: ${header.rows.length.toLocaleString()} plots, their fields read from the species table (first record per plot)`
    : [`${header.rows.length.toLocaleString()} plots`, ...(link?.notes ?? [])].join("; ");
  const missing = species && !(chosen.species[COLUMNS.plot] && chosen.species[COLUMNS.species]);
  const missingHeader = header && !header.fromSpecies && !chosen.header[COLUMNS.plot];
  const unlinked = link?.state === "broken";
  $("species-error").textContent = missing ? "Choose the plot ID and species columns of the species table." : "";
  $("header-error").textContent = missingHeader ? "Choose the plot ID column of the header table."
    : unlinked ? "No plot ID is in both tables: choose the plot ID columns that identify the same plots." : "";
  $("run").disabled = !species || missing || missingHeader || unlinked;
  if (state.suites) {
    renderTargets();
    if (state.tablesRead) startPreload();
  }
  showExampleNote();
}

// Darwin Core's organismQuantity holds cover only where organismQuantityType
// says so; the models read cover in percent.
function quantityNote() {
  const col = chosen.species?.[COLUMNS.abundance];
  const typeCol = raw.species?.cols.find((c) => key(c) === "organismquantitytype");
  if (!col || key(col) !== "organismquantity" || !typeCol) return "";
  const types = [...new Set(raw.species.rows.map((r) => r[typeCol]).filter(Boolean))];
  const other = types.filter((t) => !/%|percent/i.test(t));
  return other.length ? `. organismQuantityType is ${other.slice(0, 3).map((t) => `"${t}"`).join(", ")}; the models read cover in percent` : "";
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Workers: a pool of inference workers, each with its own ONNX Runtime, that
// share out a run's models. A worker starts when a model is waiting and every
// worker is busy, so a small run needs few. Unless a number is entered, the
// pool grows to one less than the processor cores, at most 4.
// ---------------------------------------------------------------------------

const CORES = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1));
const pool = { workers: [], queue: [], jobs: new Map(), next: 0, manual: null };
const poolSize = () => Math.max(1, pool.manual ?? Math.min(CORES, 4));

function startWorker() {
  const w = { worker: new Worker(new URL("infer_worker.js", import.meta.url), { type: "module" }),
              job: null, seen: new Set() };
  w.worker.onmessage = ({ data }) => {
    const job = pool.jobs.get(data.id);
    if (!job) return;
    if ("fraction" in data) { job.onFraction?.(data.fraction); return; }
    pool.jobs.delete(data.id);
    w.job = null;
    if (data.error) job.reject(new Error(data.error)); else job.resolve(data.out);
    dispatch();
  };
  w.worker.onerror = (e) => {
    e.preventDefault();
    const job = w.job;
    pool.workers.splice(pool.workers.indexOf(w), 1);
    w.worker.terminate();
    if (job) { pool.jobs.delete(job.id); job.reject(new Error(e.message || "an inference worker failed")); }
    showWorkers();
    dispatch();
  };
  pool.workers.push(w);
  showWorkers();
  return w;
}

function dispatch() {
  for (const w of pool.workers.filter((x) => !x.job).slice(poolSize())) {
    pool.workers.splice(pool.workers.indexOf(w), 1);
    w.worker.terminate();
  }
  while (pool.queue.length) {
    const w = pool.workers.find((x) => !x.job) ?? (pool.workers.length < poolSize() ? startWorker() : null);
    if (!w) break;
    const job = pool.queue.shift();
    w.job = job;
    const key = `${job.runId}|${job.sig}`;
    const message = { id: job.id, runId: job.runId, sig: job.sig, model: job.model, meta: job.meta };
    if (!w.seen.has(key)) { message.encoded = job.encoded; w.seen.add(key); }
    w.worker.postMessage(message, [job.model.buffer]);
  }
  showWorkers();
}

// One model's predictions on an encoding, from whichever worker is free.
function infer(job) {
  return new Promise((resolve, reject) => {
    const id = ++pool.next;
    const entry = { ...job, id, resolve, reject };
    pool.jobs.set(id, entry);
    pool.queue.push(entry);
    dispatch();
  });
}

function showWorkers() {
  const n = pool.workers.length, busyWorkers = pool.workers.filter((w) => w.job).length;
  $("o-workers").placeholder = `auto: ${Math.min(CORES, 4)}`;
  $("workers").textContent = n
    ? `${n} worker${n > 1 ? "s" : ""}${busyWorkers ? `, ${busyWorkers} busy` : ""}`
    : "workers start with the first run";
}

// Words shown beside the progress bar while the models work, one picked at
// random every few seconds.
const VERBS = [
  "Botanising", "Keying out", "Pressing specimens", "Counting quadrats", "Photosynthesising",
  "Consulting the herbarium", "Squinting at sedges", "Marking the plot corners","Checking the flora",
  "Untangling grasses", "Reading the understorey", "Sniffing the mint", "Measuring the turf",
  "Asking the lichens", "Weighing the evidence", "Rooting around", "Leafing through", "Branching out",
  "Sorting the seedlings", "Cross-checking Ellenberg", "Pacing out the plot", "Germinating",
];

// The heartbeat beside the verb: an asterisk that turns and breathes once
// per PULSE_PERIOD, set every PULSE_MS, and the time since the run started.
// It is driven from the page's own event loop, never a CSS animation, so it
// stops when the page hangs and keeps going only while the page is alive.
// The asterisk turns to the warning colour once the share done has not moved
// for STALL_MS.
const PULSE_MS = 80, PULSE_PERIOD = 1600, VERB_MS = 2800, STALL_MS = 10000;
const STILL = window.matchMedia("(prefers-reduced-motion: reduce)");

const busy = { timer: null, members: [], start: new Map(), done: 0, partial: new Map(), targets: 0, combined: 0,
               t0: 0, lastMove: 0, share: -1, tick: 0 };

const elapsed = (ms) => {
  const s = Math.floor(ms / 1000);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
};

function beat() {
  const now = performance.now();
  if (busy.tick % Math.round(VERB_MS / PULSE_MS) === 0) {
    $("busy-verb").textContent = `${VERBS[Math.floor(Math.random() * VERBS.length)]}…`;
  }
  const pulse = $("busy-pulse"), phase = ((now - busy.t0) % PULSE_PERIOD) / PULSE_PERIOD;
  pulse.style.transform = STILL.matches ? ""
    : `rotate(${60 * phase}deg) scale(${0.75 + 0.25 * Math.sin(Math.PI * phase)})`;
  pulse.classList.toggle("stalled", now - busy.lastMove > STALL_MS);
  $("busy-time").textContent = elapsed(now - busy.t0);
  busy.tick += 1;
}

function startBusy(members, targets) {
  const now = performance.now();
  Object.assign(busy, { members, start: new Map(members.map((m) => [m, receivedOf(m)])), done: 0,
                        partial: new Map(), targets, combined: 0, t0: now, lastMove: now, share: -1, tick: 0 });
  $("busy").hidden = false;
  $("status").hidden = true;
  $("download-status").hidden = true;
  beat();
  busy.timer = setInterval(beat, PULSE_MS);
  showBusy();
}

// The share of the run done, in steps: each model's download still to come
// at the start (the share of its graph received since), each model's
// predictions (finished, or the share of its batches run), and each
// variable's combining of its seeds with its gap fitting and terrain
// correction.
function showBusy() {
  if ($("busy").hidden) return;
  let steps = busy.members.length + busy.targets, done = busy.done + busy.combined;
  for (const f of busy.partial.values()) done += f;
  for (const m of busy.members) {
    const r0 = busy.start.get(m);
    if (r0 >= 1) continue;
    steps += 1;
    done += Math.max(0, receivedOf(m) - r0) / (1 - r0);
  }
  const share = Math.min(1, done / Math.max(1, steps));
  if (share !== busy.share) Object.assign(busy, { share, lastMove: performance.now() });
  const percent = Math.floor(100 * share);
  $("busy-bar").style.width = `${percent}%`;
  $("busy-bar").parentElement.setAttribute("aria-valuenow", percent);
  $("busy-share").textContent = `${busy.done} of ${busy.members.length} models · ${percent}%`;
}

function stopBusy() {
  clearInterval(busy.timer);
  $("busy").hidden = true;
  $("status").hidden = false;
  $("download-status").hidden = false;
}

async function run() {
  const targets = chosenTargets();
  const header = headerRows(), species = state.tables.species.rows;
  $("run").disabled = true;
  status("Starting…");
  const result = { plotIds: null, columns: [], recognised: null, coords: null };
  state.result = result;
  const started = performance.now();
  $("s-time").textContent = "…";
  preload(targets).catch((e) => status(`Download failed: ${e.message}`));
  startBusy(targets.flatMap((t) => t.members), targets.length);
  try {
    await predictTargets(result, targets, header, species);
  } finally {
    stopBusy();
    $("run").disabled = false;
  }
  $("s-time").textContent = duration((performance.now() - started) / 1000);
  status(result.notes ? result.notes.join(" ") : "");
}

const duration = (s) => (s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

let runs = 0;

// Every model of every target is handed to the pool at once; the targets are
// combined and shown in order as their seeds come back.
async function predictTargets(result, targets, header, species) {
  const runId = ++runs;
  const aliases = await nameLookup();
  const encodings = new Map();
  const member = async (name) => {
    const meta = JSON.parse(new TextDecoder().decode(await file(`${name}/model.json`)));
    // Members sharing a vocabulary and settings share one encoding.
    const sig = `${spellingEdits()}|${meta.species.length}|${meta.genera.length}|${meta.pool_species_cap}|${meta.covariates}|${meta.has_coordinates}|${meta.species[1]}|${meta.species.at(-1)}`;
    if (!encodings.has(sig)) {
      const names = renamed(meta, species, aliases);
      encodings.set(sig, { ...names, encoded: encode(meta, header, names.rows) });
    }
    const { encoded } = encodings.get(sig);
    const model = new Uint8Array((await file(`${name}/${graphOf(meta)}`)).slice(0));
    const out = await infer({
      runId, sig, encoded, model,
      meta: { targets: meta.targets, continuous_width: meta.continuous_width },
      onFraction: (f) => { busy.partial.set(name, f); showBusy(); },
    });
    busy.partial.delete(name);
    busy.done += 1;
    showBusy();
    return { meta, sig, out };
  };
  const pending = targets.map((t) => Promise.all(t.members.map(member)));
  pending.forEach((p) => p.catch(() => {}));
  result.variables = { done: 0, total: targets.length };
  for (const [i, target] of targets.entries()) {
    const done = await pending[i];
    const meta0 = done[0].meta, outputs = done.map((d) => d.out);
    const { encoded, rows, moved } = encodings.get(done[0].sig);
    result.plotIds ??= encoded.plotIds;
    result.recognised ??= Float64Array.from(encoded.unknownFraction, (f) => 1 - f);
    result.unknown ??= unrecognised(meta0, rows, new Set(encoded.plotIds));
    result.moved ??= moved;
    await addTarget(result, target, meta0, outputs, encoded);
    if (!result.coords && meta0.has_coordinates) {
      result.coords = encoded.plotIds.map((_, p) => [encoded.continuous[p * meta0.continuous_width + 1],
                                                     encoded.continuous[p * meta0.continuous_width]]);
    }
    result.variables.done = i + 1;
    busy.combined += 1;
    showBusy();
    render();
  }
}

// Whether each plot records a target in the tables, in the prediction's plot
// order (1 / 0); null where the tables record it on no plot.
function recordedMask(result, target) {
  const count = recordedCount(target);
  if (!count || count.recorded === 0) return null;
  const { num } = gapPlots(result);
  const values = target.outputs.map(num);
  return Uint8Array.from(values[0], (_, p) => values.every((v) => Number.isFinite(v[p])));
}

// A target the tables record on some plots but not all, whose gaps were not
// fitted: whether each plot records it, as the engine's `<target>_recorded`
// column (TRUE / FALSE). A gap fit's method column says it otherwise, and a
// target recorded on no plot is imputed on every one.
function pushRecorded(result, target, mask = recordedMask(result, target)) {
  if (!mask || mask.every(Boolean)) return;
  result.columns.push({ name: `${label(target)}, recorded`, values: Array.from(mask, (m) => (m ? "TRUE" : "FALSE")) });
}

// A unit as the manifest writes it (m^2), in the page's typography (m²).
const unitText = (u) => u.replace(/\^2\b/g, "²").replace(/\^3\b/g, "³");

async function addTarget(result, target, meta, outputs, encoded) {
  const n = encoded.n;
  const units = target.units ? ` (${unitText(target.units)})` : "";
  if (target.combine === "vote") {
    const head = meta.targets[0];
    const coarse = target.readAt ? target.read_at[target.readAt] : null;
    const classes = coarse ? coarse.classes : head.classes;
    const probs = outputs.map((o) => (coarse ? sumByAncestor(o[head.name], n, coarse.index, classes.length) : o[head.name]));
    const c = combineVote(probs, classes.length, n);
    result.columns.push({ name: `${label(target)}`, values: Array.from(c.value, (v) => classes[v]) },
                        { name: `${label(target)} seed agreement`, values: Array.from(c.agreement), digits: 2 },
                        { name: `${label(target)} probability`, digits: 3,
                          values: Array.from(c.value, (v, p) => c.mean[p * classes.length + v]) });
    (result.summary ??= []).push({ kind: "class", name: target.name, label: label(target), classes, value: c.value,
                                   names: coarse ? coarse.names : target.class_names ?? {} });
  } else if (target.combine === "spherical_mean") {
    const [latName, lonName] = target.outputs;
    const c = combineSpherical(outputs.map((o) => o[latName]), outputs.map((o) => o[lonName]), n);
    result.columns.push({ name: "Imputed latitude", values: Array.from(c.lat), digits: 4 },
                        { name: "Imputed longitude", values: Array.from(c.lon), digits: 4 },
                        { name: "Location seed spread (km)", values: Array.from(c.km), digits: 1 });
    result.predictedCoords = Array.from(c.lat, (la, p) => [la, c.lon[p]]);
    const raw = new Float64Array(2 * n);
    c.lat.forEach((la, p) => { raw[2 * p] = la; raw[2 * p + 1] = c.lon[p]; });
    const fit = await fillGaps(result, target, "geographic", raw);
    if (fit) {
      result.columns.push({ name: "Imputed latitude, calibrated", values: Array.from(c.lat, (_, p) => fit.fitted[2 * p]), digits: 4 },
                          { name: "Imputed longitude, calibrated", values: Array.from(c.lat, (_, p) => fit.fitted[2 * p + 1]), digits: 4 },
                          { name: "Location, source", values: Array.from(fit.method, gapSource) });
      result.filledCoords = Array.from(c.lat, (_, p) => [fit.fitted[2 * p], fit.fitted[2 * p + 1]]);
    } else {
      pushRecorded(result, target);
    }
  } else {
    const name = target.outputs[0];
    const c = combineMean(outputs.map((o) => o[name]), n);
    result.columns.push({ name: `${label(target)}${units}`, values: Array.from(c.value), digits: 1 },
                        { name: `${label(target)} seed SD`, values: Array.from(c.sd), digits: 1 });
    const entry = { kind: "number", name: target.name, label: label(target), units: unitText(target.units ?? ""),
                    value: c.value };
    (result.summary ??= []).push(entry);
    const fit = await fillGaps(result, target, meta.targets[0].log1p ? "log" : "linear", c.value);
    const mask = recordedMask(result, target);
    if (fit) {
      result.columns.push({ name: `${label(target)}${units}, calibrated`, values: Array.from(fit.fitted), digits: 1 },
                          { name: `${label(target)}, source`, values: Array.from(fit.method, gapSource) });
    } else {
      pushRecorded(result, target, mask);
    }
    // Each plot's value as the map shows it: the recorded one where the
    // tables hold it, else the gap fit's, else the combined prediction.
    const recorded = mask && gapPlots(result).num(target.outputs[0]);
    const imputed = fit ? fit.fitted : c.value;
    Object.assign(entry, { recorded: mask,
                           plotValue: Float64Array.from(imputed, (v, p) => (mask?.[p] ? recorded[p] : v)) });
    if (target.terrain && terrainOn()) await addTerrain(result, target, c, encoded, units);
  }
}

// ---------------------------------------------------------------------------
// Terrain correction (terrain.js)
// ---------------------------------------------------------------------------

const terrainOn = () => $("o-terrain").checked;

// The suite's terrain field, read block by block from where the page names it
// (data-terrain-field), with its index and block lengths from the model pack;
// kept across runs.
async function terrainField(suite) {
  if (state.terrain?.suite === suite.name) return state.terrain.field;
  const index = JSON.parse(new TextDecoder().decode(await file(suite.terrain_field)));
  const [lengthsAt] = await packRange(packPath(index.lengths.file, suite.terrain_field));
  const lengths = { read: (ranges) => fromPack((url) => new HttpRangeSource(url, { offset: lengthsAt }).read(ranges)) };
  const field = new TerrainField(index, new HttpRangeSource(TERRAIN_FIELD), lengths);
  state.terrain = { suite: suite.name, field };
  return field;
}

// The plots' terrain descriptors, read once per run for every corrected
// target, from the header's coordinates in the prediction's plot order.
async function describeTerrain(suite, coordinates) {
  const field = await terrainField(suite);
  const n = coordinates.length / 2;
  const lat = new Float64Array(n), lon = new Float64Array(n);
  for (let p = 0; p < n; p++) { lat[p] = coordinates[2 * p]; lon[p] = coordinates[2 * p + 1]; }
  return describePlots(field, lat, lon);
}

// A target's terrain correction applied to its combined prediction, as the
// engine's suite applies it (suite.cpp, correct_terrain): the prediction, the
// elevation model's value at the plot and over the 1 km around it, the
// plot's location uncertainty, the seeds' spread, and the share of the plot's
// species the target's models recognise and their number.
async function addTerrain(result, target, c, encoded, units) {
  const { coordinates, num } = gapPlots(result);
  if (!coordinates) return;
  try {
    status("Reading the terrain…");
    result.terrain ??= describeTerrain(target.suite, coordinates);
    const d = await result.terrain;
    const correction = new TerrainCorrection(await file(target.terrain.file));
    const elevation = correction.descriptor === "elevation";
    const recognised = Float64Array.from(encoded.nSpecies, (s, p) =>
      (s > 0 ? Math.fround(Math.fround(s - encoded.unknownCount[p]) / s) : NaN));
    const r = correction.apply({
      prediction: c.value, dem: elevation ? d.elevation : d.slope, dem_1km: elevation ? d.elevation_1km : d.slope_1km,
      uncertainty_m: num(UNCERTAINTY), spread: c.sd, recognised_share: recognised, n_species: encoded.nSpecies });
    const model = "elevation model";
    result.columns.push(
      { name: `${label(target)}, terrain corrected${units}`, values: Array.from(r.corrected), digits: 1 },
      { name: `${label(target)}, ${model}${units}`, values: Array.from(elevation ? d.elevation : d.slope), digits: 1 },
      { name: `${label(target)}, ${model} over 1 km${units}`, values: Array.from(elevation ? d.elevation_1km : d.slope_1km), digits: 1 });
  } catch (e) {
    (result.notes ??= []).push(`Terrain correction of ${label(target).toLowerCase()} failed: ${e.message}`);
    status(`Terrain correction of ${label(target).toLowerCase()} failed: ${e.message}`);
  }
}

// What gap fitting reads about the plots, in the order of the prediction:
// every species named (whatever a model's vocabulary holds), the header's
// coordinates and survey year, and the header row of each plot.
function gapPlots(result) {
  if (result.gapPlots) return result.gapPlots;
  const index = new Map(result.plotIds.map((id, p) => [String(id), p]));
  const ids = new Map(), present = result.plotIds.map(() => new Set());
  for (const r of state.tables.species.rows) {
    const p = index.get(String(r[COLUMNS.plot])), name = r[COLUMNS.species];
    if (p === undefined || name == null || name === "") continue;
    if (!ids.has(name)) ids.set(name, ids.size);
    present[p].add(ids.get(name));
  }
  const rows = new Array(result.plotIds.length).fill(null);
  for (const r of headerRows()) {
    const p = index.get(String(r[COLUMNS.plot]));
    if (p !== undefined && rows[p] === null) rows[p] = r;
  }
  const num = (role) => Float64Array.from(rows, (r) => parseDouble(r?.[role]) ?? NaN);
  const found = new Set(state.tables.header?.found ?? []);
  let coordinates = null;
  if (found.has(COLUMNS.latitude) && found.has(COLUMNS.longitude)) {
    const lat = num(COLUMNS.latitude), lon = num(COLUMNS.longitude);
    coordinates = new Float64Array(2 * rows.length);
    lat.forEach((la, p) => { coordinates[2 * p] = la; coordinates[2 * p + 1] = lon[p]; });
  }
  result.gapPlots = {
    species: present.map((s) => Int32Array.from(s).sort()),
    coordinates,
    years: found.has("Year") ? num("Year") : null,
    num,
  };
  return result.gapPlots;
}

// Gap fitting runs in its own worker (gaps_worker.js), one fit at a time; its
// neighbour searches use one less worker than the processor cores, or the
// number entered.
const GAP_THREADS = Math.max(1, (navigator.hardwareConcurrency || 2) - 1);
let gapWorker = null, gapJobs = 0;
const gapPending = new Map();
function fitInWorker(message) {
  if (!gapWorker) {
    gapWorker = new Worker(new URL("gaps_worker.js", import.meta.url), { type: "module" });
    gapWorker.onmessage = ({ data }) => {
      const job = gapPending.get(data.id);
      gapPending.delete(data.id);
      if (data.error) job.reject(new Error(data.error)); else job.resolve(data);
    };
  }
  return new Promise((resolve, reject) => {
    const id = ++gapJobs;
    gapPending.set(id, { resolve, reject });
    gapWorker.postMessage({ id, threads: pool.manual ?? GAP_THREADS, ...message });
  });
}

const gapsOn = () => $("o-gaps").checked;

// A target recorded on some plots, where at least two record it: gap fitting
// has complete plots to choose its method on.
const partlyRecorded = (t) => {
  const c = t.combine === "vote" ? null : recordedCount(t);
  return c !== null && c.recorded >= 2 && c.recorded < c.total;
};

// A target the tables record on some plots: the gaps fitted from the
// complete plots (gaps.js, the engine's gap fitting), its report kept for the
// summary. Null when the tables record it on no plot or on every plot.
async function fillGaps(result, target, scale, raw) {
  if (!gapsOn()) return null;
  const count = recordedCount(target);
  if (!count || count.recorded === 0 || count.recorded === count.total) return null;
  const { num, ...plots } = gapPlots(result);
  const columns = target.outputs.map(num);
  const recorded = new Float64Array(raw.length);
  const d = columns.length;
  for (let p = 0; p < raw.length / d; p++) for (let j = 0; j < d; j++) recorded[p * d + j] = columns[j][p];
  status(`Calibrating the imputed ${label(target).toLowerCase()} on the plots recording it…`);
  const fit = await fitInWorker({ target: { name: target.name, scale, period: target.period ?? 360 },
                                  raw: Float64Array.from(raw), recorded, plots });
  (result.gaps ??= []).push({ target, scale, report: fit.report });
  return fit;
}

const LABELS = { eunis: "EUNIS habitat", eunis_l2: "EUNIS habitat, level 2", eunis_l3: "EUNIS habitat, level 3",
                 alliance: "Alliance (EuroVegChecklist)",
                 altitude: "Altitude", slope: "Slope", area: "Plot area", year: "Survey year", location: "Location" };
const label = (t) => (t.readAt ? `EUNIS habitat, level ${t.readAt}` : LABELS[t.name] ?? t.name);
const habitatName = (code, names) => (names[code] ? `${code} (${names[code]})` : code);

function status(text) { $("status").textContent = text; }

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

// The results table shows one page of ROWS_PAGE plots at a time; the pager
// above it moves between pages and names the range on screen.
const ROWS_PAGE = 200;
const tablePage = { cols: [], n: 0, page: 0 };

function format(v, digits) {
  if (typeof v === "string") return v;
  if (v === undefined || Number.isNaN(v)) return "";
  return digits === undefined ? String(v) : v.toFixed(digits);
}

function tablePageCount() { return Math.max(1, Math.ceil(tablePage.n / ROWS_PAGE)); }

function showTablePage(page) {
  const { cols, n } = tablePage;
  const last = tablePageCount() - 1;
  tablePage.page = Math.min(Math.max(page, 0), last);
  const start = tablePage.page * ROWS_PAGE;
  const end = Math.min(n, start + ROWS_PAGE);
  const rows = [];
  for (let p = start; p < end; p++) {
    rows.push(`<tr>${cols.map((c) => `<td>${escape(format(c.values[p], c.digits))}</td>`).join("")}</tr>`);
  }
  $("table").tBodies[0].innerHTML = rows.join("");
  $("table").parentElement.scrollTop = 0;
  $("pager").hidden = n <= ROWS_PAGE;
  $("pager-range").textContent =
    `Plots ${(start + 1).toLocaleString()}–${end.toLocaleString()} of ${n.toLocaleString()}`;
  $("pager-first").disabled = $("pager-prev").disabled = tablePage.page === 0;
  $("pager-next").disabled = $("pager-last").disabled = tablePage.page === last;
}

function render() {
  const r = state.result;
  const cols = [{ name: "Plot", values: r.plotIds },
                { name: "Species recognised (share of cover)", values: Array.from(r.recognised), digits: 2 },
                ...r.columns];
  const head = `<tr>${cols.map((c) => `<th scope="col">${escape(c.name)}</th>`).join("")}</tr>`;
  $("table").innerHTML = `<thead>${head}</thead><tbody></tbody>`;
  Object.assign(tablePage, { cols, n: r.plotIds.length });
  showTablePage(0);
  $("results").hidden = false;
  renderStats(r);
  renderVariable(r);
  renderGaps(r);
  renderUnknown(r);
  renderMoved(r);
  renderMap(r);
  $("download").disabled = false;
}

// The name lookup (names.json.gz beside the models, names.json gzipped): names
// no vocabulary holds, each with the model token it stands for and the source
// of the link ("eva" or a backbone). Without the file, names are matched as
// written.
let lookup = null, nameBackbones = [];
async function nameLookup() {
  lookup ??= file("names.json.gz")
    .then((b) => new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"))).json())
    .then((names) => {
      nameBackbones = backbonesUsed(names);
      return names.aliases;
    })
    .catch(() => ({}));
  return lookup;
}

// The backbones that link at least one name to a model token, in the
// lookup's priority order.
function backbonesUsed(names) {
  const used = new Set(Object.values(names.aliases).map(([, source]) => source));
  return (names.sources ?? []).filter((s) => s !== "eva" && used.has(s));
}

// The backbones named in the text; the others are behind one link to the
// species names article's list of backbones.
const NAMED_BACKBONES = ["euromed", "wfo"];

// Where a name is looked for before it counts as not recognised, as HTML.
function nameRoutes() {
  const named = NAMED_BACKBONES.filter((b) => nameBackbones.includes(b)).map((b) => escape(SOURCE_NAMES[b]));
  const others = nameBackbones.length > named.length
    ? `the synonyms in the <a href="species-names.html#backbones">other taxonomies</a>` : "";
  const routes = ["EVA's records", ...named, ...(others ? [others] : [])];
  const k = spellingEdits();
  if (k) routes.push(`a fuzzy match within ${k} edit${k > 1 ? "s" : ""}`);
  return routes.length > 1 ? `${routes.slice(0, -1).join(", ")} or ${routes.at(-1)}` : routes[0];
}

// The edits a misspelled name may be away from a name the models know:
// one by default; two or three also rename some species the models never saw.
const spellingEdits = () => Number($("o-spelling").value);

// Each vocabulary's spelling index (resolveName's cache), kept across runs.
const spellingIndexes = new Map();

// The species rows with each name the model's vocabulary lacks replaced by
// the token it resolves to (resolveName), and the names moved, with their
// route and record counts.
function renamed(meta, species, aliases) {
  const known = new Set(meta.species.slice(1).filter(Boolean)), moved = new Map(), resolved = new Map();
  const vocabulary = `${meta.species.length}|${meta.species[1]}|${meta.species.at(-1)}`;
  if (!spellingIndexes.has(vocabulary)) spellingIndexes.set(vocabulary, {});
  const spelling = spellingIndexes.get(vocabulary);
  const k = spellingEdits();
  const rows = species.map((r) => {
    const name = r[COLUMNS.species];
    if (name == null || name === "") return r;
    if (!resolved.has(name)) resolved.set(name, resolveName(name, known, aliases, k, spelling));
    const hit = resolved.get(name);
    if (!hit) return r;
    const m = moved.get(name) ?? { ...hit, records: 0 };
    m.records += 1;
    moved.set(name, m);
    return { ...r, [COLUMNS.species]: hit.token };
  });
  return { rows, moved };
}

// The species names of the plots read that the models' vocabulary does not
// hold, with their record counts: the engine matches names exactly, and these
// records are left out of the prediction.
function unrecognised(meta, species, plots) {
  const known = new Set(meta.species), counts = new Map();
  for (const r of species) {
    if (!plots.has(String(r[COLUMNS.plot]))) continue;
    const name = r[COLUMNS.species] ?? "";
    if (!known.has(name)) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
}

function renderUnknown(r) {
  const box = $("unknown");
  if (!r.unknown.length) { box.hidden = true; return; }
  box.hidden = false;
  const records = r.unknown.reduce((a, [, k]) => a + k, 0);
  box.querySelector("summary").textContent =
    `${r.unknown.length.toLocaleString()} species names not recognised: ${records.toLocaleString()} records not used, their plots imputed from the other species`;
  box.querySelector("p").innerHTML =
    `None of these names was found through ${nameRoutes()}.`;
  box.querySelector("ul").innerHTML = r.unknown.slice(0, 200)
    .map(([name, k]) => `<li>${escape(name || "(empty)")} <span class="count">${k.toLocaleString()}</span></li>`).join("") +
    (r.unknown.length > 200 ? `<li class="count">and ${(r.unknown.length - 200).toLocaleString()} more</li>` : "");
}

// Each name source's short label (in the text and beside a renamed name) and
// full name (on hover).
const SOURCES = {
  eva: "EVA", wfo: "WFO", euromed: "Euro+Med", wcvp: "WCVP", lcvp: "LCVP", col: "COL", colxr: "COL XR",
  gbif: "GBIF", itis: "ITIS", ncbi: "NCBI", ott: "Open Tree", worms: "WoRMS",
};
const SOURCE_NAMES = {
  wfo: "World Flora Online", euromed: "Euro+Med PlantBase", wcvp: "World Checklist of Vascular Plants",
  lcvp: "Leipzig Catalogue of Vascular Plants", col: "Catalogue of Life", colxr: "Catalogue of Life Extended Release",
  gbif: "GBIF Backbone Taxonomy", itis: "Integrated Taxonomic Information System", ncbi: "NCBI Taxonomy",
  ott: "Open Tree of Life", worms: "World Register of Marine Species",
};

function route(m) {
  if (m.route !== "spelling") return SOURCES[m.source] ?? m.source;
  const edits = m.distance === 0 ? "letter case" : `${m.distance} edit${m.distance === 1 ? "" : "s"}`;
  return m.via === m.token ? `spelling, ${edits}` : `spelling, ${edits} from ${m.via}`;
}

function renderMoved(r) {
  const box = $("moved");
  const moved = [...(r.moved ?? new Map())].sort((a, b) => b[1].records - a[1].records || (a[0] < b[0] ? -1 : 1));
  if (!moved.length) { box.hidden = true; return; }
  box.hidden = false;
  const records = moved.reduce((a, [, m]) => a + m.records, 0);
  box.querySelector("summary").textContent =
    `${moved.length.toLocaleString()} species names renamed (${records.toLocaleString()} records)`;
  box.querySelector("p").innerHTML =
    "The models know species by the World Flora Online name the European Vegetation Archive (EVA) matched its " +
    "records to; EVA files Anemone nemorosa under Anemone altaica, for example. These names were renamed through " +
    `${nameRoutes()}; the source is given beside each name.`;
  box.querySelector("ul").innerHTML = moved.slice(0, 200)
    .map(([name, m]) => `<li>${escape(name)} → ${escape(m.token)} <span class="count">${escape(route(m))}, ${m.records.toLocaleString()}</span></li>`).join("") +
    (moved.length > 200 ? `<li class="count">and ${(moved.length - 200).toLocaleString()} more</li>` : "");
}

// The map and variable panels share a row; a panel with nothing to show is
// hidden and the other takes the whole row.
function showPanel(id, shown) {
  $(id).hidden = !shown;
  const panels = ["map-panel", "variable-panel"].map($);
  for (const p of panels) p.classList.toggle("col-lg-12", panels.some((q) => q.hidden));
}

function renderStats(r) {
  const { done, total } = r.variables;
  $("s-plots").textContent = r.plotIds.length.toLocaleString();
  $("s-variables").textContent = done === total ? String(total) : `${done} of ${total}`;
  $("s-unknown").textContent = r.unknown.length.toLocaleString();
}

// ---------------------------------------------------------------------------
// Map variables: the panel beside the map and the map's colours show one
// imputed variable at a time, chosen with the panel's arrows.
// ---------------------------------------------------------------------------

// A habitat's colour is its level-1 group's, the first letter of its code.
const PALETTE = { M: "#2b6cb0", N: "#d69e2e", P: "#3182ce", Q: "#2c7a7b", R: "#68a33f",
                  S: "#b7791f", T: "#276749", U: "#a0aec0", V: "#c05621" };
const colour = (c) => PALETTE[c?.[0]] ?? "#718096";
const NO_VALUE = "#4a5568";

// Fixed class breaks per target, in its units, so that a colour stands for
// the same values in every run: plot area on a roughly logarithmic scale,
// survey years by decade. A target without an entry is cut at round numbers
// over its range (roundBreaks); aspect is cut into eight compass sectors.
const BREAKS = {
  altitude: { breaks: [200, 500, 1000, 1500, 2000, 2500] },
  slope: { breaks: [2, 5, 10, 20, 30, 45] },
  area: { breaks: [1, 4, 10, 25, 100, 400, 1000] },
  year: { breaks: [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020],
          label: (lo, hi) => (lo === null ? `before ${hi}` : `${lo}s`) },
};
const SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

// About six breaks at multiples of 1, 2 or 5 times a power of ten, strictly
// inside the range of the finite values.
function roundBreaks(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!(hi > lo)) return [];
  const raw = (hi - lo) / 6, power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const breaks = [];
  for (let b = Math.floor(lo / step + 1) * step; b < hi; b += step) breaks.push(Number(b.toPrecision(12)));
  return breaks;
}

// Viridis from its dark end to 90% of the way to its light end, so the
// lightest class still shows on the map tiles.
const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
function sequential(n) {
  const channel = (hex, j) => parseInt(hex.slice(1 + 2 * j, 3 + 2 * j), 16);
  return Array.from({ length: n }, (_, k) => {
    const x = (n === 1 ? 0.45 : (0.9 * k) / (n - 1)) * (VIRIDIS.length - 1);
    const i = Math.min(Math.floor(x), VIRIDIS.length - 2), f = x - i;
    return `#${[0, 1, 2].map((j) => Math.round(channel(VIRIDIS[i], j) * (1 - f) + channel(VIRIDIS[i + 1], j) * f)
      .toString(16).padStart(2, "0")).join("")}`;
  });
}

// A unit as written after a number in a class label or tooltip: degrees as
// °, years without a unit.
const binUnit = (u) => ({ degrees: "°", years: "" })[u] ?? u;
const withUnit = (text, u) => (!u ? text : u === "°" ? `${text}°` : `${text} ${u}`);

// The classes of a numeric variable, each with its label and colour, and the
// class of a value (-1 when it is not finite).
function numberClasses(s) {
  if (s.name === "aspect") {
    return { classes: SECTORS.map((label, k) => ({ label, colour: `hsl(${45 * k}, 60%, 45%)` })),
             classify: (v) => (Number.isFinite(v) ? Math.floor((((v % 360) + 360) % 360 + 22.5) / 45) % 8 : -1) };
  }
  const spec = BREAKS[s.name] ?? { breaks: roundBreaks(s.plotValue) };
  const b = spec.breaks, u = binUnit(s.units);
  const text = spec.label ?? ((lo, hi) => (lo === null ? `< ${withUnit(hi.toLocaleString(), u)}`
    : hi === null ? `≥ ${withUnit(lo.toLocaleString(), u)}` : withUnit(`${lo.toLocaleString()}–${hi.toLocaleString()}`, u)));
  const classes = sequential(b.length + 1)
    .map((c, k) => ({ label: text(k ? b[k - 1] : null, k < b.length ? b[k] : null), colour: c }));
  return { classes, classify: (v) => (Number.isFinite(v) ? b.filter((x) => x <= v).length : -1) };
}

// A variable as the panel and the map draw it: its classes in the order the
// panel lists them (habitats by plots, numeric classes by value), each with
// the legend entry it falls under (a habitat's level-1 group, a numeric class
// itself); per plot its class (-1 without a value), its tooltip text and
// whether the tables record it. A numeric variable also carries each plot's
// value, the class of any value and a value written with its unit.
function variableView(s) {
  if (s.kind === "class") {
    const groups = new Map();
    const classes = s.classes.map((c) => {
      if (!groups.has(c[0])) groups.set(c[0], { label: habitatName(c[0], s.names), colour: colour(c) });
      return { code: c, label: s.names[c] ?? "", colour: colour(c), group: groups.get(c[0]) };
    });
    return { kind: "class", label: s.label, byCount: true, classes, classOf: Int32Array.from(s.value), recorded: null,
             tip: (p) => habitatName(s.classes[s.value[p]], s.names) };
  }
  const { classes, classify } = numberClasses(s);
  for (const c of classes) c.group = c;
  const u = binUnit(s.units);
  const format = { maximumFractionDigits: s.name === "year" ? 0 : 1, useGrouping: s.name !== "year" };
  const text = (v) => withUnit(v.toLocaleString(undefined, format), u);
  return {
    kind: "number", label: s.label, byCount: false, classes, classOf: Int32Array.from(s.plotValue, classify),
    recorded: s.recorded, value: s.plotValue, classify, text, circular: s.name === "aspect",
    tip: (p) => (Number.isFinite(s.plotValue[p]) ? `${s.label} ${text(s.plotValue[p])}${s.recorded?.[p] ? ", recorded" : ""}` : ""),
  };
}

// The variables that can colour the map: every one imputed but the location,
// which places the plots. The one shown is kept by label across runs.
const mapVariables = (r) => r.summary ?? [];
let variableShown = null;
function shownView(r) {
  const all = mapVariables(r);
  const s = all.find((x) => x.label === variableShown) ?? all[0];
  return s ? variableView(s) : null;
}

function stepVariable(by) {
  const r = state.result, all = r ? mapVariables(r) : [];
  if (all.length < 2) return;
  const at = Math.max(0, all.findIndex((x) => x.label === variableShown));
  variableShown = all[(at + by + all.length) % all.length].label;
  renderVariable(r);
  renderMap(r, false);
}

// The shown variable's classes with the number of plots in each and their
// share of all plots, as a bar scaled to the fullest class.
function renderVariable(r) {
  const view = shownView(r);
  showPanel("variable-panel", !!view);
  $("variable-step").hidden = mapVariables(r).length < 2;
  if (!view) { $("summary").innerHTML = ""; return; }
  $("variable-title").textContent = view.label;
  const counts = new Array(view.classes.length).fill(0);
  for (const k of view.classOf) if (k >= 0) counts[k] += 1;
  const n = r.plotIds.length;
  const rows = view.classes.map((c, k) => [c, counts[k]]).filter(([, k]) => k > 0);
  if (view.byCount) rows.sort((a, b) => b[1] - a[1]);
  const most = Math.max(1, ...rows.map(([, k]) => k));
  $("summary").classList.toggle("bins", !view.byCount);
  $("summary").innerHTML = rows
    .map(([c, k]) => `<li><span class="sw" style="background:${c.colour}"></span>${c.code ? `<b>${escape(c.code)}</b>` : ""}` +
                     `<span class="mid"><span class="nm">${escape(c.label)}</span>` +
                     `<span class="bar" style="width:${((100 * k) / most).toFixed(1)}%"></span></span>` +
                     `<span class="n">${k.toLocaleString()} <span class="count">(${((100 * k) / n).toFixed(1)}%)</span></span></li>`)
    .join("");
}

// ---------------------------------------------------------------------------
// Map marks: the plots one by one, or summarised on the EEA reference grid
// (geo.js) at a cell size chosen or set by the zoom.
// ---------------------------------------------------------------------------

const GRID_KM = [2, 5, 10, 25, 50, 100, 200];
const CELL_EDGE = "#000000";
const mapMode = () => document.querySelector('#resolve-app input[name="map-view"]:checked')?.value ?? "points";
const tally = (counts, key, by) => counts.set(key, (counts.get(key) ?? 0) + by);

// The map's scale at its centre, in metres per pixel.
const metresPerPixel = () => 40075016.686 * Math.cos(map.getCenter().lat * Math.PI / 180) / (256 * 2 ** map.getZoom());

// The cell size in kilometres: the smallest whose cells span at least 12
// pixels at the map's centre.
function gridKm() {
  return GRID_KM.find((km) => (1000 * km) / metresPerPixel() >= 12) ?? GRID_KM[GRID_KM.length - 1];
}

// The median of finite values (the mean of the middle two for an even count).
function median(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// The mean direction of angles in degrees, in [0, 360).
function circularMean(degrees) {
  let s = 0, c = 0;
  for (const d of degrees) { s += Math.sin(d * Math.PI / 180); c += Math.cos(d * Math.PI / 180); }
  return ((Math.atan2(s, c) * 180) / Math.PI + 360) % 360;
}

// The legend entries under the counted marks: habitat groups by plots,
// numeric classes by value.
function legendGroups(view, counts) {
  if (!view) return [];
  return view.byCount ? [...counts].sort((a, b) => b[1] - a[1]).map(([g]) => g)
    : [...new Set(view.classes.map((c) => c.group))].filter((g) => counts.has(g));
}

// What the map draws of the located plots that `keep` accepts, in latitude
// and longitude, with its legend: the groups shown and the keys to the marks.
function mapMarks(r, view, coords, keep = () => true) {
  const located = [];
  coords.forEach(([la, lo], p) => { if (Number.isFinite(la) && Number.isFinite(lo) && keep(p)) located.push(p); });
  if (mapMode() !== "grid") return plotMarks(r, view, coords, located);
  const km = gridKm();
  return gridMarks(view, coords, located, km, (1000 * km) / metresPerPixel());
}

// The width of a cell's outline: 1.5 pixels, thinner on cells under 15
// pixels across, so that the outline never covers the cell's colour.
const cellEdge = (pixels) => Math.min(1.5, Math.max(0.4, pixels / 10));

// One mark per plot in its class's colour: a dot, or a ring where the tables
// record the value; the two are keyed only where both occur.
function plotMarks(r, view, coords, located) {
  const counts = new Map();
  let rings = false, dots = false;
  const marks = located.map((p) => {
    const k = view ? view.classOf[p] : -1;
    const ring = !!view?.recorded?.[p];
    if (ring) rings = true; else dots = true;
    if (k >= 0) tally(counts, view.classes[k].group, 1);
    const tip = view?.tip(p);
    return { kind: ring ? "ring" : "dot", at: coords[p], fill: k >= 0 ? view.classes[k].colour : NO_VALUE,
             tip: `${r.plotIds[p]}${tip ? `: ${tip}` : ""}` };
  });
  const keys = rings ? [...(dots ? [{ shape: "dot", label: "imputed" }] : []), { shape: "ring", label: "recorded" }] : [];
  return { marks, legend: { groups: legendGroups(view, counts), keys } };
}

// One mark per grid cell holding plots, outlined in black. A habitat cell
// takes the colour of its most frequent level-1 group, paler the smaller that
// group's share; a numeric cell the class of its plots' median value (their
// mean direction for aspect). The outline's width follows the cell's size on
// screen in pixels.
function gridMarks(view, coords, located, km, pixels) {
  const cells = new Map();
  for (const p of located) {
    const [e, n] = eeaCell(coords[p][0], coords[p][1], km);
    const code = eeaCellCode(e, n, km);
    if (!cells.has(code)) cells.set(code, { e, n, plots: [] });
    cells.get(code).plots.push(p);
  }
  const counts = new Map();
  const marks = [...cells].map(([code, { e, n, plots }]) => {
    const head = `${code}: ${plots.length.toLocaleString()} plot${plots.length === 1 ? "" : "s"}`;
    const mark = { kind: "cell", outline: eeaCellOutline(e, n, km), fill: NO_VALUE, opacity: 0.75,
                   edge: cellEdge(pixels), tip: head };
    const valued = view ? plots.filter((p) => view.classOf[p] >= 0) : [];
    if (!valued.length) return mark;
    let group;
    if (view.kind === "class") {
      const within = new Map();
      for (const p of valued) tally(within, view.classes[view.classOf[p]].group, 1);
      const [top, k] = [...within].sort((a, b) => b[1] - a[1])[0];
      group = top;
      Object.assign(mark, { fill: top.colour, opacity: 0.3 + (0.6 * k) / valued.length,
                            tip: `${head}; ${Math.round((100 * k) / valued.length)}% ${top.label}` });
    } else {
      const values = valued.map((p) => view.value[p]);
      const centre = view.circular ? circularMean(values) : median(values);
      group = view.classes[view.classify(centre)];
      const recorded = view.recorded ? plots.filter((p) => view.recorded[p]).length : 0;
      Object.assign(mark, {
        fill: group.colour,
        tip: `${head}${view.recorded ? `, ${recorded.toLocaleString()} recorded` : ""}; ` +
             `${view.circular ? "mean direction" : "median"} ${view.text(centre)}`,
      });
    }
    tally(counts, group, valued.length);
    return mark;
  });
  return { marks, legend: { groups: legendGroups(view, counts), keys: [] } };
}

// Per fitted target: how many gaps, which method filled them, and the error
// estimated for the whole choice (nested cross-validation on the plots
// recording the value) beside the model's own cross-validated error.
function renderGaps(r) {
  const box = $("gaps");
  box.innerHTML = (r.gaps ?? []).map(({ target, scale, report }) => {
    const unit = { linear: target.units ? ` ${target.units}` : "", log: " (absolute error of log1p)",
                   circular: "°", geographic: " km (median)" }[scale];
    const digits = scale === "log" ? 3 : 1;
    const head = `${label(target)}: ${report.n_gaps.toLocaleString()} of ${(report.n_gaps + report.n_complete).toLocaleString()} plots without it`;
    if (report.skipped) return `<li>${escape(head)}, imputed with the model's value (${escape(report.skipped)})</li>`;
    const groups = report.groups.map((g) => {
      const modelError = g.cv_error.find(([m]) => m === 1)?.[1];
      const where = report.groups.length > 1 ? `${g.n_gaps.toLocaleString()} ${g.group === "located" ? "with" : "without"} coordinates: ` : "";
      const how = `${where}imputed from ${describeGapMethod(g.method, g.k, scale)}`;
      if (!Number.isFinite(g.estimated_error)) return how;
      const versus = g.method === 1 || !Number.isFinite(modelError) ? "" : ` against ${modelError.toFixed(digits)} for the model's value`;
      return `${how}, estimated error ${g.estimated_error.toFixed(digits)}${unit}${versus}`;
    });
    return `<li>${escape(head)}; ${escape(groups.join("; "))}, judged on the ${report.n_complete.toLocaleString()} plots recording it</li>`;
  }).join("");
  box.hidden = !box.innerHTML;
}

// The plots on the map in the chosen variable's colours, one by one or on
// the grid; `fit` zooms to them. On the grid with the cell size set by the
// zoom, a zoom redraws the cells.
let map = null, layer = null;
function renderMap(r, fit = true) {
  const coords = r.filledCoords ?? r.coords ?? r.predictedCoords;
  showPanel("map-panel", !!coords && !!window.L);
  if (!coords || !window.L) return;
  if (!map) {
    map = L.map("map", { preferCanvas: true });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                { maxZoom: 18, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
    map.on("zoomend", () => {
      if (state.result && mapMode() === "grid") renderMap(state.result, false);
    });
  }
  const pts = coords.filter(([la, lo]) => Number.isFinite(la) && Number.isFinite(lo));
  if (fit && pts.length) map.fitBounds(pts, { padding: [16, 16], maxZoom: 9 });
  map.invalidateSize();
  layer?.remove();
  layer = L.layerGroup().addTo(map);
  const view = shownView(r);
  const grid = mapMode() === "grid";
  const { marks, legend } = mapMarks(r, view, coords);
  for (const m of marks) {
    const shape = m.kind === "cell"
      ? L.polygon(m.outline, { fillColor: m.fill, fillOpacity: m.opacity, color: CELL_EDGE, opacity: 1, weight: m.edge })
      : m.kind === "ring" ? L.circleMarker(m.at, { radius: 3, weight: 1.5, color: m.fill, opacity: 0.9, fill: false })
      : L.circleMarker(m.at, { radius: 3, weight: 0, fillOpacity: 0.8, fillColor: m.fill });
    shape.bindTooltip(m.tip).addTo(layer);
  }
  $("map").setAttribute("aria-label", `Map of the plots${grid ? ` on a ${gridKm()} km grid` : ""}${view ? `, coloured by ${view.label}` : ""}`);
  $("legend").classList.toggle("grid", grid);
  $("legend").innerHTML = [
    ...legend.groups.map((l) => `<span><i style="background:${l.colour}"></i>${escape(l.label)}</span>`),
    ...legend.keys.map((k) => `<span class="key"><i class="${k.shape}"></i>${escape(k.label)}</span>`),
  ].join("");
  $("download-map").disabled = !pts.length;
}

// Country outlines for the map download: Eurostat GISCO's countries (2020,
// WGS84), fetched from GISCO when first drawn at a scale. The scale is the
// coarsest whose detail the view still shows: 1:20 million, 1:10 million or
// 1:3 million, at 0.26 mm per pixel about 5.3, 2.6 and 0.8 km.
const GISCO = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/";
const GISCO_SCALES = [[5300, "20M"], [2600, "10M"], [0, "03M"]];
const outlines = new Map();
const countryOutlines = (scale) => {
  if (!outlines.has(scale)) {
    outlines.set(scale, fetch(`${GISCO}CNTR_RG_${scale}_2020_4326.geojson`)
      .then((r) => { if (!r.ok) throw new Error(`country outlines: ${r.status}`); return r.json(); })
      .catch((e) => { outlines.delete(scale); throw e; }));
  }
  return outlines.get(scale);
};

// The rings of the outlines as SVG path data in the map's pixels: rings off
// the view left out, and a vertex dropped while it lies within half a pixel
// of the last one kept.
function outlinePaths(geojson, W, H) {
  const paths = [];
  for (const feature of geojson.features) {
    const g = feature.geometry;
    if (!g) continue;
    const polygons = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
    let d = "";
    for (const rings of polygons) {
      for (const ring of rings) {
        const pts = [];
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, last = null;
        for (const [lon, lat] of ring) {
          const p = map.latLngToContainerPoint([lat, lon]);
          if (last && Math.abs(p.x - last.x) < 0.5 && Math.abs(p.y - last.y) < 0.5) continue;
          pts.push(p);
          last = p;
          x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
        }
        if (pts.length < 3 || x1 < 0 || y1 < 0 || x0 > W || y0 > H) continue;
        d += `M${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join("L")}Z`;
      }
    }
    if (d) paths.push(d);
  }
  return paths;
}

// The legend's symbol for a shape at height y: a dot, ring or grid cell.
function legendSymbol(shape, fill, y) {
  if (shape === "ring") return `<circle cx="6" cy="${y}" r="3.5" fill="none" stroke="${fill}" stroke-width="1.5"/>`;
  if (shape === "dot") return `<circle cx="6" cy="${y}" r="4" fill="${fill}"/>`;
  return `<rect x="1.5" y="${y - 4.5}" width="9" height="9" fill="${fill}" fill-opacity="0.75" stroke="${CELL_EDGE}" stroke-width="1"/>`;
}

// The map as it is shown, as an SVG: the country outlines, the marks of the
// plots in view in the shown variable's colours, and the legend. Marks with
// the same style share one path and a spot is drawn once per style, so the
// file stays small with many plots.
async function downloadMap() {
  const r = state.result;
  if (!map || !r) return;
  const button = $("download-map");
  button.disabled = true;
  try {
    const { x: W, y: H } = map.getSize();
    const scale = GISCO_SCALES.find(([m]) => metresPerPixel() >= m)[1];
    const land = outlinePaths(await countryOutlines(scale), W, H);
    const coords = r.filledCoords ?? r.coords ?? r.predictedCoords;
    const view = shownView(r);
    const inView = (p) => {
      const q = map.latLngToContainerPoint(coords[p]);
      return q.x >= -4 && q.y >= -4 && q.x <= W + 4 && q.y <= H + 4;
    };
    const { marks, legend } = mapMarks(r, view, coords, inView);
    const styles = new Map();
    const at = (latLon) => { const q = map.latLngToContainerPoint(latLon); return `${q.x.toFixed(1)} ${q.y.toFixed(1)}`; };
    for (const m of marks) {
      let style, d;
      if (m.kind === "cell") {
        style = `fill="${m.fill}" fill-opacity="${m.opacity.toFixed(2)}" stroke="${CELL_EDGE}" ` +
                `stroke-width="${m.edge.toFixed(2)}" stroke-linejoin="round"`;
        d = `M${m.outline.map(at).join("L")}Z`;
      } else if (m.kind === "ring") {
        const q = map.latLngToContainerPoint(m.at);
        style = `fill="none" stroke="${m.fill}" stroke-width="1.5" stroke-opacity="0.9"`;
        d = `M${(q.x - 3).toFixed(1)} ${q.y.toFixed(1)}a3 3 0 1 0 6 0a3 3 0 1 0-6 0`;
      } else {
        style = `fill="none" stroke="${m.fill}" stroke-width="6" stroke-linecap="round" stroke-opacity="0.85"`;
        d = `M${at(m.at)}h0`;
      }
      if (!styles.has(style)) styles.set(style, new Set());
      styles.get(style).add(d);
    }
    const grid = mapMode() === "grid";
    const items = [...legend.groups.map((g) => ({ label: g.label, colour: g.colour, shape: grid ? "cell" : "dot" })),
                   ...legend.keys.map((k) => ({ label: k.label, colour: "#6b7178", shape: k.shape }))];
    const notice = "© EuroGeographics for the administrative boundaries";
    const total = H + 12 + 16 * (items.length + 1) + 6;
    const source = `Country boundaries: Eurostat GISCO, ${notice}`;
    const width = Math.ceil(Math.max(W, 22 + 6.2 * Math.max(0, ...items.map((l) => l.label.length)), 5.1 * source.length));
    const svg = [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${total}" viewBox="0 0 ${width} ${total}" font-family="Helvetica, Arial, sans-serif">`,
      `<rect width="${width}" height="${total}" fill="#fff"/>`,
      `<clipPath id="view"><rect width="${W}" height="${H}"/></clipPath>`,
      `<g clip-path="url(#view)">`,
      `<rect width="${W}" height="${H}" fill="#e6edf2"/>`,
      `<g fill="#fbfbf8" stroke="#9aa4ad" stroke-width="0.6" stroke-linejoin="round" fill-rule="evenodd">${land.map((d) => `<path d="${d}"/>`).join("")}</g>`,
      ...[...styles].map(([style, set]) => `<path ${style} d="${[...set].join("")}"/>`),
      "</g>",
      `<g transform="translate(0 ${H + 12})" font-size="11">`,
      ...items.map((l, k) => legendSymbol(l.shape, l.colour, 4 + 16 * k) +
                             `<text x="16" y="${8 + 16 * k}" fill="#333">${escape(l.label)}</text>`),
      `<text x="0" y="${8 + 16 * items.length}" font-size="9" fill="#6b7178">${escape(source)}</text>`,
      "</g>",
      "</svg>",
    ];
    const blob = new Blob([svg.join("\n")], { type: "image/svg+xml" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "map.svg" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } catch (e) {
    status(`Could not draw the map: ${e.message}`);
  } finally {
    button.disabled = false;
  }
}

function download() {
  const r = state.result;
  const cols = [{ name: "PlotObservationID", values: r.plotIds },
                { name: "Species recognised (share of cover)", values: Array.from(r.recognised), digits: 4 },
                ...r.columns.map((c) => ({ ...c, digits: c.digits === undefined ? undefined : c.digits + 2 }))];
  const quote = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const lines = [cols.map((c) => quote(c.name)).join(",")];
  for (let p = 0; p < r.plotIds.length; p++) {
    lines.push(cols.map((c) => quote(format(c.values[p], c.digits))).join(","));
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"),
                          { href: URL.createObjectURL(blob), download: "imputed.csv" });
  a.click();
  URL.revokeObjectURL(a.href);
}

const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

// One choice per variable; which models predict it follows from the fields
// the tables give, and is named beside it.
function renderTargets() {
  const box = $("targets");
  const off = new Set([...box.querySelectorAll("input[name=target]:not(:checked)")].map((i) => i.value));
  const models = targetNames().map((name) => [name, chosenModel(name)]).filter(([, t]) => t);
  $("gaps-option").hidden = !models.some(([, t]) => partlyRecorded(t));
  box.innerHTML = models.map(([name, t]) => {
    const c = partlyRecorded(t) ? recordedCount(t) : null;
    const gaps = c && gapsOn()
      ? `; ${(c.total - c.recorded).toLocaleString()} imputed values calibrated on the ${c.recorded.toLocaleString()} plots recording it` : "";
    return `<label class="choice"><input class="form-check-input" type="checkbox" name="target" value="${escape(name)}"` +
      `${off.has(name) ? "" : " checked"}> ${escape(LABELS[name] ?? name)}` +
      `<span class="hint">from ${escape(reads(t))}${escape(gaps)}</span>` +
      `${name === "eunis" ? levelChoice() : ""}</label>`;
  }).join("");
  for (const input of box.querySelectorAll("input[name=target]")) input.addEventListener("change", startPreload);
  $("o-level")?.addEventListener("change", (e) => {
    state.level = Number(e.target.value);
    renderTargets();
    startPreload();
  });
  progress();
}

// The EUNIS levels on offer, inside the habitat choice; while the release
// holds level 1 alone, a note that the finer levels are still to come, and
// for models released from their canonical cells (no production fits yet),
// what they are.
function levelChoice() {
  const levels = habitatChoices();
  if (levels.length < 2) {
    const n = Object.keys(levels[0]?.target.class_names ?? {}).length;
    return `<span class="hint">level 1${n ? ` (${n} groups)` : ""}</span>`;
  }
  const chosen = habitatChoice();
  const options = levels.map((c) => {
    const n = c.readAt ? c.target.read_at[c.readAt].classes.length : Object.keys(c.target.class_names ?? {}).length;
    const read = c.readAt ? `, read from level ${c.target.level}` : "";
    return `<option value="${c.level}"${c.level === chosen.level ? " selected" : ""}>level ${c.level}` +
      `${n ? ` (${n} ${c.level === 3 ? "habitats" : "groups"}${read})` : ""}</option>`;
  }).join("");
  const provisional = chosen.target.source === "canonical"
    ? `<span class="hint">from level ${chosen.target.level} models (${escape(chosen.target.encoder ?? "")}) trained on ` +
      `${Math.round(100 * chosen.target.train_share)}% of the plots; the release's own fits are still to come</span>`
    : "";
  return `<select class="form-select form-select-sm" id="o-level" aria-label="EUNIS level">${options}</select>${provisional}`;
}

// What a target's models read, in words: species, and coordinates and date
// where they read them.
function reads(t) {
  const words = ["species"];
  if (t.inputs.includes(COLUMNS.latitude)) words.push("coordinates");
  if (t.inputs.includes("Year")) words.push("date");
  return words.length === 1 ? "species only" : `${words.slice(0, -1).join(", ")} and ${words.at(-1)}`;
}

let preloaded = "";

function startPreload() {
  const order = ["eunis", "alliance", "altitude", "year", "area", "slope", "location"];
  const rank = (t) => order.indexOf(t.level ? "eunis" : t.name);
  const targets = chosenTargets().sort((a, b) => rank(a) - rank(b));
  const members = targets.flatMap((t) => t.members).join();
  if (members === preloaded) return progress();
  preloaded = members;
  preload(targets, () => preloaded === members).catch((e) => { $("download-status").textContent = `Download failed: ${e.message}`; });
}

// The loader: the logo fills from the bottom as the page starts, one step at a
// time, each filling to its share over the time it is expected to take; the
// page appears when the last step is done.
let loaderDone = false;

function loaderStep(text, share, seconds) {
  if (loaderDone) return;
  $("loader-step").textContent = text;
  $("hex-fill").style.setProperty("--dur", `${seconds}s`);
  $("hex-fill").style.setProperty("--p", `${(100 * share).toFixed(1)}%`);
}

function loaderFinish() {
  if (loaderDone) return;
  loaderStep("Ready.", 1, 0.3);
  setTimeout(() => { loaderDone = true; $("loader").classList.add("done"); }, 350);
  setTimeout(() => { $("loader").hidden = true; $("tool").hidden = false; }, 750);
}

function loaderFail(text) {
  if (loaderDone) { $("download-status").textContent = text; return; }
  $("loader-step").textContent = text;
  $("loader").classList.add("error");
}

async function start() {
  loaderStep("Reading the model list…", 0.5, 4);
  showWorkers();
  $("o-workers").addEventListener("change", () => {
    const v = $("o-workers").value.trim();
    pool.manual = v === "" ? null : Math.max(1, Math.min(16, Math.round(Number(v)) || 1));
    $("o-workers").value = pool.manual ?? "";
    dispatch();
  });
  $("o-terrain").addEventListener("change", renderTargets);
  $("o-gaps").addEventListener("change", renderTargets);
  for (const which of Object.keys(DATA_INPUTS)) $(DATA_INPUTS[which]).addEventListener("change", () => onDataInput(which));
  $("reset-data").addEventListener("click", resetData);
  $("run").addEventListener("click", () => run().catch((e) => { status(`Error: ${e.message}`); $("run").disabled = false; }));
  $("download").addEventListener("click", download);
  $("pager-first").addEventListener("click", () => showTablePage(0));
  $("pager-prev").addEventListener("click", () => showTablePage(tablePage.page - 1));
  $("pager-next").addEventListener("click", () => showTablePage(tablePage.page + 1));
  $("pager-last").addEventListener("click", () => showTablePage(tablePageCount() - 1));
  $("download-map").addEventListener("click", () => downloadMap());
  $("variable-prev").addEventListener("click", () => stepVariable(-1));
  $("variable-next").addEventListener("click", () => stepVariable(1));
  for (const control of document.querySelectorAll('#resolve-app input[name="map-view"]')) {
    control.addEventListener("change", () => { if (state.result) renderMap(state.result, false); });
  }
  try {
    state.suites = JSON.parse(new TextDecoder().decode(await file("suites.json"))).suites;
  } catch (e) {
    loaderFail(`Could not read the model list: ${e.message}`);
    return;
  }
  $("terrain-option").hidden = !state.suites.some((s) => s.terrain_field);
  refreshInputs();
  loaderStep("Reading the example plots…", 0.9, 2);
  const params = PARAMS;
  if (params.has("species")) {
    for (const which of ["species", "header"]) {
      if (params.has(which)) await loadTable(which, params.get(which));
    }
    refreshInputs();
  } else {
    await loadExample().catch((e) => { $("species-error").textContent = `Could not load the example data: ${e.message}`; });
  }
  // Models are preloaded only once the tables are read: before, the suite
  // chosen is the species-only one, which tables with coordinates replace.
  state.tablesRead = true;
  refreshInputs();
  loaderFinish();
  if (params.has("run") && state.tables.species) await run();
}

start().catch((e) => loaderFail(`Could not start: ${e.message}`));
