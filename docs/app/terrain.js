// Terrain correction on the page: the engine's coded terrain field read
// block by block over HTTP range requests (terrain_codec.cpp), its terrain
// descriptors (terrain_source.cpp, terrain_descriptors) and the gated
// terrain correction of a suite target (terrain_correction.cpp, apply).
//
// The field is the file the suite ships. The page reads it through a small
// index written by app/build_terrain.py (the tiles, where each tile's stream
// starts in the field, and every block's length as a uint16 in a second
// file), so a run fetches only the blocks its plots reach. For each plot it
// asks for every block within reach of the plot's block, whatever the plot's
// place inside that block, so the requests tell the server which block each
// plot lies in and nothing finer.

const kRadians = Math.PI / 180;
const kEarthRadiusM = 6371000.0;   // kTerrainEarthRadiusM
const kWindowM = 1000.0;           // kTerrainWindowM
const kLattice = 33;               // kTerrainLattice
const kNaN = NaN;

// ---------------------------------------------------------------------------
// Byte sources: read([[begin, end), ...]) -> Uint8Array per range
// ---------------------------------------------------------------------------

// A server's answer other than the one asked for.
export class HttpStatusError extends Error {
  constructor(url, status, expected) {
    super(`${url}: the server answered ${status}, not ${expected}`);
    this.status = status;
  }
}

// A response that sent no bytes for `ms` milliseconds; the request is
// cancelled and counts as a dropped connection.
export class StalledError extends Error {
  constructor(url, ms) {
    super(`${url}: no data for ${ms / 1000} s`);
  }
}

export const STALL_MS = 30000;

// The body of a 206 answer to one Range request, its byte count so far
// reported as it arrives (`onBytes(received, total)`). A response that
// stalls for `stallMs` is cancelled with a StalledError.
export async function fetchRange(fetchImpl, url, range, { onBytes, stallMs = STALL_MS } = {}) {
  const abort = new AbortController();
  let timer = null;
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => abort.abort(new StalledError(url, stallMs)), stallMs);
  };
  arm();
  try {
    const response = await fetchImpl(url, { headers: { Range: range }, signal: abort.signal, cache: "no-store" });
    if (response.status !== 206) {
      abort.abort();
      throw new HttpStatusError(url, response.status, 206);
    }
    if (!response.body) return new Uint8Array(await response.arrayBuffer());
    const total = Number(response.headers.get("Content-Length")) || 0;
    const reader = response.body.getReader();
    const parts = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      arm();
      parts.push(value);
      received += value.length;
      onBytes?.(received, Math.max(total, received));
    }
    const bytes = new Uint8Array(received);
    let at = 0;
    for (const part of parts) { bytes.set(part, at); at += part.length; }
    onBytes?.(received, received);
    return bytes;
  } catch (e) {
    throw abort.signal.reason instanceof StalledError ? abort.signal.reason : e;
  } finally {
    clearTimeout(timer);
  }
}

// A request run again after a server error (5xx), a dropped connection
// (fetch and reading a body fail with a TypeError) or a stalled response,
// which a hosting site gives now and then under load: up to `tries` times,
// waiting 1, 2, 4 ... s between.
export async function withRetry(task, { tries = 4, wait = (k) => 1000 * 2 ** k } = {}) {
  for (let k = 0; ; k++) {
    try {
      return await task();
    } catch (e) {
      const transient = e instanceof TypeError || e instanceof StalledError ||
        (e instanceof HttpStatusError && e.status >= 500);
      if (!transient || k + 1 >= tries) throw e;
      await new Promise((r) => setTimeout(r, wait(k)));
    }
  }
}

// Ranges of one file over HTTP: many ranges per request (multipart/byteranges,
// RFC 9110 section 14.6), at most `maxRanges` per request and `concurrency`
// requests at a time, so a hosting site's request limit is met. A server that
// answers with the whole file is cut off at once. `offset` places the source
// inside a larger file (a file of the model pack): ranges are asked for that
// many bytes further on.
export class HttpRangeSource {
  constructor(url, { offset = 0, maxRanges = 300, concurrency = 2, fetchImpl = (...args) => globalThis.fetch(...args) } = {}) {
    this.url = url;
    this.offset = offset;
    this.maxRanges = maxRanges;
    this.concurrency = concurrency;
    this.fetch = fetchImpl;
    this.requests = 0;
    this.bytes = 0;
  }

  async read(ranges) {
    const out = new Array(ranges.length);
    const order = ranges.map((_, i) => i).sort((a, b) => ranges[a][0] - ranges[b][0]);
    const batches = [];
    for (let i = 0; i < order.length; i += this.maxRanges) batches.push(order.slice(i, i + this.maxRanges));
    let next = 0;
    const worker = async () => {
      while (next < batches.length) {
        const batch = batches[next++];
        const parts = await this.request(batch.map((i) => ranges[i]));
        batch.forEach((i, k) => { out[i] = parts[k]; });
      }
    };
    await Promise.all(Array.from({ length: Math.min(this.concurrency, batches.length) }, worker));
    return out;
  }

  async request(asked) {
    const ranges = asked.map(([b, e]) => [b + this.offset, e + this.offset]);
    const header = "bytes=" + ranges.map(([b, e]) => `${b}-${e - 1}`).join(",");
    return withRetry(async () => {
      this.requests += 1;
      const body = await fetchRange(this.fetch, this.url, header);
      this.bytes += body.length;
      return splitRangeResponse(body, ranges);
    });
  }
}

// The parts of a 206 response in the order the ranges were asked for. A
// multipart body is told apart by its first line (some servers name another
// content type in the header); a single part is the one range asked, or the
// span of all of them when the server merged them.
export function splitRangeResponse(body, ranges) {
  const parts = parseMultipart(body);
  if (parts) {
    return ranges.map(([b, e]) => {
      const p = parts.find((q) => q.begin <= b && q.end >= e);
      if (!p) throw new Error(`range ${b}-${e - 1} is missing from the server's answer`);
      return p.data.subarray(b - p.begin, e - p.begin);
    });
  }
  if (ranges.length === 1 && body.length === ranges[0][1] - ranges[0][0]) return [body];
  const begin = Math.min(...ranges.map((r) => r[0])), end = Math.max(...ranges.map((r) => r[1]));
  if (body.length === end - begin) return ranges.map(([b, e]) => body.subarray(b - begin, e - begin));
  throw new Error(`the server's answer (${body.length} bytes) does not hold the ${ranges.length} ranges asked for`);
}

const ascii = (bytes, from, to) => String.fromCharCode(...bytes.subarray(from, to));

// One text line of a multipart body from `at`, ended by LF or CRLF (servers
// differ): [line, start of the next line], or null past the body's end.
function line(body, at) {
  if (at >= body.length) return null;
  let end = at;
  while (end < body.length && body[end] !== 10) end++;
  const text = ascii(body, at, end > at && body[end - 1] === 13 ? end - 1 : end);
  return [text, end + 1];
}

// multipart/byteranges: [{begin, end, data}], or null for a body that is not
// one. Each part's length is read from its Content-Range, so the data may
// hold any bytes, the boundary's included.
function parseMultipart(body) {
  let at = 0, l;
  while ((l = line(body, at)) && l[0] === "") at = l[1];
  if (!l || !l[0].startsWith("--")) return null;
  const boundary = l[0].slice(2);
  at = l[1];
  const parts = [];
  for (;;) {
    let range = null;
    while ((l = line(body, at)) && l[0] !== "") {
      const m = l[0].match(/^content-range:\s*bytes\s+(\d+)-(\d+)\/(\d+|\*)/i);
      if (m) range = [Number(m[1]), Number(m[2]) + 1];
      at = l[1];
    }
    if (!l || !range) return null;
    at = l[1];
    const [begin, end] = range;
    if (at + end - begin > body.length) return null;
    parts.push({ begin, end, data: body.subarray(at, at + end - begin) });
    at += end - begin;
    while ((l = line(body, at)) && l[0] === "") at = l[1];
    if (!l) return null;
    if (l[0] === `--${boundary}--`) return parts;
    if (l[0] !== `--${boundary}`) return null;
    at = l[1];
  }
}

// ---------------------------------------------------------------------------
// Tensor packs (tensor_pack.hpp) held whole in memory
// ---------------------------------------------------------------------------

const DTYPES = {
  float16: null, float32: Float32Array, float64: Float64Array, int16: Int16Array,
  int32: Int32Array, int64: BigInt64Array, uint8: Uint8Array,
};

export function readTensorPack(buffer) {
  const bytes = new Uint8Array(buffer);
  if (ascii(bytes, 0, 11) !== "RESOLVEPACK" || bytes[11] !== 0) throw new Error("not a tensor pack");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(12, true) !== 1) throw new Error("tensor pack version is not 1");
  const headerLength = Number(view.getBigUint64(16, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(24, 24 + headerLength)));
  const dataStart = Math.ceil((24 + headerLength) / 64) * 64;
  const tensors = new Map();
  for (const t of header.tensors) {
    const Type = DTYPES[t.dtype];
    if (!Type) throw new Error(`tensor pack dtype ${t.dtype} is not read here`);
    const at = bytes.byteOffset + dataStart + t.offset;
    const data = new Type(bytes.buffer.slice(at, at + t.bytes));
    tensors.set(t.name, { shape: t.shape, data });
  }
  return { meta: header.meta, tensors };
}

// ---------------------------------------------------------------------------
// Block decoding (terrain_codec.cpp, decode_block)
// ---------------------------------------------------------------------------

const kEscape = 32;

// The stream BitWriter writes, most significant bit first, through a 32-bit
// window: a unary run is counted with Math.clz32 instead of bit by bit (the
// engine's BitReader does the same over 64 bits).
const ENDS_EARLY = "terrain codec: block stream ends early";

class BitReader {
  constructor(bytes) {
    this.bytes = bytes;
    this.next = 0;
    this.window = 0;  // valid bits left-aligned, the rest zero
    this.valid = 0;
  }
  refill() {
    while (this.valid <= 24 && this.next < this.bytes.length) {
      this.window = (this.window | (this.bytes[this.next++] << (24 - this.valid))) >>> 0;
      this.valid += 8;
    }
  }
  consume(count) {
    this.window = count >= 32 ? 0 : (this.window << count) >>> 0;
    this.valid -= count;
  }
  // count in [0, 24].
  small(count) {
    if (count === 0) return 0;
    this.refill();
    if (count > this.valid) throw new Error(ENDS_EARLY);
    const v = this.window >>> (32 - count);
    this.consume(count);
    return v;
  }
  bit() { return this.small(1); }
  // count in [0, 32]; the value as a non-negative Number.
  bits(count) {
    if (count <= 24) return this.small(count);
    const high = this.small(count - 16);
    return high * 65536 + this.small(16);
  }
  // Leading ones up to `limit`, consumed.
  ones(limit) {
    let n = 0;
    for (;;) {
      this.refill();
      if (this.valid === 0) throw new Error(ENDS_EARLY);
      const run = Math.min(Math.clz32(~this.window), this.valid, limit - n);
      this.consume(run);
      n += run;
      if (n === limit || this.valid > 0) return n;
    }
  }
  // Leading zeros up to `limit`, consumed.
  zeros(limit) {
    let n = 0;
    for (;;) {
      this.refill();
      if (this.valid === 0) throw new Error(ENDS_EARLY);
      const run = Math.min(Math.clz32(this.window), this.valid, limit - n);
      this.consume(run);
      n += run;
      if (n === limit || this.valid > 0) return n;
    }
  }
  rice(k) {
    const ones = this.ones(kEscape);
    if (ones >= kEscape) return this.bits(32);
    this.consume(1);
    return ones * (1 << k) + this.small(k);
  }
  expGolomb() {
    const len = this.zeros(32);
    if (len > 31) throw new Error("terrain codec: an exponential-Golomb code is too long");
    return this.bits(len + 1) - 1;
  }
}

const unzigzag = (z) => (z % 2 === 1 ? -(z + 1) / 2 : z / 2);

// One block's bytes to its elevations (quantised value x quantum, as float32),
// h x w, row-major.
export function decodeBlock(bytes, h, w, quantum) {
  const br = new BitReader(bytes);
  const n = h * w;
  const q = new Int32Array(n);
  let first = br.bits(32);
  if (first >= 2 ** 31) first -= 2 ** 32;
  const runMode = br.bit() === 1;
  const k = br.bits(4);
  if (!runMode) {
    // The Rice loop inlined over the reader's window: one clz32 per unary
    // run, one shift per remainder.
    const data = bytes, len = data.length;
    let { window, valid, next } = br;
    for (let i = 1; i < n; i++) {
      while (valid <= 24 && next < len) { window = (window | (data[next++] << (24 - valid))) >>> 0; valid += 8; }
      let ones = Math.clz32(~window);
      if (ones < valid && ones < kEscape && ones + 1 + k <= valid) {
        window = ones === 31 ? 0 : (window << (ones + 1)) >>> 0;  // a shift by 32 is a shift by 0
        valid -= ones + 1;
        let z = ones << k;
        if (k) { z |= window >>> (32 - k); window = (window << k) >>> 0; valid -= k; }
        q[i] = (z >>> 1) ^ -(z & 1);
        continue;
      }
      br.window = window; br.valid = valid; br.next = next;
      q[i] = unzigzag(br.rice(k));
      ({ window, valid, next } = br);
    }
  } else {
    let i = 1;
    for (;;) {
      const run = br.expGolomb();
      if (run > n - i) throw new Error("terrain codec: a zero run passes the block's end");
      i += run;
      if (i >= n) break;
      q[i++] = unzigzag(br.rice(k) + 1);
    }
  }
  q[0] = first;
  for (let c = 1; c < w; c++) q[c] += q[c - 1];
  for (let r = 1; r < h; r++) {
    const row = r * w, above = row - w;
    q[row] += q[above];
    for (let c = 1; c < w; c++) q[row + c] += q[row + c - 1] + q[above + c] - q[above + c - 1];
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = q[i] * quantum;
  return out;
}

// ---------------------------------------------------------------------------
// Tiles (terrain_source.cpp, TerrainTileIndex)
// ---------------------------------------------------------------------------

const packKey = (south, west) => (south + 1000) * 4000 + (west + 2000);

export class TileIndex {
  constructor(tiles) {
    this.tiles = tiles;
    this.byKey = new Map();
    tiles.forEach((t, i) => {
      const [s, w] = TileIndex.key(t);
      t.south = s;
      t.west = w;
      if (this.byKey.has(packKey(s, w))) throw new Error(`tiles ${t.name} and another cover the same degree square`);
      this.byKey.set(packKey(s, w), i);
    });
  }

  // A quarter cell inward from the south-west cell centre (TerrainTile::key).
  static key(t) {
    const south = t.lat_first - t.rows * t.dlat + 0.25 * t.dlat;
    const west = t.lon_first + 0.25 * t.dlon;
    return [Math.floor(south), Math.floor(west)];
  }

  find(south, west) { return this.byKey.get(packKey(south, west)) ?? -1; }

  tileOf(lat, lon) {
    return Number.isFinite(lat) && Number.isFinite(lon) ? this.find(Math.floor(lat), Math.floor(lon)) : -1;
  }

  // A cell moved across tile edges, rows first, then columns; [-1, ...] when
  // it leaves every tile (TerrainTileIndex::normalise).
  normalise(t, r, c) {
    if (t < 0) return [-1, r, c];
    for (let axis = 0; axis < 2; axis++) {
      const tt = this.tiles[t];
      const v = axis === 0 ? r : c;
      const extent = axis === 0 ? tt.rows : tt.cols;
      if (v >= 0 && v < extent) continue;
      const after = v >= extent;
      const next = axis === 0 ? this.find(after ? tt.south - 1 : tt.south + 1, tt.west)
                              : this.find(tt.south, after ? tt.west + 1 : tt.west - 1);
      if (next < 0) return [-1, r, c];
      const nt = this.tiles[next];
      const nextExtent = axis === 0 ? nt.rows : nt.cols;
      const moved = after ? v - extent : v + nextExtent;
      if (moved < 0 || moved >= nextExtent) return [-1, r, c];
      if (axis === 0) {
        r = moved;
        if (nt.dlon !== tt.dlon) c = llround((tt.lon_first + c * tt.dlon - nt.lon_first) / nt.dlon);
      } else {
        c = moved;
      }
      t = next;
    }
    return [t, r, c];
  }

  nearestCell(lat, lon) {
    const t = this.tileOf(lat, lon);
    if (t < 0) return [-1, 0, 0];
    const tt = this.tiles[t];
    return this.normalise(t, llround((tt.lat_first - lat) / tt.dlat), llround((lon - tt.lon_first) / tt.dlon));
  }

  centre(t, r, c) {
    if (t < 0) return [kNaN, kNaN];
    const tt = this.tiles[t];
    return [tt.lat_first - r * tt.dlat, tt.lon_first + c * tt.dlon];
  }
}

// std::llround: halfway cases away from zero.
const llround = (x) => (x < 0 ? -Math.round(-x) : Math.round(x));

// ---------------------------------------------------------------------------
// The coded field over a byte source
// ---------------------------------------------------------------------------

// index: the parsed index.json of app/build_terrain.py; field and lengths:
// byte sources over the field and the block-length file.
export class TerrainField {
  constructor(index, field, lengths) {
    if (index.format !== "resolve-terrain-index" || index.version !== 1) {
      throw new Error("not a terrain index of version 1");
    }
    this.block = index.block;
    this.quantum = index.quantum;
    const T = index.tiles;
    const tiles = T.name.map((name, i) => ({
      name, rows: T.rows[i], cols: T.cols[i], lat_first: T.lat_first[i], lon_first: T.lon_first[i],
      dlat: T.dlat[i], dlon: T.dlon[i], stream: T.stream[i], lengthsAt: T.lengths[i],
    }));
    this.index = new TileIndex(tiles);
    this.field = field;
    this.lengths = lengths;
    this.offsets = new Map();  // tile -> Float64Array of block offsets in its stream
    this.blocks = new Map();   // block key -> Float32Array
    this.base = [];
    let base = 0;
    for (const t of tiles) {
      this.base.push(base);
      base += this.blockCols(t) * Math.ceil(t.rows / this.block);
    }
  }

  blockCols(tile) { return Math.ceil(tile.cols / this.block); }

  blockKey(t, r, c) {
    return this.base[t] + Math.floor(r / this.block) * this.blockCols(this.index.tiles[t]) + Math.floor(c / this.block);
  }

  // The blocks a plot's request names: every block reached by a cell within
  // the window's half width (and four cells more) of the plot's block, so the
  // set is the same for every plot in that block.
  requestFor(lat, lon) {
    const [t, r, c] = this.index.nearestCell(lat, lon);
    if (t < 0) return [];
    const tt = this.index.tiles[t], B = this.block;
    const r0 = Math.floor(r / B) * B, c0 = Math.floor(c / B) * B;
    const r1 = Math.min(r0 + B, tt.rows) - 1, c1 = Math.min(c0 + B, tt.cols) - 1;
    const north = tt.lat_first - (r0 - 1) * tt.dlat;
    const mr = Math.ceil(kWindowM / 2 / (tt.dlat * kRadians * kEarthRadiusM)) + 4;
    const mc = Math.ceil(kWindowM / 2 / (tt.dlon * kRadians * kEarthRadiusM * Math.cos(Math.min(89.9, Math.abs(north) + 0.05) * kRadians))) + 4;
    const keys = new Set();
    // Rows grouped by the tile and block row they land in; columns walked
    // once per group.
    const groups = new Map();
    for (let rr = r0 - mr; rr <= r1 + mr; rr++) {
      const [nt, nr] = this.index.normalise(t, rr, c0);
      if (nt < 0) continue;
      const g = `${nt}:${Math.floor(nr / B)}`;
      if (!groups.has(g)) groups.set(g, rr);
    }
    for (const rr of groups.values()) {
      for (let cc = c0 - mc; cc <= c1 + mc; cc++) {
        const [nt, nr, nc] = this.index.normalise(t, rr, cc);
        if (nt >= 0) keys.add(this.blockKey(nt, nr, nc));
      }
    }
    return [...keys];
  }

  tileOfKey(key) {
    let lo = 0, hi = this.base.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.base[mid] <= key) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  // Each tile's block offsets, from its block lengths.
  async loadOffsets(tiles) {
    const missing = tiles.filter((t) => !this.offsets.has(t));
    if (!missing.length) return;
    const ranges = missing.map((t) => {
      const tile = this.index.tiles[t];
      const n = this.blockCols(tile) * Math.ceil(tile.rows / this.block);
      return [tile.lengthsAt, tile.lengthsAt + 2 * n];
    });
    const parts = await this.lengths.read(ranges);
    missing.forEach((t, i) => {
      const p = parts[i];
      const view = new DataView(p.buffer, p.byteOffset, p.byteLength);
      const n = p.byteLength / 2;
      const off = new Float64Array(n + 1);
      for (let b = 0; b < n; b++) off[b + 1] = off[b] + view.getUint16(2 * b, true);
      this.offsets.set(t, off);
    });
  }

  // Fetch and decode the blocks under `keys` not yet held. Consecutive blocks
  // of a tile are read as one range.
  async loadBlocks(keys) {
    const wanted = [...new Set(keys)].filter((k) => !this.blocks.has(k)).sort((a, b) => a - b);
    if (!wanted.length) return;
    const tiles = [...new Set(wanted.map((k) => this.tileOfKey(k)))];
    await this.loadOffsets(tiles);
    const runs = [];
    for (const k of wanted) {
      const t = this.tileOfKey(k), b = k - this.base[t];
      const last = runs.at(-1);
      if (last && last.t === t && last.b1 === b) { last.b1 = b + 1; continue; }
      runs.push({ t, b0: b, b1: b + 1 });
    }
    const ranges = runs.map(({ t, b0, b1 }) => {
      const off = this.offsets.get(t), s = this.index.tiles[t].stream;
      return [s + off[b0], s + off[b1]];
    });
    const parts = await this.field.read(ranges);
    runs.forEach(({ t, b0, b1 }, i) => {
      const off = this.offsets.get(t), tile = this.index.tiles[t], bc = this.blockCols(tile), B = this.block;
      for (let b = b0; b < b1; b++) {
        const h = Math.min(B, tile.rows - Math.floor(b / bc) * B), w = Math.min(B, tile.cols - (b % bc) * B);
        const bytes = parts[i].subarray(off[b] - off[b0], off[b + 1] - off[b0]);
        this.blocks.set(this.base[t] + b, decodeBlock(bytes, h, w, this.quantum));
      }
    });
  }

  // Every block the plots' requests name, fetched; the cells the descriptors
  // read are among them (descriptors() checks each).
  async prepare(lat, lon) {
    const keys = new Set();
    for (let i = 0; i < lat.length; i++) for (const k of this.requestFor(lat[i], lon[i])) keys.add(k);
    await this.loadBlocks([...keys]);
  }

  cellValue(t, r, c) {
    if (t < 0) return kNaN;
    const tile = this.index.tiles[t];
    if (r < 0 || r >= tile.rows || c < 0 || c >= tile.cols) {
      throw new Error(`cell (${r}, ${c}) lies outside tile ${tile.name}`);
    }
    const key = this.blockKey(t, r, c);
    const block = this.blocks.get(key);
    if (!block) {
      throw Object.assign(new Error(`terrain block of cell (${r}, ${c}) in tile ${tile.name} was not fetched`),
                          { missingKey: key });
    }
    const B = this.block, w = Math.min(B, tile.cols - Math.floor(c / B) * B);
    return block[(r % B) * w + (c % B)];
  }

  // bilinear_elevation: the two rows around a point, each in the tile that
  // holds it and read at the point's longitude in that tile's columns,
  // weighted; NaN cells left out.
  elevation(lat, lon) {
    const t = this.index.tileOf(lat, lon);
    const w = [0, 0, 0, 0], v = [0, 0, 0, 0];
    if (t >= 0) {
      const tt = this.index.tiles[t];
      const fr = (tt.lat_first - lat) / tt.dlat;
      const r0 = Math.floor(fr), wr = fr - r0;
      for (let dr = 0; dr < 2; dr++) {
        const [rt, rr] = this.index.normalise(t, r0 + dr, 0);
        if (rt < 0) continue;
        const rtile = this.index.tiles[rt];
        const fc = (lon - rtile.lon_first) / rtile.dlon;
        const c0 = Math.floor(fc), wc = fc - c0;
        for (let dc = 0; dc < 2; dc++) {
          const k = 2 * dr + dc;
          const [nt, nr, nc] = this.index.normalise(rt, rr, c0 + dc);
          const value = this.cellValue(nt, nr, nc);
          if (Number.isFinite(value)) {
            w[k] = (dr ? wr : 1.0 - wr) * (dc ? wc : 1.0 - wc);
            v[k] = value;
          }
        }
      }
    }
    const wsum = w[0] + w[1] + w[2] + w[3];
    return wsum > 0 ? (w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3]) / wsum : kNaN;
  }
}

// ---------------------------------------------------------------------------
// Descriptors (terrain_source.cpp, descriptors_chunk)
// ---------------------------------------------------------------------------

const LATTICE = (() => {
  const axis = Array.from({ length: kLattice }, (_, i) => -kWindowM / 2 + (kWindowM / (kLattice - 1)) * i);
  const ex = [], ny = [];
  for (let i = 0; i < kLattice; i++) for (let j = 0; j < kLattice; j++) { ex.push(axis[j]); ny.push(axis[i]); }
  return { ex, ny };
})();

// Solve the 3 x 3 system A x = b by Gaussian elimination with partial pivoting.
function solve3(A, b) {
  const M = [[A[0], A[1], A[2], b[0]], [A[3], A[4], A[5], b[1]], [A[6], A[7], A[8], b[2]]];
  for (let col = 0; col < 3; col++) {
    let p = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[p][col])) p = r;
    [M[col], M[p]] = [M[p], M[col]];
    for (let r = col + 1; r < 3; r++) {
      const f = M[r][col] / M[col][col];
      for (let k = col; k < 4; k++) M[r][k] -= f * M[col][k];
    }
  }
  const x = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = M[r][3];
    for (let k = r + 1; k < 3; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// The elevation at each coordinate, the mean elevation over the 1 km square
// around it, the slope of the nearest cell from its four edge neighbours and
// the slope of the least-squares plane over the 1 km square (degrees).
export function terrainDescriptors(field, lat, lon) {
  const n = lat.length;
  const out = { elevation: new Float64Array(n), elevation_1km: new Float64Array(n),
                slope: new Float64Array(n), slope_1km: new Float64Array(n) };
  const { ex, ny } = LATTICE, p = ex.length, need = Math.floor((p + 1) / 2);
  const index = field.index;
  for (let i = 0; i < n; i++) {
    const la = lat[i], lo = lon[i];
    // The 1 km lattice: mean and least-squares plane.
    const coslat = kEarthRadiusM * Math.cos(la * kRadians);
    let count = 0, sum = 0;
    const A = [0, 0, 0, 0, 0, 0, 0, 0, 0], b = [0, 0, 0];
    for (let k = 0; k < p; k++) {
      const z = field.elevation(la + ny[k] / kEarthRadiusM / kRadians, lo + ex[k] / coslat / kRadians);
      if (!Number.isFinite(z)) continue;
      count++;
      sum += z;
      const x = [1, ex[k], ny[k]];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) A[3 * r + c] += x[r] * x[c];
        b[r] += x[r] * z;
      }
    }
    const enough = count >= need;
    out.elevation_1km[i] = enough ? sum / Math.max(count, 1) : kNaN;
    if (enough) {
      const coef = solve3(A, b);
      out.slope_1km[i] = Math.atan(Math.sqrt(coef[1] ** 2 + coef[2] ** 2)) / kRadians;
    } else {
      out.slope_1km[i] = kNaN;
    }

    out.elevation[i] = field.elevation(la, lo);

    // Slope of the nearest cell from its four edge neighbours.
    const [t, r, c] = index.nearestCell(la, lo);
    if (t < 0) { out.slope[i] = kNaN; continue; }
    const [clat] = index.centre(t, r, c);
    const z = [[-1, 0], [1, 0], [0, -1], [0, 1]].map(([dr, dc]) => {
      const [nt, nr, nc] = index.normalise(t, r + dr, c + dc);
      const [nlat, nlon] = index.centre(nt, nr, nc);
      return nt < 0 ? kNaN : field.elevation(nlat, nlon);
    });
    const tt = index.tiles[t];
    const dy = tt.dlat * kRadians * kEarthRadiusM;
    const dx = tt.dlon * kRadians * kEarthRadiusM * Math.cos(clat * kRadians);
    const gy = (z[0] - z[1]) / (2.0 * dy), gx = (z[3] - z[2]) / (2.0 * dx);
    out.slope[i] = Math.atan(Math.sqrt(gx * gx + gy * gy)) / kRadians;
  }
  return out;
}

// The descriptors of the plots, their blocks fetched first. A cell outside
// every requested block (none on the check plots) is fetched on its own and
// reported, so the descriptors are always the engine's.
export async function describePlots(field, lat, lon, { onExtra = (key) => console.warn(`terrain block ${key} fetched outside the plots' requests`) } = {}) {
  await field.prepare(lat, lon);
  for (;;) {
    try {
      return terrainDescriptors(field, lat, lon);
    } catch (e) {
      if (e.missingKey === undefined) throw e;
      onExtra(e.missingKey);
      await field.loadBlocks([e.missingKey]);
    }
  }
}

// ---------------------------------------------------------------------------
// Terrain correction (terrain_correction.cpp)
// ---------------------------------------------------------------------------

export const TERRAIN_WEIGHT_NAMES = ["resolve", "dem", "dem_1km"];
const kExperts = 3;

export class TerrainCorrection {
  // buffer: the bytes of a terrain correction file (a tensor pack).
  constructor(buffer) {
    const { meta, tensors } = readTensorPack(buffer);
    if (meta.format !== "resolve-terrain-correction") throw new Error("not a terrain correction file");
    if (meta.version !== 1) throw new Error(`terrain correction version ${meta.version}`);
    this.target = meta.target;
    this.descriptor = meta.descriptor;
    this.scale = meta.config.scale;
    this.offsetFraction = meta.config.offset_fraction;
    this.fills = meta.fills;
    this.report = meta.report;
    this.layers = [];
    for (let l = 0; ; l += 2) {
      const W = tensors.get(`${l}.weight`), bias = tensors.get(`${l}.bias`);
      if (!W) break;
      this.layers.push({ W: W.data, b: bias.data, out: W.shape[0], in: W.shape[1] });
    }
    if (this.layers.at(-1).out !== kExperts + 1) throw new Error("terrain correction: unexpected network output");
  }

  // inputs: {prediction, dem, dem_1km, uncertainty_m, spread,
  // recognised_share, n_species}, arrays of one value per plot (any but the
  // first two may be absent). Returns {corrected, weights (n x 3), offset,
  // applied}.
  apply(inputs) {
    const n = inputs.prediction.length, scale = this.scale, f32 = Math.fround;
    const col = (a) => (a ? Float64Array.from(a, Number) : new Float64Array(n).fill(kNaN));
    const pred = col(inputs.prediction), dem = col(inputs.dem), dem1 = col(inputs.dem_1km);
    const u = col(inputs.uncertainty_m), spread = col(inputs.spread);
    const rec = col(inputs.recognised_share), ns = col(inputs.n_species);
    const filled = (v, fill) => (Number.isFinite(v) ? v : fill);
    const corrected = Float64Array.from(pred), weights = new Float64Array(n * kExperts).fill(kNaN);
    const offset = new Float64Array(n).fill(kNaN), applied = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const has1 = Number.isFinite(dem1[i]);
      const d1raw = has1 ? dem1[i] : dem[i];
      const hasU = Number.isFinite(u[i]);
      const lu = hasU ? Math.log10(Math.max(u[i], 0.0) + 1.0) : 0;
      const sp = Math.max(filled(spread[i], this.fills.spread), 0.0);
      const rc = filled(rec[i], this.fills.recognised_share);
      const nsp = Math.max(filled(ns[i], this.fills.n_species), 0.0);
      const ok = Number.isFinite(pred[i]) && Number.isFinite(dem[i]);
      applied[i] = ok ? 1 : 0;
      if (!ok) continue;
      const p = pred[i] / scale, d = dem[i] / scale, d1 = d1raw / scale;
      const x = Float32Array.of(p, d, d1, d - p, d1 - p, lu, hasU ? 1 : 0, Math.log1p(sp / scale * 100.0), rc,
                          Math.log1p(nsp), has1 ? 1 : 0);
      const E = [f32(p), f32(d), f32(d1)];
      let h = x;
      this.layers.forEach((layer, l) => {
        const y = new Float32Array(layer.out);
        for (let o = 0; o < layer.out; o++) {
          let s = 0;
          for (let j = 0; j < layer.in; j++) s += layer.W[o * layer.in + j] * h[j];
          s = f32(s + layer.b[o]);
          y[o] = l < this.layers.length - 1 ? f32(s / (1 + Math.exp(-s))) : s;
        }
        h = y;
      });
      const m = Math.max(h[0], h[1], h[2]);
      const e = [0, 1, 2].map((k) => f32(Math.exp(h[k] - m)));
      const z = f32(e[0] + e[1] + e[2]);
      const w = e.map((v) => f32(v / z));
      const off = f32(this.offsetFraction * Math.tanh(h[kExperts]));
      const yv = f32(f32(f32(w[0] * E[0]) + f32(w[1] * E[1])) + f32(w[2] * E[2]));
      corrected[i] = f32(yv + off) * scale;
      for (let k = 0; k < kExperts; k++) weights[i * kExperts + k] = w[k];
      offset[i] = off * scale;
    }
    return { corrected, weights, offset, applied };
  }
}
