// Positions on the Earth, the engine's geo.cpp: great-circle distance, the
// geographic median and the Lambert azimuthal equal-area projection of the
// GRS80 ellipsoid. Angles in degrees, projected coordinates in metres.

const EARTH_RADIUS_KM = 6371;
const GRS80_A = 6378137.0, GRS80_INV_F = 298.257222101;
const RAD = Math.PI / 180;

// ---------------------------------------------------------------------------

export function greatCircleKm(lat1, lon1, lat2, lon2) {
  const p1 = lat1 * RAD, p2 = lat2 * RAD, dl = (lon2 - lon1) * RAD;
  const s1 = Math.sin((p2 - p1) / 2), s2 = Math.sin(dl / 2);
  const a = s1 * s1 + Math.cos(p1) * Math.cos(p2) * s2 * s2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(Math.min(1, Math.max(0, a))));
}

// The point minimising the summed great-circle distance (Weiszfeld on the
// unit sphere), started from the normalised mean.
export function geographicMedian(lats, lons, iterations = 200) {
  const xyz = [];
  lats.forEach((la, i) => {
    const lo = lons[i];
    if (!Number.isFinite(la) || !Number.isFinite(lo)) return;
    const phi = la * RAD, lambda = lo * RAD;
    xyz.push([Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)]);
  });
  if (!xyz.length) throw new Error("geographic median: no position has finite coordinates");
  const unit = (p) => { const n = Math.hypot(...p); return p.map((x) => x / n); };
  let point = unit([0, 1, 2].map((j) => xyz.reduce((a, q) => a + q[j], 0) / xyz.length));
  for (let it = 0; it < iterations; it++) {
    const next = [0, 0, 0];
    for (const q of xyz) {
      const dot = Math.min(1, Math.max(-1, q[0] * point[0] + q[1] * point[1] + q[2] * point[2]));
      const w = 1 / Math.max(Math.acos(dot), 1e-9);
      for (let j = 0; j < 3; j++) next[j] += q[j] * w;
    }
    const n = unit(next);
    const step = Math.hypot(n[0] - point[0], n[1] - point[1], n[2] - point[2]);
    if (step < 1e-12) break;
    point = n;
  }
  return [Math.asin(point[2]) / RAD, Math.atan2(point[1], point[0]) / RAD];
}

// Lambert azimuthal equal-area on the GRS80 ellipsoid (Snyder 1987, ch. 24).
export class Laea {
  constructor(lat0, lon0) {
    const f = 1 / GRS80_INV_F;
    this.lat0 = lat0; this.lon0 = lon0; this.a = GRS80_A;
    this.e2 = f * (2 - f); this.e = Math.sqrt(this.e2);
    this.qPole = this.q(1);
    this.rq = this.a * Math.sqrt(this.qPole / 2);
    const phi0 = lat0 * RAD, sin0 = Math.sin(phi0);
    this.beta0 = Math.asin(Math.min(1, Math.max(-1, this.q(sin0) / this.qPole)));
    this.dd = this.a * Math.cos(phi0) / (Math.sqrt(1 - this.e2 * sin0 * sin0) * this.rq * Math.cos(this.beta0));
  }

  q(s) {
    const es = this.e * s;
    return (1 - this.e2) * (s / (1 - this.e2 * s * s) - Math.log((1 - es) / (1 + es)) / (2 * this.e));
  }

  forward(lat, lon) {
    const phi = lat * RAD, dl = (lon - this.lon0) * RAD;
    const beta = Math.asin(Math.min(1, Math.max(-1, this.q(Math.sin(phi)) / this.qPole)));
    const sb = Math.sin(beta), cb = Math.cos(beta), sb0 = Math.sin(this.beta0), cb0 = Math.cos(this.beta0);
    const b = this.rq * Math.sqrt(2 / (1 + sb0 * sb + cb0 * cb * Math.cos(dl)));
    return [b * this.dd * cb * Math.sin(dl), (b / this.dd) * (cb0 * sb - sb0 * cb * Math.cos(dl))];
  }

  // The geodetic latitude of an authalic one, by Newton's method; the engine
  // iterates its whole vector until every step is below the tolerance, so a
  // value may take an extra step here or there (below 1e-13 rad).
  geodetic(beta) {
    const target = this.qPole * Math.sin(beta);
    let phi = beta;
    for (let i = 0; i < 30; i++) {
      const s = Math.sin(phi), den = 1 - this.e2 * s * s;
      const deriv = 2 * (1 - this.e2) * Math.cos(phi) / (den * den);
      const step = Math.abs(deriv) > 1e-12 ? (this.q(s) - target) / deriv : 0;
      phi -= step;
      if (!(Math.abs(step) >= 1e-13)) break;
    }
    return Math.min(Math.PI / 2, Math.max(-Math.PI / 2, phi));
  }

  inverse(x, y) {
    const d = this.dd, rho = Math.hypot(x / d, y * d);
    if (rho < 1e-9) return [this.lat0, this.lon0];
    const c = 2 * Math.asin(Math.min(1, Math.max(-1, rho / (2 * this.rq))));
    const sc = Math.sin(c), cc = Math.cos(c), sb0 = Math.sin(this.beta0), cb0 = Math.cos(this.beta0);
    const beta = Math.asin(Math.min(1, Math.max(-1, cc * sb0 + (d * y * sc * cb0) / rho)));
    return [this.geodetic(beta) / RAD,
            this.lon0 + Math.atan2(x * sc, d * rho * cb0 * cc - d * d * y * sb0 * sc) / RAD];
  }
}
