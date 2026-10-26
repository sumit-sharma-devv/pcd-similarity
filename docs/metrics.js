// Point cloud similarity metrics. Primary: Density-aware Chamfer Distance (Wu et al., NeurIPS 2021,
// arXiv:2111.12702):
//   DCD = 1/2 * ( mean_x [1 - exp(-alpha*||x-y||)/n_y] + mean_y [1 - exp(-alpha*||y-x||)/n_x] )
// where y is x's nearest neighbour in the other set and n_y counts how many querying points pick y.
// Mirrors pcd_similarity/metrics.py.
(function (root) {
  const DEFAULT_ALPHA = 1000, DEFAULT_TAU = 0.01, MAX_POINTS = 200000;

  // Static 3-D kd-tree over a Float64Array of xyz triples, stored as an implicit balanced tree.
  class KDTree {
    constructor(pts) {
      this.pts = pts; this.n = pts.length / 3;
      this.idx = new Uint32Array(this.n); for (let i = 0; i < this.n; i++) this.idx[i] = i;
      this.axis = new Uint8Array(this.n);
      this._build(0, this.n, 0);
    }
    _select(lo, hi, k, ax) { // quickselect on idx[lo,hi) by coordinate `ax` so idx[k] is the median
      const p = this.pts, id = this.idx;
      hi--;
      while (lo < hi) {
        const pv = p[id[(lo + hi) >> 1] * 3 + ax]; let i = lo, j = hi;
        while (i <= j) {
          while (p[id[i] * 3 + ax] < pv) i++;
          while (p[id[j] * 3 + ax] > pv) j--;
          if (i <= j) { const t = id[i]; id[i] = id[j]; id[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else return;
      }
    }
    _build(lo, hi, depth) {
      const stack = [[lo, hi]];
      while (stack.length) {
        const [l, h] = stack.pop(); if (h - l <= 0) continue;
        // split on the axis of greatest spread
        let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        for (let i = l; i < h; i++) for (let a = 0; a < 3; a++) {
          const v = this.pts[this.idx[i] * 3 + a]; if (v < mn[a]) mn[a] = v; if (v > mx[a]) mx[a] = v;
        }
        let ax = 0; if (mx[1] - mn[1] > mx[ax] - mn[ax]) ax = 1; if (mx[2] - mn[2] > mx[ax] - mn[ax]) ax = 2;
        const m = (l + h) >> 1; this._select(l, h, m, ax); this.axis[m] = ax;
        stack.push([l, m], [m + 1, h]);
      }
    }
    // nearest neighbour of (qx,qy,qz): returns the point index; squared distance in this.d2
    nearest(qx, qy, qz) {
      const p = this.pts, id = this.idx, ax = this.axis;
      let best = -1, bd = Infinity;
      const st = this._st || (this._st = new Float64Array(256)); let sp = 0;
      st[sp++] = 0; st[sp++] = this.n; st[sp++] = 0; // [lo, hi, minimum squared distance to that region]
      const q = [qx, qy, qz];
      while (sp) {
        const bound = st[--sp], h = st[--sp], l = st[--sp];
        if (h <= l || bound >= bd) continue; // far side re-checked against the current best on pop
        const m = (l + h) >> 1, pi = id[m] * 3, a = ax[m];
        const dx = p[pi] - qx, dy = p[pi + 1] - qy, dz = p[pi + 2] - qz, d = dx * dx + dy * dy + dz * dz;
        if (d < bd) { bd = d; best = id[m]; }
        const diff = q[a] - p[pi + a], d2 = diff * diff;
        // push the far side first, the near side last so it is explored first
        if (diff < 0) { st[sp++] = m + 1; st[sp++] = h; st[sp++] = d2; st[sp++] = l; st[sp++] = m; st[sp++] = bound; }
        else { st[sp++] = l; st[sp++] = m; st[sp++] = d2; st[sp++] = m + 1; st[sp++] = h; st[sp++] = bound; }
      }
      this.d2 = bd; return best;
    }
  }

  // Small deterministic PRNG so subsampling is reproducible.
  function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

  function subsample(pts, max, seed) {
    const n = pts.length / 3; if (n <= max) return { pts, sampled: false };
    const r = rng(seed), idx = new Uint32Array(n); for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = 0; i < max; i++) { const j = i + Math.floor(r() * (n - i)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    const out = new Float64Array(max * 3);
    for (let i = 0; i < max; i++) { out[i * 3] = pts[idx[i] * 3]; out[i * 3 + 1] = pts[idx[i] * 3 + 1]; out[i * 3 + 2] = pts[idx[i] * 3 + 2]; }
    return { pts: out, sampled: true };
  }

  // Centre on A and scale both by A's bounding-sphere radius (A is the reference frame).
  function normalizePair(a, b) {
    const n = a.length / 3; let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) { cx += a[i * 3]; cy += a[i * 3 + 1]; cz += a[i * 3 + 2]; }
    cx /= n; cy /= n; cz /= n;
    let r = 0;
    for (let i = 0; i < n; i++) r = Math.max(r, Math.hypot(a[i * 3] - cx, a[i * 3 + 1] - cy, a[i * 3 + 2] - cz));
    if (!(r > 0)) r = 1;
    const f = p => { const o = new Float64Array(p.length); for (let i = 0; i < p.length; i += 3) { o[i] = (p[i] - cx) / r; o[i + 1] = (p[i + 1] - cy) / r; o[i + 2] = (p[i + 2] - cz) / r; } return o; };
    return [f(a), f(b)];
  }

  function directional(src, tree, nDst) {
    const n = src.length / 3, d = new Float64Array(n), idx = new Uint32Array(n), counts = new Uint32Array(nDst);
    for (let i = 0; i < n; i++) {
      const j = tree.nearest(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
      d[i] = Math.sqrt(tree.d2); idx[i] = j; counts[j]++;
    }
    return { d, idx, counts };
  }

  function computeMetrics(a, b, opts = {}) {
    const alpha = opts.alpha ?? DEFAULT_ALPHA, tau = opts.tau ?? DEFAULT_TAU, normalize = opts.normalize ?? true;
    const sa = subsample(a, MAX_POINTS, 0), sb = subsample(b, MAX_POINTS, 0);
    a = sa.pts; b = sb.pts;
    if (normalize) [a, b] = normalizePair(a, b);
    const na = a.length / 3, nb = b.length / 3;
    const ab = directional(a, new KDTree(b), nb), ba = directional(b, new KDTree(a), na);

    let dcdA = 0, dcdB = 0, l1a = 0, l1b = 0, l2a = 0, l2b = 0, hd = 0, recall = 0, precision = 0;
    for (let i = 0; i < na; i++) {
      const d = ab.d[i]; dcdA += 1 - Math.exp(-alpha * d) / ab.counts[ab.idx[i]];
      l1a += d; l2a += d * d; if (d > hd) hd = d; if (d < tau) recall++;
    }
    for (let i = 0; i < nb; i++) {
      const d = ba.d[i]; dcdB += 1 - Math.exp(-alpha * d) / ba.counts[ba.idx[i]];
      l1b += d; l2b += d * d; if (d > hd) hd = d; if (d < tau) precision++;
    }
    const dcd = 0.5 * (dcdA / na + dcdB / nb);
    recall /= na; precision /= nb;
    return {
      similarity: 1 - dcd, dcd,
      chamfer_l1: 0.5 * (l1a / na + l1b / nb), chamfer_l2: 0.5 * (l2a / na + l2b / nb),
      hausdorff: hd, fscore: precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall),
      points_a: na, points_b: nb, alpha, tau, normalized: normalize, subsampled: sa.sampled || sb.sampled,
      _a: a, _b: b, // normalised clouds, reused for the 3-D preview
    };
  }

  // Up to `max` points, as a Float32Array, in the A-normalised frame.
  function preview(a, b, max = 20000) {
    const [na, nb] = normalizePair(a, b);
    const f = p => { const s = subsample(p, max, 0).pts; return Float32Array.from(s); };
    return { a: f(na), b: f(nb) };
  }

  const api = { KDTree, computeMetrics, normalizePair, preview, DEFAULT_ALPHA, DEFAULT_TAU };
  if (typeof module !== 'undefined') module.exports = api; else root.Metrics = api;
})(typeof self !== 'undefined' ? self : this);
