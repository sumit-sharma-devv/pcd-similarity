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

  const DEFAULT_SCALE = 0.05, DEFAULT_VOXEL = 0.05, DEFAULT_EMD_POINTS = 2048;

  // Exact optimal assignment (Hungarian / shortest augmenting path, O(n^3)) on an n x n cost matrix.
  // Returns the total cost of the optimal one-to-one matching.
  function assignmentCost(cost, n) {
    const u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
    const minv = new Float64Array(n + 1), used = new Uint8Array(n + 1);
    for (let i = 1; i <= n; i++) {
      p[0] = i; let j0 = 0; minv.fill(Infinity); used.fill(0);
      do {
        used[j0] = 1; const i0 = p[j0]; let delta = Infinity, j1 = 0;
        const row = (i0 - 1) * n - 1;
        for (let j = 1; j <= n; j++) if (!used[j]) {
          const cur = cost[row + j] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= n; j++) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    let total = 0;
    for (let j = 1; j <= n; j++) { const i = p[j] - 1, k = j - 1; total += cost[i * n + k]; }
    return total;
  }

  // Earth Mover's Distance between two equal-size clouds: mean Euclidean length of the optimal matching.
  function emd(a, b) {
    const n = a.length / 3, cost = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
      cost[i * n + j] = Math.hypot(a[i * 3] - b[j * 3], a[i * 3 + 1] - b[j * 3 + 1], a[i * 3 + 2] - b[j * 3 + 2]);
    return assignmentCost(cost, n) / n;
  }

  // Voxel IoU: occupied cells of a shared grid, |A ∩ B| / |A ∪ B|.
  function voxelIoU(a, b, size) {
    const lim = 1 << 30, cells = p => {
      const o = new Int32Array(p.length);
      for (let i = 0; i < p.length; i++) {
        const c = Math.floor(p[i] / size);
        if (!(Math.abs(c) < lim)) throw new Error('Voxel size is too small for these coordinates.');
        o[i] = c;
      }
      return o;
    };
    const ca = cells(a), cb = cells(b), mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const c of [ca, cb]) for (let i = 0; i < c.length; i++) { const k = i % 3; if (c[i] < mn[k]) mn[k] = c[i]; if (c[i] > mx[k]) mx[k] = c[i]; }
    const dx = mx[0] - mn[0] + 1, dy = mx[1] - mn[1] + 1, dz = mx[2] - mn[2] + 1;
    if (dx * dy * dz > 2 ** 52) throw new Error('Voxel size is too small for these coordinates.');
    const keys = c => { const s = new Set(); for (let i = 0; i < c.length; i += 3) s.add((c[i] - mn[0]) + dx * ((c[i + 1] - mn[1]) + dy * (c[i + 2] - mn[2]))); return s; };
    const sa = keys(ca), sb = keys(cb); let inter = 0;
    const [small, big] = sa.size <= sb.size ? [sa, sb] : [sb, sa];
    for (const k of small) if (big.has(k)) inter++;
    const union = sa.size + sb.size - inter;
    return { iou: inter / union, voxels_a: sa.size, voxels_b: sb.size, intersection: inter, union };
  }

  const METHODS = ['dcd', 'cd', 'emd', 'voxel'];

  function computeMetrics(a, b, opts = {}) {
    const alpha = opts.alpha ?? DEFAULT_ALPHA, tau = opts.tau ?? DEFAULT_TAU, normalize = opts.normalize ?? true;
    const method = opts.method ?? 'dcd', scale = opts.scale ?? DEFAULT_SCALE, voxel = opts.voxel ?? DEFAULT_VOXEL;
    if (!METHODS.includes(method)) throw new Error(`Unknown method '${method}'.`);
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
    const dcd = 0.5 * (dcdA / na + dcdB / nb), cdL1 = 0.5 * (l1a / na + l1b / nb);
    recall /= na; precision /= nb;

    const vox = voxelIoU(a, b, voxel);
    const scores = {
      dcd: { value: dcd, similarity: 1 - dcd },
      cd: { value: cdL1, similarity: Math.exp(-cdL1 / scale) },
      voxel: { value: vox.iou, similarity: vox.iou },
    };
    let emdPoints = 0;
    if (method === 'emd') { // the only expensive method, so computed on request, on an equal-size subsample
      emdPoints = Math.min(na, nb, opts.emdPoints ?? DEFAULT_EMD_POINTS);
      const e = emd(subsample(a, emdPoints, 1).pts, subsample(b, emdPoints, 2).pts);
      scores.emd = { value: e, similarity: Math.exp(-e / scale) };
    }
    return {
      method, similarity: scores[method].similarity, scores,
      dcd, chamfer_l1: cdL1, chamfer_l2: 0.5 * (l2a / na + l2b / nb),
      hausdorff: hd, fscore: precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall),
      voxel: vox, emd_points: emdPoints,
      points_a: na, points_b: nb, alpha, tau, scale, voxel_size: voxel, normalized: normalize, subsampled: sa.sampled || sb.sampled,
      _a: a, _b: b, // normalised clouds, reused for the 3-D preview
    };
  }

  // Up to `max` points, as a Float32Array, in the A-normalised frame.
  function preview(a, b, max = 20000) {
    const [na, nb] = normalizePair(a, b);
    const f = p => { const s = subsample(p, max, 0).pts; return Float32Array.from(s); };
    return { a: f(na), b: f(nb) };
  }

  const api = { KDTree, computeMetrics, normalizePair, preview, emd, voxelIoU, assignmentCost, METHODS,
    DEFAULT_ALPHA, DEFAULT_TAU, DEFAULT_SCALE, DEFAULT_VOXEL, DEFAULT_EMD_POINTS };
  if (typeof module !== 'undefined') module.exports = api; else root.Metrics = api;
})(typeof self !== 'undefined' ? self : this);
