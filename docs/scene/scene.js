// Label-free scene analysis for road/driving point clouds, and a semantic scene-similarity score.
//
// No machine-learning model is used. Each cloud is segmented with geometric rules:
//   1. ground  : per-cell lower height envelope with a slope limit -> points near it are ground
//   2. objects : the remaining points are voxelised and grouped into connected clusters
//   3. classes : clusters are labelled building / vegetation / vehicle / pedestrian / pole-like by size and shape
//   4. junctions: road intersections are found by sampling a ring of free ground space around candidate cells
// The two scenes are then compared class by class (see compareScenes). Heuristics are approximate.
(function (root) {
  const NAMES = ['unlabeled', 'ground', 'building', 'vegetation', 'vehicle', 'pedestrian', 'pole', 'other'];
  const GROUND = 1, BUILDING = 2, VEGETATION = 3, VEHICLE = 4, PEDESTRIAN = 5, POLE = 6, OTHER = 7;
  const DEFAULTS = {
    up: 'z',            // which input axis points up
    groundTol: 0.25,    // m above the ground surface still counted as ground
    groundCell: 1.0,    // m, cell size of the ground-height grid
    slope: 0.2,         // max ground slope (rise/run) of the height envelope
    voxel: 0.4,         // m, voxel size used to cluster non-ground points
    ringR: 12,          // m, radius of the ring sampled for junction detection
    bev: 2.0,           // m, cell size of the bird's-eye semantic grid used for comparison
    matchDist: 2.0,     // m, max centre distance to match vehicles / pedestrians / poles
    junctionDist: 10.0, // m, max centre distance to match intersections
    maxCells: 2.5e6,    // cap on any grid, to bound memory
  };
  // How much each class contributes to the overall score (only classes present in A count).
  const WEIGHTS = { ground: 2, building: 1, vegetation: 1, vehicle: 1, pedestrian: 1, pole: 0.5, junction: 2 };

  // Deterministic random subsample to at most `max` points.
  function subsample(pts, max) {
    const n = pts.length / 3; if (n <= max) return pts;
    let seed = 0; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    const idx = new Uint32Array(n); for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = 0; i < max; i++) { const j = i + Math.floor(rnd() * (n - i)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    const out = new Float64Array(max * 3);
    for (let i = 0; i < max; i++) { out[i * 3] = pts[idx[i] * 3]; out[i * 3 + 1] = pts[idx[i] * 3 + 1]; out[i * 3 + 2] = pts[idx[i] * 3 + 2]; }
    return out;
  }

  // Reorder axes so that z is "up".
  function toZUp(p, up) {
    if (up === 'z') return p;
    const o = new Float64Array(p.length);
    for (let i = 0; i < p.length; i += 3) {
      if (up === 'y') { o[i] = p[i]; o[i + 1] = -p[i + 2]; o[i + 2] = p[i + 1]; }
      else { o[i] = p[i + 1]; o[i + 1] = p[i + 2]; o[i + 2] = p[i]; } // x up
    }
    return o;
  }

  // Eigenvalues (descending) of the symmetric matrix [[a,b,c],[b,d,e],[c,e,f]].
  function eig3(a, b, c, d, e, f) {
    const p1 = b * b + c * c + e * e;
    if (p1 < 1e-18) return [a, d, f].sort((x, y) => y - x);
    const q = (a + d + f) / 3, A = a - q, D = d - q, F = f - q;
    const p = Math.sqrt((A * A + D * D + F * F + 2 * p1) / 6);
    const det = (A * (D * F - e * e) - b * (b * F - e * c) + c * (b * e - D * c)) / (p * p * p);
    const phi = Math.acos(Math.max(-1, Math.min(1, det / 2))) / 3;
    const l1 = q + 2 * p * Math.cos(phi), l3 = q + 2 * p * Math.cos(phi + 2 * Math.PI / 3);
    return [l1, 3 * q - l1 - l3, l3];
  }

  // Ground height per cell: low percentile of z, then a slope-limited lower envelope so that cells covered
  // by a roof or an object cannot sit higher than the surrounding ground allows.
  function estimateGround(p, n, gc, slope, maxCells) {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = p[i * 3], y = p[i * 3 + 1];
      if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    let gx = Math.floor((maxx - minx) / gc) + 1, gy = Math.floor((maxy - miny) / gc) + 1;
    if (gx * gy > maxCells) { gc *= Math.sqrt(gx * gy / maxCells); gx = Math.floor((maxx - minx) / gc) + 1; gy = Math.floor((maxy - miny) / gc) + 1; }
    const cell = new Int32Array(n), counts = new Uint32Array(gx * gy + 1);
    for (let i = 0; i < n; i++) {
      const c = Math.floor((p[i * 3 + 1] - miny) / gc) * gx + Math.floor((p[i * 3] - minx) / gc);
      cell[i] = c; counts[c + 1]++;
    }
    for (let k = 0; k < gx * gy; k++) counts[k + 1] += counts[k];
    const fill = counts.slice(0, gx * gy), zs = new Float64Array(n);
    for (let i = 0; i < n; i++) zs[fill[cell[i]]++] = p[i * 3 + 2];
    const z0 = new Float64Array(gx * gy).fill(Infinity);
    for (let k = 0; k < gx * gy; k++) {
      const cnt = counts[k + 1] - counts[k]; if (!cnt) continue;
      const sub = zs.subarray(counts[k], counts[k + 1]).sort();
      z0[k] = sub[Math.floor(0.05 * (cnt - 1))];
    }
    const d1 = slope * gc, d2 = d1 * Math.SQRT2;
    for (let it = 0; it < 2; it++) {
      for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
        const k = j * gx + i; let v = z0[k];
        if (i > 0) v = Math.min(v, z0[k - 1] + d1);
        if (j > 0) { v = Math.min(v, z0[k - gx] + d1); if (i > 0) v = Math.min(v, z0[k - gx - 1] + d2); if (i < gx - 1) v = Math.min(v, z0[k - gx + 1] + d2); }
        z0[k] = v;
      }
      for (let j = gy - 1; j >= 0; j--) for (let i = gx - 1; i >= 0; i--) {
        const k = j * gx + i; let v = z0[k];
        if (i < gx - 1) v = Math.min(v, z0[k + 1] + d1);
        if (j < gy - 1) { v = Math.min(v, z0[k + gx] + d1); if (i < gx - 1) v = Math.min(v, z0[k + gx + 1] + d2); if (i > 0) v = Math.min(v, z0[k + gx - 1] + d2); }
        z0[k] = v;
      }
    }
    return { cell, z0, minx, miny, maxx, maxy };
  }

  function classify(c) {
    const { n, H, minH, L, W, pf, pfs } = c;
    if (n < 8 || H < 0.3) return OTHER;
    if (H >= 2.5 && L <= 1.0) return POLE;                                   // signal / sign / lamp posts
    if (H >= 1.0 && H <= 2.3 && L <= 1.3 && W <= 1.3 && H / L >= 1.2) return PEDESTRIAN;
    if (H >= 0.9 && H <= 4.2 && L >= 2.0 && L <= 13 && W >= 1.0 && W <= 3.3 && minH < 1.0 && pfs >= 0.2) return VEHICLE;
    if (H >= 3.0 && (L >= 6 || L * W >= 25) && pf >= 0.35) return BUILDING;
    if (H >= 0.4 && pf < 0.35) return VEGETATION;
    if (H >= 3.0) return BUILDING;
    return OTHER;
  }

  // Free road space -> intersections, by counting the separate free "arms" crossing a ring around a cell.
  function detectJunctions(p, n, h, labels, o) {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = p[i * 3], y = p[i * 3 + 1];
      if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    let fs = 1.0, gx = Math.floor((maxx - minx) / fs) + 1, gy = Math.floor((maxy - miny) / fs) + 1;
    if (gx * gy > o.maxCells) { fs *= Math.sqrt(gx * gy / o.maxCells); gx = Math.floor((maxx - minx) / fs) + 1; gy = Math.floor((maxy - miny) / fs) + 1; }
    const road = new Uint8Array(gx * gy), blocked = new Uint8Array(gx * gy), veh = new Uint8Array(gx * gy);
    for (let i = 0; i < n; i++) {
      const k = Math.floor((p[i * 3 + 1] - miny) / fs) * gx + Math.floor((p[i * 3] - minx) / fs), l = labels[i];
      if (l === GROUND) road[k] = 1;
      else if ((l === BUILDING || l === VEGETATION) && h[i] >= 0.3 && h[i] <= 2.5) blocked[k] = 1;
      else if (l === VEHICLE || l === PEDESTRIAN) veh[k] = 1;
    }
    const free = new Uint8Array(gx * gy);
    for (let k = 0; k < gx * gy; k++) free[k] = (road[k] || veh[k]) && !blocked[k] ? 1 : 0;
    // distance (in cells) to the nearest non-free cell, 3-4 chamfer
    const dt = new Float32Array(gx * gy);
    for (let k = 0; k < gx * gy; k++) dt[k] = free[k] ? 1e9 : 0;
    for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
      const k = j * gx + i; if (!free[k]) continue; let v = dt[k];
      if (i > 0) v = Math.min(v, dt[k - 1] + 1);
      if (j > 0) { v = Math.min(v, dt[k - gx] + 1); if (i > 0) v = Math.min(v, dt[k - gx - 1] + 1.414); if (i < gx - 1) v = Math.min(v, dt[k - gx + 1] + 1.414); }
      dt[k] = v;
    }
    for (let j = gy - 1; j >= 0; j--) for (let i = gx - 1; i >= 0; i--) {
      const k = j * gx + i; if (!free[k]) continue; let v = dt[k];
      if (i < gx - 1) v = Math.min(v, dt[k + 1] + 1);
      if (j < gy - 1) { v = Math.min(v, dt[k + gx] + 1); if (i < gx - 1) v = Math.min(v, dt[k + gx + 1] + 1.414); if (i > 0) v = Math.min(v, dt[k + gx - 1] + 1.414); }
      dt[k] = v;
    }
    const R = o.ringR / fs, S = 72, dx = new Int32Array(S), dy = new Int32Array(S);
    for (let s = 0; s < S; s++) { dx[s] = Math.round(R * Math.cos(2 * Math.PI * s / S)); dy[s] = Math.round(R * Math.sin(2 * Math.PI * s / S)); }
    const cand = new Uint8Array(gx * gy), arms = new Uint8Array(gx * gy), ring = new Uint8Array(S), dil = new Uint8Array(S);
    const minClear = 2, minRun = 4, maxRun = 30;
    for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++) {
      const k = j * gx + i; if (!free[k] || dt[k] < minClear) continue;
      let anyFree = 0;
      for (let s = 0; s < S; s++) {
        const x = i + dx[s], y = j + dy[s];
        ring[s] = x >= 0 && y >= 0 && x < gx && y < gy ? free[y * gx + x] : 0; anyFree += ring[s];
      }
      if (anyFree === 0 || anyFree === S) continue;
      // morphological closing: a thin obstacle (tree trunk, post) must not split one road arm into two
      for (let s = 0; s < S; s++) dil[s] = ring[(s + S - 1) % S] | ring[s] | ring[(s + 1) % S];
      for (let s = 0; s < S; s++) ring[s] = dil[(s + S - 1) % S] & dil[s] & dil[(s + 1) % S];
      anyFree = 0; for (let s = 0; s < S; s++) anyFree += ring[s];
      if (anyFree === S) continue;
      let start = 0; while (ring[start]) start++;           // begin at a blocked sample so runs don't wrap
      let runs = 0, len = 0, ok = true;
      for (let t = 1; t <= S; t++) {
        if (ring[(start + t) % S]) len++;
        else { if (len) { if (len > maxRun) ok = false; if (len >= minRun) runs++; } len = 0; }
      }
      if (ok && runs >= 3) { cand[k] = 1; arms[k] = Math.min(runs, 6); }
    }
    const seen = new Uint8Array(gx * gy), out = [];
    for (let k0 = 0; k0 < gx * gy; k0++) {
      if (!cand[k0] || seen[k0]) continue;
      const stack = [k0], cells = []; seen[k0] = 1;
      while (stack.length) {
        const k = stack.pop(); cells.push(k);
        const i = k % gx, j = (k - i) / gx;
        for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
          const x = i + a, y = j + b; if (x < 0 || y < 0 || x >= gx || y >= gy) continue;
          const m = y * gx + x; if (cand[m] && !seen[m]) { seen[m] = 1; stack.push(m); }
        }
      }
      if (cells.length < 3) continue;
      let sx = 0, sy = 0; const hist = {};
      for (const k of cells) { const i = k % gx; sx += i; sy += (k - i) / gx; hist[arms[k]] = (hist[arms[k]] || 0) + 1; }
      const arm = +Object.keys(hist).sort((a, b) => hist[b] - hist[a])[0];
      out.push({ x: minx + (sx / cells.length + 0.5) * fs, y: miny + (sy / cells.length + 0.5) * fs, arms: arm });
    }
    return out;
  }

  function analyze(ptsIn, opts = {}) {
    const o = { ...DEFAULTS, ...opts };
    const p = toZUp(ptsIn, o.up), n = p.length / 3;
    const g = estimateGround(p, n, o.groundCell, o.slope, o.maxCells);
    const h = new Float32Array(n), labels = new Uint8Array(n), ng = [];
    for (let i = 0; i < n; i++) {
      h[i] = p[i * 3 + 2] - g.z0[g.cell[i]];
      if (h[i] < o.groundTol) labels[i] = GROUND; else ng.push(i);
    }
    const objects = [];
    if (ng.length) {
      // voxelise non-ground points in (x, y, height-above-ground)
      const vs = o.voxel, m = ng.length, vi = new Int32Array(m);
      let maxIz = 0;
      for (const i of ng) maxIz = Math.max(maxIz, Math.floor(h[i] / vs));
      const nx = Math.floor((g.maxx - g.minx) / vs) + 1, ny = Math.floor((g.maxy - g.miny) / vs) + 1, nz = maxIz + 1;
      if (nx * ny * nz > 2 ** 52) throw new Error('Scene is too large for the chosen voxel size.');
      const vmap = new Map(), vx = [], vy = [], vz = [];
      for (let t = 0; t < m; t++) {
        const i = ng[t], ix = Math.floor((p[i * 3] - g.minx) / vs), iy = Math.floor((p[i * 3 + 1] - g.miny) / vs), iz = Math.floor(h[i] / vs);
        const key = ix + nx * (iy + ny * iz); let v = vmap.get(key);
        if (v === undefined) { v = vx.length; vmap.set(key, v); vx.push(ix); vy.push(iy); vz.push(iz); }
        vi[t] = v;
      }
      const V = vx.length, mom = new Float64Array(V * 10);
      for (let t = 0; t < m; t++) {
        const i = ng[t], x = p[i * 3], y = p[i * 3 + 1], z = h[i], q = vi[t] * 10;
        mom[q] += 1; mom[q + 1] += x; mom[q + 2] += y; mom[q + 3] += z;
        mom[q + 4] += x * x; mom[q + 5] += x * y; mom[q + 6] += x * z; mom[q + 7] += y * y; mom[q + 8] += y * z; mom[q + 9] += z * z;
      }
      // connected components over the 26-neighbourhood (union-find)
      const par = new Int32Array(V); for (let v = 0; v < V; v++) par[v] = v;
      const find = a => { while (par[a] !== a) { par[a] = par[par[a]]; a = par[a]; } return a; };
      for (let v = 0; v < V; v++) {
        for (let c = -1; c <= 1; c++) for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
          if (!a && !b && !c) continue;
          const jx = vx[v] + a, jy = vy[v] + b, jz = vz[v] + c;
          if (jx < 0 || jy < 0 || jz < 0) continue;
          const w = vmap.get(jx + nx * (jy + ny * jz));
          if (w !== undefined && w > v) { const ra = find(v), rb = find(w); if (ra !== rb) par[ra] = rb; }
        }
      }
      const cid = new Int32Array(V).fill(-1); let C = 0;
      for (let v = 0; v < V; v++) { const r = find(v); if (cid[r] < 0) cid[r] = C++; cid[v] = cid[r]; }
      const cm = new Float64Array(C * 10), planar = new Uint32Array(C), withPts = new Uint32Array(C), vplan = new Int8Array(V).fill(-1);
      const planarS = new Uint32Array(C), withPtsS = new Uint32Array(C);
      const nb = new Float64Array(10), nbS = new Float64Array(10), PR = 2;
      // true when the points summarised by moments `m` lie on a plane (not a line, not a blob)
      const isPlanar = (m, minCount) => {
        const cnt = m[0]; if (cnt < minCount) return -1;
        const mx = m[1] / cnt, my = m[2] / cnt, mz = m[3] / cnt;
        const e = eig3(m[4] / cnt - mx * mx, m[5] / cnt - mx * my, m[6] / cnt - mx * mz, m[7] / cnt - my * my, m[8] / cnt - my * mz, m[9] / cnt - mz * mz);
        return e[1] >= 0.1 * e[0] && e[2] <= 0.12 * e[1] ? 1 : 0;
      };
      for (let v = 0; v < V; v++) {
        const c = cid[v], q = v * 10;
        for (let k = 0; k < 10; k++) cm[c * 10 + k] += mom[q + k];
        // Local planarity at two scales: the surrounding 5^3 voxels (~2 m; facades, roads) and 3^3 voxels
        // (~1.2 m; the surfaces of a vehicle), pooled so that sparse regions still count.
        nb.fill(0); nbS.fill(0);
        for (let dz = -PR; dz <= PR; dz++) for (let dy = -PR; dy <= PR; dy++) for (let dx = -PR; dx <= PR; dx++) {
          const jx = vx[v] + dx, jy = vy[v] + dy, jz = vz[v] + dz; if (jx < 0 || jy < 0 || jz < 0) continue;
          const w = vmap.get(jx + nx * (jy + ny * jz)); if (w === undefined) continue;
          const small = Math.abs(dx) <= 1 && Math.abs(dy) <= 1 && Math.abs(dz) <= 1;
          for (let k = 0; k < 10; k++) { nb[k] += mom[w * 10 + k]; if (small) nbS[k] += mom[w * 10 + k]; }
        }
        const pl = isPlanar(nb, 15), ps = isPlanar(nbS, 8);
        if (pl >= 0) { withPts[c]++; planar[c] += pl; vplan[v] = pl; }
        if (ps >= 0) { withPtsS[c]++; planarS[c] += ps; }
      }
      // principal axis of each cluster's footprint, then extents along it
      const cx = new Float64Array(C), cy = new Float64Array(C), th = new Float64Array(C);
      for (let c = 0; c < C; c++) {
        const cnt = cm[c * 10]; cx[c] = cm[c * 10 + 1] / cnt; cy[c] = cm[c * 10 + 2] / cnt;
        const sxx = cm[c * 10 + 4] / cnt - cx[c] ** 2, sxy = cm[c * 10 + 5] / cnt - cx[c] * cy[c], syy = cm[c * 10 + 7] / cnt - cy[c] ** 2;
        th[c] = 0.5 * Math.atan2(2 * sxy, sxx - syy);
      }
      const umin = new Float64Array(C).fill(Infinity), umax = new Float64Array(C).fill(-Infinity);
      const wmin = new Float64Array(C).fill(Infinity), wmax = new Float64Array(C).fill(-Infinity);
      const hmin = new Float64Array(C).fill(Infinity), hmax = new Float64Array(C).fill(-Infinity), pc = new Uint32Array(C);
      for (let t = 0; t < m; t++) {
        const i = ng[t], c = cid[vi[t]], dx = p[i * 3] - cx[c], dy = p[i * 3 + 1] - cy[c];
        const u = dx * Math.cos(th[c]) + dy * Math.sin(th[c]), w = -dx * Math.sin(th[c]) + dy * Math.cos(th[c]);
        if (u < umin[c]) umin[c] = u; if (u > umax[c]) umax[c] = u; if (w < wmin[c]) wmin[c] = w; if (w > wmax[c]) wmax[c] = w;
        if (h[i] < hmin[c]) hmin[c] = h[i]; if (h[i] > hmax[c]) hmax[c] = h[i]; pc[c]++;
      }
      const clsOf = new Uint8Array(C);
      for (let c = 0; c < C; c++) {
        const ru = umax[c] - umin[c], rw = wmax[c] - wmin[c];
        const info = { n: pc[c], H: hmax[c], minH: hmin[c], L: Math.max(ru, rw), W: Math.min(ru, rw), pf: withPts[c] ? planar[c] / withPts[c] : 0, pfs: withPtsS[c] ? planarS[c] / withPtsS[c] : 0 };
        clsOf[c] = classify(info);
        if (clsOf[c] === VEHICLE || clsOf[c] === PEDESTRIAN || clsOf[c] === POLE)
          objects.push({ cls: clsOf[c], x: cx[c], y: cy[c], height: info.H, length: info.L, width: info.W, points: info.n });
      }
      // A big building/vegetation cluster may be a building fused with a tree: refine it voxel by voxel,
      // labelling a voxel by how planar its neighbourhood is (facades are planar, foliage is not).
      const vcls = new Uint8Array(V), clusterTop = hmax, clusterLen = c => Math.max(umax[c] - umin[c], wmax[c] - wmin[c]);
      for (let v = 0; v < V; v++) {
        const c = cid[v]; vcls[v] = clsOf[c];
        if ((clsOf[c] !== BUILDING && clsOf[c] !== VEGETATION) || clusterTop[c] < 3 || clusterLen(c) < 5) continue;
        let valid = 0, pl = 0;
        for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const jx = vx[v] + dx, jy = vy[v] + dy, jz = vz[v] + dz; if (jx < 0 || jy < 0 || jz < 0) continue;
          const w = vmap.get(jx + nx * (jy + ny * jz)); if (w === undefined || vplan[w] < 0) continue;
          valid++; pl += vplan[w];
        }
        if (valid >= 3) vcls[v] = pl / valid >= 0.45 ? BUILDING : VEGETATION;
      }
      for (let t = 0; t < m; t++) labels[ng[t]] = vcls[vi[t]];
    }
    const junctions = detectJunctions(p, n, h, labels, o);
    return { p, n, labels, h, objects, junctions, opts: o };
  }

  // Bird's-eye grid of per-class point counts over a bounding box shared by both scenes.
  function classGrid(s, ox, oy, minx, miny, cs, gx, gy) {
    const cnt = new Uint16Array(gx * gy * 8);
    for (let i = 0; i < s.n; i++) {
      const k = Math.floor((s.p[i * 3 + 1] + oy - miny) / cs) * gx + Math.floor((s.p[i * 3] + ox - minx) / cs);
      const q = k * 8 + s.labels[i]; if (cnt[q] < 65535) cnt[q]++;
    }
    return cnt;
  }

  // Greedy nearest-first one-to-one matching of points within maxD. Returns the matched index pairs.
  function match(A, B, maxD) {
    const pairs = [];
    for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) {
      const d = Math.hypot(A[i].x - B[j].x, A[i].y - B[j].y); if (d <= maxD) pairs.push([d, i, j]);
    }
    pairs.sort((a, b) => a[0] - b[0]);
    const ua = new Uint8Array(A.length), ub = new Uint8Array(B.length), out = [];
    for (const [, i, j] of pairs) if (!ua[i] && !ub[j]) { ua[i] = ub[j] = 1; out.push([i, j]); }
    return out;
  }

  // Compare two analysed scenes. The score is directional: how much of what is in A is also in B.
  // Anything extra in B (more buildings, trees, cars, ...) is not penalised.
  function compareScenes(sa, sb, opts = {}) {
    const o = { ...DEFAULTS, ...opts };
    let ox = 0, oy = 0;
    if (opts.alignCentroids) {
      const cen = s => { let x = 0, y = 0; for (let i = 0; i < s.n; i++) { x += s.p[i * 3]; y += s.p[i * 3 + 1]; } return [x / s.n, y / s.n]; };
      const [ax, ay] = cen(sa), [bx, by] = cen(sb); ox = ax - bx; oy = ay - by;
    }
    const sB = (b) => ({ ...b, x: b.x + ox, y: b.y + oy });
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (const [s, dx, dy] of [[sa, 0, 0], [sb, ox, oy]]) for (let i = 0; i < s.n; i++) {
      const x = s.p[i * 3] + dx, y = s.p[i * 3 + 1] + dy;
      if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    let cs = o.bev, gx = Math.floor((maxx - minx) / cs) + 1, gy = Math.floor((maxy - miny) / cs) + 1;
    if (gx * gy > 4e6) { cs *= Math.sqrt(gx * gy / 4e6); gx = Math.floor((maxx - minx) / cs) + 1; gy = Math.floor((maxy - miny) / cs) + 1; }
    const ga = classGrid(sa, 0, 0, minx, miny, cs, gx, gy), gb = classGrid(sb, ox, oy, minx, miny, cs, gx, gy);
    const MIN_PTS = 3, rows = [];
    for (const [cls, name] of [[GROUND, 'ground'], [BUILDING, 'building'], [VEGETATION, 'vegetation']]) {
      let na = 0, nb = 0, both = 0;
      for (let k = 0; k < gx * gy; k++) {
        const a = ga[k * 8 + cls] >= MIN_PTS, b = gb[k * 8 + cls] >= MIN_PTS;
        if (a) na++; if (b) nb++; if (a && b) both++;
      }
      rows.push({ key: name, name, kind: 'area', a: na, b: nb, matched: both, cov_ab: na ? both / na : null, cov_ba: nb ? both / nb : null, weight: WEIGHTS[name] });
    }
    for (const [cls, name] of [[VEHICLE, 'vehicle'], [PEDESTRIAN, 'pedestrian'], [POLE, 'pole']]) {
      const A = sa.objects.filter(t => t.cls === cls), B = sb.objects.filter(t => t.cls === cls).map(sB);
      const m = match(A, B, o.matchDist);
      rows.push({ key: name, name, kind: 'object', a: A.length, b: B.length, matched: m.length, cov_ab: A.length ? m.length / A.length : null, cov_ba: B.length ? m.length / B.length : null, weight: WEIGHTS[name] });
    }
    const JA = sa.junctions, JB = sb.junctions.map(sB), jm = match(JA, JB, o.junctionDist);
    const sameArms = jm.filter(([i, j]) => JA[i].arms === JB[j].arms).length;
    rows.push({ key: 'junction', name: 'intersection', kind: 'object', a: JA.length, b: JB.length, matched: jm.length, cov_ab: JA.length ? jm.length / JA.length : null, cov_ba: JB.length ? jm.length / JB.length : null, weight: WEIGHTS.junction, sameArms });

    const wavg = (key) => {
      let s = 0, w = 0; for (const r of rows) if (r[key] !== null) { s += r.weight * r[key]; w += r.weight; }
      return w ? s / w : 0;
    };
    const score = wavg('cov_ab'), reverse = wavg('cov_ba');
    return { score, reverse, symmetric: 0.5 * (score + reverse), rows, offset: [ox, oy], cell: cs,
             points: { a: countLabels(sa), b: countLabels(sb) } };
  }

  function countLabels(s) {
    const c = {}; for (const nme of NAMES.slice(1)) c[nme] = 0;
    for (let i = 0; i < s.n; i++) c[NAMES[s.labels[i]]]++;
    return c;
  }

  // Up to `max` points per scene, with class labels, in the A-normalised frame and y-up for the viewer.
  function previewLabeled(sa, sb, offset = [0, 0], max = 20000) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < sa.n; i++) { cx += sa.p[i * 3]; cy += sa.p[i * 3 + 1]; cz += sa.p[i * 3 + 2]; }
    cx /= sa.n; cy /= sa.n; cz /= sa.n;
    let r = 0; for (let i = 0; i < sa.n; i++) r = Math.max(r, Math.hypot(sa.p[i * 3] - cx, sa.p[i * 3 + 1] - cy, sa.p[i * 3 + 2] - cz));
    if (!(r > 0)) r = 1;
    const one = (s, ox, oy) => {
      const step = Math.max(1, s.n / max), m = Math.min(s.n, max), pts = new Float32Array(m * 3), lab = new Uint8Array(m);
      for (let t = 0; t < m; t++) {
        const i = Math.min(s.n - 1, Math.floor(t * step));
        pts[t * 3] = (s.p[i * 3] + ox - cx) / r; pts[t * 3 + 1] = (s.p[i * 3 + 2] - cz) / r; pts[t * 3 + 2] = -(s.p[i * 3 + 1] + oy - cy) / r;
        lab[t] = s.labels[i];
      }
      return [pts, lab];
    };
    const [a, la] = one(sa, 0, 0), [b, lb] = one(sb, offset[0], offset[1]);
    return { a, b, la, lb };
  }

  const api = { analyze, compareScenes, previewLabeled, subsample, toZUp, eig3, NAMES, DEFAULTS, WEIGHTS };
  if (typeof module !== 'undefined') module.exports = api; else root.Scene = api;
})(typeof self !== 'undefined' ? self : this);
