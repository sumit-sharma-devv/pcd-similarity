// Run with: node tests/scene.test.js
// Builds synthetic street scenes (cross intersection, buildings, cars, pedestrians, trees, a pole) and
// checks that scene.js labels them and that the semantic score behaves as described.
const assert = require('assert');
const S = require('../docs/scene/scene.js');

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

function makeScene({ seed = 1, cars = [[2, -20], [-2, 25], [20, 2]], peds = [[-3, 12], [3, -9]], trees = [[0.5, 20], [-0.5, -30]],
                     extraBuildings = [], extraTrees = [], extraCars = [] } = {}) {
  const r = rng(seed), P = [];
  const jit = s => (r() - 0.5) * s;
    const blocks = [[5, 5, 45, 45], [-45, 5, -5, 45], [5, -45, 45, -5], [-45, -45, -5, -5], ...extraBuildings]; // 10 m wide cross streets
  for (let x = -60; x <= 60; x += 0.4) for (let y = -60; y <= 60; y += 0.4) {
    const inside = blocks.some(([x0, y0, x1, y1]) => x > x0 && x < x1 && y > y0 && y < y1);
    if (!inside) P.push(x + jit(0.1), y + jit(0.1), jit(0.04));
  }
  for (const [x0, y0, x1, y1] of blocks) {            // vertical faces, 12 m high
    for (let z = 0; z <= 12; z += 0.4) {
      for (let t = x0; t <= x1; t += 0.4) { P.push(t, y0, z + jit(0.04)); P.push(t, y1, z + jit(0.04)); }
      for (let t = y0; t <= y1; t += 0.4) { P.push(x0, t, z + jit(0.04)); P.push(x1, t, z + jit(0.04)); }
    }
  }
  const box = (cx, cy, L, W, H) => {                  // vehicle-like box, surface samples
    for (let a = 0; a <= L; a += 0.12) for (let b = 0; b <= W; b += 0.12) { P.push(cx - L / 2 + a, cy - W / 2 + b, H + jit(0.02)); }
    for (let a = 0; a <= L; a += 0.12) for (let z = 0.25; z <= H; z += 0.12) { P.push(cx - L / 2 + a, cy - W / 2, z); P.push(cx - L / 2 + a, cy + W / 2, z); }
    for (let b = 0; b <= W; b += 0.12) for (let z = 0.25; z <= H; z += 0.12) { P.push(cx - L / 2, cy - W / 2 + b, z); P.push(cx + L / 2, cy - W / 2 + b, z); }
  };
  for (const [x, y] of [...cars, ...extraCars]) box(x, y, 4.5, 1.9, 1.5);
  for (const [x, y] of peds) for (let i = 0; i < 400; i++) { const a = r() * 6.283; P.push(x + 0.25 * Math.cos(a), y + 0.25 * Math.sin(a), r() * 1.7); }
  for (const [x, y] of [...trees, ...extraTrees]) {
    for (let i = 0; i < 150; i++) { const a = r() * 6.283; P.push(x + 0.15 * Math.cos(a), y + 0.15 * Math.sin(a), r() * 3); }
    for (let i = 0; i < 1500; i++) {
      let u, v, w; do { u = r() * 2 - 1; v = r() * 2 - 1; w = r() * 2 - 1; } while (u * u + v * v + w * w > 1);
      P.push(x + 2.5 * u, y + 2.5 * v, 5.5 + 2.5 * w);
    }
  }
  for (let i = 0; i < 400; i++) { const a = r() * 6.283; P.push(4 + 0.1 * Math.cos(a), 4 + 0.1 * Math.sin(a), r() * 6); } // pole
  return Float64Array.from(P);
}

const count = (s, name) => s.objects.filter(o => S.NAMES[o.cls] === name).length;
const A = S.analyze(makeScene());
console.log('A objects: vehicles', count(A, 'vehicle'), 'peds', count(A, 'pedestrian'), 'poles', count(A, 'pole'),
  '| junctions', JSON.stringify(A.junctions.map(j => [Math.round(j.x), Math.round(j.y), j.arms])));
const lab = {}; for (const l of A.labels) lab[S.NAMES[l]] = (lab[S.NAMES[l]] || 0) + 1; console.log('A labels', lab);

assert.strictEqual(count(A, 'vehicle'), 3, 'three vehicles');
assert.strictEqual(count(A, 'pedestrian'), 2, 'two pedestrians');
assert.strictEqual(count(A, 'pole'), 1, 'one pole');
assert(lab.building > 5000 && lab.vegetation > 2000 && lab.ground > 20000, 'building/vegetation/ground labelled');
assert(A.junctions.some(j => Math.hypot(j.x, j.y) < 8 && j.arms === 4), 'four-way intersection near origin');
assert(A.junctions.filter(j => Math.abs(j.x) < 40 && Math.abs(j.y) < 40).length === 1, 'exactly one junction among the streets');

// B = A plus extra content, with one car moved away. The directional score should stay high,
// the reverse score (B covered by A) should be lower.
const B = S.analyze(makeScene({ seed: 2, cars: [[2, -20], [-2, 25]], extraCars: [[-20, -2], [30, 2]],
                                extraTrees: [[-0.5, 35]], extraBuildings: [[50, 5, 58, 40]] }));
const cmp = S.compareScenes(A, B);
console.log('A vs B', cmp.score.toFixed(3), 'reverse', cmp.reverse.toFixed(3));
for (const r of cmp.rows) console.log('  ', r.name.padEnd(13), r.a, r.b, r.matched, r.cov_ab === null ? '-' : r.cov_ab.toFixed(2));
assert(cmp.score > 0.75, 'A is mostly contained in B');
assert(cmp.rows.find(r => r.key === 'vehicle').matched === 2, '2 of A\'s 3 cars are in B');

// A scene with the buildings and trees removed should score lower against A than B does.
const C = S.analyze(makeScene({ seed: 3, cars: [], peds: [], trees: [] }));
const sC = S.compareScenes(A, C).score;
console.log('A vs sparse C', sC.toFixed(3));
assert(sC < cmp.score, 'missing content lowers the score');
assert(S.compareScenes(A, A).score > 0.999, 'identical scenes score 1');
console.log('scene tests passed');
