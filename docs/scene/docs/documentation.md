# PCD Scene Similarity – Documentation

How the scene-similarity tool works end to end: what it does, the full algorithm that works out what is in a point cloud without labels, how two scenes are compared and scored, and how the page is implemented, file by file and function by function.

## 1. Overview

The [main tool](../../) compares two point clouds by geometry (DCD, CD, EMD, Voxel IoU). This page documents a separate tool, **PCD Scene Similarity**, for road scenes. It first works out **what is present** in each file (road, buildings, vegetation, vehicles, pedestrians, poles, intersections) and then asks how much of A's content is also found in B.

Why a separate tool: if file A is a street with a road, an intersection, cars and pedestrians, and file B is the same street plus buildings and trees, any geometric distance reports the extra buildings and trees as a mismatch, and a voxel overlap can be fooled the other way when different objects occupy the same space. Comparing what each scene *contains* avoids both problems.

Key properties:

- **No labels and no machine learning.** Your files carry no per-point classes, so the tool labels the points itself with geometric rules (sizes, shapes, heights). It is approximate.
- **Runs in the browser.** Files are read locally and never uploaded. The analysis runs in a Web Worker, so the page stays responsive.
- **Directional score.** The headline answers "how much of A is in B". Extra content in B is not penalised.
- **Checkable.** The 3D view can colour points by detected class, so you can see what the rules decided.

## 2. Using the tool

1. Choose file **A** (the reference scene) and file **B**.
2. Set the options if needed. The defaults suit a typical road scene in metres.

| Option | Meaning | Default |
|---|---|---|
| Up axis | Which coordinate axis points up in your files | Z |
| Ground tolerance | Height (m) above the estimated ground surface still counted as ground | 0.25 |
| Grid cell | Cell size (m) of the bird's-eye grid used to compare area classes | 2 |
| Match distance | Largest distance (m) between the centres of two objects (vehicle, pedestrian, pole) for them to count as the same object | 2 |
| Align scene centres | Shifts B so both scenes have the same centre, for files that do not share a coordinate frame | off |

3. Press **Compare scenes**. You get a headline percentage, a per-class table, a points-per-class table and a 3D view.

The 3D view has a **Colour** switch (by detected class, or by cloud: A blue, B orange), per-cloud toggles, a point-size slider, preset views, auto-rotate, fullscreen, and mouse, touch and keyboard navigation (hover the view for the key list on the page).

## 3. The pipeline at a glance

```
file A, file B (.pcd)
   │  PCD.readPCD                      parse x,y,z (ascii / binary / LZF-compressed)
   │  Scene.subsample                  at most 200,000 points per file
   ▼
Scene.analyze (once per file)
   ├─ toZUp                            reorder axes so z is up
   ├─ estimateGround                   ground height per 1 m cell, slope-limited
   ├─ label ground points              height above ground below the tolerance
   ├─ voxelise the rest (0.4 m)        in (x, y, height above ground)
   ├─ connected clusters               26-neighbour union-find
   ├─ cluster features                 height, footprint, planarity at two scales
   ├─ classify                         pole / pedestrian / vehicle / building / vegetation / other
   ├─ refine big clusters per voxel    building fused with a tree
   └─ detectJunctions                  ring test on free road space
   ▼
Scene.compareScenes                    class coverage of A in B, weighted score
Scene.previewLabeled                   ≤ 20,000 labelled points per scene for the viewer
   ▼
page: tables + class-coloured 3D viewer
```

## 4. Scene analysis in detail

All distances are in the files' own units, assumed to be **metres**.

### 4.1 Input preparation

- **Subsampling.** A file with more than 200,000 points is randomly reduced to 200,000 using a seeded partial Fisher–Yates shuffle, so repeated runs give the same result.
- **Up axis.** Everything below assumes $z$ is up. For other up axes the coordinates are reordered first: Y-up maps $(x, y, z) \mapsto (x, -z, y)$ and X-up maps $(x, y, z) \mapsto (y, z, x)$.

### 4.2 Ground estimation

The cloud is viewed from above and split into square cells of 1 m (the cell size is enlarged automatically if the grid would exceed 2.5 million cells). Points are bucketed per cell with a counting sort.

1. **First estimate.** In every non-empty cell the ground height $z_0$ is the 5th percentile of the point heights (the value at index $\lfloor 0.05\,(n_c-1) \rfloor$ of the sorted heights, with $n_c$ points in the cell). A low percentile rather than the minimum resists points below the road (noise, reflections). Empty cells start at $+\infty$.
2. **Slope-limited lower envelope.** Ground cannot rise faster than a slope $\sigma$ (default 0.2, about 11°) between neighbouring cells. Each cell is lowered if a neighbour implies a lower bound:

$$z_0(c) \leftarrow \min\bigl(z_0(c),\; z_0(c') + \sigma\, d(c,c')\bigr) \quad \text{for the 8 neighbours } c',$$

where $d(c,c')$ is the distance between cell centres. This is a two-pass chamfer sweep (forward then backward over the grid), repeated twice. It stops a car roof or building from being taken as ground (their cells are pulled down towards the surrounding road), and it fills empty cells.
3. **Labelling.** For each point, its height above ground is $h = z - z_0(\text{cell})$. If $h < $ **ground tolerance** (default 0.25 m) the point is labelled *ground*; otherwise it is a *non-ground* point. Ground is a single class: road, pavement and grass are indistinguishable by geometry alone.

### 4.3 Grouping non-ground points into objects

Non-ground points are quantised into 0.4 m voxels in the coordinates $(x, y, h)$, so slopes are straightened by using height above ground. A hash map from voxel key to voxel index stores only occupied voxels. For each occupied voxel the tool accumulates ten running **moments**: the point count, the sums of $x$, $y$, $h$, and the sums of their pairwise products. From moments, the mean and covariance of any union of voxels can be computed without revisiting the points.

Occupied voxels that touch (all 26 neighbours, including diagonals) are merged with **union-find** (path compression), giving connected **clusters**. Each cluster is one candidate object: a car, a person, a tree, a building, or several touching things.

### 4.4 Cluster features

For each cluster the tool computes:

- **Height** $H$ (largest $h$) and **lowest height** $H_{\min}$ (smallest $h$).
- **Footprint** $L \times W$. The principal direction of the cluster seen from above is $\theta = \tfrac{1}{2}\operatorname{atan2}(2\sigma_{xy},\ \sigma_{xx} - \sigma_{yy})$ from the cluster's $x$–$y$ covariance. Points are projected on that axis and its perpendicular, and $L$ and $W$ are the larger and smaller extents.
- **Point count** $n$.
- **Planarity** at two scales. For a voxel, the moments of all voxels within a neighbourhood are pooled, and the eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ of the pooled covariance are computed (closed-form, via the trigonometric solution for symmetric 3×3 matrices). The neighbourhood is **planar** when it is not a line and is thin:

$$\lambda_2 \ge 0.1\,\lambda_1 \quad\text{and}\quad \lambda_3 \le 0.12\,\lambda_2 .$$

  Walls, roads and vehicle panels pass this test; foliage (where $\lambda_3 \approx \lambda_2$) does not. It is evaluated over the surrounding $5^3$ voxels (about 2 m, needs at least 15 points) to give $p$, the cluster's fraction of planar voxels, used for **buildings**; and over the surrounding $3^3$ voxels (about 1.2 m, at least 8 points) to give $p_s$, used for **vehicles** whose surfaces are small. A voxel without enough points in its neighbourhood is simply not counted.

### 4.5 Classification rules

Applied to each cluster in this order; the first rule that matches wins (sizes in metres):

| Class | Rule |
|---|---|
| other (noise) | fewer than 8 points, or $H < 0.3$ |
| pole-like (signal, sign, lamp post) | $H \ge 2.5$ and $L \le 1.0$ |
| pedestrian | $1.0 \le H \le 2.3$, $L \le 1.3$, $W \le 1.3$ and $H/L \ge 1.2$ |
| vehicle | $0.9 \le H \le 4.2$, $2 \le L \le 13$, $1 \le W \le 3.3$, $H_{\min} < 1$ and $p_s \ge 0.2$ |
| building | $H \ge 3$, ($L \ge 6$ or $L W \ge 25$) and $p \ge 0.35$ |
| vegetation | $H \ge 0.4$ and $p < 0.35$ |
| building (fallback) | $H \ge 3$ |
| other | anything else |

Vehicles, pedestrians and pole-like clusters are also recorded as **objects** with centre $(x, y)$, height, length, width and point count; these are what the comparison matches between scenes.

**Refinement of big clusters.** A building fused with a tree touches, so it forms one cluster that a single rule would label wrongly. Clusters classified as building or vegetation with $H \ge 3$ and $L \ge 5$ are therefore relabelled voxel by voxel: a voxel looks at its $3^3$ neighbours that have a planarity verdict; if at least three do, it becomes *building* when at least 45% of them are planar and *vegetation* otherwise. Other voxels keep the cluster's label. Every point then takes the label of its voxel.

### 4.6 Intersection detection

This works on a second grid of 1 m cells covering the scene.

1. **Free road space.** A cell is *free* when it contains ground points or vehicle/pedestrian points (a car on the road hides the ground below it) and it is not **blocked**. A cell is blocked when it contains building or vegetation points between 0.3 m and 2.5 m above ground (a tall canopy above the road does not block it).
2. **Clearance.** A two-pass 3–4 chamfer distance transform gives each free cell its distance to the nearest non-free cell.
3. **Ring test.** For each free cell with clearance of at least 2 m, 72 points (every 5°) are sampled on a ring of radius $R$ (default 12 m) around it. A sample is free if its cell is free; samples outside the grid count as not free. Cells where the ring is entirely free or entirely blocked are skipped.
4. **Closing.** Free runs are dilated and then eroded by one sample, so a thin obstacle (a tree trunk, a post) of up to two samples does not split one road arm into two.
5. **Arms.** The ring is read as circular runs of free samples. A run counts as an **arm** when it is between 4 and 30 samples long (20° to 150°). The cell is an intersection candidate when none of its runs is wider than 30 samples (so an open plaza is rejected) and it has at least three arms. Its arm count is stored (capped at 6).
6. **Merging.** Candidate cells that touch (8 neighbours) are grouped. Groups of fewer than 3 cells are discarded. Each remaining group becomes one intersection at the group's centre, with **arms** equal to the most common arm count among its cells (3 means T-junction, 4 means cross).

## 5. Comparing two scenes

### 5.1 The shared bird's-eye grid

Both scenes are placed on one grid (default 2 m cells, enlarged if it would exceed 4 million cells) covering the union of their extents, in their own coordinates. If **Align scene centres** is on, B is first shifted by the difference of the two scenes' centres (mean $x, y$), and the same shift is applied to B's objects and intersections. For each cell and class the tool counts the points of that class (per scene, saturating at 65,535).

### 5.2 Coverage per class

For each class $k$ the **coverage** is the share of A's content of that class that is also present in B:

$$
\mathrm{cov}_k(A \to B) =
\begin{cases}
\dfrac{\lvert \{c : n^A_k(c) \ge 3 \ \wedge\ n^B_k(c) \ge 3\} \rvert}{\lvert \{c : n^A_k(c) \ge 3\} \rvert} & k \in \{\text{ground, building, vegetation}\} \\[2ex]
\dfrac{\text{matched objects}}{\text{objects in } A} & k \in \{\text{vehicle, pedestrian, pole, intersection}\}
\end{cases}
$$

where $n^S_k(c)$ is the number of class-$k$ points of scene $S$ in grid cell $c$. For the **area classes** a cell counts as occupied from 3 points; coverage is the fraction of A's occupied cells that are also occupied in B. For the **object classes**, A's and B's objects of the same class are matched one-to-one: all pairs within the match distance are collected, sorted by distance, and taken nearest first, skipping pairs whose members are already used. The match distance is 2 m for vehicles, pedestrians and poles, and 10 m for intersections. Coverage is the number of matches divided by the number of objects in A. The same formula with the roles swapped gives $\mathrm{cov}_k(B \to A)$.

### 5.3 The overall score

$$S_{\mathrm{scene}} = \frac{\sum_{k \in K_A} w_k\, \mathrm{cov}_k(A \to B)}{\sum_{k \in K_A} w_k}, \qquad K_A = \{k : A \text{ contains class } k\}$$

| Class | ground | intersection | building | vegetation | vehicle | pedestrian | pole |
|---|---|---|---|---|---|---|---|
| Weight $w_k$ | 2 | 2 | 1 | 1 | 1 | 1 | 0.5 |

- **Extra content in B is not penalised.** A class absent from A is dropped from both numerator and denominator, and B containing more of a class than A does not lower the coverage.
- **Directional.** $S_{\mathrm{scene}}$ answers "how much of A is in B". The tool also reports the **reverse** score (the same formula with A and B swapped, which tells you how much of B is in A) and their mean, the **symmetric** score.
- The page also reports how many matched intersections have the same number of arms.

### 5.4 Worked example

The synthetic test scene A is a cross intersection (10 m wide streets) with four buildings, two trees, three cars, two pedestrians and a pole. Scene B is similar but has one car fewer, two extra cars, an extra tree and an extra building. The tool reports:

| Class | In A | In B | A found in B | Weight |
|---|---|---|---|---|
| ground | 2,276 cells | 2,225 cells | 97.8% | 2 |
| building | 308 cells | 349 cells | 99.4% | 1 |
| vegetation | 42 cells | 58 cells | 97.6% | 1 |
| vehicle | 3 | 4 | 66.7% | 1 |
| pedestrian | 2 | 2 | 100% | 1 |
| pole | 1 | 1 | 100% | 0.5 |
| intersection | 9 | 10 | 100% | 2 |

The weights sum to 8.5, so

$$S_{\mathrm{scene}} = \frac{2(0.978) + 0.994 + 0.976 + 0.667 + 1.0 + 0.5(1.0) + 2(1.0)}{8.5} = \frac{8.093}{8.5} \approx 95.2\% .$$

Only two of A's three cars have a counterpart in B, which is the main loss. B's extra building, tree and cars do not reduce the score. The synthetic open ground around the scene produces several extra intersection candidates; they appear in both scenes so they match.

## 6. Parameters and constants

**Set on the page:**

| Name | Default | Used in |
|---|---|---|
| Up axis (`up`) | Z | axis reordering |
| Ground tolerance (`groundTol`) | 0.25 m | ground labelling |
| Grid cell (`bev`) | 2 m | comparison grid |
| Match distance (`matchDist`) | 2 m | vehicle, pedestrian, pole matching |
| Align scene centres (`align`) | off | comparison |

**Fixed in `scene.js` (the `DEFAULTS` object and rule constants):**

| Name | Value | Meaning |
|---|---|---|
| `groundCell` | 1 m | ground-height grid cell |
| `slope` | 0.2 | ground slope limit |
| `voxel` | 0.4 m | clustering voxel size |
| `ringR` | 12 m | intersection ring radius |
| `junctionDist` | 10 m | intersection matching distance |
| `maxCells` | 2.5 million | grid size cap (analysis) |
| subsample cap | 200,000 points | per file |
| preview cap | 20,000 points | per scene, for the viewer |
| area-class occupancy | 3 points | cell counts as containing a class |
| `WEIGHTS` | see section 5.3 | class weights |

## 7. Implementation

### 7.1 Files

| File | Role |
|---|---|
| `docs/scene/index.html` | The page: form, results tables, 3D viewer, and the glue to the worker. |
| `docs/scene/worker.js` | Web Worker: parses both files, runs the analysis and comparison, returns the result. |
| `docs/scene/scene.js` | All the scene logic (works in a browser, a worker and Node). |
| `docs/pcd.js` | PCD parser, shared with the main tool. |
| `tests/scene.test.js` | Synthetic-scene tests, run with `node tests/scene.test.js`. |

### 7.2 What happens when you press Compare

1. `index.html` reads both files as `ArrayBuffer`s and validates the numeric options.
2. It creates a Web Worker `worker.js?v=<VERSION>` and posts the two buffers and options, **transferring** the buffers (no copy).
3. `worker.js` calls `PCD.readPCD` on each buffer, `Scene.subsample` (cap 200,000), then `Scene.analyze` for A and for B.
4. It calls `Scene.compareScenes(sa, sb, options)` for the score and tables, and `Scene.previewLabeled` for the viewer points.
5. It posts the result back, transferring the four typed arrays of the preview. On any error it posts `{ ok: false, error }` and the page shows the message.
6. The page fills the tables and hands the preview to the viewer.

The `?v=` suffix is appended by `worker.js` to the `importScripts` URLs as well, so the page, worker and scripts always come from the same release; this avoids a stale cached script being paired with a newer page. `VERSION` in `index.html` is bumped whenever those files change. If the result lacks the expected fields the page tells you to hard refresh.

### 7.3 `scene.js` function by function

| Function | Purpose |
|---|---|
| `subsample(pts, max)` | Seeded random reduction to at most `max` points. |
| `toZUp(p, up)` | Reorders axes so that z is up. |
| `eig3(a,b,c,d,e,f)` | Eigenvalues (descending) of a symmetric 3×3 matrix, closed form. |
| `estimateGround(p, n, gc, slope, maxCells)` | Per-cell 5th-percentile heights, then the slope-limited lower envelope; returns each point's cell index and the ground-height grid. |
| `classify(c)` | The rule table of section 4.5, applied to a cluster's features. |
| `detectJunctions(p, n, h, labels, o)` | Free-space grid, distance transform, ring test, merging; returns intersections `{x, y, arms}`. |
| `analyze(points, opts)` | The full per-scene analysis; returns `{ p, n, labels, h, objects, junctions, opts }`. |
| `classGrid(s, ox, oy, minx, miny, cs, gx, gy)` | Per-class point counts per bird's-eye cell. |
| `match(A, B, maxD)` | Greedy nearest-first one-to-one matching of objects within `maxD`. |
| `compareScenes(sa, sb, opts)` | Coverage per class, weighted scores, and the per-class rows shown on the page. |
| `countLabels(s)` | Points per class, for the points table. |
| `previewLabeled(sa, sb, offset, max)` | Up to `max` evenly strided points per scene with their labels, in A's normalised frame, y-up for the viewer. |

### 7.4 Data structures

- **Analysed scene** (`analyze` result): `p` (xyz, z up, `Float64Array`), `n` (point count), `labels` (`Uint8Array`, class id per point: 1 ground, 2 building, 3 vegetation, 4 vehicle, 5 pedestrian, 6 pole, 7 other), `h` (height above ground, `Float32Array`), `objects` (`{ cls, x, y, height, length, width, points }`), `junctions` (`{ x, y, arms }`).
- **Comparison result** (`compareScenes`): `score`, `reverse`, `symmetric`, `rows` (one per class: `name`, `kind` area or object, `a`, `b`, `matched`, `cov_ab`, `cov_ba`, `weight`; intersections also `sameArms`), `offset` (the alignment shift), `cell` (grid cell actually used) and `points` (counts per class for each scene).
- **Preview** (`previewLabeled`): `a`, `b` (`Float32Array` xyz) and `la`, `lb` (`Uint8Array` class ids). Coordinates are centred on A's centroid and divided by A's radius, then mapped $(x, y, z) \mapsto (x, z, -y)$ so that up is the viewer's vertical axis. B is shifted by the alignment offset first.

Heavy loops use typed arrays and flat grids rather than nested objects, and large intermediate structures are sized once.

### 7.5 The 3D viewer

The viewer is a small dependency-free canvas renderer inside `index.html`. Each point is rotated by the yaw and pitch angles, scaled by the zoom, translated by the pan offset and drawn as a small square. In class mode points are drawn in one pass per class using the class colours (ground grey, building brown-orange, vegetation green, vehicle blue, pedestrian pink, pole yellow, other slate). It supports mouse drag to rotate, right or shift drag to pan, wheel to zoom, pinch on touch screens, preset views (front, side, top, iso), auto-rotate, fullscreen, and a keyboard handler that is active while the view is hovered or focused (arrows rotate, WASD pan, Q/E or −/+ zoom, Shift for speed, 1–4 views, Space spin, 0 or R reset, Z/X toggle A/B, F fullscreen). Keys are processed in a time-based animation loop so speed does not depend on frame rate.

### 7.6 Cost

With $n$ points, $V$ occupied voxels and $G$ grid cells: ground estimation is $O(n \log n)$ in the worst case (sorting within cells); clustering is $O(26V)$; planarity is $O(125V)$ lookups; the ring test is $O(72G)$ at worst; and the comparison is $O(n + G + N_A N_B)$, where $N_A$ and $N_B$ are the object counts of a class. Grids are capped at a few million cells, and inputs at 200,000 points per file. On the synthetic scenes (about 110,000 points each) a full comparison takes well under a second in the browser; real scenes with more objects and larger extents will take longer.

## 8. Reading the results

| You see | It suggests |
|---|---|
| High overall score, high "B found in A" too | The two scenes contain much the same things in the same places. |
| High overall score, low "B found in A" | B contains everything A has, plus more (extra buildings, trees, vehicles). |
| Low ground coverage | The scenes cover different areas, or are not in the same coordinate frame (try *Align scene centres*). |
| A class shows "not in A, ignored" | A has none of that class, so it does not affect the score. |
| Vehicle or pedestrian coverage low | Those objects are in different places (beyond the match distance), moved, or merged into other objects and not detected. |
| Intersection arms differ | Matched junctions have a different number of road arms (T versus cross). |

Look at the class-coloured 3D view before trusting a number: if buildings are coloured as vegetation, or cars are missing, the score reflects those detection mistakes.

## 9. Limitations

- **Heuristic, not learned.** The rules assume sizes typical of road scenes, in **metres** and with a known up axis. Check the class-coloured 3D view on your own files before trusting a number. Rules will mislabel unusual objects (a bus shelter as a building, a hedge as a vehicle).
- **Touching objects merge.** A car parked against a wall, or a pedestrian touching a vehicle, is part of one cluster and is not detected separately. Big building-or-vegetation clusters are refined, but vehicles and pedestrians are not split out of them.
- **Cannot see what geometry does not show.** Lane lines, painted markings, traffic-light state and road-versus-pavement all need intensity, colour or camera data, so they are not detected. Pole-like objects are one class (signals, signs and lamp posts look alike in geometry). Ground is one class.
- **Intersections are indicative.** The ring test finds places where several open arms meet, so a street that opens onto a car park or the edge of the scan can be reported as a junction. It needs buildings or vegetation on both sides of the road to see the road's width, and the 12 m ring suits streets up to roughly 10–12 m wide.
- **Spatial comparison needs a shared frame.** Area classes and objects are compared cell by cell in the files' coordinates. Files in different frames need *Align scene centres*, which only fixes translation, not rotation.
- **Sparse or one-sided scans.** Far-away or single-sweep data gives few points per surface, which weakens the planarity test that separates buildings from trees.
- **Fixed thresholds.** Most rule thresholds are constants in `scene.js`, not page options.
- **A trained segmentation model** (for example one trained on SemanticKITTI or Waymo) would be more accurate and could add finer classes, but needs a server or a large in-browser model. See section 10 for where to swap it in.

## 10. Testing and extending

**Testing.** SCENE is JavaScript only (there is no Python version). `tests/scene.test.js` builds synthetic street scenes (a cross intersection with four buildings, trees, cars, pedestrians and a pole) and checks that:

- exactly 3 vehicles, 2 pedestrians and 1 pole are found, and buildings, vegetation and ground are labelled;
- a four-arm intersection is found at the crossing, and only one among the streets;
- a scene scored against a copy with extra buildings, trees and cars keeps a high directional score (above 0.75; about 0.95 in practice) and finds 2 of A's 3 cars;
- a scene with the buildings and trees removed scores lower than the copy with extra content, and identical scenes score 1.

These are synthetic and easy: clean surfaces, no occlusion, no sensor noise beyond a small jitter. Accuracy on real LiDAR data is **unverified**.

**Extending.**

- *Different class rules or thresholds:* edit `classify` and the constants at the top of `scene.js`.
- *Different weights:* edit `WEIGHTS`.
- *A trained model:* replace the labelling inside `analyze` so that it produces the same outputs (`labels`, `objects`, `junctions`); `compareScenes` and the page only use those.
- *A new class:* add a name and id in `NAMES`, a rule in `classify`, a weight, and a row in `compareScenes`; the page renders whatever rows it receives, and the legend and colours are in the viewer's `CLASS` table.

The main tool at the site root is unchanged and independent of this page.
