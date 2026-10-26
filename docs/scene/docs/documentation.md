# PCD Scene Similarity – Documentation

How the scene-similarity tool works: how it works out what is in a point cloud without labels, how two scenes are compared, how to read the result, and where it can go wrong.

## 1. Overview

The [main tool](../../) compares two point clouds by geometry (DCD, CD, EMD, Voxel IoU). This page documents a separate tool, **PCD Scene Similarity**, for road scenes. It first works out **what is present** in each file (road, buildings, vegetation, vehicles, pedestrians, poles, intersections) and then asks how much of A's content is also found in B.

Why a separate tool: if file A is a street with a road, an intersection, cars and pedestrians, and file B is the same street plus buildings and trees, any geometric distance reports the extra buildings and trees as a mismatch, and a voxel overlap can be fooled the other way when different objects occupy the same space. Comparing what each scene *contains* avoids both problems.

Use it when the two files contain different things, for example one scene has extra buildings and trees. Geometric distances would report that extra content as a mismatch; this tool does not.

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

3. Press **Compare scenes**. You get a headline percentage, a per-class table, and a 3D view that can colour points by detected class so you can check the detection by eye.

Everything runs in your browser; the files are never uploaded.

## 3. How a scene is analysed

### No labels, no machine learning

Your files have no per-point classes, so the tool labels the points itself with geometric rules. This runs entirely in the browser, needs no model download and no server, and is **approximate**: it relies on how things are shaped and sized, not on a trained model. Each cloud goes through four steps.

**1. Ground.** The cloud is split into 1 m cells (viewed from above). In each cell the 5th-percentile height gives a first ground estimate. A slope-limited lower envelope then enforces that the ground cannot rise faster than a slope $\sigma$ (default 0.2, about 11°) between neighbouring cells:

$$z_0(c) \leftarrow \min\bigl(z_0(c),\; z_0(c') + \sigma\, d(c,c')\bigr) \quad \text{for neighbouring cells } c'.$$

This stops a car roof or a building from being mistaken for ground. A point whose height above $z_0$ is below the **ground tolerance** (default 0.25 m) is labelled *ground* (road, pavement and any open flat surface; the tool cannot tell these apart without intensity or colour).

**2. Objects.** The remaining points are placed in 0.4 m voxels and grouped into connected clusters (26-neighbour connectivity). For each cluster the tool measures:

- height $H$ above ground, and the lowest height $H_{\min}$;
- footprint length $L$ and width $W$, along the cluster's principal axis seen from above;
- **planarity** $p$: the fraction of voxels whose local neighbourhood lies on a plane. For a neighbourhood with covariance eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$, it is planar when $\lambda_2 \ge 0.1\lambda_1$ and $\lambda_3 \le 0.12\lambda_2$. Walls and vehicle panels are planar; foliage is not. It is measured at two scales: about 2 m ($p$, for buildings) and about 1.2 m ($p_s$, for vehicles).

**3. Classes.** Rules are applied in this order (all sizes in metres):

| Class | Rule |
|---|---|
| pole-like (signal, sign, lamp post) | $H \ge 2.5$ and $L \le 1.0$ |
| pedestrian | $1.0 \le H \le 2.3$, $L, W \le 1.3$ and $H/L \ge 1.2$ |
| vehicle | $0.9 \le H \le 4.2$, $2 \le L \le 13$, $1 \le W \le 3.3$, $H_{\min} < 1$ and $p_s \ge 0.2$ |
| building | $H \ge 3$, ($L \ge 6$ or $L W \ge 25$) and $p \ge 0.35$ |
| vegetation | $H \ge 0.4$ and $p < 0.35$ |
| other | anything else (small clutter, tiny clusters) |

A big building-or-vegetation cluster can be a building fused with a tree. Such clusters are refined voxel by voxel: a voxel is labelled *building* when at least 45% of its neighbourhood is planar, otherwise *vegetation*.

**4. Intersections.** Free road space is the ground that is not blocked by building or vegetation points at 0.3–2.5 m height (vehicles and pedestrians do not block). For each free cell that is at least 2 m from any obstacle, 72 points are sampled on a ring of radius $R$ (default 12 m) around it. Small gaps of up to two samples (a tree trunk, a post) are closed so they cannot split one road into two. The cell is an intersection candidate when the ring crosses three or more separate free arms, each between 20° and 150° wide. Neighbouring candidates are merged into one intersection whose **arms** count (3 = T-junction, 4 = cross) is the most common arm count of its cells.

## 4. How two scenes are compared

Both scenes are placed on one **bird's-eye grid** (default 2 m cells) in their own coordinates, so the two files must share a coordinate frame (tick *Align scene centres* if they only share an orientation). For each class $k$ the **coverage** is the share of A's content of that class that is also present in B:

$$
\mathrm{cov}_k(A \to B) =
\begin{cases}
\dfrac{\lvert \{c : n^A_k(c) \ge 3 \ \wedge\ n^B_k(c) \ge 3\} \rvert}{\lvert \{c : n^A_k(c) \ge 3\} \rvert} & k \in \{\text{ground, building, vegetation}\} \\[2ex]
\dfrac{\text{matched objects}}{\text{objects in } A} & k \in \{\text{vehicle, pedestrian, pole, intersection}\}
\end{cases}
$$

where $n^S_k(c)$ is the number of class-$k$ points of scene $S$ in grid cell $c$. For the area classes a cell counts as occupied from 3 points. For objects, A's and B's objects of the same class are matched one-to-one, nearest first, when their centres are within the **match distance** (default 2 m; intersections use 10 m). The overall score is the weighted mean over the classes that **are present in A**:

$$S_{\mathrm{scene}} = \frac{\sum_{k \in K_A} w_k\, \mathrm{cov}_k(A \to B)}{\sum_{k \in K_A} w_k}, \qquad K_A = \{k : A \text{ contains class } k\}$$

with weights ground 2, intersection 2, building 1, vegetation 1, vehicle 1, pedestrian 1, pole 0.5.

- **Extra content in B is not penalised.** A class that is absent from A is ignored, and B containing more of a class than A does not lower its coverage.
- **Directional.** $S_{\mathrm{scene}}$ answers "how much of A is in B". The reverse score (B in A) and their mean are shown too, which tells you how much B contains beyond A.
- The results table lists, per class, how much is in A and in B, the coverage in both directions and the weight, and the 3D viewer can colour the points by detected class so you can check the detection by eye.

## 5. Reading the results

| You see | It suggests |
|---|---|
| High overall score, high "B found in A" too | The two scenes contain much the same things in the same places. |
| High overall score, low "B found in A" | B contains everything A has, plus more (extra buildings, trees, vehicles). |
| Low ground coverage | The scenes cover different areas, or are not in the same coordinate frame (try *Align scene centres*). |
| A class shows "not in A, ignored" | A has none of that class, so it does not affect the score. |
| Vehicle or pedestrian coverage low | Those objects are in different places (beyond the match distance), moved, or merged into other objects and not detected. |
| Intersection arms differ | Matched junctions have a different number of road arms (T versus cross). |

Look at the class-coloured 3D view before trusting a number: if buildings are coloured as vegetation, or cars are missing, the score reflects those detection mistakes.

## 6. Limitations

- **Heuristic, not learned.** The rules assume sizes typical of road scenes, in **metres** and with a known up axis. Check the class-coloured 3D view on your own files before trusting a number. Rules will mislabel unusual objects (a bus shelter as a building, a hedge as a vehicle).
- **Touching objects merge.** A car parked against a wall, or a pedestrian touching a vehicle, is part of one cluster and is not detected separately.
- **Cannot see what geometry does not show.** Lane lines, painted markings, traffic-light state and road-versus-pavement all need intensity, colour or camera data, so they are not detected. Pole-like objects are one class (signals, signs and lamp posts look alike in geometry). Ground is one class.
- **Intersections are indicative.** The ring test finds places where several open arms meet, so a street that opens onto a car park or the edge of the scan can be reported as a junction. It needs buildings or vegetation on both sides of the road to see the road's width.
- **Sparse or one-sided scans.** Far-away or single-sweep data gives few points per surface, which weakens the planarity test that separates buildings from trees.
- **A trained segmentation model** (for example one trained on SemanticKITTI or Waymo) would be more accurate and could add finer classes, but needs a server or a large in-browser model. The labelling step here is isolated in `scene.js`, so it could be swapped later without changing the scoring.

## 7. Testing

SCENE is JavaScript only (there is no Python version). It was tested on synthetic street scenes (`tests/scene.test.js`): a cross intersection with four buildings, trees, cars, pedestrians and a pole. On those it found every object and the intersection, scored a scene against a copy with extra buildings, trees and cars at about 95%, and scored a bare scene lower. Its accuracy on real LiDAR data is **unverified**.

## 8. Implementation

| File | Role |
|---|---|
| `docs/scene/scene.js` | Ground estimation, clustering, class rules, intersection detection, scene comparison. |
| `docs/scene/worker.js` | Runs the analysis in a Web Worker so the page stays responsive. |
| `docs/pcd.js` | PCD parser, shared with the main tool. |
| `docs/scene/index.html` | The tool page and 3D viewer. |
| `tests/scene.test.js` | Synthetic-scene tests (`node tests/scene.test.js`). |

The main tool at the site root is unchanged and independent of this page.
