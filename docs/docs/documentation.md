# PCD Similarity – Documentation

What the project does, which similarity methods it offers, the logic and formula behind each, and how to read the numbers.

## 1. Overview

PCD Similarity compares two 3D point clouds stored as `.pcd` files (the Point Cloud Library format) and reports how similar they are. You upload file **A** (the reference) and file **B**, choose a **similarity method**, and get:

- a headline similarity percentage for the chosen method,
- the scores of the other cheap methods side by side,
- supporting distance metrics (Chamfer L1/L2, Hausdorff, F-score, voxel counts),
- an interactive 3D view with both clouds overlaid.

Everything runs in your browser. The files are read locally and are never uploaded to a server.

### Available methods

| Method | What it measures | Similarity score |
|---|---|---|
| **DCD** – Density-aware Chamfer Distance | Nearest-neighbour distance *and* density agreement | $1 - \mathrm{DCD}$ |
| **CD** – Chamfer Distance only | Average nearest-neighbour distance, density ignored | $\exp(-\mathrm{CD}/s)$ |
| **EMD** – Earth Mover's Distance only | Cost of optimally moving one cloud onto the other | $\exp(-\mathrm{EMD}/s)$ |
| **Voxel IoU** | Overlap of occupied cells on a shared 3D grid | $\mathrm{IoU}$ |

DCD is the default and follows [Wu et al., NeurIPS 2021 (arXiv:2111.12702)](https://arxiv.org/abs/2111.12702), which proposes it as an improvement on CD and EMD. The other three are the classic baselines the paper compares against, plus a volumetric measure.

## 2. How it works

1. **Parse.** Each file's header is read (`FIELDS`, `SIZE`, `TYPE`, `POINTS`, `DATA`) and the `x`, `y`, `z` values are extracted from `ascii`, `binary` or `binary_compressed` (LZF) data. Points with non-finite coordinates (NaN padding) are dropped. Other fields such as colour or intensity are ignored.
2. **Subsample.** A cloud with more than 200,000 points is randomly reduced to 200,000 (fixed seed, so repeated runs agree).
3. **Normalise** (optional, on by default). Both clouds are translated so the centroid of A is the origin, then divided by the largest distance from that centroid to any point of A. A therefore fits inside a unit sphere and B is expressed in the same frame. Because the same transform is applied to both, a real difference in size or position between A and B is still penalised.
4. **Nearest neighbours.** For every point of A the closest point of B is found, and vice versa, using a kd-tree.
5. **Score.** The chosen method is evaluated (sections 3–6). DCD, CD and Voxel IoU are cheap, so their scores are always shown; EMD is only computed when selected.
6. **Display.** Up to 20,000 points per cloud are drawn in the viewer, in the same normalised frame.

### Notation

For point sets $A$ and $B$ in $\mathbb{R}^3$, the distance from a point $x$ to the other set is

$$d_B(x) = \min_{y \in B} \lVert x - y \rVert_2 .$$

All distances below are Euclidean ($L_2$) and, with normalisation on, are measured in units of A's radius (so A has radius 1).

## 3. DCD – Density-aware Chamfer Distance

### Background: the weakness of plain Chamfer Distance

Chamfer Distance only looks at *nearest-neighbour distance*. Two clouds can score a perfect CD while having very different point densities: if a thousand points of A all sit on one point of B, every distance is still tiny and CD does not notice. Its values are also unbounded, so a few outliers can dominate.

### Definition

Let $\hat y$ be the nearest neighbour in $B$ of a point $x \in A$, and $\hat x$ the nearest neighbour in $A$ of a point $y \in B$. Then

$$
\mathrm{DCD}(A,B) = \frac{1}{2}\left[ \frac{1}{|A|}\sum_{x \in A}\left(1 - \frac{1}{n_{\hat y}}\, e^{-\alpha \lVert x-\hat y\rVert_2}\right) + \frac{1}{|B|}\sum_{y \in B}\left(1 - \frac{1}{n_{\hat x}}\, e^{-\alpha \lVert y-\hat x\rVert_2}\right)\right]
$$

- $e^{-\alpha d}$ turns a distance into a closeness value in $(0,1]$: it is 1 when the points coincide and decays towards 0 as they separate. $\alpha$ sets how quickly (default $\alpha = 1000$, the paper's value).
- $n_{\hat y}$ is the **density term**: the number of points of the querying set whose nearest neighbour is that same $\hat y$. If many points pile onto one target, each contribution is divided by that count, so the clump is penalised. For a one-to-one matching $n = 1$ and the term has no effect.

### Similarity score

$$S_{\mathrm{DCD}} = 1 - \mathrm{DCD}(A,B) \in [0,1]$$

100% means the clouds coincide point for point; 0% means no point has a close neighbour.

### Properties

- **Density-aware:** mismatched local density lowers the score even when shapes line up.
- **Bounded:** every term lies in $[0,1]$, so outliers cannot dominate.
- **Cheap:** only nearest-neighbour queries, unlike EMD.

> **The score is deliberately strict.** With $\alpha = 1000$ and normalised coordinates, a nearest neighbour at distance $0.001$ contributes $e^{-1} \approx 0.37$, and at $0.01$ only $e^{-10} \approx 4.5\times10^{-5}$. Two scans of the *same* object that sample different points will score close to 0%. Lower $\alpha$ (for example 10–100) for noisier real-world data, and compare scores only between runs that use the same $\alpha$.

## 4. CD – Chamfer Distance only

### Definition

Chamfer Distance averages the nearest-neighbour distance in both directions:

$$
\mathrm{CD}(A,B) = \frac{1}{2}\left[ \frac{1}{|A|}\sum_{x \in A} d_B(x) + \frac{1}{|B|}\sum_{y \in B} d_A(y) \right]
$$

This is the $L_1$ form (plain distances) and is what the "CD only" method uses. The $L_2$ form averages squared distances instead and is shown under *Other metrics*:

$$
\mathrm{CD}_2(A,B) = \frac{1}{2}\left[ \frac{1}{|A|}\sum_{x \in A} d_B(x)^2 + \frac{1}{|B|}\sum_{y \in B} d_A(y)^2 \right]
$$

Some papers sum the two directions instead of averaging; such figures are twice as large as the values here.

### Mapping distance to a similarity

CD is a distance in $[0,\infty)$, not a score, so it is mapped to $(0,1]$ with a decaying exponential:

$$S_{\mathrm{CD}} = \exp\!\left(-\frac{\mathrm{CD}(A,B)}{s}\right)$$

$s$ is the **distance scale** (default $0.05$, i.e. 5% of A's radius): a CD equal to $s$ gives a similarity of $e^{-1} \approx 37\%$, and CD $= 0$ gives 100%. The mapping is a presentation choice, not part of the original metric; smaller $s$ makes the score stricter.

### Behaviour

Cheap and intuitive, but blind to density: a cloud with duplicated points, or one that covers the same surface far more sparsely, can have a CD of zero.

## 5. EMD – Earth Mover's Distance only

### Definition

EMD treats the clouds as piles of earth and measures the work needed to reshape one into the other. For two clouds of the same size $n$, it is the average length of the best one-to-one matching:

$$
\mathrm{EMD}(A,B) = \min_{\phi : A \to B}\; \frac{1}{n}\sum_{x \in A} \lVert x - \phi(x) \rVert_2
$$

where the minimum runs over all bijections $\phi$. Unlike CD, every point of $B$ must be used exactly once, so EMD is sensitive to the **global distribution** of points and therefore to density.

### How it is computed

- The optimal matching is an assignment problem, solved **exactly** with the Hungarian algorithm (shortest augmenting paths), which costs $O(n^3)$ time and $O(n^2)$ memory.
- That is too slow for large clouds, so both clouds are randomly subsampled to the same size $m = \min(|A|, |B|, 2048)$ points (fixed seed). The page reports how many points were used.
- If one cloud is much larger than the other, only $m$ points of each are compared, so EMD is an estimate of the full-cloud value.

### Similarity score

$$S_{\mathrm{EMD}} = \exp\!\left(-\frac{\mathrm{EMD}(A,B)}{s}\right)$$

with the same distance scale $s$ as for CD (default $0.05$).

### Behaviour

More sensitive to density and global arrangement than CD, but also the most expensive method (roughly 0.5–1 s for 2,048 points in a browser) and subject to sampling noise.

## 6. Voxel IoU

### Definition

Space is divided into cubes of edge length $v$ (the **voxel size**, default $0.05$ in normalised units, roughly 40 cells across A). A point $p$ falls in the cell

$$c(p) = \left\lfloor \frac{p}{v} \right\rfloor \in \mathbb{Z}^3 ,$$

and the *occupied set* of a cloud is the set of cells containing at least one of its points: $V_A = \{c(x) : x \in A\}$, and likewise $V_B$. Both clouds use the same grid. The score is the overlap of the two occupied sets:

$$S_{\mathrm{IoU}} = \mathrm{IoU}(A,B) = \frac{\lvert V_A \cap V_B \rvert}{\lvert V_A \cup V_B \rvert} \in [0,1]$$

### Behaviour

- **Resolution-controlled:** a larger $v$ tolerates more noise and misalignment; a smaller $v$ is stricter. The score is only comparable between runs that use the same $v$.
- **Volumetric and coarse:** it ignores sub-voxel detail and, like CD, ignores how many points are in each cell.
- **Grid-sensitive:** a tiny shift can push points across a cell boundary, so scores can jump slightly when the clouds are nudged.
- Very small $v$ on large coordinates is rejected, because the number of grid cells would overflow.

## 7. Supporting metrics

These are always shown under *Other metrics* and are not selectable as the headline score.

| Metric | Definition | Meaning | Better |
|---|---|---|---|
| Hausdorff Distance | $\max\left(\max_{x\in A} d_B(x),\ \max_{y\in B} d_A(y)\right)$ | Worst-case gap: how far the most isolated point is from the other cloud. Very sensitive to outliers. | Lower |
| F-score | $\dfrac{2PR}{P+R}$ | How much of each cloud is covered by the other within tolerance $\tau$. | Higher |

where precision $P = \dfrac{\lvert\{y\in B : d_A(y) < \tau\}\rvert}{\lvert B\rvert}$ and recall $R = \dfrac{\lvert\{x\in A : d_B(x) < \tau\}\rvert}{\lvert A\rvert}$, with $\tau$ defaulting to $0.01$ in normalised units.

## 8. Comparing the methods

| | DCD | CD | EMD | Voxel IoU |
|---|---|---|---|---|
| Sees density mismatch | Yes | No | Yes (globally) | Only per-cell occupancy |
| Natural range | $[0,1]$ | $[0,\infty)$ | $[0,\infty)$ | $[0,1]$ |
| Needs a mapping to a score | No | Yes ($s$) | Yes ($s$) | No |
| Cost | Fast | Fast | Slow ($O(n^3)$, subsampled) | Fast |
| Main parameter | $\alpha$ | $s$ | $s$ | $v$ |
| Strictness | Very high at $\alpha=1000$ | Tunable by $s$ | Tunable by $s$ | Tunable by $v$ |

**Choosing a method**

- Comparing a reconstruction against ground truth where point density matters: **DCD**.
- A quick, intuitive "how close are the surfaces": **CD**.
- Comparing overall distributions when point counts are comparable and a few thousand points are enough: **EMD**.
- Robust, resolution-controlled overlap, for example for occupancy or coverage: **Voxel IoU**.

Different methods give different numbers for the same pair by design. A percentage from one method should not be compared with a percentage from another.

## 9. Worked example

Three synthetic clouds of 900 points each: **A** lies on a unit sphere; **B** is A with Gaussian noise ($\sigma = 0.01$ per coordinate); **C** is a fresh sample of a sphere scaled by 1.05 and shifted by $0.03$ along every axis. Settings: normalisation on, $\alpha = 100$, $s = 0.05$, $v = 0.1$.

| Pair | DCD | CD | EMD | Voxel IoU |
|---|---|---|---|---|
| A vs B (same shape, slight noise) | 24.50% | 73.48% | 73.18% | 70.22% |
| A vs C (larger, shifted, resampled) | 0.44% | 19.14% | 7.51% | 15.09% |

All four methods rank A–B above A–C, but they disagree on how much. DCD is the harshest (even a mild $\sigma = 0.01$ perturbation costs it three quarters of the score at $\alpha = 100$); CD and EMD land close together for the near pair; EMD punishes the mismatched pair more than CD because it must account for every point; Voxel IoU is slightly lower than CD and EMD on the near pair and falls between them on the far pair.

## 10. Reading the results

| You see | It suggests |
|---|---|
| High similarity, low Chamfer, high F-score | The clouds match closely in shape and density. |
| Low DCD but high CD | Shapes agree but density or point multiplicity differs, or $\alpha$ is too strict for the noise level. |
| Low Chamfer but large Hausdorff | Mostly aligned, with a few outliers or a missing region. |
| High Chamfer and low F-score | The clouds differ in shape, scale or position (or are not aligned). |
| High Voxel IoU, low DCD | Same overall volume, but fine detail or density differs. |

## 11. Limitations and notes

- **No alignment.** The tool does not register the clouds. If B is rotated or shifted relative to A, scores drop. Align them first (for example with ICP) if pose should not matter.
- **A is the reference.** Normalisation uses A's centroid and extent, so swapping A and B can change the numbers slightly. With normalisation off, raw coordinates are compared and $\alpha$, $\tau$, $s$ and $v$ are in the files' own units.
- **Geometry only.** Colour, intensity and normals are ignored.
- **Subsampling.** Clouds above 200,000 points are randomly subsampled, EMD works on at most 2,048 points per cloud, and the viewer shows at most 20,000.
- **Score mappings.** The exponential mappings for CD and EMD, and the default values of $\alpha$, $s$ and $v$, are choices made for this tool. Percentages are relative measures for comparing runs with the same settings, not absolute amounts of "sameness".
- **Testing.** The code was checked on synthetic clouds, including exact agreement between the JavaScript and Python implementations for all four methods, and unit tests for EMD and Voxel IoU on cases with known answers. It has not been benchmarked on real scanner data, and the compressed (LZF) format was verified only on a small hand-made case.
- DCD was implemented from the definition in the paper rather than from the authors' code, so tiny numerical differences from other implementations are possible.

## 12. Implementation

| File | Role |
|---|---|
| `docs/pcd.js` | PCD parser (ascii, binary, LZF-compressed). |
| `docs/metrics.js` | kd-tree, normalisation, subsampling, DCD, CD, EMD (Hungarian), Voxel IoU, Chamfer L2, Hausdorff, F-score. |
| `docs/worker.js` | Runs parsing and metrics in a Web Worker so the page stays responsive. |
| `docs/index.html` | The tool page and the 3D viewer. |
| `docs/docs/` | This documentation (Markdown, rendered in the browser with marked and KaTeX). |
| `pcd_similarity/`, `app.py` | Equivalent Python reference implementation (SciPy) and a Flask server that hosts the site and exposes `POST /compare`. |

## 13. Reference

Wu et al., *Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion*, NeurIPS 2021. [arXiv:2111.12702](https://arxiv.org/abs/2111.12702).
