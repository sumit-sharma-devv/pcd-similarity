# pcd-similarity

Upload two `.pcd` files in the browser and get a similarity score. Pick the method:

| Method | Similarity score |
|---|---|
| **DCD** – Density-aware Chamfer Distance ([Wu et al., NeurIPS 2021](https://arxiv.org/abs/2111.12702)) | `1 − DCD` |
| **CD** – Chamfer Distance only | `exp(−CD / s)` |
| **EMD** – Earth Mover's Distance only (exact, on ≤ 2,048 sampled points) | `exp(−EMD / s)` |
| **Voxel IoU** | `|A ∩ B| / |A ∪ B|` |

Chamfer L1/L2, Hausdorff and F-score are shown alongside, and a 3D viewer overlays both
clouds (A blue, B orange). Full logic and formulas: [`docs/docs/documentation.md`](docs/docs/documentation.md)
(rendered with KaTeX at `/docs/` on the site).

**Scene similarity (separate page, `/scene/`):** for road scenes with different contents. Detects ground,
buildings, vegetation, vehicles, pedestrians, poles and intersections from raw points (no labels, rule-based,
no ML) and scores how much of A's content is found in B. Docs at `/scene/docs/`; code in `docs/scene/`;
tests: `node tests/scene.test.js`. Approximate and only tested on synthetic scenes.

Supported PCD formats: `ascii`, `binary`, `binary_compressed` (needs `x y z` fields).

## Layout

- `docs/` – the static site (GitHub Pages). Parser and metrics in plain JavaScript (`pcd.js`,
  `metrics.js`, run in a Web Worker); everything runs in the visitor's browser, nothing is uploaded.
- `pcd_similarity/`, `app.py` – equivalent Python reference implementation (NumPy/SciPy) and a
  Flask server that hosts `docs/` and exposes `POST /compare` (form fields `file_a`, `file_b`, `method`,
  `alpha`, `scale`, `voxel`, `tau`, `normalize`). JS and Python agree to ~1e-15 on all four methods.

## Run

Static site only: `cd docs && python3 -m http.server 8099` → http://localhost:8099

Flask (site + API):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py        # http://localhost:5001
```

## Test

```bash
pip install pytest && pytest
```

## Deploy

- **GitHub Pages:** Settings → Pages → Deploy from a branch → `main` / `/docs`.
- **Flask:** `render.yaml` (Render) or the `Dockerfile`. Uploads are capped by `MAX_UPLOAD_MB`.
  Production-style run: `gunicorn app:app --workers 1 --threads 2 --timeout 120`.

## Notes

- With normalisation on (default), both clouds are centred on A and scaled by A's radius, so A is
  the reference frame and `α`, `s`, `v`, `τ` are in units of A's radius.
- `α` (default 1000, as in the paper) makes DCD strict; lower it for noisy real data.
- Clouds over 200k points are randomly subsampled (fixed seed).
