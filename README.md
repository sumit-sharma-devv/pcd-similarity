# pcd-similarity

Upload two `.pcd` files in the browser and get a similarity score.

The score is based on the **Density-aware Chamfer Distance (DCD)** from
[Wu et al., NeurIPS 2021 (arXiv:2111.12702)](https://arxiv.org/abs/2111.12702).
DCD is bounded in [0, 1] and, unlike plain Chamfer Distance, penalises density
mismatch. The headline figure is `similarity = 1 − DCD`. Chamfer (L1/L2),
Hausdorff and F-score are shown alongside for reference.

After comparing, a 3D viewer overlays both clouds (A blue, B orange). Mouse: drag rotates, right/shift-drag pans, scroll zooms. Keyboard (while hovering the viewer): arrows rotate, WASD pan, Q/E or -/+ zoom, Shift = faster, 1-4 preset views, Space auto-rotate, 0/R reset, Z/X toggle A/B, F fullscreen.

Supported PCD formats: `ascii`, `binary`, `binary_compressed` (needs `x y z` fields).

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py        # http://localhost:5001
```

## Test

```bash
pip install pytest && pytest
```

## Notes

- Clouds are centred and scaled by cloud A's bounding-sphere radius (toggleable), so
  A is the reference frame; size or position differences between A and B are penalised.
- `α` (default 1000, as in the paper) controls how fast the score drops with distance;
  the F-score threshold `τ` is in normalised units.
- Clouds over 200k points are randomly subsampled (fixed seed).

## Static site (GitHub Pages)

`docs/` is a dependency-free port of the same parser and metrics to JavaScript
(`pcd.js`, `metrics.js`, run in a Web Worker). Everything runs in the visitor's browser, so
nothing is uploaded and no server is needed. Its results match the Python version to ~1e-15.

To publish: repo **Settings → Pages → Build and deployment → Deploy from a branch →
`main` / `/docs`**. The site appears at `https://<user>.github.io/pcd-similarity/`.
To preview locally: `cd docs && python3 -m http.server 8099`.

## Deploy the Flask version (Render / Docker)

`render.yaml` and the `Dockerfile` deploy the Python app instead. Uploads are capped by
`MAX_UPLOAD_MB`. Local production-style run: `gunicorn app:app --workers 1 --threads 2 --timeout 120`.
