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

## Deploy (Render)

1. Push this folder to a GitHub repo.
2. On Render: **New → Blueprint**, pick the repo; `render.yaml` configures everything.
3. Uploads are capped by `MAX_UPLOAD_MB` (100 on the free tier's 512 MB RAM).

Local production-style run: `gunicorn app:app --workers 1 --threads 2 --timeout 120`.
