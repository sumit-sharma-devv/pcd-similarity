"""Flask host for the static site in docs/, plus a JSON API (`POST /compare`) backed by the Python
reference implementation in pcd_similarity/. The browser UI computes everything client-side; the
API is for scripts and for cross-checking the JavaScript port."""
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from pcd_similarity.metrics import (DEFAULT_ALPHA, DEFAULT_EMD_POINTS, DEFAULT_SCALE, DEFAULT_TAU,
                                    DEFAULT_VOXEL, compute_metrics, normalize_pair)
from pcd_similarity.pcd import PCDError, read_pcd

app = Flask(__name__, static_folder="docs", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 512)) * 1024 * 1024


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload too large."), 413


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/docs/")
def documentation():
    return send_from_directory(os.path.join(app.static_folder, "docs"), "index.html")


def _num(name, default):
    value = float(request.form.get(name, default))
    if not value > 0:
        raise ValueError(f"{name} must be a positive number.")
    return value


@app.post("/compare")
def compare():
    files = [request.files.get("file_a"), request.files.get("file_b")]
    if not all(files) or not all(f.filename for f in files):
        return jsonify(error="Please upload two PCD files."), 400
    try:
        clouds = [read_pcd(f.read()) for f in files]
        result = compute_metrics(
            *clouds,
            alpha=_num("alpha", DEFAULT_ALPHA), tau=_num("tau", DEFAULT_TAU),
            scale=_num("scale", DEFAULT_SCALE), voxel=_num("voxel", DEFAULT_VOXEL),
            emd_points=int(_num("emd_points", DEFAULT_EMD_POINTS)),
            method=request.form.get("method", "dcd"),
            normalize=request.form.get("normalize", "on") != "off",
        )
    except (PCDError, ValueError) as e:
        return jsonify(error=str(e)), 400
    result["preview"] = preview(*clouds)
    return jsonify(result)


def preview(a, b, max_points=20_000):
    """Subsampled, co-registered (A-normalised) points for a 3D viewer."""
    rng = np.random.default_rng(0)
    out = []
    for c in normalize_pair(a, b):
        if len(c) > max_points:
            c = c[rng.choice(len(c), max_points, replace=False)]
        out.append(np.round(c, 4).ravel().tolist())
    return {"a": out[0], "b": out[1]}


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1", port=5001)
