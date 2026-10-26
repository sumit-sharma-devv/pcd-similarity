import os

import numpy as np
from flask import Flask, jsonify, render_template, request

from pcd_similarity.metrics import DEFAULT_ALPHA, DEFAULT_TAU, compute_metrics, normalize_pair
from pcd_similarity.pcd import PCDError, read_pcd

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 512)) * 1024 * 1024


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload too large."), 413


@app.get("/")
def index():
    return render_template("index.html", alpha=DEFAULT_ALPHA, tau=DEFAULT_TAU)


@app.post("/compare")
def compare():
    files = [request.files.get("file_a"), request.files.get("file_b")]
    if not all(files) or not all(f.filename for f in files):
        return jsonify(error="Please upload two PCD files."), 400
    try:
        clouds = [read_pcd(f.read()) for f in files]
        alpha = float(request.form.get("alpha", DEFAULT_ALPHA))
        tau = float(request.form.get("tau", DEFAULT_TAU))
        if alpha <= 0 or tau <= 0:
            raise ValueError
    except PCDError as e:
        return jsonify(error=str(e)), 400
    except ValueError:
        return jsonify(error="alpha and tau must be positive numbers."), 400
    normalize = request.form.get("normalize", "on") != "off"
    result = compute_metrics(*clouds, alpha=alpha, tau=tau, normalize=normalize)
    result["preview"] = preview(*clouds)
    return jsonify(result)


def preview(a, b, max_points=20_000):
    """Subsampled, co-registered (A-normalised) points for the 3D viewer."""
    rng = np.random.default_rng(0)
    out = []
    for c in normalize_pair(a, b):
        if len(c) > max_points:
            c = c[rng.choice(len(c), max_points, replace=False)]
        out.append(np.round(c, 4).ravel().tolist())
    return {"a": out[0], "b": out[1]}


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1", port=5001)
