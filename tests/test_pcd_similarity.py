import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402
from pcd_similarity.metrics import compute_metrics  # noqa: E402
from pcd_similarity.pcd import read_pcd  # noqa: E402


def make_pcd(pts, fmt="ascii"):
    n = len(pts)
    head = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA {fmt}\n").encode()
    if fmt == "ascii":
        return head + "\n".join(" ".join(map(str, p)) for p in pts).encode() + b"\n"
    return head + pts.astype("<f4").tobytes()


def sphere(n, seed=0):
    p = np.random.default_rng(seed).normal(size=(n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_parse_formats_agree():
    pts = sphere(100)
    a, b = read_pcd(make_pcd(pts, "ascii")), read_pcd(make_pcd(pts, "binary"))
    assert a.shape == (100, 3)
    assert np.allclose(a, b, atol=1e-5)


def test_identical_is_one():
    p = sphere(2000)
    m = compute_metrics(p, p.copy())
    assert m["similarity"] > 0.999 and m["chamfer_l1"] == 0 and m["fscore"] == 1


def test_noise_lowers_score_monotonically():
    p = sphere(2000)
    scores = [compute_metrics(p, p + np.random.default_rng(1).normal(scale=s, size=p.shape))["similarity"]
              for s in (0.001, 0.01, 0.1)]
    assert scores[0] > scores[1] > scores[2]


def test_density_mismatch_penalised():
    # Identical support (Chamfer = 0), but A piles 900 extra points onto one location.
    p = sphere(100)
    clumped = np.concatenate([p, np.repeat(p[:1], 900, axis=0)])
    m = compute_metrics(clumped, p)
    assert m["chamfer_l1"] == 0 and m["similarity"] < 0.6


def test_endpoint():
    c = app.test_client()
    data = {"file_a": (io.BytesIO(make_pcd(sphere(500))), "a.pcd"),
            "file_b": (io.BytesIO(make_pcd(sphere(500), "binary")), "b.pcd")}
    r = c.post("/compare", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and 0 <= r.json["similarity"] <= 1
    assert c.post("/compare", data={}, content_type="multipart/form-data").status_code == 400


def test_methods_identical_clouds():
    p = sphere(500)
    for method in ("dcd", "cd", "emd", "voxel"):
        r = compute_metrics(p, p.copy(), method=method)
        assert r["method"] == method and r["similarity"] > 0.99, method


def test_methods_order_noise():
    p = sphere(800)
    near = p + np.random.default_rng(1).normal(scale=0.005, size=p.shape)
    far = p + np.random.default_rng(1).normal(scale=0.1, size=p.shape)
    for method in ("cd", "emd", "voxel"):
        assert compute_metrics(p, near, method=method)["similarity"] > \
            compute_metrics(p, far, method=method)["similarity"], method


def test_emd_exact_on_known_matching():
    # Two points each, shifted by 1 along x: optimal matching moves every point exactly 1.
    from pcd_similarity.metrics import emd
    a = np.array([[0.0, 0, 0], [5, 0, 0]])
    assert abs(emd(a, a + [1.0, 0, 0]) - 1.0) < 1e-12


def test_voxel_iou_known_value():
    from pcd_similarity.metrics import voxel_iou
    a = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    b = np.array([[1.5, 0.5, 0.5], [2.5, 0.5, 0.5]])
    r = voxel_iou(a, b, 1.0)
    assert (r["intersection"], r["union"]) == (1, 3) and abs(r["iou"] - 1 / 3) < 1e-12


def test_endpoint_method_and_pages():
    c = app.test_client()
    assert c.get("/").status_code == 200 and c.get("/docs/").status_code == 200
    data = {"file_a": (io.BytesIO(make_pcd(sphere(300))), "a.pcd"),
            "file_b": (io.BytesIO(make_pcd(sphere(300, 1))), "b.pcd"), "method": "emd"}
    r = c.post("/compare", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and r.json["method"] == "emd" and "emd" in r.json["scores"]
    data = {"file_a": (io.BytesIO(make_pcd(sphere(30))), "a.pcd"),
            "file_b": (io.BytesIO(make_pcd(sphere(30))), "b.pcd"), "method": "nope"}
    assert c.post("/compare", data=data, content_type="multipart/form-data").status_code == 400
