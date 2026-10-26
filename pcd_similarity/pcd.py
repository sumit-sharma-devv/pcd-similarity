"""Minimal PCD (Point Cloud Data) reader: ascii, binary and binary_compressed."""
import struct

import numpy as np

_TYPES = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
}


class PCDError(ValueError):
    pass


def _lzf_decompress(data: bytes, out_len: int) -> bytes:
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:  # literal run
            out += data[i:i + ctrl + 1]
            i += ctrl + 1
        else:  # back reference
            length = ctrl >> 5
            if length == 7:
                length += data[i]
                i += 1
            ref = len(out) - ((ctrl & 0x1F) << 8) - data[i] - 1
            i += 1
            if ref < 0:
                raise PCDError("Corrupt LZF data")
            for k in range(length + 2):
                out.append(out[ref + k])
    if len(out) != out_len:
        raise PCDError("LZF decompressed size mismatch")
    return bytes(out)


def read_pcd(raw: bytes) -> np.ndarray:
    """Parse PCD bytes and return an (N, 3) float64 array of XYZ points."""
    pos = 0
    header = {}
    while True:
        end = raw.find(b"\n", pos)
        if end == -1:
            raise PCDError("Invalid PCD: header has no DATA line")
        line = raw[pos:end].decode("ascii", "ignore").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            break

    try:
        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        npts = int(header["POINTS"][0])
        fmt = header["DATA"][0].lower()
    except (KeyError, IndexError, ValueError):
        raise PCDError("Invalid PCD: missing or malformed header fields")
    if not all(f in fields for f in ("x", "y", "z")):
        raise PCDError("PCD file has no x/y/z fields")

    dtype = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        if (t, s) not in _TYPES:
            raise PCDError(f"Unsupported field type {t}{s}")
        dtype.append((f, _TYPES[(t, s)]) if c == 1 else (f, _TYPES[(t, s)], (c,)))
    dtype = np.dtype(dtype)

    body = raw[pos:]
    if fmt == "ascii":
        rows = np.loadtxt(body.decode("ascii", "ignore").splitlines(), ndmin=2) if npts else np.empty((0, 0))
        col, cols = 0, {}
        for f, c in zip(fields, counts):
            cols[f] = col
            col += c
        if rows.size and rows.shape[1] < col:
            raise PCDError("ASCII PCD rows have too few columns")
        pts = rows[:, [cols["x"], cols["y"], cols["z"]]] if rows.size else np.empty((0, 3))
    elif fmt == "binary":
        if len(body) < npts * dtype.itemsize:
            raise PCDError("Binary PCD body is truncated")
        arr = np.frombuffer(body, dtype=dtype, count=npts)
        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
    elif fmt == "binary_compressed":
        comp, uncomp = struct.unpack("<II", body[:8])
        buf = _lzf_decompress(body[8:8 + comp], uncomp)
        # Data is stored column-wise (all x, then all y, ...).
        off, cols = 0, {}
        for f, s, c in zip(fields, sizes, counts):
            nbytes = s * c * npts
            cols[f] = buf[off:off + nbytes]
            off += nbytes
        pts = np.stack(
            [np.frombuffer(cols[f], dtype=dtype[f].base, count=npts) for f in ("x", "y", "z")],
            axis=1,
        )
    else:
        raise PCDError(f"Unsupported DATA format '{fmt}'")

    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]  # drop NaN padding points
    if len(pts) == 0:
        raise PCDError("PCD file contains no valid points")
    return pts
