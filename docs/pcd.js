// PCD reader (ascii, binary, binary_compressed). Returns a Float64Array of xyz triples.
(function (root) {
  const TYPES = {
    F4: [Float32Array, 'getFloat32', 4], F8: [Float64Array, 'getFloat64', 8],
    U1: [Uint8Array, 'getUint8', 1], U2: [Uint16Array, 'getUint16', 2], U4: [Uint32Array, 'getUint32', 4],
    I1: [Int8Array, 'getInt8', 1], I2: [Int16Array, 'getInt16', 2], I4: [Int32Array, 'getInt32', 4],
  };

  function lzfDecompress(data, outLen) {
    const out = new Uint8Array(outLen);
    let i = 0, o = 0;
    while (i < data.length) {
      const ctrl = data[i++];
      if (ctrl < 32) {
        const n = ctrl + 1;
        if (o + n > outLen) throw new Error('Corrupt LZF data');
        out.set(data.subarray(i, i + n), o); i += n; o += n;
      } else {
        let len = ctrl >> 5;
        if (len === 7) len += data[i++];
        let ref = o - ((ctrl & 0x1f) << 8) - data[i++] - 1;
        if (ref < 0 || o + len + 2 > outLen) throw new Error('Corrupt LZF data');
        for (let k = 0; k < len + 2; k++) out[o++] = out[ref++];
      }
    }
    if (o !== outLen) throw new Error('LZF decompressed size mismatch');
    return out;
  }

  function readPCD(buffer) {
    const bytes = new Uint8Array(buffer);
    const header = {};
    let pos = 0;
    for (;;) {
      let end = bytes.indexOf(10, pos);
      if (end === -1) throw new Error('Invalid PCD: header has no DATA line');
      const line = new TextDecoder('ascii').decode(bytes.subarray(pos, end)).trim();
      pos = end + 1;
      if (!line || line[0] === '#') continue;
      const [key, ...vals] = line.split(/\s+/);
      header[key.toUpperCase()] = vals;
      if (key.toUpperCase() === 'DATA') break;
    }
    const fields = header.FIELDS, sizes = (header.SIZE || []).map(Number), types = header.TYPE;
    const counts = (header.COUNT || fields.map(() => '1')).map(Number);
    const npts = parseInt((header.POINTS || [])[0], 10), fmt = ((header.DATA || [])[0] || '').toLowerCase();
    if (!fields || !types || !(npts >= 0)) throw new Error('Invalid PCD: missing or malformed header fields');
    if (!['x', 'y', 'z'].every(f => fields.includes(f))) throw new Error('PCD file has no x/y/z fields');
    const spec = fields.map((f, i) => {
      const t = TYPES[types[i] + sizes[i]];
      if (!t) throw new Error(`Unsupported field type ${types[i]}${sizes[i]}`);
      return { name: f, T: t[0], get: t[1], size: t[2], count: counts[i] };
    });
    const body = bytes.subarray(pos);
    const xyz = new Float64Array(npts * 3);
    const idx = { x: 0, y: 1, z: 2 };

    if (fmt === 'ascii') {
      const cols = {}; let c = 0;
      spec.forEach(s => { cols[s.name] = c; c += s.count; });
      const lines = new TextDecoder('ascii').decode(body).split('\n');
      let n = 0;
      for (const ln of lines) {
        if (n >= npts) break;
        const t = ln.trim(); if (!t) continue;
        const p = t.split(/\s+/);
        if (p.length < c) throw new Error('ASCII PCD rows have too few columns');
        xyz[n * 3] = +p[cols.x]; xyz[n * 3 + 1] = +p[cols.y]; xyz[n * 3 + 2] = +p[cols.z]; n++;
      }
      if (n < npts) throw new Error('ASCII PCD has fewer rows than POINTS');
    } else if (fmt === 'binary') {
      const stride = spec.reduce((a, s) => a + s.size * s.count, 0);
      if (body.length < stride * npts) throw new Error('Binary PCD body is truncated');
      const dv = new DataView(body.buffer, body.byteOffset, body.byteLength);
      const offs = {}; let o = 0;
      spec.forEach(s => { offs[s.name] = s; s.off = o; o += s.size * s.count; });
      for (let i = 0; i < npts; i++)
        for (const k of ['x', 'y', 'z'])
          xyz[i * 3 + idx[k]] = dv[offs[k].get](i * stride + offs[k].off, true);
    } else if (fmt === 'binary_compressed') {
      const dv0 = new DataView(body.buffer, body.byteOffset, 8);
      const comp = dv0.getUint32(0, true), uncomp = dv0.getUint32(4, true);
      const buf = lzfDecompress(body.subarray(8, 8 + comp), uncomp);
      const dv = new DataView(buf.buffer);
      let off = 0; // column-wise: all x, then all y, ...
      for (const s of spec) {
        if (s.name in idx)
          for (let i = 0; i < npts; i++) xyz[i * 3 + idx[s.name]] = dv[s.get](off + i * s.size, true);
        off += s.size * s.count * npts;
      }
    } else throw new Error(`Unsupported DATA format '${fmt}'`);

    // Drop non-finite (NaN padding) points.
    let n = 0;
    for (let i = 0; i < npts; i++) {
      const x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
      if (isFinite(x) && isFinite(y) && isFinite(z)) { xyz[n * 3] = x; xyz[n * 3 + 1] = y; xyz[n * 3 + 2] = z; n++; }
    }
    if (n === 0) throw new Error('PCD file contains no valid points');
    return xyz.slice(0, n * 3);
  }

  const api = { readPCD, lzfDecompress };
  if (typeof module !== 'undefined') module.exports = api; else root.PCD = api;
})(typeof self !== 'undefined' ? self : this);
