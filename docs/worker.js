// Runs parsing + metrics off the main thread so the page stays responsive.
importScripts('pcd.js', 'metrics.js');
self.onmessage = ({ data }) => {
  try {
    const a = PCD.readPCD(data.bufA), b = PCD.readPCD(data.bufB);
    const r = Metrics.computeMetrics(a, b, data);
    delete r._a; delete r._b;
    r.preview = Metrics.preview(a, b);
    self.postMessage({ ok: true, result: r }, [r.preview.a.buffer, r.preview.b.buffer]);
  } catch (e) {
    self.postMessage({ ok: false, error: e.message });
  }
};
