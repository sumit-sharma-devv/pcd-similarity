// Runs parsing + metrics off the main thread so the page stays responsive.
// Version suffix busts caches so the page, worker and scripts always come from the same release.
const V = new URL(self.location.href).search || '';
importScripts('pcd.js' + V, 'metrics.js' + V);
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
