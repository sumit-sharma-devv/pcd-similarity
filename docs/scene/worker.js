// Scene analysis in a Web Worker so the page stays responsive.
// The version suffix busts caches so the page, worker and scripts always come from the same release.
const V = new URL(self.location.href).search || '';
importScripts('../pcd.js' + V, 'scene.js' + V);
self.onmessage = ({ data }) => {
  try {
    const opts = { up: data.up, groundTol: data.groundTol, bev: data.bev, matchDist: data.matchDist, alignCentroids: data.align };
    const sa = Scene.analyze(Scene.subsample(PCD.readPCD(data.bufA), 200000), opts);
    const sb = Scene.analyze(Scene.subsample(PCD.readPCD(data.bufB), 200000), opts);
    const r = Scene.compareScenes(sa, sb, opts);
    r.objects = { a: sa.objects.length, b: sb.objects.length };
    r.preview = Scene.previewLabeled(sa, sb, r.offset);
    self.postMessage({ ok: true, result: r },
      [r.preview.a.buffer, r.preview.b.buffer, r.preview.la.buffer, r.preview.lb.buffer]);
  } catch (e) {
    self.postMessage({ ok: false, error: e.message });
  }
};
