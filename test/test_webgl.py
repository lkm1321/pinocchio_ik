"""Browser-side WebGL diagnostic.

Run with: `pixi run test-webgl`

Generates `test/webgl_check.html`. Open it in the same browser you use for
`meshcat_wrapper.html`; the page prints "OK" with renderer info if WebGL works,
or "FAIL" with whatever error it could collect if it doesn't.

Use this to distinguish "meshcat HTML is broken" from "WebGL doesn't work in
this browser/GPU" when meshcat_wrapper.html renders as a blank canvas.
"""
import argparse
import os
import webbrowser


DIAGNOSTIC_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WebGL Diagnostic</title>
<style>
  body { font-family: monospace; padding: 24px; max-width: 900px; }
  h1 { font-size: 18px; }
  .ok { color: #1a7f1a; font-weight: bold; font-size: 16px; }
  .fail { color: #c00000; font-weight: bold; font-size: 16px; }
  pre { background: #f4f4f4; padding: 12px; border: 1px solid #ddd; overflow-x: auto; }
  canvas { border: 1px solid #ccc; margin-top: 8px; }
  ul { line-height: 1.6; }
</style>
</head>
<body>
<h1>WebGL Diagnostic</h1>
<div id="result">Probing&hellip;</div>
<canvas id="gl-probe" width="80" height="80"></canvas>
<div id="hints"></div>
<script>
(function() {
  const result = document.getElementById('result');
  const hints  = document.getElementById('hints');
  const canvas = document.getElementById('gl-probe');

  const attempts = [];
  function tryContext(name) {
    try {
      const ctx = canvas.getContext(name, { failIfMajorPerformanceCaveat: false });
      attempts.push({ name, ok: !!ctx });
      return ctx;
    } catch (e) {
      attempts.push({ name, ok: false, error: String(e) });
      return null;
    }
  }

  let gl = tryContext('webgl2') || tryContext('webgl') || tryContext('experimental-webgl');

  if (!gl) {
    result.innerHTML = '<p class="fail">FAIL &mdash; WebGL is not available in this browser.</p>'
      + '<pre>' + JSON.stringify(attempts, null, 2) + '</pre>';
    hints.innerHTML = '<h2>Common causes</h2><ul>'
      + '<li>Hardware acceleration disabled in browser settings.</li>'
      + '<li>GPU/driver blocklisted by the browser (check <code>chrome://gpu</code> or <code>about:support</code>).</li>'
      + '<li>Browser running in a sandbox/container without GPU access (e.g. SSH X-forwarding, some Wayland setups).</li>'
      + '<li>Outdated GPU drivers, or no GPU at all (try a software renderer / Mesa llvmpipe).</li>'
      + '</ul>';
    return;
  }

  const info = {
    contextChosen: attempts.find(a => a.ok).name,
    version:           gl.getParameter(gl.VERSION),
    shadingLanguage:   gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
    vendor:            gl.getParameter(gl.VENDOR),
    renderer:          gl.getParameter(gl.RENDERER),
    maxTextureSize:    gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxViewportDims:   gl.getParameter(gl.MAX_VIEWPORT_DIMS),
  };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  if (dbg) {
    info.unmaskedVendor   = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL);
    info.unmaskedRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
  }
  info.extensions = gl.getSupportedExtensions();

  // Try to actually draw something — context creation can succeed but draws fail.
  gl.clearColor(0.2, 0.5, 0.8, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const pixel = new Uint8Array(4);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  info.clearPixel = Array.from(pixel);
  const drewOK = pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0;

  result.innerHTML = (drewOK
    ? '<p class="ok">OK &mdash; WebGL works and a test draw succeeded.</p>'
    : '<p class="fail">PARTIAL &mdash; WebGL context exists but draw produced black pixels.</p>')
    + '<pre>' + JSON.stringify(info, null, 2) + '</pre>';
})();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true", help="Try to open the diagnostic in the default browser")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "webgl_check.html")
    with open(out, "w") as f:
        f.write(DIAGNOSTIC_HTML)
    print(f"wrote {out}")
    print(f"open file://{out} in the same browser you use for meshcat_wrapper.html")

    if args.open:
        webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    main()
