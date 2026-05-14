"""Helpers for working with meshcat from Python-only code.

The auto-opened meshcat viewer shows a blank canvas if the browser can't
create a WebGL context, and there is no public hook to inject custom JS into
the page meshcat-server serves. ``open_meshcat_with_webgl_probe`` works
around that by writing a thin local wrapper HTML that iframes the meshcat
URL and runs a WebGL probe in the outer page; if WebGL is unavailable the
wrapper renders a visible banner above the iframe instead of leaving the
user staring at an empty viewer.
"""
import os
import tempfile
import webbrowser


WRAPPER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Meshcat (with WebGL probe)</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; font-family: monospace; }
  #banner { display: none; padding: 14px; background: #fff3cd; color: #8a4b00;
            border-bottom: 2px solid #c00; font-size: 14px; }
  #frame  { display: block; width: 100vw; height: 100vh; border: 0; }
</style>
</head>
<body>
<div id="banner"></div>
<iframe id="frame" src="__MESHCAT_URL__"></iframe>
<script>
(function() {
  var gl = null;
  try {
    var c = document.createElement('canvas');
    gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
  } catch (e) {}
  if (gl) return;
  var b = document.getElementById('banner');
  b.style.display = 'block';
  b.innerHTML = '<b>WebGL is not available in this browser.</b> '
    + 'Meshcat (loaded below) cannot render the scene. '
    + 'Check <code>chrome://gpu</code> / <code>about:support</code> for details.';
  document.getElementById('frame').style.height = 'calc(100vh - 60px)';
})();
</script>
</body>
</html>
"""


def write_meshcat_wrapper(meshcat_url, out_path=None):
    """Write a wrapper HTML that iframes ``meshcat_url`` and runs a WebGL probe.

    If ``out_path`` is None, a temporary file is created.
    Returns the absolute path to the written wrapper.
    """
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_meshcat_wrapper.html")
        os.close(fd)
    with open(out_path, "w") as f:
        f.write(WRAPPER_TEMPLATE.replace("__MESHCAT_URL__", meshcat_url))
    return os.path.abspath(out_path)


def open_meshcat_with_webgl_probe(meshcat_url, out_path=None):
    """Write the wrapper and open it in the default browser.

    Returns the absolute path to the wrapper file.
    """
    path = write_meshcat_wrapper(meshcat_url, out_path)
    webbrowser.open("file://" + path)
    return path
