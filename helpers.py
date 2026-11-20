"""Shared plumbing: settings, the design build cache, B-rep helpers, normative gait curves, cached parallel scenario runs."""
import functools, hashlib, json, os, subprocess, sys, traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np

CACHE = os.path.expanduser(os.environ.get("RESIPS_CACHE", "~/.cache/resips"))
WORKERS = int(os.environ.get("RESIPS_WORKERS", os.cpu_count() or 1))


def save(path, obj):
  json.dump(obj, open(path, "w"), indent=1, default=float)


def build(path, script, cache, timeout=600):
  """Runs a design file through script in a fresh process (design files are untrusted code); cached by the file's hash."""
  src = open(path, "rb").read()
  sha = hashlib.sha256(src).hexdigest()[:16]
  d = os.path.join(cache, sha)
  meta = os.path.join(d, "meta.json")
  os.makedirs(d, exist_ok=True)
  if not os.path.exists(meta):
    open(os.path.join(d, "design.py"), "wb").write(src)
    try:
      subprocess.run([sys.executable, "-c", script, os.path.join(d, "design.py"), d], cwd=d, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
      json.dump({"error": f"build() took longer than {timeout} s"}, open(meta, "w"))
    if not os.path.exists(meta):
      json.dump({"error": "build process crashed"}, open(meta, "w"))
  return {**json.load(open(meta)), "dir": d, "sha": sha}


def xcyl(r, x0, x1, y=0.0, z=0.0):
  from build123d import Cylinder, Pos, Rot
  return Pos((x0 + x1) / 2, y, z) * Rot(0, 90, 0) * Cylinder(r, x1 - x0)


def box(x, y, z):
  from build123d import Box, Pos
  return Pos(sum(x) / 2, sum(y) / 2, sum(z) / 2) * Box(x[1] - x[0], y[1] - y[0], z[1] - z[0])


def vol(s):
  if s is None:
    return 0.0
  try:
    return sum(v.volume for v in s.solids())
  except (AttributeError, ValueError):
    return 0.0


# normative sagittal joint angles over a gait cycle: (percent of cycle, degrees)
HIP = [(0, 30), (10, 26), (25, 10), (40, -3), (52, -10), (60, -4), (70, 12), (80, 26), (88, 32), (95, 31)]
KNEE = [(0, 3), (8, 12), (15, 18), (25, 12), (40, 4), (50, 8), (60, 35), (70, 58), (75, 62), (82, 50), (90, 22), (96, 6)]
ANKLE = [(0, 0), (6, -5), (15, 2), (40, 10), (50, 8), (58, -8), (63, -15), (70, -8), (80, 0), (90, 2)]


@functools.cache
def gait():
  from scipy.interpolate import CubicSpline

  def periodic(points):
    x = np.array([p[0] for p in points] + [100.0]) / 100.0
    y = np.radians([p[1] for p in points] + [points[0][1]])
    return CubicSpline(x, y, bc_type="periodic")
  return {"hip": periodic(HIP), "knee": periodic(KNEE), "ankle": periodic(ANKLE)}


def minjerk(t, t0, t1, a, b):
  x = np.clip((t - t0) / (t1 - t0), 0, 1)
  return a + (b - a) * (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5)


def pmap(fn, jobs, workers=None):
  if len(jobs) <= 1:
    return [fn(j) for j in jobs]
  with ProcessPoolExecutor(min(workers or WORKERS, len(jobs))) as ex:
    return list(ex.map(fn, jobs))


def scenario_path(meta, sc, tag):
  d = os.path.join(meta["dir"], "scenarios")
  os.makedirs(d, exist_ok=True)
  return os.path.join(d, hashlib.sha256((json.dumps(sc, sort_keys=True) + tag).encode()).hexdigest()[:16] + ".json")


def run_cached(path, fn, sc, *args):
  """fn(*args) for scenario sc, cached at path; an exception becomes an error result (not cached)."""
  if os.path.exists(path):
    return json.load(open(path))
  try:
    r = fn(*args)
  except Exception as e:
    r = {"scenario": sc, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:], "fails": None}
  if r.get("error") is None:
    json.dump(r, open(path, "w"), default=float)
  return r


def verdicts(rs, brief):
  return {"confirmed": [brief(r) for r in rs if r.get("fails")], "not_confirmed": [brief(r) for r in rs if r.get("fails") is False],
          "errors": [r for r in rs if r.get("error")]}


def scenario_motive(m, check_scenario, attack, base, path, full=True):
  """Two-sided test with a named scenario: it must fail on the baseline and pass on the change."""
  err = check_scenario(m["scenario"])
  if err:
    return {"motive": m, "rejected": err, "holds": False}
  rb, rd = attack(base, [m["scenario"]]), attack(path, [m["scenario"]])
  out = {"motive": m, "baseline_fails": bool(rb.get("confirmed")), "design_passes": bool(rd.get("not_confirmed"))}
  if full:
    out.update(baseline=(rb.get("confirmed") or rb.get("not_confirmed") or [None])[0],
               design=(rd.get("confirmed") or rd.get("not_confirmed") or [None])[0])
  return {**out, "holds": out["baseline_fails"] and out["design_passes"]}
