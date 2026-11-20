"""The judge. Checks claims only by simulation.

  score(design)          105 PVT points (3 process x 7 temperatures x 5 supplies) + worst |offset| over all 1,000,000
                         mismatch seeds: linear screen from 8 sensitivities, the top 1000 seeds re-simulated exactly
  attack(design, a)      an attacker's counterexamples {"points": [{"process", "temp", "vdd"}], "seeds": [int]}:
                         a point or seed counts only if it is in the specified range and the simulation fails there

  python -m debaterl.chip.judge design.json [--attack attack.json]
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from debaterl.chip import sim
from debaterl.helpers import CACHE, WORKERS

TEMPS = [-40.0, -12.5, 15.0, 42.5, 70.0, 97.5, 125.0]
VDDS = [1.7, 1.75, 1.8, 1.85, 1.9]
GRID = [(p, t, v) for p in ("TT", "FF", "SS") for t in TEMPS for v in VDDS]
POOL = ThreadPoolExecutor(WORKERS)  # ngspice runs are subprocesses, so threads are enough
ZFILE = os.path.join(CACHE, "chip", "mismatch_z.npy")


def all_z():
  """The mismatch draws of all 1,000,000 seeds, built once into the cache."""
  if not os.path.exists(ZFILE):
    print("building the 1,000,000-seed mismatch table (once)...", file=sys.stderr)
    os.makedirs(os.path.dirname(ZFILE), exist_ok=True)
    with open(ZFILE + ".part", "wb") as fh:
      np.save(fh, np.stack([sim.mismatch_z(s) for s in range(sim.N_SEEDS)]))
    os.replace(ZFILE + ".part", ZFILE)
  return np.load(ZFILE, mmap_mode="r")


def extreme(res, k, f):
  return f((r[k] for r in res if r.get(k) is not None), default=None)


def outside(r):
  return any("outside" in f for f in r["failures"])


def grid(design):
  res = list(POOL.map(lambda pt: sim.simulate(design, *pt), GRID))
  fails = [r for r in res if not r["pass"]]
  return {"grid_points": len(res), "grid_failures": len(fails),
          "first_failures": [{k: r.get(k) for k in ("process", "temp", "vdd", "failures")} for r in fails[:5]],
          "min_gain_db": extreme(res, "gain_db", min), "min_gbw_hz": extreme(res, "gbw_hz", min), "min_pm_deg": extreme(res, "pm_deg", min),
          "max_power_w": extreme(res, "power_w", max)}


def sensitivities(design, h=1e-3):
  base = sim.offset_with_shifts(design, {d: 0.0 for d in sim.DEVICES})

  def one(d):
    v = sim.offset_with_shifts(design, {k: (h if k == d else 0.0) for k in sim.DEVICES})
    return None if (v is None or base is None) else (v - base) / h
  return base, list(POOL.map(one, sim.DEVICES))


def offset_tail(design, top=1000):
  if errs := sim.check_bounds(design):
    return {"pass": False, "failures": errs}
  base, s = sensitivities(design)
  if base is None or any(v is None for v in s):
    return {"pass": False, "failures": ["offset simulation failed"]}
  sig = np.array([sim.AVT_MV_UM * 1e-3 / (design[sim.AREA_OF[d][0]] * design[sim.AREA_OF[d][1]]) ** 0.5 for d in sim.DEVICES])
  lin = base + all_z() @ (np.array(s) * sig)
  idx = np.argsort(-np.abs(lin))[:top]
  exact = list(POOL.map(lambda i: sim.offset(design, int(i)), idx))
  worst = max(exact, key=lambda r: abs(r["offset_v"]) if r["offset_v"] is not None else 1e9)
  return {"systematic_offset_v": base, "sigma_offset_v": float(np.sqrt(np.sum((np.array(s) * sig) ** 2))),
          "linear_max_abs_v": float(np.max(np.abs(lin))), "worst_seed": worst["seed"], "worst_offset_v": worst["offset_v"],
          "seeds_failing_among_top": sum(not r["pass"] for r in exact), "pass": all(r["pass"] for r in exact)}


def score(design):
  if errs := sim.check_bounds(design):
    return {"area_um2": None, "pass": False, "failures": errs}
  g, o = grid(design), offset_tail(design)
  return {"area_um2": round(sim.area_um2(design), 4), "pvt": g, "offset": o, "pass": g["grid_failures"] == 0 and o.get("pass", False)}


def attack(design, a):
  pts = [p if isinstance(p, dict) else dict(zip(("process", "temp", "vdd"), p)) for p in (a.get("points") or [])[:3]]
  seeds = (a.get("seeds") or [])[:5]
  res = list(POOL.map(lambda p: sim.simulate(design, str(p["process"]), float(p["temp"]), float(p["vdd"])), pts))
  res += list(POOL.map(lambda s: sim.offset(design, int(s)), seeds))
  return {"confirmed": [r for r in res if not r["pass"] and not outside(r)], "rejected": [r for r in res if r["pass"] or outside(r)]}


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("design")
  ap.add_argument("--attack", help="attack.json: verify these counterexamples instead of scoring the design")
  a = ap.parse_args(argv)
  d = json.load(open(a.design))
  print(json.dumps(attack(d, json.load(open(a.attack))) if a.attack else score(d), indent=1, default=float))


if __name__ == "__main__":
  main()
