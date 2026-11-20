"""Exact B-rep checks (OpenCASCADE via build123d): solid validity, printer fit, keep-outs, interfaces, hard stops, mass."""
import json, math, os
from build123d import Pos, Rot, import_step
from debaterl import helpers
from debaterl.helpers import box, vol, xcyl

SPEC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_claims.json")))
G = SPEC["geometry"]
TOL = G["volume_tol_mm3"]

BUILD = r"""
import json, runpy, sys
from build123d import export_step
src, out = sys.argv[1], sys.argv[2]
meta = {}
try:
    ns = runpy.run_path(src)
    part = ns["build"]()
    solids = part.solids()
    meta = {"axis": [float(v) for v in ns["AXIS"]], "build_dir": str(ns["BUILD_DIR"]), "n_solids": len(solids),
            "valid": bool(part.is_valid), "volume_mm3": float(part.volume)}
    export_step(part, out + "/part.step")
except Exception as e:
    meta = {"error": f"{type(e).__name__}: {e}"}
json.dump(meta, open(out + "/meta.json", "w"))
"""


def build(path, timeout=600):
  return helpers.build(path, BUILD, os.path.join(helpers.CACHE, "hip"), timeout)


def link(theta, axis):
  L = G["link"]
  x0, x1 = L["slab_x"]
  shape = xcyl(L["hub_r"], x0, x1) + box((x0, x1), (-L["bar_width"] / 2, L["bar_width"] / 2), (-L["bar_length"], 0))
  return Pos(0, axis[0], axis[1]) * Rot(theta, 0, 0) * shape


def load(meta):
  return import_step(os.path.join(meta["dir"], "part.step")).solids()[0]


def stop_angle(slab, axis, sign, step=2.0, limit=180.0, tol_deg=0.01):
  """First angle (deg) from 0 in direction sign at which the link touches the bracket; None if it never does."""
  def hit(th):
    return vol(slab & link(th, axis)) > TOL
  prev = 0.0
  th = step
  while th <= limit + 1e-9:
    if hit(sign * th):
      lo, hi = prev, th
      while hi - lo > tol_deg:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if hit(sign * mid) else (mid, hi)
      return round(sign * lo, 3)
    prev, th = th, th + step
  return None


def check(meta):
  """Scenario-free claims G1..G7. Returns (results, info); info carries the stop angles K1 needs."""
  r, info = {}, {}
  if "error" in meta:
    return {"G1_solid": False}, {"error": meta["error"]}
  r["G1_solid"] = meta["n_solids"] == 1 and meta["valid"] and meta["volume_mm3"] > 0
  if not r["G1_solid"]:
    return r, {"error": f"{meta['n_solids']} solids, valid={meta['valid']}"}
  p = load(meta)
  ya, za = axis = meta["axis"]
  bb = p.bounding_box()
  size = {"x": bb.size.X, "y": bb.size.Y, "z": bb.size.Z}
  bd = meta["build_dir"]
  if bd not in ("+x", "-x", "+y", "-y", "+z", "-z"):
    r["G2_printer"] = False
  else:
    up = size[bd[1]]
    flat = sorted(v for k, v in size.items() if k != bd[1])
    bed = sorted(SPEC["printer_bed_mm"]["xy"])
    r["G2_printer"] = up <= SPEC["printer_bed_mm"]["z"] and flat[0] <= bed[0] and flat[1] <= bed[1]
  info["bbox_mm"] = {k: round(v, 2) for k, v in size.items()}

  E, K, A, L = G["envelope"], G["keepout"], G["actuator"], G["link"]
  out = {}
  out["envelope"] = vol(p - box(E["x"], E["y"], E["z"]))
  out["body"] = vol(p & box((-1000, K["body_x_below"]), (-1000, 1000), (-1000, 1000)))
  bp = K["belt_plate"]
  out["belt_plate"] = vol(p & box((bp["x"][0], bp["x"][1] - 0.05), bp["y"], bp["z"]))
  out["actuator"] = vol(p & xcyl(A["body_r"], A["body_x"][0] + 0.05, A["body_x"][1], ya, za))
  out["link_hub"] = vol(p & xcyl(L["hub_r"], L["slab_x"][0], L["slab_x"][1], ya, za))
  r["G3_keepout"] = all(v <= TOL for v in out.values())
  info["keepout_violation_mm3"] = {k: round(v, 3) for k, v in out.items() if v > TOL}

  B, bad = G["belt_bolts"], {}
  for y, z in B["yz"]:
    bad[f"belt_bore_{y:+.0f}"] = vol(p & xcyl(B["bore_r"], *B["bore_x"], y, z))
    bad[f"belt_seat_{y:+.0f}"] = vol((xcyl(B["seat_r"][1], *B["seat_x"], y, z) - xcyl(B["seat_r"][0], *B["seat_x"], y, z)) - p)
  seat = xcyl(A["seat_r"][1], *A["seat_x"], ya, za) - xcyl(A["seat_r"][0], *A["seat_x"], ya, za)
  for k in A["bolt_angles_deg"]:
    a = math.radians(k)
    yb, zb = ya + A["bolt_pcd_r"] * math.sin(a), za - A["bolt_pcd_r"] * math.cos(a)
    bad[f"act_bore_{k}"] = vol(p & xcyl(A["bolt_r"], *A["bolt_x"], yb, zb))
    seat -= xcyl(A["bolt_r"], *A["bolt_x"], yb, zb)
  bad["act_seat"] = vol(seat - p)
  bad["shaft"] = vol(p & xcyl(A["shaft_r"], *A["shaft_x"], ya, za))
  r["G4_interfaces"] = all(v <= TOL for v in bad.values())
  info["interface_violation_mm3"] = {k: round(v, 3) for k, v in bad.items() if v > TOL}

  ab = G["axis_bounds"]
  r["G5_axis"] = ab["y"][0] <= ya <= ab["y"][1] and ab["z"][0] <= za <= ab["z"][1]

  slab = p & box(L["slab_x"], (-1000, 1000), (-1000, 1000))
  free0 = vol(slab & link(0.0, axis)) <= TOL
  flex = stop_angle(slab, axis, +1) if free0 else 0.0
  ext = stop_angle(slab, axis, -1) if free0 else 0.0
  S = G["stops"]
  r["G6_stops"] = free0 and flex is not None and ext is not None and flex <= S["flexion_stop_max_deg"] and ext >= S["extension_stop_min_deg"]
  info["stops_deg"] = {"flexion": flex, "extension": ext}

  mass = p.volume * SPEC["material"]["density_g_cm3"] / 1000.0
  r["G7_mass"] = mass <= G["mass_max_g"]
  info["mass_g"] = round(mass, 3)
  info["volume_mm3"] = round(p.volume, 2)
  return r, info


def rom(info, theta_min, theta_max):
  s, m = info["stops_deg"], G["stops"]["rom_margin_deg"]
  if s["flexion"] is None or s["extension"] is None:
    return False
  return s["flexion"] > theta_max + m and s["extension"] < theta_min - m
