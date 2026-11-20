"""Judge for the actuator debate: the round-3 bracket is fixed; the design is an actuator spec (five numbers, see hip_motor.py).

Claims: M1 the actuator fits its keep-out (outer diameter <= 92 mm, length <= 48 mm); and for every scenario on the grid:
M2 the drive delivers the controller's demanded torque within 1 N*m at every instant (ngspice), M3 the winding stays at or
below 110 C after 30 minutes of the activity at 35 C ambient (walking continuously; sit-to-stand once every 30 s), M4 while
walking the actuator's drag at the hip (reflected rotor inertia times link acceleration, plus gear friction) stays at or
below 3 N*m, and the bracket keeps its claims K1, K2, S1, S3 under the actuator's mass, inertia and torque.
Objective: the lightest actuator that passes.
"""
import json, os
from resips.helpers import scenario_motive, verdicts
from resips.cad import hip_geom as geom, hip_judge as judge, hip_motor as motor

BRACKET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_bracket.py")
CLAIMS = ("M2_torque", "M3_thermal", "M4_drag", "K1_rom", "K2_cuff_slide", "S1_strength", "S3_stiffness")


def bracket():
  meta = geom.build(BRACKET)
  checks, info = geom.check(meta)
  return meta, info


def envelope(A):
  return A["od_mm"] <= motor.ENVELOPE["od_mm"] and A["length_mm"] <= motor.ENVELOPE["length_mm"]


def brief(r):
  keep = ("scenario", "claims", "torque_shortfall_Nm", "winding_C", "drag_peak_Nm", "sf_inplane", "sf_interlayer",
          "seat_deflection_mm", "seat_tilt_deg", "cuff_slide_max_mm")
  return {k: r[k] for k in keep if k in r}


def check(path, scenarios=None):
  spec = json.load(open(path))
  err = motor.check(spec)
  if err:
    return {"pass": False, "error": err}
  A = motor.derive(spec)
  meta, info = bracket()
  rs = judge.run_many(meta, info, scenarios if scenarios is not None else judge.grid(), act_spec=spec)
  ok = [r for r in rs if r.get("error") is None]
  claims = {"M1_envelope": envelope(A)}
  for c in CLAIMS:
    claims[c] = all(r["claims"][c] for r in ok)
  out = {"spec": spec, "actuator": {k: round(v, 5) if isinstance(v, float) else v for k, v in A.items()},
         "mass_g": round(A["mass_kg"] * 1000, 1), "claims": claims, "pass": all(claims.values()) and len(ok) == len(rs),
         "n_scenarios": len(rs), "n_failing": sum(1 for r in ok if r["fails"]), "errors": [r for r in rs if r.get("error")][:3],
         "failing": [brief(r) for r in ok if r["fails"]][:10]}
  if ok:
    out["worst"] = {k: brief(f(ok, key=lambda r: r[k])) for k, f in (("torque_shortfall_Nm", max), ("winding_C", max), ("drag_peak_Nm", max),
                                                                      ("sf_inplane", min), ("sf_interlayer", min), ("seat_deflection_mm", max))}
  return out


def run(path, scenarios):
  spec = json.load(open(path))
  err = motor.check(spec)
  if err:
    return {"error": err}
  meta, info = bracket()
  bad = [{"scenario": s, "rejected": judge.check_scenario(s)} for s in scenarios if judge.check_scenario(s)]
  good = [s for s in scenarios if not judge.check_scenario(s)]
  return {**verdicts(judge.run_many(meta, info, good, act_spec=spec) if good else [], brief), "rejected": bad}


def motive(base, path, m):
  if not isinstance(m, dict):
    return {"motive": m, "rejected": "a motive must be a JSON object", "holds": False}
  if m.get("kind") == "mass":
    mb, md = (motor.derive(json.load(open(p)))["mass_kg"] * 1000 for p in (base, path))
    return {"motive": m, "baseline_mass_g": round(mb, 1), "design_mass_g": round(md, 1), "holds": md <= m["target_g"] < mb}
  if m.get("kind") == "scenario":
    return scenario_motive(m, judge.check_scenario, run, base, path, full=False)
  return {"motive": m, "rejected": "kind must be mass or scenario", "holds": False}
