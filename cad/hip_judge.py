"""The hip bracket judge: rebuilds the design, runs the scenario chain and returns pass/fail per claim, with the failing scenario.

Chain per scenario: MuJoCo (commanded exo torque, joint speeds) -> ngspice (delivered torque) -> MuJoCo again with the
delivered torque -> bracket load history -> superposition of the six CalculiX unit cases -> safety factors and seat motion;
the exo joint range goes to the hard-stop check and the cuff slide to its limit. Settings are fixed unless a scenario sets
them inside the agreed ranges.
"""
import itertools, json, math, os, time
import numpy as np
from debaterl.helpers import pmap, run_cached, scenario_motive, scenario_path, verdicts
from debaterl.cad import fem, hip_body as body, hip_drive as drive, hip_geom as geom, hip_motor as motor

SPEC = geom.SPEC
TH = SPEC["thresholds"]
TH_MOTOR = {"shortfall_max_Nm": 1.0, "winding_max_C": 110.0, "drag_max_Nm": 3.0, "ambient_C": 35.0, "duration_s": 1800.0,
            "sts_period_s": 30.0}
END_TRIM = 0.3
VERSION = "springs-ip-v3"
CASE_KEYS = {"walk": ("height_m", "speed_mps", "slope_deg", "assist_Nm", "battery_V"),
             "sts": ("height_m", "chair_height_m", "assist_Nm", "battery_V")}
LOADS = ("Fx_N", "Fy_N", "Fz_N", "Mx_Nmm", "My_Nmm", "Mz_Nmm")


def load_region(X, T, on, axis):
  """The actuator interface: the seat face out to the housing radius, and the four actuator bolt bores."""
  A, (ya, za) = geom.G["actuator"], axis
  x, y, z = X.T
  r = np.hypot(y - ya, z - za)
  load = on & (np.abs(x - A["seat_x"][1]) < 1e-4) & (r >= A["seat_r"][0] - 1e-6) & (r <= A["body_r"])
  for k in A["bolt_angles_deg"]:
    a = math.radians(k)
    rb = np.hypot(y - ya - A["bolt_pcd_r"] * math.sin(a), z - za + A["bolt_pcd_r"] * math.cos(a))
    load |= fem.expand(T, on & (np.abs(rb - A["bolt_r"]) < 1e-3))
  return load


def unit(meta, cfg=None):
  """The six unit load cases at the actuator seat centre."""
  x = geom.G["actuator"]["seat_x"][1]
  return fem.solve(meta, geom.G, SPEC["belt_clamp"], SPEC["material"], load_region, [x, *meta["axis"]], x, VERSION, cfg)


def check_scenario(sc):
  """Returns an error string for an out-of-range or malformed scenario, else None."""
  if not isinstance(sc, dict):
    return "a scenario must be a JSON object"
  act = sc.get("activity")
  if act not in CASE_KEYS:
    return f"activity must be one of {list(CASE_KEYS)}"
  rng = SPEC["scenarios"][act]
  for k in CASE_KEYS[act]:
    if not isinstance(sc.get(k), (int, float)):
      return f"missing number {k}"
    if not rng[k][0] <= sc[k] <= rng[k][1]:
      return f"{k}={sc[k]} outside {rng[k]}"
  extra = set(sc) - set(CASE_KEYS[act]) - {"activity", "settings"}
  if extra:
    return f"unknown fields {sorted(extra)}"
  st = sc.get("settings", {}) or {}
  if set(st) - {"mesh", "mujoco_timestep_s", "spice_step_s"}:
    return f"unknown settings {sorted(set(st) - {'mesh', 'mujoco_timestep_s', 'spice_step_s'})}"
  ms = SPEC["model_settings"]
  if "mujoco_timestep_s" in st and not ms["mujoco_timestep_s"]["range"][0] <= st["mujoco_timestep_s"] <= ms["mujoco_timestep_s"]["range"][1]:
    return "mujoco_timestep_s outside range"
  if "spice_step_s" in st and not ms["spice_step_s"]["range"][0] <= st["spice_step_s"] <= ms["spice_step_s"]["range"][1]:
    return "spice_step_s outside range"
  if "mesh" in st:
    try:
      if not fem.mesh_ok(fem.mesh_cfg(st["mesh"])):
        return "mesh settings outside range"
    except (TypeError, KeyError, ValueError):
      return "malformed mesh settings"
  return None


def window(o, act):
  return (o["t"] >= body.SETTLE) & (o["t"] <= body.DURATION[act] - END_TRIM)


def run_scenario(meta, info, sc, act_spec=None):
  """Runs one in-range scenario on a built design; returns the claim verdicts and the numbers behind them.
  With an actuator spec, the actuator's mass, reflected inertia, friction, torque cap and winding drive the run, and
  the actuator claims M2-M4 are added."""
  t0 = time.time()
  st = sc.get("settings", {}) or {}
  act = sc["activity"]
  A = motor.derive(act_spec) if act_spec else None
  mj = {"timestep": st.get("mujoco_timestep_s", body.SETTINGS_DEFAULT["timestep"])}
  o1 = body.simulate(sc, meta["axis"], mj, act=A)
  dr = drive.deliver(o1["t"], o1["tau_cmd"], o1["exo_vel"], sc["battery_V"], st.get("spice_step_s", drive.STEP_DEFAULT), act=A)
  o = body.simulate(sc, meta["axis"], mj, tau_override=(dr["t"], dr["tau"]), act=A)
  k = window(o, act)
  W = o["W"][k]
  s = fem.sample(W)
  ev = fem.evaluate(unit(meta, st.get("mesh")), W[s], meta["build_dir"], SPEC["material"])
  tw = o["t"][k][s]
  for key in ("vm_t_index", "interlayer_t_index"):
    ev[key.replace("_index", "_s")] = None if ev[key] is None else round(float(tw[ev[key]]), 3)
    ev.pop(key)
  lo, hi = float(np.degrees(o["exo"][k].min())), float(np.degrees(o["exo"][k].max()))
  slide = float(np.abs(o["slide"][k]).max() * 1000)
  res = {
    "K1_rom": geom.rom(info, lo, hi),
    "K2_cuff_slide": slide <= TH["slide_max_mm"],
    "S1_strength": ev["sf_inplane"] >= TH["sf_min"] and ev["sf_interlayer"] >= TH["sf_min"],
    "S3_stiffness": ev["seat_deflection_mm"] <= TH["seat_deflection_max_mm"] and ev["seat_tilt_deg"] <= TH["seat_tilt_max_deg"],
  }
  extra = {}
  if A:
    k1 = window(o1, act)
    short = float(np.max(np.abs(o1["tau_dem"][k1]) - np.abs(dr["tau"][k1])))
    p_cu = float(np.mean(dr["i"][k1] ** 2) * A["R"])
    if act == "sts":
      p_cu *= (o1["t"][k1][-1] - o1["t"][k1][0]) / TH_MOTOR["sts_period_s"]
    t_wind = TH_MOTOR["ambient_C"] + p_cu * A["R_th"] * (1 - np.exp(-TH_MOTOR["duration_s"] / (A["R_th"] * A["C_th"])))
    alpha = np.gradient(o["exo_vel"], o["t"])
    drag = float(np.max(np.abs(A["J_reflected"] * alpha[k]) + A["friction_Nm"]))
    res["M2_torque"] = short <= TH_MOTOR["shortfall_max_Nm"]
    res["M3_thermal"] = t_wind <= TH_MOTOR["winding_max_C"]
    res["M4_drag"] = drag <= TH_MOTOR["drag_max_Nm"] if act == "walk" else True
    extra = {"torque_shortfall_Nm": round(short, 3), "copper_loss_W": round(p_cu, 2), "winding_C": round(float(t_wind), 1),
             "drag_peak_Nm": round(drag, 3), "actuator_mass_g": round(A["mass_kg"] * 1000, 1), "tau_cap_Nm": round(A["tau_cap"], 2)}
  return {"scenario": sc, "claims": res, "fails": not all(res.values()), **extra,
          "exo_joint_range_deg": [round(lo, 2), round(hi, 2)], "stops_deg": info["stops_deg"], "cuff_slide_max_mm": round(slide, 2),
          **ev, "peak_abs_load": {n: round(float(np.abs(W[:, i]).max()), 2) for i, n in enumerate(LOADS)},
          "exo_torque_peak_Nm": round(float(np.abs(o["tau"][k]).max()), 3), "drive_shortfall_Nm": round(dr["shortfall_Nm"], 3),
          "drive_current_peak_A": round(dr["i_peak_A"], 2), "seconds": round(time.time() - t0, 1)}


def _unit(args):
  unit(*args)


def _one(args):
  meta, info, sc, act_spec = args
  tag = VERSION + (json.dumps(act_spec, sort_keys=True) + motor.VERSION if act_spec else "")
  return run_cached(scenario_path(meta, sc, tag), run_scenario, sc, meta, info, sc, act_spec)


def run_many(meta, info, scenarios, act_spec=None):
  meshes = sorted({json.dumps((s.get("settings") or {}).get("mesh"), sort_keys=True) for s in scenarios})
  pmap(_unit, [(meta, json.loads(m)) for m in meshes], len(meshes))
  return pmap(_one, [(meta, info, s, act_spec) for s in scenarios])


def grid():
  out = []
  for act in ("walk", "sts"):
    g = SPEC["grid"][act]
    for vals in itertools.product(*[g[k] for k in CASE_KEYS[act]]):
      out.append({"activity": act, **dict(zip(CASE_KEYS[act], vals))})
  return out


def worst(rs):
  ok = [r for r in rs if r.get("error") is None]
  if not ok:
    return {}

  def pick(key, f):
    r = f(ok, key=lambda r: r[key])
    return {"scenario": r["scenario"], key: r[key]}
  r = max(ok, key=lambda r: r["exo_joint_range_deg"][1])
  return {"sf_inplane": pick("sf_inplane", min), "sf_interlayer": pick("sf_interlayer", min),
          "seat_deflection_mm": pick("seat_deflection_mm", max), "seat_tilt_deg": pick("seat_tilt_deg", max),
          "cuff_slide_max_mm": pick("cuff_slide_max_mm", max),
          "exo_joint_max_deg": {"scenario": r["scenario"], "exo_joint_range_deg": r["exo_joint_range_deg"]}}


def brief(r):
  keep = ("scenario", "claims", "sf_inplane", "sf_interlayer", "vm_at_mm", "interlayer_at_mm", "seat_deflection_mm",
          "seat_tilt_deg", "exo_joint_range_deg", "stops_deg", "cuff_slide_max_mm")
  return {k: r[k] for k in keep if k in r}


def geometry(path):
  """Scenario-free claims only (fast)."""
  meta = geom.build(path)
  checks, info = geom.check(meta)
  return {"design": os.path.basename(path), "sha": meta["sha"], "axis": meta.get("axis"), "geometry": checks, **info}


def check(path, scenarios=None):
  """Every claim: geometry, then the judge's grid (or the given scenarios)."""
  meta = geom.build(path)
  checks, info = geom.check(meta)
  out = {"design": os.path.basename(path), "sha": meta["sha"], "axis": meta.get("axis"), "build_dir": meta.get("build_dir"),
         "geometry": checks, **info}
  if not checks.get("G1_solid"):
    out["pass"] = False
    return out
  rs = run_many(meta, info, scenarios if scenarios is not None else grid())
  failing = [r for r in rs if r.get("fails")]
  errors = [r for r in rs if r.get("error")]
  claims = dict(checks)
  for c in ("K1_rom", "K2_cuff_slide", "S1_strength", "S3_stiffness"):
    claims[c] = all(r["claims"][c] for r in rs if r.get("error") is None)
  out.update({"claims": claims, "pass": all(claims.values()) and not errors, "n_scenarios": len(rs),
              "n_failing": len(failing), "failing": [brief(r) for r in failing[:10]], "errors": errors[:5], "worst": worst(rs)})
  return out


def run(path, scenarios):
  """Runs NO's scenarios on a design (an attack). Out-of-range scenarios are rejected; the rest are run and reported."""
  meta = geom.build(path)
  checks, info = geom.check(meta)
  if not checks.get("G1_solid"):
    return {"error": "design does not build", "geometry": checks}
  bad = [{"scenario": s, "rejected": check_scenario(s)} for s in scenarios if check_scenario(s)]
  good = [s for s in scenarios if not check_scenario(s)]
  rs = run_many(meta, info, good) if good else []
  v = verdicts(rs, brief)
  return {"design": os.path.basename(path), "sha": meta["sha"], "confirmed": v["confirmed"], "not_confirmed": v["not_confirmed"],
          "errors": v["errors"], "rejected": bad}


def motive(base, path, m):
  """Two-sided test: the named claim instance must fail on the baseline and hold on the change."""
  if not isinstance(m, dict):
    return {"motive": m, "rejected": "a motive must be a JSON object", "holds": False}
  if m.get("kind") == "mass":
    mb, md = (geom.check(geom.build(p))[1].get("mass_g") for p in (base, path))
    holds = md is not None and mb is not None and md <= m["target_g"] < mb
    return {"motive": m, "baseline_mass_g": mb, "design_mass_g": md, "holds": holds}
  if m.get("kind") == "geometry":
    cb, cd = (geom.check(geom.build(p))[0] for p in (base, path))
    c = m.get("claim")
    return {"motive": m, "baseline": cb.get(c), "design": cd.get(c), "holds": cb.get(c) is False and cd.get(c) is True}
  if m.get("kind") == "scenario":
    return scenario_motive(m, check_scenario, run, base, path)
  return {"motive": m, "rejected": "kind must be mass, geometry or scenario", "holds": False}
