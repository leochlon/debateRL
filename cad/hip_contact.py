"""Exact push-only belt clamp at a bracket's worst instants: the face inside the belt-plate footprint (outline cut into
the mesh) rests on the claims' spring bed, but a spring that would pull is released, and the solve repeats until the set of
touching nodes is stable. Two treatments of a lifted node's tangential springs bracket the friction question: "stick"
keeps them (shear carried even where the face has lifted, the stiff bound) and "free" releases them with the normal spring.

  debaterl hip contact design.py [scenarios.json] [--actuator actuator.json] [--variants free,stick] [--export out.json]

Without scenarios it runs the three hardest moves below. With an actuator spec, the MuJoCo and ngspice runs carry that
actuator (hip_motor.py) instead of the bracket debate's default. Model settings in a scenario are not used: the check runs the
default mesh, MuJoCo step and ngspice step. "pass" is judged on the "free" variant: seat motion <= 1.0 mm, tilt <= 1.5 deg,
both safety factors >= 2.0 at every checked instant.
"""
import argparse, base64, json, math, multiprocessing, os, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from debaterl.cad import fem, hip_body as body, hip_drive as drive, hip_geom as geom, hip_judge as judge, hip_motor as motor

SC = [
  ("uphill fast", {"activity": "walk", "height_m": 1.5, "speed_mps": 1.8, "slope_deg": 10, "assist_Nm": 20, "battery_V": 19.8}),
  ("downhill fast", {"activity": "walk", "height_m": 1.5, "speed_mps": 1.8, "slope_deg": -10, "assist_Nm": 20, "battery_V": 25.2}),
  ("stand up, low chair", {"activity": "sts", "height_m": 1.5, "chair_height_m": 0.38, "assist_Nm": 20, "battery_V": 25.2}),
]
ITERS, TOL = 30, 1e-7
C, MAT = geom.SPEC["belt_clamp"], geom.SPEC["material"]


def imprint(gmsh):
  """Cut the belt-plate footprint's outline into the back face, so the support ends on mesh edges."""
  occ = gmsh.model.occ
  bp = geom.G["keepout"]["belt_plate"]
  (y0, y1), (z0, z1) = bp["y"], bp["z"]
  r = occ.addRectangle(0, 0, 0, y1 - y0, z1 - z0)
  occ.rotate([(2, r)], 0, 0, 0, 1, 1, 1, 2 * math.pi / 3)
  occ.translate([(2, r)], 0, y0, z0)
  occ.fragment(occ.getEntities(3), [(2, r)])
  occ.synchronize()
  vols = gmsh.model.getEntities(3)
  bnd = {t for _, t in gmsh.model.getBoundary(vols, combined=False, oriented=False)}
  extra = [(2, t) for _, t in gmsh.model.getEntities(2) if t not in bnd]
  if extra:
    occ.remove(extra, recursive=True)
    occ.synchronize()


def setup(meta):
  X, E, T = fem.mesh(os.path.join(meta["dir"], "part.step"), fem.mesh_cfg(None), imprint)
  E = fem.abaqus_order(X, E)
  on = np.zeros(len(X), bool)
  on[T.ravel()] = True
  x, y, z = X.T
  bp, e = geom.G["keepout"]["belt_plate"], 1e-6
  face = on & (np.abs(x) < 1e-4) & (y >= bp["y"][0] - e) & (y <= bp["y"][1] + e) & (z >= bp["z"][0] - e) & (z <= bp["z"][1] + e)
  B = geom.G["belt_bolts"]
  bores = [fem.expand(T, on & (np.abs(np.hypot(y - yb, z - zb) - B["bore_r"]) < 1e-3)) for yb, zb in B["yz"]]
  load = judge.load_region(X, T, on, meta["axis"])
  w = fem.area_lumped(X, T, face)
  fi = np.flatnonzero(w > 0)
  bolts = []
  for b in bores:
    wb = fem.area_lumped(X, T, b)
    bolts += [(int(i), d, C["k_bolt_N_per_mm"] * wb[i] / wb.sum()) for i in np.flatnonzero(wb > 0) for d in (1, 2, 3)]
  tw = fem.tri_weights(X, T)
  tw[~load] = 0.0
  ref = np.array([geom.G["actuator"]["seat_x"][1], *meta["axis"]])
  snodes, inv = np.unique(T[:, :3], return_inverse=True)
  return {"X": X, "E": E, "fi": fi, "wf": w[fi], "bolts": bolts, "forces": np.array(fem.coupling_forces(X, tw, ref)),
          "seat": np.flatnonzero(load & (np.abs(X[:, 0] - ref[0]) < 1e-4)), "ref": ref, "snodes": snodes, "stri": inv.reshape(-1, 3),
          "nn": fem.layer_normal(meta["build_dir"])}


def springs(m, normal, tang):
  out = list(m["bolts"])
  for i, a, n, t in zip(m["fi"], m["wf"], normal, tang):
    if n:
      out.append((int(i), 1, C["k_normal_N_per_mm3"] * a))
    if t:
      out += [(int(i), 2, C["k_tangent_N_per_mm3"] * a), (int(i), 3, C["k_tangent_N_per_mm3"] * a)]
  return out


def ccx(m, f, normal, tang, wd):
  fem.write_inp(os.path.join(wd, "c.inp"), m["X"], m["E"], np.zeros(len(m["X"]), bool), m["seat"], f, MAT, springs(m, normal, tang), face=m["fi"])
  p = subprocess.run(["ccx", "-i", "c"], cwd=wd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
  if p.returncode != 0:
    raise RuntimeError(p.stdout[-1500:])
  return fem.read_dat(os.path.join(wd, "c.dat"), len(m["X"]), len(m["E"]))


def contact(job):
  m, w, variant, start = job
  f = np.tensordot(w, m["forces"], axes=1)
  act, seen = start.copy(), []
  wd = tempfile.mkdtemp(prefix="cc_")
  try:
    for it in range(1, ITERS + 1):
      U, S = ccx(m, f, act, act if variant == "free" else np.ones(len(act), bool), wd)
      ux = U[m["fi"], 0]
      pull, press = act & (ux > TOL), ~act & (ux < -TOL)
      done = not pull.any() and not press.any()
      seen.append(act.copy())
      nxt = act & ~pull if pull.any() else act | press
      if done or it == ITERS or any(np.array_equal(nxt, a) for a in seen):
        break
      act = nxt
  finally:
    shutil.rmtree(wd, ignore_errors=True)
  rig = fem.rigid_fit(m["X"][m["seat"]], [U[m["seat"]]], m["ref"])[0]
  vm, sn = fem.von_mises(S), S @ m["nn"]
  surf = []
  for v in (vm, sn):
    a = np.zeros(len(m["X"]))
    for c in range(4):
      np.maximum.at(a, m["E"][:, c], v[:, c])
    surf.append(a[m["snodes"]].astype(np.float32))
  return {"_surf": surf, "variant": variant, "converged": bool(done), "iters": it, "touching": round(float(m["wf"][act].sum() / m["wf"].sum()), 3),
          "seat_deflection_mm": round(float(np.linalg.norm(rig[:3])), 3), "seat_tilt_deg": round(float(np.degrees(np.hypot(rig[4], rig[5]))), 3),
          "sf_inplane": round(MAT["strength_inplane_MPa"] / float(vm.max()), 3),
          "sf_interlayer": round(MAT["strength_interlayer_MPa"] / float(sn.max()), 3)}


def contact_unit(job):
  m, k = job
  wd = tempfile.mkdtemp(prefix="cu_")
  try:
    U, _ = ccx(m, m["forces"][k], np.ones(len(m["fi"]), bool), np.ones(len(m["fi"]), bool), wd)
  finally:
    shutil.rmtree(wd, ignore_errors=True)
  return U[m["fi"]], U[m["seat"]]


def check(path, scenarios=None, act_path=None, variants=("free",), export=None):
  A = motor.derive(json.load(open(act_path))) if act_path else None
  todo = SC if scenarios is None else [(f"scenario {k + 1}", {kk: v for kk, v in sc.items() if kk != "settings"}) for k, sc in enumerate(scenarios)]
  bad = [(n, judge.check_scenario(sc)) for n, sc in todo if judge.check_scenario(sc)]
  if bad:
    return {"error": "out-of-range scenarios", "rejected": bad}
  meta = geom.build(path)
  checks, info = geom.check(meta)
  m = setup(meta)
  spawn = multiprocessing.get_context("spawn")
  with ProcessPoolExecutor(6, mp_context=spawn) as ex:
    unit = list(ex.map(contact_unit, [(m, k) for k in range(6)]))
  Uf = np.array([u[0] for u in unit])
  Us = np.array([u[1] for u in unit])
  rig = fem.rigid_fit(m["X"][m["seat"]], Us, m["ref"])
  jobs, labels = [], []
  for name, sc in todo:
    o1 = body.simulate(sc, meta["axis"], act=A)
    dr = drive.deliver(o1["t"], o1["tau_cmd"], o1["exo_vel"], sc["battery_V"], act=A)
    o = body.simulate(sc, meta["axis"], tau_override=(dr["t"], dr["tau"]), act=A)
    W = o["W"][judge.window(o, sc["activity"])]
    W = W[fem.sample(W)]
    ux = W @ Uf[:, :, 0]
    pull = (np.clip(ux, 0, None) * m["wf"] * C["k_normal_N_per_mm3"]).sum(1)
    mo = W @ rig
    defl = np.linalg.norm(mo[:, :3], axis=1)
    cand = []
    for v in (pull, defl):
      for i in np.argsort(-v):
        if all(abs(int(i) - j) >= 8 for j in cand):
          cand.append(int(i))
          break
    i2 = [int(i) for i in np.argsort(-pull) if all(abs(int(i) - j) >= 8 for j in cand)][:1]
    for i in cand + i2:
      for variant in variants:
        jobs.append((m, W[i], variant, ux[i] <= 0.0))
        labels.append({"scenario": name, "conditions": sc, "instant": i, "bilateral_deflection_mm": round(float(defl[i]), 3)})
  with ProcessPoolExecutor(min(len(jobs), 30), mp_context=spawn) as ex:
    res = list(ex.map(contact, jobs))
  surfs = [r.pop("_surf") for r in res]
  rows = [{**l, **r} for l, r in zip(labels, res)]
  if export:
    def b64(a):
      return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    k = min((i for i, r in enumerate(rows) if r["variant"] == "free"), key=lambda i: rows[i]["sf_inplane"])
    vm, sn = surfs[k]
    json.dump({"row": rows[k], "pos": b64(m["X"][m["snodes"]].astype(np.float32)), "tri": b64(m["stri"].astype(np.uint32)),
               "vm": b64(vm), "sn": b64(sn), "hot": m["X"][m["snodes"][int(np.argmax(vm))]].round(2).tolist(), "mass_g": info["mass_g"]},
              open(export, "w"))
  TH = judge.TH
  for r in rows:
    r["fails"] = (r["seat_deflection_mm"] > TH["seat_deflection_max_mm"] or r["seat_tilt_deg"] > TH["seat_tilt_max_deg"]
                  or r["sf_inplane"] < TH["sf_min"] or r["sf_interlayer"] < TH["sf_min"])
  worst = {}
  for v in variants:
    rv = [r for r in rows if r["variant"] == v]
    worst[v] = {"seat_deflection_mm": max(r["seat_deflection_mm"] for r in rv), "seat_tilt_deg": max(r["seat_tilt_deg"] for r in rv),
                "sf_inplane": min(r["sf_inplane"] for r in rv), "sf_interlayer": min(r["sf_interlayer"] for r in rv),
                "unconverged": sum(not r["converged"] for r in rv)}
  failing = sorted({r["scenario"] for r in rows if r["variant"] == "free" and r["fails"]})
  return {"design": os.path.basename(path), "actuator": os.path.basename(act_path) if act_path else None,
          "pass": not failing if "free" in variants else None, "failing_scenarios": failing,
          "failing_conditions": [dict(todo)[n] for n in failing], "mass_g": info["mass_g"], "face_nodes": len(m["fi"]), "worst": worst, "rows": rows}


def cli(args):
  ap = argparse.ArgumentParser(prog="debaterl hip contact")
  ap.add_argument("design")
  ap.add_argument("scenarios", nargs="?")
  ap.add_argument("--actuator")
  ap.add_argument("--variants", default="free")
  ap.add_argument("--export", help="write the worst free-variant instant's surface stress field here")
  a = ap.parse_args(args)
  sc = json.load(open(a.scenarios)) if a.scenarios else None
  return check(a.design, sc if sc is None or isinstance(sc, list) else [sc], a.actuator, tuple(a.variants.split(",")), a.export)
