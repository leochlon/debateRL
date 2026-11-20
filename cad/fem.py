"""Linear-elastic FEA: gmsh meshes the exact STEP into 10-node tets, CalculiX (ccx) solves six unit load cases.

Supports: an elastic belt clamp. The bracket surface lying on the belt plate (x=0, inside the plate footprint) rests on a
bed of grounded springs (normal and tangential, per unit area), and each belt bolt is a spring spread over its bore. With
no rigid supports the stress stays bounded everywhere, so every integration point is scored (stress is read at the four
integration points of each tet, which converge better than values extrapolated to the corners). Each unit load case is a
force or moment at a reference point on the axis, spread over the loaded interface (the caller's load_region) as a
distributing coupling (uniform weights: forces F/W plus a linear field that carries the moment).
Any load history is then a linear combination of the six cases.
"""
import hashlib, json, os, subprocess
import numpy as np

MESH_DEFAULT = {"h": 2.5, "curvature_elems": 12, "refine": []}
MESH_LIMITS = {"h": [1.2, 3.0], "refine_max": 3, "refine_radius_max": 8.0, "refine_h_min": 0.5}
CASES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
IP_A, IP_B = 0.5854101966249685, 0.1381966011250105  # barycentric coordinates of the 4-point tet rule


def mesh_cfg(cfg=None):
  c = {**MESH_DEFAULT, **(cfg or {})}
  c["refine"] = [dict(r) for r in c.get("refine", [])]
  return c


def mesh_ok(c):
  if not (MESH_LIMITS["h"][0] <= c["h"] <= MESH_LIMITS["h"][1]) or c["curvature_elems"] != MESH_DEFAULT["curvature_elems"]:
    return False
  if len(c["refine"]) > MESH_LIMITS["refine_max"]:
    return False
  return all(len(r["center"]) == 3 and 0 < r["radius"] <= MESH_LIMITS["refine_radius_max"] and
             MESH_LIMITS["refine_h_min"] <= r["h"] <= c["h"] for r in c["refine"])


def mesh(step, c, imprint=None):
  """Nodes X, tet10 elements E (gmsh order) and surface tri6 T. imprint(gmsh) may cut faces into the part before meshing."""
  import gmsh
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 0)
  gmsh.option.setNumber("General.NumThreads", 1)
  gmsh.model.occ.importShapes(step)
  gmsh.model.occ.synchronize()
  if imprint:
    imprint(gmsh)
  gmsh.option.setNumber("Mesh.MeshSizeMax", c["h"])
  gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
  gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", c["curvature_elems"])
  gmsh.option.setNumber("Mesh.ElementOrder", 2)
  gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
  gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
  gmsh.option.setNumber("Mesh.Algorithm3D", 1)
  gmsh.option.setNumber("Mesh.RandomSeed", 1)
  if c["refine"]:
    fs = []
    for r in c["refine"]:
      f = gmsh.model.mesh.field.add("Ball")
      gmsh.model.mesh.field.setNumber(f, "Radius", r["radius"])
      for k, v in zip("XYZ", r["center"]):
        gmsh.model.mesh.field.setNumber(f, k + "Center", v)
      gmsh.model.mesh.field.setNumber(f, "VIn", r["h"])
      gmsh.model.mesh.field.setNumber(f, "VOut", 1e9)
      gmsh.model.mesh.field.setNumber(f, "Thickness", 2 * r["radius"])
      fs.append(f)
    m = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(m, "FieldsList", fs)
    gmsh.model.mesh.field.setAsBackgroundMesh(m)
  gmsh.model.mesh.generate(3)
  tags, xyz, _ = gmsh.model.mesh.getNodes()
  X = xyz.reshape(-1, 3)
  idx = np.full(int(tags.max()) + 1, -1)
  idx[tags.astype(int)] = np.arange(len(tags))
  etypes, etags, enodes = gmsh.model.mesh.getElements(3)
  if len(etags) != 1:
    gmsh.finalize()
    raise RuntimeError("mesh: expected a single tet10 block")
  q = gmsh.model.mesh.getElementQualities(etags[0], "minSJ")
  if q.min() < 0.1:
    gmsh.finalize()
    raise RuntimeError(f"mesh: {int((q < 0.1).sum())} elements with scaled Jacobian < 0.1")
  E = idx[np.asarray(enodes[0], dtype=int)].reshape(-1, 10)
  tri = []
  for et, en in zip(*gmsh.model.mesh.getElements(2)[0::2]):
    tri.append(idx[np.asarray(en, dtype=int)].reshape(-1, 6))
  gmsh.finalize()
  return X, E, np.vstack(tri)


def abaqus_order(X, E):
  """gmsh tet10 midside order -> Abaqus/ccx C3D10 order, matched by coordinates."""
  pairs = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
  out = E.copy()
  mids = E[:, 4:]
  for k, (a, b) in enumerate(pairs):
    target = (X[E[:, a]] + X[E[:, b]]) / 2
    d = np.linalg.norm(X[mids] - target[:, None, :], axis=2)
    out[:, 4 + k] = mids[np.arange(len(E)), d.argmin(1)]
    if (d.min(1) > 0.05 * np.linalg.norm(X[E[:, a]] - X[E[:, b]], axis=1)).any():
      raise RuntimeError("mesh: midside node far from its edge")
  if any(len(set(r)) < 6 for r in out[:, 4:]):
    raise RuntimeError("mesh: midside nodes matched twice")
  v6 = np.einsum("ij,ij->i", np.cross(X[out[:, 1]] - X[out[:, 0]], X[out[:, 2]] - X[out[:, 0]]), X[out[:, 3]] - X[out[:, 0]])
  if (v6 <= 0).any():
    raise RuntimeError("mesh: inverted tets")
  return out


def tri_weights(X, T):
  """Consistent nodal weights of uniform traction on 6-node triangles (corners 0, midsides area/3)."""
  w = np.zeros(len(X))
  a = 0.5 * np.linalg.norm(np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]), axis=1)
  for k in (3, 4, 5):
    np.add.at(w, T[:, k], a / 3)
  return w


def expand(T, m):
  """All nodes of the surface triangles whose three corners are in m (straight-sided midsides sit off curved faces)."""
  out = np.zeros_like(m)
  out[T[m[T[:, :3]].all(1)].ravel()] = True
  return out


def area_lumped(X, T, mask):
  """Nodal areas of the triangles lying wholly in mask (corners a/12, midsides a/4)."""
  tri = T[mask[T].all(1)]
  a = 0.5 * np.linalg.norm(np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]]), axis=1)
  w = np.zeros(len(X))
  for c in (0, 1, 2):
    np.add.at(w, tri[:, c], a / 12)
  for c in (3, 4, 5):
    np.add.at(w, tri[:, c], a / 4)
  return w


def clamp_regions(X, T, G):
  """Belt-plate contact face and each belt bolt bore (G: the claims' geometry with keepout.belt_plate and belt_bolts)."""
  on = np.zeros(len(X), bool)
  on[T.ravel()] = True
  x, y, z = X.T
  bp = G["keepout"]["belt_plate"]
  face = on & (np.abs(x) < 1e-4) & (y >= bp["y"][0]) & (y <= bp["y"][1]) & (z >= bp["z"][0]) & (z <= bp["z"][1])
  return face, [expand(T, on & (np.abs(np.hypot(y - yb, z - zb) - G["belt_bolts"]["bore_r"]) < 1e-3)) for yb, zb in G["belt_bolts"]["yz"]]


def clamp_springs(X, T, face, bores, C):
  """Elastic belt clamp: a Winkler bed under the contact face and a bolt spring spread over each bore."""
  out = []
  w = area_lumped(X, T, face)
  for i in np.flatnonzero(w > 0):
    out += [(i, 1, C["k_normal_N_per_mm3"] * w[i]), (i, 2, C["k_tangent_N_per_mm3"] * w[i]), (i, 3, C["k_tangent_N_per_mm3"] * w[i])]
  for b in bores:
    wb = area_lumped(X, T, b)
    if wb.sum() <= 0:
      raise RuntimeError("belt bolt bore not found")
    for i in np.flatnonzero(wb > 0):
      out += [(i, d, C["k_bolt_N_per_mm"] * wb[i] / wb.sum()) for d in (1, 2, 3)]
  return out


def coupling_forces(X, w, ref):
  """Nodal forces of the six unit cases (N, N*mm) applied at ref through a uniform distributing coupling."""
  m = w > 0
  W = w[m].sum()
  c = (w[m][:, None] * X[m]).sum(0) / W
  R = X[m] - c
  J = np.einsum("i,ij,ik->jk", w[m], R, R)
  J = np.trace(J) * np.eye(3) - J
  Ji = np.linalg.inv(J)
  out = []
  for k in range(6):
    F, M = np.zeros(3), np.zeros(3)
    (F if k < 3 else M)[k % 3] = 1.0
    t = F / W + np.cross(Ji @ (M + np.cross(ref - c, F)), R)
    f = np.zeros_like(X)
    f[m] = w[m][:, None] * t
    out.append(f)
  return out


def write_inp(path, X, E, fixed, seat, force, mat, springs=(), face=None):
  """One CalculiX input per unit case, so the six cases can be solved by separate single-threaded processes.
  Supports are either fixed nodes or grounded springs (node, dof, stiffness). U is printed at the seat, and at the
  nodes in face when given (hip_contact.py)."""
  with open(path, "w") as fh:
    fh.write("*NODE, NSET=NALL\n")
    fh.writelines(f"{i + 1},{p[0]:.9g},{p[1]:.9g},{p[2]:.9g}\n" for i, p in enumerate(X))
    fh.write("*ELEMENT, TYPE=C3D10, ELSET=EALL\n")
    fh.writelines(f"{i + 1}," + ",".join(str(n + 1) for n in e) + "\n" for i, e in enumerate(E))
    for j, (i, dof, k) in enumerate(springs):
      eid = len(E) + j + 1
      fh.write(f"*ELEMENT, TYPE=SPRING1, ELSET=SP{eid}\n{eid},{i + 1}\n*SPRING, ELSET=SP{eid}\n{dof}\n{k:.9g}\n")
    fh.write("*NSET, NSET=NFIX\n")
    fh.writelines(f"{i + 1},\n" for i in np.flatnonzero(fixed))
    fh.write("*NSET, NSET=NSEAT\n")
    fh.writelines(f"{i + 1},\n" for i in seat)
    if face is not None:
      fh.write("*NSET, NSET=NFACE\n")
      fh.writelines(f"{i + 1},\n" for i in face)
    fh.write(f"*MATERIAL, NAME=M\n*ELASTIC\n{mat['E_MPa']},{mat['nu']}\n*SOLID SECTION, ELSET=EALL, MATERIAL=M\n")
    if np.any(fixed):
      fh.write("*BOUNDARY\nNFIX,1,3\n")
    fh.write("*STEP\n*STATIC\n*CLOAD, OP=NEW\n")
    for i in np.flatnonzero(np.abs(force).sum(1) > 0):
      for d in range(3):
        if force[i, d] != 0.0:
          fh.write(f"{i + 1},{d + 1},{force[i, d]:.12g}\n")
    fh.write("*NODE PRINT, NSET=NSEAT\nU\n")
    if face is not None:
      fh.write("*NODE PRINT, NSET=NFACE\nU\n")
    fh.write("*EL PRINT, ELSET=EALL\nS\n*END STEP\n")


def read_dat(path, n_nodes, n_elem):
  """Displacements and integration-point stresses of a single-step .dat file."""
  lines = open(path).read().split("\n")
  heads = [i for i, l in enumerate(lines) if "displacements" in l or "stresses" in l] + [len(lines)]
  U, S = np.zeros((n_nodes, 3)), np.zeros((n_elem, 4, 6))
  for a, b in zip(heads[:-1], heads[1:]):
    body = " ".join(l for l in lines[a + 1:b] if l.strip())
    v = np.array(body.split(), dtype=float)
    if "displacements" in lines[a]:
      v = v.reshape(-1, 4)
      U[v[:, 0].astype(int) - 1] = v[:, 1:4]
    else:
      v = v.reshape(-1, 8)
      S[v[:, 0].astype(int) - 1, v[:, 1].astype(int) - 1] = v[:, 2:8]
  return U, S


def ip_matrix():
  """Integration-point values from corner values of a linear field (rows: points, columns: corners)."""
  return IP_B * np.ones((4, 4)) + (IP_A - IP_B) * np.eye(4)


def corner_stress(S):
  """Linear extrapolation of the 4 tet integration-point stresses to the 4 corners, unaveraged."""
  return np.einsum("ij,...jk->...ik", np.linalg.inv(ip_matrix()), S)


def solve(meta, G, C, mat, load_region, ref, seat_x, version, cfg=None, threads=1):
  """Mesh + six unit cases on the built part in meta["dir"]. load_region(X, T, on, axis) marks the loaded interface,
  ref is the load point, the seat (for the rigid fit) is the loaded face at x = seat_x. Cached per design, mesh settings
  and version. Returns a dict of numpy arrays."""
  c = mesh_cfg(cfg)
  key = hashlib.sha256((json.dumps(c, sort_keys=True) + version).encode()).hexdigest()[:12]
  d = os.path.join(meta["dir"], f"fea_{key}")
  npz = os.path.join(d, "unit.npz")
  if os.path.exists(npz):
    return dict(np.load(npz, allow_pickle=False))
  os.makedirs(d, exist_ok=True)
  import fcntl
  lock = open(os.path.join(d, "lock"), "w")
  fcntl.flock(lock, fcntl.LOCK_EX)
  if os.path.exists(npz):
    return dict(np.load(npz, allow_pickle=False))
  X, E, T = mesh(os.path.join(meta["dir"], "part.step"), c)
  E = abaqus_order(X, E)
  face, bores = clamp_regions(X, T, G)
  on = np.zeros(len(X), bool)
  on[T.ravel()] = True
  load = load_region(X, T, on, meta["axis"])
  if face.sum() < 10 or load.sum() < 10:
    raise RuntimeError(f"boundary regions not found (clamp face {face.sum()}, loaded {load.sum()} nodes)")
  springs = clamp_springs(X, T, face, bores, C)
  fixed = np.zeros(len(X), bool)
  w = tri_weights(X, T)
  w[~load] = 0.0
  ref = np.asarray(ref, float)
  forces = coupling_forces(X, w, ref)
  seat = np.flatnonzero(load & (np.abs(X[:, 0] - seat_x) < 1e-4))
  env = {**os.environ, "OMP_NUM_THREADS": str(threads), "CCX_NPROC_EQUATION_SOLVER": str(threads)}
  procs = []
  for k, f in enumerate(forces):
    write_inp(os.path.join(d, f"case{k}.inp"), X, E, fixed, seat, f, mat, springs)
    procs.append(subprocess.Popen(["ccx", "-i", f"case{k}"], cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
  outs = [p.communicate()[0] for p in procs]
  Us, Ss = [], []
  for k, (p, o) in enumerate(zip(procs, outs)):
    dat = os.path.join(d, f"case{k}.dat")
    if p.returncode != 0 or not os.path.exists(dat):
      raise RuntimeError(f"ccx failed on case {k}: " + o[-2000:])
    u, sk = read_dat(dat, len(X), len(E))
    Us.append(u)
    Ss.append(sk)
  U, S = np.array(Us), np.array(Ss)
  for k in range(6):
    for ext in ("frd", "dat"):
      if os.path.exists(os.path.join(d, f"case{k}.{ext}")):
        os.remove(os.path.join(d, f"case{k}.{ext}"))
  corners = np.einsum("ij,ejk->eik", ip_matrix(), X[E[:, :4]])
  out = {"S": S.astype(np.float64), "U_seat": U[:, seat, :], "X_seat": X[seat], "ref": ref, "corner_xyz": corners.astype(np.float32),
         "n_nodes": np.array(len(X)), "n_elem": np.array(len(E))}
  np.savez(npz + ".tmp.npz", **out)
  os.replace(npz + ".tmp.npz", npz)
  return out


def rigid_fit(Xs, Us, ref):
  """Least-squares rigid motion (t at ref, rotation w) of the seat nodes for each unit case."""
  R = Xs - ref
  A = np.zeros((3 * len(R), 6))
  for i in range(3):
    A[i::3, i] = 1.0
  # u = t + w x r  ->  rows for ux, uy, uz
  A[0::3, 4], A[0::3, 5] = R[:, 2], -R[:, 1]
  A[1::3, 3], A[1::3, 5] = -R[:, 2], R[:, 0]
  A[2::3, 3], A[2::3, 4] = R[:, 1], -R[:, 0]
  return np.array([np.linalg.lstsq(A, U.reshape(-1), rcond=None)[0] for U in Us])


def sample(W):
  """Every fifth log sample (5 ms) plus every sample where a load component peaks."""
  idx = set(range(0, len(W), 5))
  idx |= set(np.argmax(W, 0).tolist()) | set(np.argmin(W, 0).tolist())
  return np.array(sorted(idx))


def layer_normal(build_dir):
  """Voigt weights that turn a stress vector into the normal stress across print layers stacked along build_dir."""
  n = np.zeros(3)
  n["xyz".index(build_dir[1])] = 1.0
  return np.array([n[0] ** 2, n[1] ** 2, n[2] ** 2, 2 * n[0] * n[1], 2 * n[0] * n[2], 2 * n[1] * n[2]])


def von_mises(s):
  sx, sy, sz, txy, txz, tyz = np.moveaxis(s, -1, 0)
  return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + txz ** 2 + tyz ** 2))


def evaluate(unit, W, build_dir, mat, chunk=25000):
  """W: (n_t, 6) wrench history on the bracket at the load point [Fx,Fy,Fz (N), Mx,My,Mz (N*mm)].

  Returns the minimum safety factors over time and integration points, where they occur, and the worst seat motion.
  A material without an interlayer strength (machined) has no interlayer safety factor."""
  S = unit["S"].reshape(6, -1, 6).astype(np.float32)
  nn = layer_normal(build_dir)
  W = np.atleast_2d(W).astype(np.float32)
  best = {"vm": (0.0, None, None), "nn": (0.0, None, None)}
  for a in range(0, S.shape[1], chunk):
    Sc = S[:, a:a + chunk]
    s = (W @ Sc.reshape(6, -1)).reshape(len(W), Sc.shape[1], 6)
    vm = von_mises(s)
    sn = np.einsum("tpk,k->tp", s, nn)
    for k, v in (("vm", vm), ("nn", sn)):
      i = np.unravel_index(np.argmax(v), v.shape)
      if v[i] > best[k][0]:
        best[k] = (float(v[i]), int(i[0]), a + int(i[1]))
  pts = unit["corner_xyz"].reshape(-1, 3)

  def at(p):
    return None if p is None else [round(float(v), 2) for v in pts[p]]
  rig = rigid_fit(unit["X_seat"], unit["U_seat"], unit["ref"])
  m = W @ rig
  defl = np.linalg.norm(m[:, :3], axis=1)
  tilt = np.degrees(np.hypot(m[:, 4], m[:, 5]))
  il = mat["strength_interlayer_MPa"]
  return {"sf_inplane": round(mat["strength_inplane_MPa"] / max(best["vm"][0], 1e-12), 4),
          "vm_max_MPa": round(best["vm"][0], 3), "vm_at_mm": at(best["vm"][2]), "vm_t_index": best["vm"][1],
          "sf_interlayer": None if il is None else round(il / max(best["nn"][0], 1e-12), 4),
          "interlayer_max_MPa": round(best["nn"][0], 3), "interlayer_at_mm": at(best["nn"][2]), "interlayer_t_index": best["nn"][1],
          "seat_deflection_mm": round(float(defl.max()), 4), "seat_tilt_deg": round(float(tilt.max()), 4)}


def validate():
  """The pipeline against closed-form answers: a cantilever (Timoshenko beam) and a plate with a hole (Peterson's Kt)."""
  import shutil, tempfile
  from build123d import Box, Cylinder, Pos, Rot, export_step
  E, NU = 3500.0, 0.35

  def run(shape, cfg, fixed_fn, load_fn, ref, wrench):
    d = tempfile.mkdtemp(prefix="debaterl_val_")
    export_step(shape, os.path.join(d, "p.step"))
    X, El, T = mesh(os.path.join(d, "p.step"), mesh_cfg(cfg))
    El = abaqus_order(X, El)
    on = np.zeros(len(X), bool)
    on[T.ravel()] = True
    fixed, load = on & fixed_fn(X), on & load_fn(X)
    w = tri_weights(X, T)
    w[~load] = 0
    f = sum(wk * u for wk, u in zip(wrench, coupling_forces(X, w, np.array(ref))))
    write_inp(os.path.join(d, "v.inp"), X, El, fixed, np.flatnonzero(load), f, {"E_MPa": E, "nu": NU})
    subprocess.run(["ccx", "-i", "v"], cwd=d, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1", "CCX_NPROC_EQUATION_SOLVER": "1"})
    U, S = read_dat(os.path.join(d, "v.dat"), len(X), len(El))
    shutil.rmtree(d)
    return X, El, U, S, load

  L, b, t, F = 120.0, 20.0, 10.0, 100.0
  X, El, U, S, load = run(Pos(t / 2, 0, -L / 2) * Box(t, b, L), {"h": 2.5}, lambda X: np.abs(X[:, 2]) < 1e-6, lambda X: np.abs(X[:, 2] + L) < 1e-6,
                          [t / 2, 0, -L], [0, F, 0, 0, 0, 0])
  I, G = t * b ** 3 / 12, E / (2 * (1 + NU))
  mid = np.abs(X[El[:, :4]].mean(1)[:, 2] + L / 2) < 3
  out = [{"case": "cantilever", "tip_mm": round(float(U[load][:, 1].mean()), 4),
          "beam_theory_mm": round(F * L ** 3 / (3 * E * I) + F * L / (5 / 6 * G * b * t), 4),
          "mid_span_stress_MPa": round(float(np.abs(corner_stress(S)[mid][:, :, 2]).max()), 3), "beam_theory_MPa": round(F * L / 2 * (b / 2) / I, 3)}]
  W, Lp, tp, dh = 100.0, 200.0, 2.5, 10.0
  plate = Pos(tp / 2, 0, 0) * Box(tp, W, Lp) - Pos(tp / 2, 0, 0) * Rot(0, 90, 0) * Cylinder(dh / 2, tp + 2)
  r = dh / W
  kt = (3 - 3.14 * r + 3.667 * r ** 2 - 1.527 * r ** 3) / (1 - r)
  for h in (0.8, 0.5, 0.3):
    X, El, U, S, _ = run(plate, {"h": 4.0, "refine": [{"center": [tp / 2, 0, 0], "radius": 8.0, "h": h}]}, lambda X: np.abs(X[:, 2] + Lp / 2) < 1e-6,
                         lambda X: np.abs(X[:, 2] - Lp / 2) < 1e-6, [tp / 2, 0, Lp / 2], [0, 0, W * tp, 0, 0, 0])
    out.append({"case": f"plate with hole, local h {h}", "kt_integration_points": round(float(S[:, :, 2].max()), 3),
                "kt_corners": round(float(corner_stress(S)[:, :, 2].max()), 3), "kt_peterson": round(kt, 3)})
  return out
