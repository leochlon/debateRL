"""MuJoCo run: the exo hardware is simulated; the wearer's pelvis and right thigh follow a reference motion.

Reference motion comes from a planar 7-segment body (Winter's segment fractions scaled by height): normative joint-angle
curves for walking, a min-jerk sit-to-stand, and a pelvis path that keeps the stance foot planted (forward kinematics).
In the dynamic model the pelvis and the right thigh are driven along that path by stiff servos; the actuator housing is
rigid on the pelvis through the bracket, the thigh link turns on the exo hinge, and the cuff slides along the link and is
pinned to the thigh. The bracket load is the force and moment the housing subtree puts on the pelvis (cfrc_int).
"""
import math
import numpy as np
import mujoco
from resips.helpers import gait, minjerk
from resips.cad.hip_geom import G

SEG = {"hat": (0.678, 0.288, 0.626, 0.496), "thigh": (0.100, 0.245, 0.433, 0.323),
       "shank": (0.0465, 0.246, 0.433, 0.302), "foot": (0.0145, 0.152, 0.5, 0.475)}
BELT_ABOVE_HIP = 0.055
SKIN_OFFSET = 0.095
CUFF_AT = 0.6
EXO = {"housing_kg": G["actuator"]["mass_kg"], "link_kg": 0.18, "cuff_kg": 0.08, "cuff_x": -0.045,
       "Kt": 0.032, "N": 30, "eta": 0.80, "I_max": 25.0, "dofc_delay": 0.25}
TAU_MAX = EXO["Kt"] * EXO["N"] * EXO["eta"] * EXO["I_max"]
SETTINGS_DEFAULT = {"timestep": 0.0005}
SETTINGS_LIMITS = {"timestep": [0.0001, 0.001]}
DURATION = {"walk": 4.0, "sts": 3.0}
SETTLE = 0.5
FILTER_HZ = 6.0
PELVIS_SHARE_OF_LEAN = 0.55

CURVES = gait()


def dims(M, H):
  s = {k: {"m": mf * M, "L": lf * H, "c": cf * lf * H, "I": mf * M * (rg * lf * H) ** 2} for k, (mf, lf, cf, rg) in SEG.items()}
  s["ankle_h"] = 0.039 * H
  s["hw"] = 0.085 * H / 1.75
  return s


def exo_geometry(sc, axis_mm):
  H = sc["height_m"]
  s = dims(sc.get("mass_kg", 75.0), H)
  A = np.array([s["hw"] + SKIN_OFFSET, axis_mm[0] / 1000, BELT_ABOVE_HIP * H + axis_mm[1] / 1000])
  P0 = np.array([s["hw"] + SKIN_OFFSET + EXO["cuff_x"], 0.0, -CUFF_AT * s["thigh"]["L"]])
  link_origin = A + np.array([-0.009, 0, 0])
  return {"A": A, "seat": A + np.array([G["actuator"]["seat_x"][1] / 1000, 0, 0]), "link_origin": link_origin,
          "cuff_rel": P0 - link_origin, "s": s}


def kinematic_xml(sc):
  """Full planar body, used only for forward kinematics of the reference."""
  s = dims(sc.get("mass_kg", 75.0), sc["height_m"])
  Lth, Lsh, Lf, ha = s["thigh"]["L"], s["shank"]["L"], s["foot"]["L"], s["ankle_h"]

  def leg(sd, x):
    return f"""
      <body name="thigh_{sd}" pos="{x:.4f} 0 0"><joint name="hip_{sd}" axis="1 0 0"/><geom type="sphere" size="0.01"/>
        <body name="shank_{sd}" pos="0 0 {-Lth:.5f}"><joint name="knee_{sd}" axis="-1 0 0"/><geom type="sphere" size="0.01"/>
          <body name="foot_{sd}" pos="0 0 {-Lsh:.5f}"><joint name="ankle_{sd}" axis="1 0 0"/>
            <geom name="heel_{sd}" type="sphere" pos="0 {-0.25 * Lf + 0.02:.4f} {-ha + 0.02:.4f}" size="0.02"/>
            <geom name="toe_{sd}" type="sphere" pos="0 {0.75 * Lf - 0.03:.4f} {-ha + 0.02:.4f}" size="0.02"/>
          </body></body></body>"""
  return f"""<mujoco><worldbody><body name="pelvis"><joint name="rootP" axis="1 0 0"/><geom type="sphere" size="0.01"/>
      {leg("r", s["hw"])}{leg("l", -s["hw"])}</body></worldbody></mujoco>"""


def joint_refs(sc, s, t, amp=1.0):
  out = {}
  if sc["activity"] == "walk":
    v, H, slope = sc["speed_mps"], sc["height_m"], sc["slope_deg"]
    T = 1.07 * (1.3 / v) ** 0.45 * math.sqrt(H / 1.75)
    for side, ph in (("r", 0.0), ("l", 0.5)):
      phase = (t / T + ph) % 1.0
      hip = CURVES["hip"](phase) - np.radians(6.0)
      mean = np.radians(4.0)
      out[f"hip_{side}"] = mean + (hip - mean) * amp + np.radians(0.6 * slope)
      out[f"knee_{side}"] = CURVES["knee"](phase) * (0.85 + 0.15 * v / 1.3) + np.radians(max(slope, 0) * 0.5)
      out[f"ankle_{side}"] = CURVES["ankle"](phase) + np.radians(0.5 * slope)
    out["pitch"] = np.full_like(t, -np.radians(4.0 + 0.4 * slope))
    out["period"] = T
  else:
    h, sc_ = sc["chair_height_m"], sc["height_m"] / 1.75
    z_hip = h + 0.10 * sc_
    z_knee = s["ankle_h"] + s["shank"]["L"] * math.cos(math.radians(10))
    th = math.acos(float(np.clip((z_hip - z_knee) / s["thigh"]["L"], -1, 1)))
    lean = math.radians(40.0) * PELVIS_SHARE_OF_LEAN
    t0, t1, t2 = 0.4, 1.3, 2.5
    for side in "rl":
      out[f"hip_{side}"] = np.where(t < t1, minjerk(t, t0, t1, th, th + lean), minjerk(t, t1, t2, th + lean, 0.0))
      out[f"knee_{side}"] = minjerk(t, t1, t2, th + math.radians(10), 0.0)
      out[f"ankle_{side}"] = minjerk(t, t1, t2, math.radians(10), 0.0)
    out["pitch"] = np.where(t < t1, minjerk(t, t0, t1, 0.0, -lean), minjerk(t, t1, t2, -lean, 0.0))
  return out


def root_ref(m, d, refs, t, walking):
  """Pelvis y, z keeping the stance foot on the floor without slipping. Walking: right foot carries right-leg phase
  [0, 0.5), left foot the other half; within a foot the lower of heel and toe. Sit-to-stand: right heel fixed."""
  gid = [m.geom(f"{g}_{sd}").id for sd in "rl" for g in ("heel", "toe")]
  ja = {n: m.jnt_qposadr[m.joint(n).id] for n in ("rootP", "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l")}
  P = np.zeros((len(t), 4, 3))
  for i in range(len(t)):
    for j in ja:
      d.qpos[ja[j]] = refs["pitch" if j == "rootP" else j][i]
    mujoco.mj_kinematics(m, d)
    P[i] = d.geom_xpos[gid]
  low = P[:, :, 2] - 0.02
  z = -low.min(1)
  if not walking:
    return -P[:, 0, 1] + P[0, 0, 1], z
  foot = np.where(((t / refs["period"]) % 1.0) < 0.5, 0, 2)
  idx = np.arange(len(t))
  sel = foot + (low[idx, foot + 1] < low[idx, foot]).astype(int)
  y = np.zeros(len(t))
  for i in range(1, len(t)):
    y[i] = y[i - 1] - (P[i, sel[i], 1] - P[i - 1, sel[i], 1])
  return y, z


def reference(sc, t):
  s = dims(sc.get("mass_kg", 75.0), sc["height_m"])
  m = mujoco.MjModel.from_xml_string(kinematic_xml(sc))
  d = mujoco.MjData(m)
  walking = sc["activity"] == "walk"
  amp = 1.0
  if walking:
    tc = np.arange(0, 3.0, 0.004)
    for _ in range(6):
      refs = joint_refs(sc, s, tc, amp)
      y, _ = root_ref(m, d, refs, tc, True)
      n = int(refs["period"] / 0.004)
      v = (y[-1] - y[-1 - n]) / refs["period"]
      amp *= float(np.clip(sc["speed_mps"] / max(v, 1e-3), 0.7, 1.4)) ** 0.9
  refs = joint_refs(sc, s, t, amp)
  y, z = root_ref(m, d, refs, t, walking)
  from scipy.signal import butter, filtfilt
  b, a = butter(4, FILTER_HZ / (0.5 / (t[1] - t[0])))
  refs["rootY"], refs["rootZ"] = filtfilt(b, a, y), filtfilt(b, a, z)
  refs["amp"] = amp
  return refs


def model_xml(sc, axis_mm, st, act=None):
  geo = exo_geometry(sc, axis_mm)
  s = geo["s"]
  slope = math.radians(sc.get("slope_deg", 0.0))
  gy, gz = -9.81 * math.sin(slope), -9.81 * math.cos(slope)
  A, cr = geo["A"], geo["cuff_rel"]
  u = cr[1:] / np.linalg.norm(cr[1:])
  com, mass = G["actuator"]["com_x"] / 1000, act["mass_kg"] if act else EXO["housing_kg"]
  hinge = f' armature="{act["J_reflected"]:.6f}" frictionloss="{act["friction_Nm"]:.4f}"' if act else ""
  xml = f"""<mujoco model="exo">
  <compiler angle="radian"/>
  <option timestep="{st['timestep']}" integrator="implicitfast" gravity="0 {gy:.6f} {gz:.6f}"/>
  <default><geom contype="0" conaffinity="0"/><joint armature="0.001"/></default>
  <worldbody>
    <body name="pelvis">
      <joint name="rootY" type="slide" axis="0 1 0"/><joint name="rootZ" type="slide" axis="0 0 1"/><joint name="rootP" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="10" diaginertia="0.3 0.3 0.3"/>
      <body name="housing" pos="{A[0]:.5f} {A[1]:.5f} {A[2]:.5f}">
        <inertial pos="{com:.4f} 0 0" mass="{mass:.4f}" diaginertia="0.00066 0.00045 0.00045"/>
        <body name="link" pos="-0.009 0 0">
          <joint name="exo" axis="1 0 0" damping="0.02"{hinge}/>
          <inertial pos="-0.02 0 -0.13" mass="{EXO['link_kg']}" diaginertia="0.0012 0.0012 0.0001"/>
          <body name="cuff" pos="{cr[0]:.5f} {cr[1]:.5f} {cr[2]:.5f}">
            <joint name="slide" type="slide" axis="0 {u[0]:.6f} {u[1]:.6f}" damping="2"/>
            <inertial pos="0 0 0" mass="{EXO['cuff_kg']}" diaginertia="5e-5 5e-5 5e-5"/>
          </body>
        </body>
      </body>
      <body name="thigh_r" pos="{s['hw']:.4f} 0 0">
        <joint name="hip_r" axis="1 0 0"/>
        <inertial pos="0 0 {-s['thigh']['c']:.4f}" mass="2" diaginertia="0.05 0.05 0.01"/>
      </body>
    </body>
  </worldbody>
  <equality><connect body1="cuff" body2="thigh_r" anchor="0 0 0" solref="0.004 1"/></equality>
  <actuator>
    <position name="a_rootY" joint="rootY" kp="400000" kv="4000"/>
    <position name="a_rootZ" joint="rootZ" kp="400000" kv="4000"/>
    <position name="a_rootP" joint="rootP" kp="40000" kv="400"/>
    <position name="a_hip_r" joint="hip_r" kp="40000" kv="400"/>
    <motor name="a_exo" joint="exo" gear="1"/>
  </actuator>
</mujoco>"""
  return xml, geo


def loop_closure(geo, hip):
  """Exo hinge angle and cuff slide that put the cuff on the thigh point, for a hip flexion angle (pelvis frame)."""
  Lt = CUFF_AT * geo["s"]["thigh"]["L"]
  O, c = geo["link_origin"][1:], geo["cuff_rel"][1:]
  v = np.array([Lt * math.sin(hip), -Lt * math.cos(hip)]) - O
  return math.atan2(v[0], -v[1]) - math.atan2(c[0], -c[1]), float(np.linalg.norm(v) - np.linalg.norm(c))


def simulate(sc, axis_mm, settings=None, tau_override=None, log_dt=0.001, act=None):
  """One scenario. tau_override: (t, tau) delivered exo torque from the drive simulation (second pass)."""
  st = {**SETTINGS_DEFAULT, **(settings or {})}
  xml, geo = model_xml(sc, axis_mm, st, act)
  m = mujoco.MjModel.from_xml_string(xml)
  d = mujoco.MjData(m)
  dt = m.opt.timestep
  tau_max = act["tau_cap"] if act else TAU_MAX
  n = int(round(DURATION[sc["activity"]] / dt))
  t = np.arange(n + 1) * dt
  refs = reference(sc, t)
  drive = {"rootY": "rootY", "rootZ": "rootZ", "rootP": "pitch", "hip_r": "hip_r"}
  qa = {j: m.jnt_qposadr[m.joint(j).id] for j in list(drive) + ["exo", "slide"]}
  va = {j: m.jnt_dofadr[m.joint(j).id] for j in list(drive) + ["exo", "slide"]}
  ai = {j: m.actuator("a_" + j).id for j in drive}
  aexo = m.actuator("a_exo").id
  ref = {j: refs[k] for j, k in drive.items()}
  dref = {j: np.gradient(ref[j], dt) for j in drive}
  for j in drive:
    d.qpos[qa[j]], d.qvel[va[j]] = ref[j][0], dref[j][0]
  d.qpos[qa["exo"]], d.qpos[qa["slide"]] = loop_closure(geo, ref["hip_r"][0])
  d.qvel[va["exo"]] = dref["hip_r"][0]
  mujoco.mj_forward(m, d)
  pelvis, housing = m.body("pelvis").id, m.body("housing").id
  root = m.body_rootid[housing]
  kappa = sc["assist_Nm"]
  delay = int(round(EXO["dofc_delay"] / dt))
  hist = np.zeros(n + 1)
  step_log = max(1, int(round(log_dt / dt)))
  L = {k: [] for k in ("t", "W", "exo", "exo_vel", "slide", "tau_dem", "tau_cmd", "tau", "hip_r", "track_err")}
  for i in range(n + 1):
    for j in drive:
      kp, kv = m.actuator_gainprm[ai[j], 0], -m.actuator_biasprm[ai[j], 2]
      d.ctrl[ai[j]] = ref[j][i] + (kv / kp) * dref[j][i]
    qe = d.qpos[qa["exo"]]
    hist[i] = math.sin(qe) - math.sin(refs["hip_l"][i])
    if sc["activity"] == "sts":
      ramp = float(np.clip((t[i] - 0.4) / 0.3, 0, 1)) * float(np.clip((2.8 - t[i]) / 0.2, 0, 1))
      cmd = -kappa * math.sin(qe) * ramp
    else:
      cmd = kappa * hist[i - delay] if i >= delay else 0.0
    dem = cmd
    cmd = float(np.clip(cmd, -tau_max, tau_max))
    tau = cmd if tau_override is None else float(np.interp(t[i], *tau_override))
    d.ctrl[aexo] = tau
    if i % step_log == 0:
      mujoco.mj_rnePostConstraint(m, d)
      c = d.subtree_com[root]
      tq, f = d.cfrc_int[housing][:3], d.cfrc_int[housing][3:]
      R = d.xmat[pelvis].reshape(3, 3)
      S = d.xpos[pelvis] + R @ geo["seat"]
      tS = tq + np.cross(c - S, f)
      L["t"].append(t[i])
      L["W"].append(np.concatenate([-(R.T @ f), -(R.T @ tS) * 1000.0]))
      for k, v in (("exo", qe), ("exo_vel", d.qvel[va["exo"]]), ("slide", d.qpos[qa["slide"]]), ("tau_dem", dem), ("tau_cmd", cmd),
                   ("tau", tau), ("hip_r", d.qpos[qa["hip_r"]])):
        L[k].append(v)
      L["track_err"].append(max(abs(d.qpos[qa[j]] - ref[j][i]) for j in drive))
    mujoco.mj_step(m, d)
    if not np.isfinite(d.qpos).all():
      raise RuntimeError(f"simulation diverged at t={t[i]:.3f} s")
  out = {k: np.array(v) for k, v in L.items()}
  out["amp"] = refs["amp"]
  out["speed"] = (refs["rootY"][-1] - refs["rootY"][0]) / t[-1]
  return out

