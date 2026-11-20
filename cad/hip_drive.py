"""ngspice run of the actuator drive: battery with internal resistance and bus capacitor, averaged H-bridge, PI current
loop with back-EMF feedforward and anti-windup, and the motor winding (R, L, back-EMF from the MuJoCo joint speed).
Input: commanded output torque and joint speed over time. Output: delivered output torque = Kt * N * eta * i(t)."""
import os, subprocess, tempfile
import numpy as np
from resips.cad.hip_body import EXO

CIRCUIT = {"R_batt": 0.08, "C_bus": 470e-6, "R_motor": 0.25, "R_switch": 0.02, "L_motor": 120e-6, "duty_max": 0.95,
           "bandwidth_hz": 1000.0}
STEP_DEFAULT = 2e-5
STEP_LIMITS = [5e-6, 5e-5]


def pwl(t, x):
  pts = " ".join(f"{a:.6g} {b:.6g}" for a, b in zip(t, x))
  return "PWL(" + pts + ")"


def netlist(t, i_ref, w_motor, v_batt, step, r_motor=None, l_motor=None, ke=None):
  c = dict(CIRCUIT)
  if r_motor is not None:
    c["R_motor"], c["L_motor"] = r_motor, l_motor
  R = c["R_motor"] + c["R_switch"]
  wc = 2 * np.pi * c["bandwidth_hz"]
  kp, ki = c["L_motor"] * wc, R * wc
  ke = EXO["Kt"] if ke is None else ke
  dm = c["duty_max"]
  raw = f"({kp}*(V(iref)-I(Vsense)) + V(xi) + {ke}*V(w) + {R}*V(iref)) / max(V(busf),1)"
  return f"""* exo actuator drive
Vb bat 0 DC {v_batt}
Rb bat bus {c['R_batt']}
Cbus bus 0 {c['C_bus']} IC={v_batt}
Rf bus busf 1k
Cf busf 0 10n IC={v_batt}
Viref iref 0 {pwl(t, i_ref)}
Vw w 0 {pwl(t, w_motor)}
Braw draw 0 V = {raw}
Bd d 0 V = max(-{dm}, min({dm}, V(draw)))
Bint 0 xi I = {ki}*(V(iref)-I(Vsense)) * u({dm} - abs(V(draw)))
Cint xi 0 1 IC=0
Rint xi 0 1e9
Bterm mt 0 V = V(d)*V(bus)
Bbus bus 0 I = V(d)*I(Vsense)
Rm mt m1 {R}
Lm m1 m2 {c['L_motor']} IC=0
Vsense m2 m3 0
Bemf m3 0 V = {ke}*V(w)
.control
set filetype=ascii
tran {step} {t[-1]} 0 {step} uic
wrdata out.txt I(Vsense) V(bus) V(d)
quit
.endc
.end
"""


def deliver(t, tau_cmd, w_exo, v_batt, step=STEP_DEFAULT, act=None):
  a = act or {"Kt": EXO["Kt"], "N": EXO["N"], "eta": EXO["eta"], "I_max": EXO["I_max"], "R": None, "L": None}
  k = a["Kt"] * a["N"] * a["eta"]
  i_ref = np.clip(np.asarray(tau_cmd) / k, -a["I_max"], a["I_max"])
  w_motor = np.asarray(w_exo) * a["N"]
  with tempfile.TemporaryDirectory(prefix="resips_spice_") as d:
    open(os.path.join(d, "drive.cir"), "w").write(netlist(np.asarray(t), i_ref, w_motor, v_batt, step, a["R"], a["L"], a["Kt"]))
    p = subprocess.run(["ngspice", "-b", "drive.cir"], cwd=d, capture_output=True, text=True, timeout=600)
    out = os.path.join(d, "out.txt")
    if not os.path.exists(out):
      raise RuntimeError("ngspice failed: " + (p.stdout + p.stderr)[-1500:])
    a = np.loadtxt(out)
  ts, i, vbus, duty = a[:, 0], a[:, 1], a[:, 3], a[:, 5]
  tau = np.interp(t, ts, k * i)
  return {"t": np.asarray(t), "tau": tau, "i": np.interp(t, ts, i), "i_peak_A": float(np.abs(i).max()),
          "v_bus_min": float(vbus.min()), "duty_peak": float(np.abs(duty).max()),
          "shortfall_Nm": float(np.max(np.abs(np.asarray(tau_cmd)) - np.abs(tau)))}

