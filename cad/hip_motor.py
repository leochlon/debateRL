"""Parametric hip actuator: a frameless brushless motor and a planetary gearbox, from five numbers.

The scaling constants are assumptions, set roughly to published small robot motors: motor constant Km near pancake BLDC
actuators of the T-motor AK class, rotor inertia near frameless motors of the Kollmorgen ILM class, and the mass so that
the bracket debate's actuator (60 mm stator, 15 mm stack, Kt 0.032 N*m/A, 30:1, 25 A) comes out near its 0.6 kg. Km grows
with stator diameter squared and stack length; the winding choice (Kt) then fixes resistance R = (Kt/Km)^2 and inductance.
Gear stages are 9:1 at most, each costing efficiency, length and mass; gear friction scales with rated torque and stage count.
"""
import json, math, sys

LIMITS = {"stator_d_mm": [36.0, 86.0], "stack_mm": [4.0, 30.0], "kt_Nm_per_A": [0.008, 0.25], "ratio": [4.0, 80.0], "i_max_A": [5.0, 40.0]}
BASE = {"stator_d_mm": 60.0, "stack_mm": 15.0, "kt_Nm_per_A": 0.032, "ratio": 30.0, "i_max_A": 25.0}
ENVELOPE = {"od_mm": 92.0, "length_mm": 48.0}
VERSION = "family-v2"


def check(a):
  if not isinstance(a, dict):
    return "the spec must be a JSON object"
  for k, (lo, hi) in LIMITS.items():
    if not isinstance(a.get(k), (int, float)):
      return f"missing number {k}"
    if not lo <= a[k] <= hi:
      return f"{k}={a[k]} outside {[lo, hi]}"
  if set(a) - set(LIMITS):
    return f"unknown fields {sorted(set(a) - set(LIMITS))}"
  return None


def derive(a):
  D, Ls, Kt, N, I = (float(a[k]) for k in LIMITS)
  le = 0.2 * D
  km = 0.15 * (D / 60) ** 2 * (Ls / 15) / math.sqrt((Ls + le) / 27.0)
  R = (Kt / km) ** 2
  stages = max(1, math.ceil(math.log(N) / math.log(9) - 1e-9))
  eta = 0.92 ** stages
  tau_cap = Kt * N * eta * I
  m_motor = 2.2e-6 * D ** 2 * (Ls + le)
  m_gear = 0.06 + 0.0115 * tau_cap + 0.04 * stages
  J = 2.4e-13 * D ** 4 * Ls
  return {"Kt": Kt, "N": N, "eta": eta, "I_max": I, "R": R, "L": R * 0.48e-3 * (D / 60), "km": km, "stages": stages,
          "tau_cap": tau_cap, "mass_kg": m_motor + m_gear, "m_motor": m_motor, "J_rotor": J, "J_reflected": N ** 2 * J,
          "friction_Nm": 0.02 * tau_cap * stages, "length_mm": Ls + le + 7 * stages + 6, "od_mm": D + 6,
          "R_th": 2.5 * (60 / D) * (27 / (Ls + le)), "C_th": 450.0 * m_motor}


if __name__ == "__main__":
  a = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else BASE
  print(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in derive(a).items()}, indent=1))
