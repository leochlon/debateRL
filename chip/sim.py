"""Fixed two-stage Miller op-amp, simulated with ngspice: performance at one operating point, or the input offset of
one mismatch sample. Deterministic.

  python -m resips.chip.sim design.json --process SS --temp 125 --vdd 1.7
  python -m resips.chip.sim design.json --seed 439217

design.json: W1 L1 (input pair) W3 L3 (mirror load) W5 L5 (tail) W6 L6 (second-stage PMOS) W7 L7 (second-stage NMOS sink)
W8 L8 (bias diode) in um, Ib in uA, Cc in pF, Rz in ohm. Load 2 pF.
"""
import argparse, json, os, re, subprocess, tempfile

SPEC = {"gain_db_min": 70.0, "gbw_hz_min": 50e6, "pm_deg_min": 65.0, "power_w_max": 1e-3}
PROCESS = {"TT": (0.50, 200e-6, -0.50, 80e-6), "FF": (0.45, 220e-6, -0.45, 88e-6), "SS": (0.55, 180e-6, -0.55, 72e-6)}
BOUNDS = {"W": (0.5, 200.0), "L": (0.18, 10.0), "Ib": (1.0, 200.0), "Cc": (0.5, 10.0), "Rz": (0.0, 1e5)}
KEYS = ["W1", "L1", "W3", "L3", "W5", "L5", "W6", "L6", "W7", "L7", "W8", "L8", "Ib", "Cc", "Rz"]
LAMBDA_PER_UM = 0.08  # channel-length modulation: lambda = 0.08 / L[um]
DEVICES = ["n1a", "n1b", "p3a", "p3b", "n5", "p6", "n7", "n8"]
AREA_OF = {"n1a": ("W1", "L1"), "n1b": ("W1", "L1"), "p3a": ("W3", "L3"), "p3b": ("W3", "L3"), "n5": ("W5", "L5"),
           "p6": ("W6", "L6"), "n7": ("W7", "L7"), "n8": ("W8", "L8")}
AVT_MV_UM = 5.0          # Pelgrom: sigma(dVT) = 5 mV um / sqrt(W L)
OFFSET_MAX_V = 3e-3      # |input offset| <= 3 mV for every mismatch seed in [0, 1000000)
N_SEEDS = 1000000
AC_BLOCK = '''ac dec 50 1 10e9
let phd = 180/pi*cph(v(out))
meas ac gain_db find vdb(out) at=10
meas ac ugf when vdb(out)=0 fall=1
meas ac ph_ugf find phd at=ugf
echo "AC gain_db=$&gain_db ugf=$&ugf ph_ugf=$&ph_ugf"'''


def mismatch_z(seed):
  import numpy as np
  return np.random.default_rng(int(seed)).standard_normal(len(DEVICES))


def mismatch_shifts(p, seed):
  z = mismatch_z(seed)
  return {d: float(z[i]) * AVT_MV_UM * 1e-3 / (p[AREA_OF[d][0]] * p[AREA_OF[d][1]]) ** 0.5 for i, d in enumerate(DEVICES)}


def area_um2(p):
  gates = 2 * p["W1"] * p["L1"] + 2 * p["W3"] * p["L3"] + p["W5"] * p["L5"] + p["W6"] * p["L6"] + p["W7"] * p["L7"] + p["W8"] * p["L8"]
  return gates + p["Cc"] * 500.0  # capacitor at 2 fF/um^2


def check_bounds(p):
  errs = []
  for k in KEYS:
    if k not in p:
      errs.append(f"missing {k}")
      continue
    lo, hi = BOUNDS[k[0]] if k[0] in "WL" else BOUNDS[k]
    if not (lo <= float(p[k]) <= hi):
      errs.append(f"{k}={p[k]} outside [{lo}, {hi}]")
  return errs


def netlist(p, process, temp, vdd, shifts=None, op_only=False):
  vtn, kpn, vtp, kpp = PROCESS[process]
  dv = shifts if shifts is not None else {k: 0.0 for k in DEVICES}
  m = []
  for d in DEVICES:
    vt, kp, typ, shift = (vtn, kpn, "nmos", dv[d]) if d[0] == "n" else (vtp, kpp, "pmos", -dv[d])
    m.append(f".model {d} {typ} level=1 vto={vt + shift:.9g} kp={kp} lambda={LAMBDA_PER_UM / p[AREA_OF[d][1]]:.6g} tox=4e-9 cgso=3e-10 cgdo=3e-10")
  return f"""* two-stage Miller op-amp, process {process}, {temp} C, vdd {vdd} V
{chr(10).join(m)}
.temp {temp}
VDD vdd 0 {vdd}
Ib vdd nb {p['Ib']}e-6
M8 nb nb 0 0 n8 W={p['W8']}e-6 L={p['L8']}e-6
M5 tail nb 0 0 n5 W={p['W5']}e-6 L={p['L5']}e-6
M1 x1 inn tail 0 n1a W={p['W1']}e-6 L={p['L1']}e-6
M2 x2 inp tail 0 n1b W={p['W1']}e-6 L={p['L1']}e-6
M3 x1 x1 vdd vdd p3a W={p['W3']}e-6 L={p['L3']}e-6
M4 x2 x1 vdd vdd p3b W={p['W3']}e-6 L={p['L3']}e-6
M6 out x2 vdd vdd p6 W={p['W6']}e-6 L={p['L6']}e-6
M7 out nb 0 0 n7 W={p['W7']}e-6 L={p['L7']}e-6
Rz x2 zc {max(float(p['Rz']), 1e-3)}
Cc zc out {p['Cc']}e-12
CL out 0 2e-12
Vin inp 0 DC {vdd / 2} AC 1
Lfb out inn 1e9
Cfb inn 0 1
.control
set noaskquit
set numdgt=12
op
let pwr = -i(vdd) * {vdd}
let voutdc = v(out)
echo "OP pwr=$&pwr vout=$&voutdc"
print voutdc
{"" if op_only else AC_BLOCK}
quit
.endc
.end
"""


def spice(text):
  with tempfile.TemporaryDirectory() as d:
    f = os.path.join(d, "a.cir")
    open(f, "w").write(text)
    return subprocess.run(["ngspice", "-b", f], capture_output=True, text=True, timeout=60).stdout


def num(out, key):
  m = re.search(key + "([-+0-9.eE]+)", out)
  return float(m.group(1)) if m else None


def offset_with_shifts(p, shifts):
  v = num(spice(netlist(p, "TT", 27.0, 1.8, shifts, op_only=True)), "voutdc = ")
  return None if v is None else v - 0.9


def offset(p, seed):
  """Input-referred offset (V) for mismatch sample `seed` at TT, 27 C, 1.8 V (DC operating point only)."""
  if errs := check_bounds(p):
    return {"seed": seed, "offset_v": None, "pass": False, "failures": errs}
  if not (0 <= int(seed) < N_SEEDS):
    return {"seed": seed, "offset_v": None, "pass": False, "failures": ["seed outside [0, 1000000)"]}
  vos = offset_with_shifts(p, mismatch_shifts(p, int(seed)))
  ok = vos is not None and abs(vos) <= OFFSET_MAX_V
  return {"seed": int(seed), "offset_v": vos, "pass": ok, "failures": [] if ok else [f"offset_v={vos}"]}


def simulate(p, process="TT", temp=27.0, vdd=1.8):
  if errs := check_bounds(p):
    return {"pass": False, "failures": errs, "area_um2": None}
  if process not in PROCESS or not (-40 <= float(temp) <= 125) or not (1.7 <= float(vdd) <= 1.9):
    return {"pass": False, "failures": ["operating point outside the specified range"], "area_um2": area_um2(p)}
  out = spice(netlist(p, process, float(temp), float(vdd)))
  vout = num(out, "voutdc = ")
  r = {"process": process, "temp": float(temp), "vdd": float(vdd), "gain_db": num(out, "gain_db="), "gbw_hz": num(out, "ugf="),
       "ph_ugf": num(out, "ph_ugf="), "power_w": num(out, "pwr="), "vout_dc": num(out, "vout=") if vout is None else vout,
       "area_um2": round(area_um2(p), 3)}
  r["pm_deg"] = (180.0 + r["ph_ugf"]) if r["ph_ugf"] is not None else None
  fails = []
  if r["vout_dc"] is None or abs(r["vout_dc"] - vdd / 2) > 0.05:
    fails.append(f"not biased: vout_dc={r['vout_dc']}")
  if r["gain_db"] is None or r["gain_db"] < SPEC["gain_db_min"]:
    fails.append(f"gain_db={r['gain_db']}")
  if r["gbw_hz"] is None or r["gbw_hz"] < SPEC["gbw_hz_min"]:
    fails.append(f"gbw_hz={r['gbw_hz']}")
  if r["pm_deg"] is None or r["pm_deg"] < SPEC["pm_deg_min"]:
    fails.append(f"pm_deg={r['pm_deg']}")
  if r["power_w"] is None or r["power_w"] > SPEC["power_w_max"]:
    fails.append(f"power_w={r['power_w']}")
  r["failures"], r["pass"] = fails, not fails
  return r


def parser():
  ap = argparse.ArgumentParser()
  ap.add_argument("design")
  ap.add_argument("--process", default="TT")
  ap.add_argument("--temp", type=float, default=27.0)
  ap.add_argument("--vdd", type=float, default=1.8)
  ap.add_argument("--seed", type=int, default=None, help="mismatch sample: report the input offset at TT, 27 C, 1.8 V")
  return ap


def main(argv=None):
  a = parser().parse_args(argv)
  d = json.load(open(a.design))
  print(json.dumps(offset(d, a.seed) if a.seed is not None else simulate(d, a.process, a.temp, a.vdd)))


if __name__ == "__main__":
  main()
