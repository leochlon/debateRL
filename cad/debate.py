"""YES/NO design debates: the round messages and the official judge steps, run from code so nothing is hand-relayed.

Messages are built only from protocol fields (the current baseline, judge verdicts, the motive check and judge-confirmed
attack results); agents' free-text replies are logged, never relayed. Official judge calls use their own cache.

  debaterl debate PROBLEM start RUN [--known K.json]   baseline r0 and its report into place; K: scenarios re-run on it,
                                                        whose confirmed failures go to YES in round 1
  debaterl debate PROBLEM yes RUN R                    print YES's message for round R
  debaterl debate PROBLEM judge-yes RUN R              official check + motive on YES's round-R files (snapshotted first)
  debaterl debate PROBLEM no RUN R                     print NO's message
  debaterl debate PROBLEM judge-no RUN R               official run of NO's round-R attack; records the outcome

PROBLEM is hip (the bracket) or actuator. RUN starts with baseline_r0.py (actuator: baseline_r0.json). With a meter serving
in RUN/meter (debaterl meter serve RUN/meter --names yes,no --budget scenarios=N), each side simulates on its budget.
"""
import argparse, datetime, json, os, shutil, subprocess, sys
from debaterl import meter
from debaterl.helpers import save

HERE = os.path.dirname(os.path.abspath(__file__))

HIP = """# The bracket debate

Part: the right hip bracket of a hip exoskeleton, as parametric build123d code. Read {src}/hip_claims.json first: the frame,
the interfaces, every claim with its threshold, the agreed scenario ranges, the model settings a challenger may change, the
judge's own grid, and what the simulation does not cover.

Two sides. YES proposes one change per round to the current baseline design (edits the build123d code). NO tries to break
the change with scenarios inside the agreed ranges. The judge does not read arguments. It rebuilds the geometry and runs
each scenario through the simulators: exact OpenCASCADE geometry, MuJoCo (the exo on a wearer following walking or
sit-to-stand motion), ngspice (the actuator drive), and CalculiX FEA. A failure either reproduces or it doesn't.

A change is accepted, and becomes the new baseline, only if all three hold:
 1. the change passes every claim on the judge's grid ({cli} check design.py reports "pass": true);
 2. the motive is two-sided: a claim instance YES names fails on the baseline and holds on the change. Allowed motives:
    {{"kind": "scenario", "scenario": {{...}}}}   a scenario that fails on the baseline and passes on the change
    {{"kind": "mass", "target_g": m}}             baseline heavier than m grams, change at most m grams
    {{"kind": "geometry", "claim": "G6_stops"}}   a scenario-free claim the baseline fails and the change passes
 3. NO then submits up to 5 scenarios (each may also set model settings inside the agreed ranges: finer mesh, local
    refinement, a smaller MuJoCo timestep or ngspice step) and none of them reproduces a failure.

Objective over the rounds: the lightest bracket that passes everything. Comfort, feel and physical testing are outside
this debate.

Tools (deterministic, results cached by design file hash):
  {cli} geometry design.py                 scenario-free claims G1-G7, mass, stop angles (about 10 s)
  {cli} check design.py                    every claim on the 36-scenario grid (a new design needs ~1-2 min: FEA mesh + solve)
  {cli} run design.py scenarios.json       a JSON list of scenarios, e.g.
      [{{"activity": "walk", "height_m": 1.95, "speed_mps": 1.8, "slope_deg": 10, "assist_Nm": 20, "battery_V": 25.2}},
       {{"activity": "sts", "height_m": 1.9, "chair_height_m": 0.38, "assist_Nm": 20, "battery_V": 19.8,
         "settings": {{"mesh": {{"h": 2.0, "refine": [{{"center": [5, 0, -100], "radius": 6, "h": 0.5}}]}}, "mujoco_timestep_s": 0.0002}}}}]
  {cli} motive baseline.py design.py motive.json
The simulator source is readable in {src} (read it if useful; do not change anything outside your own folder). Each FEA
mesh setting is a fresh solve (about 30 s at the default mesh, several minutes for the finest settings), so choose
refinements deliberately. Use at most 3 debaterl commands at once.
{run_rules}"""

ACTUATOR = """# The actuator debate

Part: the hip actuator of a hip exoskeleton (a frameless brushless motor plus a planetary gearbox), described by five
numbers in a JSON spec: stator_d_mm, stack_mm, kt_Nm_per_A (winding torque constant, motor side), ratio, i_max_A.
{src}/hip_motor.py turns them into resistance, inductance, rotor inertia, gear stages, efficiency, friction, torque cap, mass,
size and thermal resistance and capacity; its scaling constants are stated assumptions. Ranges are in motor.LIMITS.

The bracket that carries the actuator is fixed: {src}/hip_bracket.py (the result of an earlier debate). The judge runs
the same scenarios as before ({src}/hip_claims.json "scenarios": walking at any height, speed, slope, assist gain and battery
voltage; sit-to-stand at any height, chair, assist gain and battery), through MuJoCo (the actuator's mass, reflected rotor
inertia and gear friction sit on the exo hinge), ngspice (battery, PI current loop, winding R and L, back-EMF) and
CalculiX FEA on the fixed bracket. Read {src}/hip_actuator.py for the exact claims:
  M1 fits the actuator keep-out: outer diameter <= 92 mm and length <= 48 mm;
  M2 the drive delivers the controller's demanded torque within 1 N*m at every instant;
  M3 winding temperature <= 110 C after 30 minutes at 35 C ambient (walking continuously; sit-to-stand once every 30 s);
  M4 while walking, the actuator's drag at the hip (reflected inertia times link acceleration, plus gear friction) <= 3 N*m;
  and the bracket keeps K1, K2, S1, S3 under the actuator's mass, inertia and torque.

Two sides. YES proposes one change of spec per round; NO tries to break it with scenarios inside the agreed ranges
(and model settings inside their ranges). A change is accepted, and becomes the new baseline, only if
 1. it passes every claim on the 36-scenario grid ({cli} check spec.json reports "pass": true);
 2. its motive is two-sided: {{"kind": "mass", "target_g": m}} (baseline heavier than m grams, change at most m) or
    {{"kind": "scenario", "scenario": {{...}}}} (fails on the baseline, passes on the change);
 3. none of NO's up to 5 scenarios reproduces a failure.
Objective: the lightest actuator that passes everything.

Tools (deterministic, cached):
  {cli} check spec.json                     all claims on the grid (~20-60 s)
  {cli} run spec.json scenarios.json        a JSON list of scenarios (same format as the bracket debate, e.g.
      [{{"activity": "walk", "height_m": 1.5, "speed_mps": 1.8, "slope_deg": 10, "assist_Nm": 20, "battery_V": 19.8}}])
  {cli} motive base.json spec.json motive.json
  python3 {src}/hip_motor.py spec.json      derived actuator properties (runs locally, instant)
{run_rules}"""

YES = """You are YES in a design debate. {rules}
Your folder is {dir}; work only there and do not read other folders under {run}.

Round {round} of {rounds}. The current baseline is {baseline}. The judge's report on it is
{report}{history}

Write your proposed {what} to {dir}/r{round}_{submit} and your motive to {dir}/r{round}_motive.json. {how}
Then expect NO to search hard for a failing scenario anywhere in the agreed ranges{search}. You have a hard limit of
{minutes} minutes: at the deadline the judge takes whatever files are in your folder and you are stopped. Reply with
{reply}, the motive, and one sentence on how you checked it against NO's likely attacks."""

NO = """You are NO in a design debate. {rules}
Your folder is {dir}; work only there and do not read other folders under {run}.

Round {round} of {rounds}. YES proposes {design} to replace the baseline {baseline}. The judge's verdict on the change:
{verdict}

Break it: find scenarios inside the agreed ranges (with model settings inside the agreed ranges if you like) where the
change fails a claim. Write up to 5 of them as a JSON list to {dir}/r{round}_attack.json. The judge runs each; only
reproduced failures count, out-of-range entries are rejected, and claimed failures that do not reproduce are recorded
against you. You have a hard limit of {minutes} minutes: at the deadline the judge takes the attack file as it stands
and you are stopped, so write your best scenarios early and improve them. Reply with the file and one sentence per
scenario."""

METERED = ("Every simulation goes through the meter, against your budget for the whole debate ({left} left now): `check` costs\n"
           "36 scenarios, `run` one per scenario, a scenario motive 2, anything else nothing. A command that would overdraw the\n"
           "budget is refused; `debaterl meter budget` shows what is left.\n")

HOW = ("Check the design with `{cli} check` (it must pass) and the motive with `{cli} motive`, because the judge runs\n"
       "exactly these.")


def summary(rep):
  return {k: rep.get(k) for k in ("pass", "claims", "mass_g", "n_failing")}


PROBLEMS = {
  "hip": {"judge": "hip", "ext": ".py", "submit": "design.py", "what": "design", "rules": HIP, "how": HOW, "search": ", including with finer meshes",
          "reply": "the design's mass, its stop angles", "summary": summary,
          "metrics": lambda r: f"mass {r.get('mass_g')} g, stops {r.get('stops_deg')}"},
  "actuator": {"judge": "actuator", "ext": ".json", "submit": "spec.json", "what": "spec", "rules": ACTUATOR, "how": HOW, "search": "",
               "reply": "the spec and its mass", "summary": summary, "metrics": lambda r: f"mass {r.get('mass_g')} g"},
}
MOTIVE_KEYS = ("motive", "holds", "rejected", "baseline_mass_g", "design_mass_g", "baseline_fails", "design_passes")
PENDING = "pending NO's attack"


def cache(side):
  return os.path.join(os.environ.get("DEBATERL_CACHE", "~/.cache/debaterl"), side)


def judge(P, *args):
  """An official judge call (local, or on DEBATERL_HOST). A judge that does not answer stops the step: nothing is recorded."""
  p = subprocess.run([sys.executable, "-m", "debaterl", P["judge"], *args], capture_output=True, text=True,
                     env={**os.environ, "DEBATERL_CACHE": cache("official"), "DEBATERL_METER": ""})
  try:
    return json.loads(p.stdout)
  except ValueError:
    raise RuntimeError(f"judge call {args[0]} failed:\n" + (p.stdout + p.stderr)[-2000:]) from None


def paths(run):
  return {k: os.path.join(run, k) for k in ("yes", "no", "judge")} | {"ledger": os.path.join(run, "ledger.json")}


def ledger(run):
  return json.load(open(paths(run)["ledger"]))


def rules(P, side, run):
  host = os.environ.get(f"DEBATERL_HOST_{side.upper()}", os.environ.get("DEBATERL_HOST"))
  env = f"DEBATERL_CACHE={cache(side)}" + (f" DEBATERL_HOST={host}" if host else "")
  remote = (f", or by ssh to {host} ({os.environ.get('DEBATERL_REMOTE_PYTHON', '~/debateRL/.venv/bin/python')} -m debaterl, in a "
            "directory of your own under /tmp there)") if host else ""
  metered, state = "", os.path.join(run, "meter")
  if os.path.exists(os.path.join(state, "url.txt")):
    env, remote = " ".join(f"{k}={v}" for k, v in meter.env(state)[side].items()), ""
    metered = METERED.format(left=", ".join(f"{n} {k}" for k, n in meter.books(state)[side]["left"].items()))
  run_rules = (f"Run all simulation through debaterl, with {env} in front of every command{remote}. Do not install packages\n"
               "or run simulations any other way, keep local files inside your folder, and do not change them after replying: the\n"
               "judge reads them as soon as you reply.")
  return P["rules"].format(cli=f"debaterl {P['judge']}", src=HERE, run_rules=metered + run_rules)


def clock(minutes):
  now = datetime.datetime.now(datetime.timezone.utc)
  end = now + datetime.timedelta(minutes=minutes)
  return (f"\n\nClock: this turn started at {now:%H:%M:%S} UTC and your deadline is {end:%H:%M:%S} UTC (check with `date -u`). "
          f"Agents in earlier rounds stopped after 2-3 minutes believing their time was up; use the time you have.")


def yes(P, run, r, rounds=3, minutes=10):
  p, L = paths(run), ledger(run)
  base, rep = os.path.join(p["yes"], f"baseline_r{r - 1}{P['ext']}"), os.path.join(p["yes"], f"baseline_r{r - 1}_report.json")
  hist = ""
  if L.get("baseline_confirmed") and os.path.basename(L["baseline"]) == "baseline_r0" + P["ext"]:
    hist += (f"\n\nThe baseline passes the grid, but the judge confirms these failing scenarios on it (known attacks, re-run by "
             f"the judge on this baseline): {json.dumps(L['baseline_confirmed'])}.")
  if L["rounds"]:
    e = L["rounds"][-1]
    hist += (f"\n\nOutcome of round {e['round']}: your change was {e['outcome']}."
             + (f" Judge check: {json.dumps(e['check_summary'])}." if e.get("check_summary") else "")
             + (f" Motive check: {json.dumps(e['motive'])}." if e.get("motive") else "")
             + (f" Judge-reported failures on the grid: {json.dumps(e['failing'])}." if e.get("failing") else "")
             + (f" NO's judge-confirmed failing scenarios: {json.dumps(e['confirmed'])}." if e.get("confirmed") else ""))
  return YES.format(rules=rules(P, "yes", run), dir=p["yes"], run=run, round=r, rounds=rounds, baseline=base, report=rep, history=hist,
                    what=P["what"], submit=P["submit"], how=P["how"].format(cli=f"debaterl {P['judge']}"), search=P["search"], minutes=minutes,
                    reply=P["reply"]) + clock(minutes)


def no(P, run, r, rounds=3, minutes=10):
  p, L = paths(run), ledger(run)
  check = json.load(open(os.path.join(p["judge"], f"r{r}_check.json")))
  verdict = (f"the change passes every claim on the judge's grid; {P['metrics'](check)}. The judge's report on it is "
             f"{os.path.join(p['no'], f'r{r}_proposed_report.json')}")
  return NO.format(rules=rules(P, "no", run), dir=p["no"], run=run, round=r, rounds=rounds, verdict=verdict, minutes=minutes,
                   design=os.path.join(p["no"], f"r{r}_proposed_design{P['ext']}"), baseline=L["baseline"]) + clock(minutes)


def start(P, run, known=None):
  p = paths(run)
  for k in ("yes", "no", "judge"):
    os.makedirs(p[k], exist_ok=True)
  base = os.path.join(run, "baseline_r0" + P["ext"])
  rep = judge(P, "check", base)
  res = None
  if known:
    snap = os.path.join(p["judge"], "baseline_r0_known.json")
    shutil.copy(known, snap)
    res = judge(P, "run", base, snap)
  save(os.path.join(p["judge"], "baseline_r0_report.json"), rep)
  shutil.copy(base, os.path.join(p["yes"], "baseline_r0" + P["ext"]))
  save(os.path.join(p["yes"], "baseline_r0_report.json"), rep)
  L = {"rounds": [], "baseline": base}
  if res is not None:
    save(os.path.join(p["judge"], "baseline_r0_known_result.json"), res)
    L["baseline_confirmed"] = res.get("confirmed", [])
    L["baseline_known_counts"] = {k: len(res.get(k, [])) for k in ("confirmed", "not_confirmed", "rejected", "errors")}
  save(p["ledger"], L)
  return P["summary"](rep) | {"known": L.get("baseline_known_counts")}


def next_baseline(P, run, r, L):
  p = paths(run)
  rep = judge(P, "check", L["baseline"])
  shutil.copy(L["baseline"], os.path.join(p["yes"], f"baseline_r{r}{P['ext']}"))
  save(os.path.join(p["yes"], f"baseline_r{r}_report.json"), rep)


def problems(P, d, m):
  """What is wrong with YES's files before the judge sees them: JSON that does not parse."""
  out = []
  for f in ([d] if d.endswith(".json") else []) + [m]:
    try:
      json.load(open(f))
    except ValueError as e:
      out.append(f"{os.path.basename(f)} is not valid JSON: {e}")
  return out


def judge_yes(P, run, r):
  p, L = paths(run), ledger(run)
  d, m = os.path.join(p["yes"], f"r{r}_{P['submit']}"), os.path.join(p["yes"], f"r{r}_motive.json")
  entry = {"round": r, "design": d}
  if not (os.path.exists(d) and os.path.exists(m)):
    entry["outcome"] = "not submitted"
  elif bad := problems(P, d, m):
    shutil.copy(d, os.path.join(p["judge"], f"r{r}_{P['submit']}"))
    entry.update(outcome="rejected: invalid submission", problems=bad)
  else:
    snap_d, snap_m = os.path.join(p["judge"], f"r{r}_design{P['ext']}"), os.path.join(p["judge"], f"r{r}_motive.json")
    shutil.copy(d, snap_d)
    shutil.copy(m, snap_m)
    check, mot = judge(P, "check", snap_d), judge(P, "motive", L["baseline"], snap_d, snap_m)
    save(os.path.join(p["judge"], f"r{r}_check.json"), check)
    save(os.path.join(p["judge"], f"r{r}_motive_result.json"), mot)
    entry.update(design=snap_d, check_summary=P["summary"](check), motive={k: mot[k] for k in MOTIVE_KEYS if k in mot})
    if not check.get("pass"):
      entry.update(outcome="rejected by the judge's check", failing=check.get("failing", [])[:5])
    elif not mot.get("holds"):
      entry["outcome"] = "rejected: the motive is not two-sided"
    else:
      entry["outcome"] = PENDING
      shutil.copy(snap_d, os.path.join(p["no"], f"r{r}_proposed_design{P['ext']}"))
      save(os.path.join(p["no"], f"r{r}_proposed_report.json"), check)
  if entry["outcome"] != PENDING:
    next_baseline(P, run, r, L)
  L["rounds"].append(entry)
  save(p["ledger"], L)
  return {k: entry.get(k) for k in ("round", "outcome", "check_summary", "motive")}


def judge_no(P, run, r):
  p, L = paths(run), ledger(run)
  entry = L["rounds"][-1]
  if entry["round"] != r or entry["outcome"] != PENDING:
    raise SystemExit(f"round {r} is not waiting for NO's attack")
  a, snap = os.path.join(p["no"], f"r{r}_attack.json"), os.path.join(p["judge"], f"r{r}_attack.json")
  note = "no attack submitted" if not os.path.exists(a) else None
  if note is None:
    shutil.copy(a, snap)
    try:
      json.load(open(snap))
    except ValueError as e:
      note = f"the attack file is not valid JSON: {e}"
  res = {"confirmed": [], "not_confirmed": [], "rejected": [], "note": note} if note else judge(P, "run", entry["design"], snap)
  if res.get("error"):
    raise RuntimeError(f"the judge could not run the attack, nothing recorded: {res['error']}")
  entry["confirmed"] = res.get("confirmed", [])
  entry["attack"] = {k: len(res.get(k, [])) for k in ("confirmed", "not_confirmed", "rejected", "errors")}
  if res.get("errors"):
    entry["judge_error"] = [e.get("error") for e in res["errors"]]
  if entry["confirmed"]:
    entry["outcome"] = "rejected: NO found a failing scenario that the judge reproduced"
  else:
    entry["outcome"] = "accepted"
    L["baseline"] = os.path.join(run, f"baseline_r{r}{P['ext']}")
    shutil.copy(entry["design"], L["baseline"])
  next_baseline(P, run, r, L)
  save(os.path.join(p["judge"], f"r{r}_attack_result.json"), res)
  save(p["ledger"], L)
  return {k: entry.get(k) for k in ("round", "outcome", "attack")}


def main(argv):
  ap = argparse.ArgumentParser(prog="debaterl debate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("problem", choices=PROBLEMS)
  ap.add_argument("step", choices=("start", "yes", "judge-yes", "no", "judge-no"))
  ap.add_argument("run")
  ap.add_argument("round", nargs="?", type=int, default=0)
  ap.add_argument("--known", help="start: scenarios to re-run on the baseline")
  ap.add_argument("--rounds", type=int, default=3)
  ap.add_argument("--minutes", type=int, default=10)
  a = ap.parse_args(argv)
  P, run = PROBLEMS[a.problem], os.path.abspath(a.run)
  if a.step in ("yes", "no"):
    print((yes if a.step == "yes" else no)(P, run, a.round, a.rounds, a.minutes))
  elif a.step == "start":
    print(json.dumps(start(P, run, a.known and os.path.abspath(a.known)), default=float))
  else:
    print(json.dumps((judge_yes if a.step == "judge-yes" else judge_no)(P, run, a.round), default=float))
  return 0
