"""Design debate: two Claude designers (A, B) plus solo baselines, each a tool-using agent with its own bash shell and
folder. Round 1 design; round 2 attack the opponent's design (and optionally submit a smaller one); round 3 final design;
round 4 attack the opponent's final design. Solo designers get two turns. The judge checks every claim by simulation;
the smallest final design that passes the judge and survives the opponent's round-4 attack wins.

  python -m debaterl.chip.debate --out runs/1 [--model claude-fable-5-2] [--effort high] [--solo 2] [--minutes 25]
                                  [--budget perf=1500,offset=1500]   each designer simulates through a meter on this budget
"""
import argparse, json, os, queue, signal, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
import anthropic
from debaterl import meter
from debaterl.chip import judge

PY = sys.executable
TOOLS = [{"type": "bash_20250124", "name": "bash"}]
CMD_TIMEOUT, MAX_OUT = 600, 30000

LOCAL_TOOLS = f"""Tools (shared simulator, deterministic):
  {PY} -m debaterl.chip.sim <design.json> --process TT|FF|SS --temp <C> --vdd <V>
      performance at one operating point: gain_db, gbw_hz, pm_deg, power_w, area_um2, pass, failures (~15 ms per run)
  {PY} -m debaterl.chip.sim <design.json> --seed <S>
      input-referred offset (offset_v) of mismatch sample S at TT, 27 C, 1.8 V
  You may also import the module (from debaterl.chip import sim) and call sim.simulate(p, process, temp, vdd)
  and sim.offset(p, seed) directly. {PY} has numpy and scipy."""

METERED_TOOLS = f"""Tools (metered simulator, deterministic):
  {PY} -m debaterl chip sim <design.json> --process TT|FF|SS --temp <C> --vdd <V>
      performance at one operating point: gain_db, gbw_hz, pm_deg, power_w, area_um2, pass, failures (costs 1 perf)
  {PY} -m debaterl chip sim <design.json> --seed <S>
      input-referred offset (offset_v) of mismatch sample S at TT, 27 C, 1.8 V (costs 1 offset)
  {PY} -m debaterl meter budget
      what is left of your budget: @BUDGET@ for the whole run, including any checks of another designer's design
  A command that would overdraw the budget is refused. Simulate only through these commands. {PY} has numpy and scipy."""

TASK = """# Task: size a two-stage Miller op-amp (fixed topology)

You choose only the numbers in a design file (JSON):
W1 L1 (input pair, um), W3 L3 (PMOS mirror load), W5 L5 (tail current source), W6 L6 (second-stage PMOS),
W7 L7 (second-stage NMOS current sink), W8 L8 (bias diode), Ib (bias current, uA), Cc (compensation cap, pF), Rz (nulling resistor, ohm).
Bounds: every W in [0.5, 200] um, every L in [0.18, 10] um, Ib in [1, 200] uA, Cc in [0.5, 10] pF, Rz in [0, 100000] ohm.

@TOOLS@

Mismatch model: sample S draws z = numpy.random.default_rng(S).standard_normal(8), one value per transistor in the order
M1, M2 (input pair), M3, M4 (mirror), M5 (tail), M6, M7 (second stage), M8 (bias diode). Each transistor's threshold
magnitude shifts by z_i * 5 mV*um / sqrt(W_i * L_i).

Spec (all required):
  1. At EVERY operating point in the range (process TT, FF or SS; any temperature from -40 to 125 C; any supply from
     1.7 to 1.9 V; load 2 pF): DC gain >= 70 dB, gain-bandwidth >= 50 MHz, phase margin >= 65 degrees, power <= 1 mW,
     output correctly biased.
  2. For EVERY mismatch sample S in [0, 1000000): |input offset| <= 3 mV (at TT, 27 C, 1.8 V).

Objective: the smallest area_um2 (gate area plus 500 um^2 per pF of Cc) that meets the whole spec.
A failing operating point or a failing seed is a counterexample to the claim that a design meets the spec.
Work only inside your own folder; do not read other folders.
"""

SYSTEM = """You work through a bash shell on a machine with ngspice and {py}. The shell starts in your folder {dir} and keeps
its state between commands. A command is stopped after {t} minutes, so run longer jobs in the background with output
redirected to a file and check on them."""

HEAD = ("Read {task} (task, spec, mismatch model and simulator). Your folder is {dir}/; work only there. The machine has "
        "{cores} cores shared with {others} other designers, so use at most 3 parallel simulation processes.\n\n")
R1 = ("You are designer {me} in a design debate. " + HEAD +
      """There is an opponent, designer {opp}, on the same task. Debate rules:
- Round 1 (now): submit your best design. A judge checks claims only by simulation.
- Round 2: you will see {opp}'s design and try to break it: name up to 3 failing operating points and/or up to 5 failing \
mismatch seeds; the judge simulates each and only confirmed failures count. You may also submit a smaller design that meets \
the whole spec to beat {opp}'s area.
- Round 3: you will see confirmed failures of your design and may revise it.
- Round 4: a final round of attacks on each other's final design.
- Your final design counts only if {opp} cannot show a confirmed failing point or seed. The smallest design that survives wins.

So check your design against the whole spec, including the mismatch claim over all one million seeds, before submitting. \
Spend at most about {min} minutes.

Write your design to {dir}/r1_design.json and reply with the JSON, its area, and one or two sentences on how you checked \
the operating range and the million-seed offset claim.""")
T1 = ("You are a circuit designer. " + HEAD +
      """A judge will check your final design by simulation against the whole spec, including the mismatch claim over all one \
million seeds, so check it thoroughly. You will get one more turn afterwards to check and improve it. Spend at most about \
{min} minutes.

Write your design to {dir}/t1_design.json and reply with the JSON, its area, and one or two sentences on how you checked \
the operating range and the million-seed offset claim.""")
ATTACK = """Designer {opp}'s {what}, area {area} um^2:
{design}
{opp} claims it meets the whole spec. Try to break it: up to 3 failing operating points and/or up to 5 failing mismatch \
seeds; the judge simulates each and only confirmed failures count (false accusations are recorded against you).{extra} \
Spend at most about {min} minutes, at most 3 parallel processes. Write {dir}/{file} as {{"points": [{{"process": "SS", \
"temp": 125, "vdd": 1.7}}, ...], "seeds": [...]{schema}}} and reply with it plus one sentence per item."""
R2 = "Round 2 (attack). " + ATTACK
R4 = "Round 4 (final attack). " + ATTACK
T2 = """Turn 2 (your last). Check your design further against the whole spec and improve it if you can (smaller area, \
still meeting everything). Spend at most about {min} minutes, at most 3 parallel processes. Write your final design to \
{dir}/t2_design.json and reply with it, its area and one sentence on what you checked."""
R3 = """Round 3 (revise). Your current design is {which}, {area} um^2. Judge: {verdict}. {opp}'s round-2 attack on your \
round-1 design: {attacked}. {opp}'s current design is {oarea} um^2:
{odesign}
Submit your final design (it may be your current one). In round 4, {opp} will attack it with up to 3 operating points and \
5 seeds; your final design counts only if no confirmed failure is found, and the smallest surviving design wins. Spend at \
most about {min} minutes, at most 3 parallel processes. Write {dir}/r3_design.json and reply with it, its area and one \
sentence on what you checked."""


class Shell:
  def __init__(self, cwd, env=None):
    self.cwd, self.env, self.p = cwd, env, None

  def start(self):
    self.p = subprocess.Popen(["bash", "--noprofile", "--norc"], cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True, env=self.env)
    self.q = queue.Queue()
    threading.Thread(target=lambda p=self.p, q=self.q: [q.put(line) for line in p.stdout], daemon=True).start()

  def kill(self):
    if self.p is not None and self.p.poll() is None:
      os.killpg(self.p.pid, signal.SIGKILL)
    self.p = None

  def run(self, cmd):
    if self.p is None or self.p.poll() is not None:
      self.start()
    mark = f"__end_{uuid.uuid4().hex}__"
    self.p.stdin.write(f"{cmd}\n__rc=$?; echo; echo {mark} $__rc\n")
    self.p.stdin.flush()
    out, end = [], time.time() + CMD_TIMEOUT
    while True:
      try:
        line = self.q.get(timeout=max(0.1, end - time.time()))
      except queue.Empty:
        self.kill()
        return "".join(out) + f"\n[stopped after {CMD_TIMEOUT // 60} minutes; the shell was restarted]"
      if line.startswith(mark):
        s, rc = "".join(out)[:-1], line.split()[-1]
        if len(s) > MAX_OUT:
          s = s[:2000] + f"\n[... {len(s) - MAX_OUT} characters omitted ...]\n" + s[-(MAX_OUT - 2000):]
        return (s or "(no output)") + ("" if rc == "0" else f"\n[exit code {rc}]")
      out.append(line)


class Agent:
  def __init__(self, name, folder, client, model, effort, env=None):
    os.makedirs(folder, exist_ok=True)
    self.name, self.dir, self.client, self.model, self.effort = name, folder, client, model, effort
    self.shell, self.messages, self.usage = Shell(folder, env), [], {}
    self.system = SYSTEM.format(py=PY, dir=folder, t=CMD_TIMEOUT // 60)
    self.log = open(os.path.join(folder, "transcript.jsonl"), "a")

  def call(self, **kw):
    with self.client.beta.messages.stream(model=self.model, max_tokens=64000, system=self.system, tools=TOOLS,
                                          messages=self.messages, output_config={"effort": self.effort},
                                          cache_control={"type": "ephemeral"},
                                          betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw) as s:
      m = s.get_final_message()
    for k, v in m.usage.model_dump().items():
      if isinstance(v, int):
        self.usage[k] = self.usage.get(k, 0) + v
    self.log.write(json.dumps({"t": time.time(), "stop": m.stop_reason, "content": [b.model_dump() for b in m.content]}, default=str) + "\n")
    self.log.flush()
    self.messages.append({"role": "assistant", "content": m.content})
    return m

  def turn(self, prompt, minutes):
    n0 = len(self.messages)
    self.messages.append({"role": "user", "content": prompt})
    self.log.write(json.dumps({"t": time.time(), "prompt": prompt}) + "\n")
    deadline, grace = time.time() + 90 * minutes, 8  # the prompt asks for `minutes`; hard stop at 1.5x
    try:
      while True:
        m = self.call() if grace > 0 else self.call(tool_choice={"type": "none"})
        calls = [b for b in m.content if b.type == "tool_use"]
        if m.stop_reason == "refusal":
          return f"[refused: {m.stop_details}]"
        if not calls:
          if m.stop_reason == "pause_turn":
            continue
          if m.stop_reason == "max_tokens":
            self.messages.append({"role": "user", "content": "Continue."})
            continue
          return "".join(b.text for b in m.content if b.type == "text")
        res = []
        for b in calls:
          if m.stop_reason == "max_tokens":
            out, err = "Your reply was cut off at the output limit; send a shorter command.", True
          elif time.time() > deadline:
            grace -= 1
            out, err = "Time is up: write your answer file now and reply.", True
          elif b.input.get("restart"):
            self.shell.kill()
            out, err = "Shell restarted.", False
          else:
            out, err = self.shell.run(b.input.get("command", "")), False
          self.log.write(json.dumps({"t": time.time(), "cmd": b.input, "out": out[:4000]}) + "\n")
          res.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        self.messages.append({"role": "user", "content": res})
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
      del self.messages[n0:]
      return f"[API error: {type(e).__name__}: {e}]"


def load(path):
  try:
    return json.load(open(path))
  except (OSError, ValueError):
    return None


def check(design, a):
  try:
    return judge.attack(design, a)
  except (TypeError, ValueError, KeyError, AttributeError) as e:
    return {"confirmed": [], "rejected": [], "error": f"unreadable attack: {e!r}"}


def best(final, names):
  """The smallest surviving design among names."""
  return min([n for n in names if n in final and final[n]["survives"]], key=lambda n: final[n]["area_um2"], default=None)


def verdict(sc):
  if sc["pass"]:
    return "passed (105 PVT points, million-seed offset screen)"
  if sc.get("failures"):
    return "failed: " + "; ".join(sc["failures"])
  o = sc["offset"]
  return (f"failed: {sc['pvt']['grid_failures']} of 105 PVT points fail {sc['pvt']['first_failures']}; "
          f"worst offset {o.get('worst_offset_v')} V at seed {o.get('worst_seed')}")


def attacked(r):
  if r is None:
    return "no attack submitted"
  if not r["confirmed"]:
    return f"no confirmed failure ({len(r['rejected'])} claimed failures did not hold)"
  keys = ("process", "temp", "vdd", "seed", "offset_v", "failures")
  return "confirmed failures: " + json.dumps([{k: x.get(k) for k in keys if k in x} for x in r["confirmed"]])


def run(out, model="claude-fable-5-2", effort="high", solo=2, minutes=25, budget=None):
  out = os.path.abspath(out)
  os.makedirs(out, exist_ok=True)
  task = os.path.join(out, "TASK.md")
  open(task, "w").write(TASK.replace("@TOOLS@", METERED_TOOLS.replace("@BUDGET@", budget) if budget else LOCAL_TOOLS))
  judge.all_z()
  client = anthropic.Anthropic(max_retries=5)
  names = ["A", "B"] + [f"S{i + 1}" for i in range(solo)]
  env = {}
  if budget:
    meter.start(os.path.join(out, "meter"), names, budget)
    env = {n: {**os.environ, **v} for n, v in meter.env(os.path.join(out, "meter")).items()}
  ag = {n: Agent(n, os.path.join(out, n), client, model, effort, env.get(n)) for n in names}
  opp, later = {"A": "B", "B": "A"}, round(minutes * 0.8)
  led = {"model": model, "effort": effort, "budget": budget, "events": []}

  def ctx(n, **k):
    return dict(me=n, opp=opp.get(n), dir=ag[n].dir, task=task, cores=os.cpu_count(), others=len(names) - 1, **k)

  def note(who, kind, **k):
    led["events"].append({"t": time.time(), "who": who, "kind": kind, **k})
    json.dump(led, open(os.path.join(out, "ledger.json"), "w"), indent=1, default=float)

  def turns(prompts, mins):
    with ThreadPoolExecutor(max(1, len(prompts))) as ex:
      replies = list(ex.map(lambda n: ag[n].turn(prompts[n], mins), prompts))
    for n, r in zip(prompts, replies):
      note(n, "reply", text=r)

  def submit(n, design, label):
    try:
      sc = judge.score(design)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
      sc = {"area_um2": None, "pass": False, "failures": [f"unreadable design: {e!r}"]}
    note(n, "design", label=label, design=design, score=sc)
    print(f"{n} {label}: {sc['area_um2']} um^2, {'PASS' if sc['pass'] else 'FAIL'}", flush=True)
    return {"design": design, "score": sc, "label": label}

  print("round 1", flush=True)
  turns({n: (R1 if n in opp else T1).format(**ctx(n, min=minutes)) for n in names}, minutes)
  cur = {}
  for n in names:
    d = load(os.path.join(ag[n].dir, "r1_design.json" if n in opp else "t1_design.json"))
    cur[n] = submit(n, d, "round-1 design" if n in opp else "turn-1 design") if d else None
  r1 = {n: cur[n] for n in opp}

  print("round 2", flush=True)
  p = {n: R2.format(**ctx(n, min=later, what="round-1 design", area=r1[opp[n]]["score"]["area_um2"], design=json.dumps(r1[opp[n]]["design"]),
                          file="r2_attack.json", extra=" You may also submit a smaller design that meets the whole spec to beat your own and "
                          f"{opp[n]}'s area.", schema=', "smaller_design": null or {...}'))
       for n in opp if r1[n] and r1[opp[n]]}
  p |= {n: T2.format(**ctx(n, min=later)) for n in names if n not in opp and cur[n]}
  turns(p, later)
  hits = {}
  for n in opp:
    a = load(os.path.join(ag[n].dir, "r2_attack.json")) if n in p else None
    if a:
      hits[opp[n]] = check(r1[opp[n]]["design"], a)
      note(n, "attack", round=2, target=opp[n], attack=a, result=hits[opp[n]])
      if isinstance(a, dict) and isinstance(a.get("smaller_design"), dict):
        cur[n] = submit(n, a["smaller_design"], "round-2 smaller design")
  for n in names:
    if n not in opp and (d := load(os.path.join(ag[n].dir, "t2_design.json"))):
      cur[n] = submit(n, d, "turn-2 design")

  print("round 3", flush=True)
  p = {n: R3.format(**ctx(n, min=later, which=f"your {cur[n]['label']}", area=cur[n]["score"]["area_um2"], verdict=verdict(cur[n]["score"]),
                          attacked=attacked(hits.get(n)), oarea=cur[opp[n]]["score"]["area_um2"], odesign=json.dumps(cur[opp[n]]["design"])))
       for n in opp if cur[n] and cur[opp[n]]}
  turns(p, later)
  for n in p:
    if d := load(os.path.join(ag[n].dir, "r3_design.json")):
      cur[n] = submit(n, d, "final design")

  print("round 4", flush=True)
  p = {n: R4.format(**ctx(n, min=later, what="final design", area=cur[opp[n]]["score"]["area_um2"], design=json.dumps(cur[opp[n]]["design"]),
                          file="r4_attack.json", extra="", schema=""))
       for n in opp if cur[n] and cur[opp[n]]}
  turns(p, later)
  final = {}
  for n in opp:
    if not cur[n]:
      continue
    a = load(os.path.join(ag[opp[n]].dir, "r4_attack.json")) if opp[n] in p else None
    r = check(cur[n]["design"], a) if a else None
    if a:
      note(opp[n], "attack", round=4, target=n, attack=a, result=r)
    final[n] = {"area_um2": cur[n]["score"]["area_um2"], "judge": cur[n]["score"]["pass"], "confirmed_failures": len(r["confirmed"]) if r else 0}
    final[n]["survives"] = final[n]["judge"] and final[n]["confirmed_failures"] == 0
  for n in names:
    if n not in opp and cur[n]:
      final[n] = {"area_um2": cur[n]["score"]["area_um2"], "judge": cur[n]["score"]["pass"], "survives": cur[n]["score"]["pass"]}
  led.update(final=final, winner=best(final, opp), best_solo=best(final, [n for n in names if n not in opp]), usage={n: ag[n].usage for n in names})
  note(None, "end")
  for a in ag.values():
    a.shell.kill()
  print(json.dumps({"final": final, "winner": led["winner"], "best_solo": led["best_solo"]}, indent=1))
  return led


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--model", default="claude-fable-5-2")
  ap.add_argument("--effort", default="high")
  ap.add_argument("--solo", type=int, default=2)
  ap.add_argument("--minutes", type=int, default=25, help="time asked for in round 1; later rounds get 80%%")
  ap.add_argument("--budget", help="KIND=N,... per designer (perf=1500,offset=1500): simulate only through a meter")
  a = ap.parse_args(argv)
  run(a.out, a.model, a.effort, a.solo, a.minutes, a.budget)


if __name__ == "__main__":
  main()
