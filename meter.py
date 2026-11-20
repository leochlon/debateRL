"""The meter: what agents simulate, against a budget for the whole run.

Each agent has a token and a budget per kind: scenarios for the hip and actuator judges, operating points (perf) and
mismatch seeds (offset) for the chip. A judge command costs what it asks to simulate, a command that would overdraw a
budget is refused whole, and every call is logged with its files to STATE/log/NAME.jsonl. Commands run as
`python -m resips` on the meter's machine (on its RESIPS_HOST, if set). A command that fails is not charged.

  resips meter serve STATE --names yes,no --budget scenarios=300   STATE/tokens.json, STATE/url.txt
  resips meter budget                                              what is left

With RESIPS_METER (the url) and RESIPS_TOKEN set, `resips hip|actuator|chip` judge commands go through the meter.
"""
import argparse, atexit, json, os, secrets, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMMANDS = {"hip": ("geometry", "check", "run", "motive", "contact"), "actuator": ("check", "run", "motive"), "chip": ("sim", "judge")}


def cost(argv, wd):
  """What a judge command asks to simulate, by budget kind; it runs in wd, so relative paths are read there."""
  prob, cmd, args = argv[0], argv[1], argv[2:]

  def load(path):
    return json.load(open(os.path.join(wd, path)))
  if prob == "chip":
    from resips.chip import judge, sim
    if cmd == "sim":
      return {"offset" if sim.parser().parse_args(args).seed is not None else "perf": 1}
    a = judge.parser().parse_args(args)
    if a.attack is None:
      return {"perf": len(judge.GRID), "offset": 1 + len(sim.DEVICES) + judge.TOP}
    at = load(a.attack)
    return {"perf": len((at.get("points") or [])[:judge.MAX_POINTS]), "offset": len((at.get("seeds") or [])[:judge.MAX_SEEDS])}
  from resips.cad import hip_judge
  if cmd == "check":
    return {"scenarios": len(hip_judge.grid())}
  if cmd == "run":
    sc = load(args[1])
    return {"scenarios": len(sc) if isinstance(sc, list) else 1}
  if cmd == "motive":
    m = load(args[2])
    return {"scenarios": 2 if isinstance(m, dict) and m.get("kind") == "scenario" else 0}
  if cmd == "contact":
    from resips.cad import hip_contact
    a = hip_contact.parser().parse_args(args)
    sc = load(a.scenarios) if a.scenarios else hip_contact.SC
    return {"scenarios": (len(sc) if isinstance(sc, list) else 1) * len(a.variants.split(","))}
  return {}


def place(req, wd):
  """The request's command with its files written under wd, and its cost."""
  argv, files = list(req["argv"]), {int(i): text for i, text in req.get("files", {}).items()}
  if len(argv) < 3 or not all(isinstance(x, str) for x in argv) or argv[1] not in COMMANDS.get(argv[0], ()):
    raise ValueError(f"not a metered command; metered: {COMMANDS}")
  for i, x in enumerate(argv[2:], 2):
    if i not in files and "/" in x:
      raise ValueError(f"no such file: {x}")
  for i, text in files.items():
    if not 2 <= i < len(argv):
      raise ValueError(f"file index {i} is not an argument")
    os.makedirs(os.path.join(wd, str(i)))
    argv[i] = os.path.join(wd, str(i), os.path.basename(argv[i]))
    open(argv[i], "w").write(text)
  try:
    c = cost(argv, wd)
  except SystemExit:
    raise ValueError(f"arguments that do not parse: {' '.join(req['argv'][2:])}") from None
  return argv, {k: n for k, n in c.items() if n}


def serve(state, names, budget):
  from resips.helpers import save
  os.makedirs(os.path.join(state, "log"), exist_ok=True)
  tf, bf = os.path.join(state, "tokens.json"), os.path.join(state, "budget.json")
  tokens = json.load(open(tf)) if os.path.exists(tf) else {}
  books = json.load(open(bf)) if os.path.exists(bf) else {}
  for n in names:
    tokens.setdefault(n, secrets.token_hex(16))
    books.setdefault(n, {"left": dict(budget), "used": dict.fromkeys(budget, 0)})
  save(tf, tokens)
  save(bf, books)
  who, lock = {t: n for n, t in tokens.items()}, threading.Lock()

  def charge(name, c, sign=1):
    for k, n in c.items():
      books[name]["left"][k] = books[name]["left"].get(k, 0) - sign * n
      books[name]["used"][k] = books[name]["used"].get(k, 0) + sign * n
    save(bf, books)

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
      pass

    def reply(self, code, obj):
      b = json.dumps(obj, default=float).encode()
      self.send_response(code)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(b)))
      self.end_headers()
      self.wfile.write(b)

    def do_POST(self):
      try:
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        name = who[req["token"]]
      except (ValueError, KeyError, TypeError):
        return self.reply(403, {"error": "unknown token"})
      if self.path == "/budget":
        with lock:
          return self.reply(200, {"name": name, **books[name]})
      if self.path != "/run":
        return self.reply(404, {"error": "use /run or /budget"})
      with tempfile.TemporaryDirectory() as wd:
        try:
          argv, c = place(req, wd)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
          return self.reply(400, {"error": f"bad request: {e}"})
        with lock:
          left = books[name]["left"]
          short = [f"{left.get(k, 0)} {k} left, {n} asked" for k, n in c.items() if n > left.get(k, 0)]
          if not short:
            charge(name, c)
          left = dict(left)
        if short:
          return self.reply(429, {"error": "budget exhausted: " + "; ".join(short), "left": left})
        p = subprocess.run([sys.executable, "-m", "resips", *argv], cwd=wd, capture_output=True, text=True,
                           env={**os.environ, "RESIPS_METER": ""})
      try:
        json.loads(p.stdout)
        ok = p.returncode == 0
      except ValueError:
        ok = False
      with lock:
        if not ok:
          charge(name, c, -1)
        left = dict(books[name]["left"])
        with open(os.path.join(state, "log", f"{name}.jsonl"), "a") as f:
          f.write(json.dumps({"t": time.time(), "argv": req["argv"], "files": req.get("files"), "cost": c, "ok": ok}) + "\n")
      if not ok:
        return self.reply(500, {"error": "the command failed (not charged):\n" + (p.stdout + p.stderr)[-2000:], "left": left})
      self.reply(200, {"stdout": p.stdout, "cost": c, "left": left})

  srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  url = f"http://127.0.0.1:{srv.server_address[1]}"
  open(os.path.join(state, "url.part"), "w").write(url + "\n")
  os.replace(os.path.join(state, "url.part"), os.path.join(state, "url.txt"))
  print(f"meter on {url}, state in {state}", flush=True)
  srv.serve_forever()


def start(state, names, budget):
  """A meter serving in the background for as long as this process runs."""
  os.makedirs(state, exist_ok=True)
  url, log = os.path.join(state, "url.txt"), os.path.join(state, "serve.log")
  if os.path.exists(url):
    os.remove(url)
  p = subprocess.Popen([sys.executable, "-m", "resips", "meter", "serve", state, "--names", ",".join(names), "--budget", budget],
                       stdout=open(log, "w"), stderr=subprocess.STDOUT)
  atexit.register(p.terminate)
  while not os.path.exists(url):
    if p.poll() is not None:
      raise RuntimeError("the meter did not start:\n" + open(log).read()[-2000:])
    time.sleep(0.2)
  return p


def env(state):
  """name -> the variables that send that agent's judge commands through the meter in state"""
  url = open(os.path.join(state, "url.txt")).read().strip()
  return {n: {"RESIPS_METER": url, "RESIPS_TOKEN": t} for n, t in json.load(open(os.path.join(state, "tokens.json"))).items()}


def books(state):
  return json.load(open(os.path.join(state, "budget.json")))


def post(path, body):
  url = os.environ.get("RESIPS_METER") or sys.exit("meter: RESIPS_METER is not set")
  req = urllib.request.Request(url + path, json.dumps({"token": os.environ.get("RESIPS_TOKEN", ""), **body}).encode(),
                               {"Content-Type": "application/json"})
  try:
    with urllib.request.urlopen(req, timeout=3600) as r:
      return json.load(r)
  except urllib.error.HTTPError as e:
    sys.exit(f"meter: {json.load(e).get('error')}")
  except urllib.error.URLError as e:
    sys.exit(f"meter: cannot reach {url}: {e.reason}")


def call(argv):
  """A judge command through the meter: its files are sent along and its output printed as the command prints it."""
  r = post("/run", {"argv": argv, "files": {i: open(a, newline="").read() for i, a in enumerate(argv) if i >= 2 and os.path.isfile(a)}})
  sys.stdout.write(r["stdout"])
  print(f"meter: spent {r['cost']}, left {r['left']}", file=sys.stderr)
  return 0


def main(argv):
  ap = argparse.ArgumentParser(prog="resips meter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("cmd", choices=("serve", "budget"))
  ap.add_argument("state", nargs="?")
  ap.add_argument("--names", default="yes,no")
  ap.add_argument("--budget", help="KIND=N,... per agent, e.g. scenarios=300 or perf=1500,offset=1500")
  a = ap.parse_args(argv)
  if a.cmd == "budget":
    print(json.dumps(post("/budget", {})))
    return 0
  if not (a.state and a.budget):
    ap.error("serve needs STATE and --budget")
  serve(a.state, a.names.split(","), {k: int(n) for k, n in (x.split("=") for x in a.budget.split(","))})
  return 0
