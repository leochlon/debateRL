"""debateRL: physics judges for design debates between agents.

  debaterl hip geometry D.py | check D.py | run D.py S.json | motive BASE.py D.py M.json | contact D.py [S.json]
  debaterl actuator check A.json | run A.json S.json | motive BASE.json A.json M.json
  debaterl debate PROBLEM start|yes|judge-yes|no|judge-no RUN [ROUND]     (debaterl debate -h)
  debaterl validate                  the FEA against closed-form answers
  debaterl chip sim|judge|debate ARGS   the op-amp contest, chip/MODULE.py   (debaterl chip MODULE -h)

Judge output is JSON on stdout. With DEBATERL_HOST set (an ssh destination) judge commands run there: the file arguments
are copied over and run with `DEBATERL_REMOTE_PYTHON -m debaterl` (default ~/debateRL/.venv/bin/python, a Python with this
package installed). DEBATERL_CACHE (default ~/.cache/debaterl) and DEBATERL_WORKERS apply on the machine that runs the judge.
"""
import importlib, json, os, shlex, subprocess, sys, uuid

JUDGES = {"hip": "debaterl.cad.hip_judge", "actuator": "debaterl.cad.hip_actuator"}
CHIP = ("sim", "judge", "debate")
SSH = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]


def remote(host, argv):
  rdir = f"/tmp/debaterl_{uuid.uuid4().hex[:10]}"
  subprocess.run(["ssh", *SSH, host, f"mkdir -p {rdir}"], check=True)
  args = []
  for i, a in enumerate(argv):
    if i >= 2 and os.path.isfile(a):
      args.append(f"{rdir}/{i}_{os.path.basename(a)}")
      subprocess.run(["scp", "-q", *SSH, a, f"{host}:{args[-1]}"], check=True)
    else:
      args.append(a)
  cache = os.environ.get("DEBATERL_CACHE", "~/.cache/debaterl")
  home = os.path.expanduser("~")
  if cache.startswith(home + "/"):
    cache = "~/" + cache[len(home) + 1:]  # a local shell expanded ~; the cache lives on the remote machine
  py = os.environ.get("DEBATERL_REMOTE_PYTHON", "~/debateRL/.venv/bin/python")
  cmd = f"cd {rdir} && DEBATERL_HOST= DEBATERL_CACHE={cache} {py} -m debaterl {shlex.join(args)}; rc=$?; rm -rf {rdir}; exit $rc"
  p = subprocess.run(["ssh", *SSH, host, cmd], capture_output=True, text=True)
  sys.stdout.write(p.stdout)
  if p.returncode != 0:
    sys.stderr.write(p.stderr[-3000:])
  return p.returncode


def judge(prob, cmd, args):
  if prob == "hip" and cmd == "contact":
    from debaterl.cad import hip_contact
    return hip_contact.cli(args)
  m = importlib.import_module(JUDGES[prob])
  if cmd == "run":
    sc = json.load(open(args[1]))
    return m.run(args[0], sc if isinstance(sc, list) else [sc])
  if cmd == "motive":
    return m.motive(args[0], args[1], json.load(open(args[2])))
  if cmd in ("geometry", "check") and hasattr(m, cmd):
    return getattr(m, cmd)(args[0])
  raise SystemExit(f"unknown command: {prob} {cmd}\n\n{__doc__}")


def main(argv=None):
  a = sys.argv[1:] if argv is None else argv
  if a[:1] == ["debate"]:
    from debaterl.cad import debate
    return debate.main(a[1:])
  if a[:1] == ["chip"] and a[1:2] and a[1] in CHIP:
    return importlib.import_module(f"debaterl.chip.{a[1]}").main(a[2:])
  if a[:1] == ["validate"]:
    from debaterl.cad import fem
    print(json.dumps(fem.validate(), indent=1))
    return 0
  if len(a) < 3 or a[0] not in JUDGES:
    print(__doc__)
    return 2
  if os.environ.get("DEBATERL_HOST"):
    return remote(os.environ["DEBATERL_HOST"], a)
  json.dump(judge(a[0], a[1], a[2:]), sys.stdout, indent=1, default=float)
  print()
  return 0


if __name__ == "__main__":
  sys.exit(main())
