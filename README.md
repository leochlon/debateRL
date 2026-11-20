# debateRL

debateRL, by Hassana Labs, runs design debates between AI agents, judged only by simulation: `cad/` holds the
hip-exoskeleton debates described first, `chip/` an op-amp contest.

Two AI agents debate a hardware design. YES changes the design; NO attacks the change with scenarios inside agreed
ranges. The judge reads no arguments: it rebuilds the part and runs each scenario through real simulators, so a failure
either reproduces or it doesn't.

| stage | tool | what it settles |
|---|---|---|
| geometry | build123d / OpenCASCADE | one valid solid, printer fit, keep-outs, interfaces, hard stops, mass |
| motion | MuJoCo | the loads on the part while the wearer walks or stands up from a chair, across body sizes |
| drive | ngspice | the torque the motor drive actually delivers (battery sag, current loop, back-EMF) |
| stress | gmsh + CalculiX | six unit load cases on quadratic tets, superposed over the load history |

A change is accepted only if it passes every claim on the judge's grid, its motive is two-sided (a claim instance that
fails on the baseline and holds on the change), and none of NO's up to 5 scenarios reproduces a failure.

Two judges: `hip` (a hip-exoskeleton bracket as build123d code) and `actuator` (the hip actuator as five numbers, on
the final bracket). `cad/hip_claims.json` is the contract: every claim and threshold, the scenario ranges, the model
settings a challenger may change, the judge's grid, and what is not covered.

## Install

CalculiX (`ccx`) and ngspice must be on the PATH (Ubuntu: `apt install calculix-ccx ngspice`).

```
pip install -e .
debaterl validate                                 # FEA against closed-form answers
```

The repository root is the `debaterl` package, so it runs only once installed; the agents' `python -m debaterl...` commands
need the install too.

## Use

```
debaterl hip geometry design.py                   # scenario-free claims, mass, stop angles (~10 s)
debaterl hip check design.py                      # every claim on the 36-scenario grid
debaterl hip run design.py scenarios.json         # a list of scenarios (an attack)
debaterl hip motive baseline.py design.py m.json  # two-sided test of a change
debaterl hip contact design.py                    # push-only belt clamp at the three hardest moves (slow)
debaterl actuator check spec.json
debaterl debate hip start RUN                     # then yes, judge-yes, no, judge-no for each round
```

Output is JSON, cached by design file hash in `DEBATERL_CACHE` (default `~/.cache/debaterl`). Runs are deterministic:
gmsh and CalculiX run single-threaded per process. An FEA solve runs six CalculiX processes of about 1 GB each. With
`DEBATERL_HOST` set to an ssh destination, judge commands run on that machine, with the Python in
`DEBATERL_REMOTE_PYTHON` (the package must be installed there too).

`debaterl debate` builds each round's messages only from protocol fields (the baseline, the judge's verdicts, the motive
check, judge-confirmed attacks) and snapshots both sides' files before judging them. Launching the agents is up to you.

## The op-amp contest

`chip/` runs a debate on a chip: size a fixed two-stage Miller op-amp (15 numbers) for the smallest area that meets a
PVT spec and keeps |offset| <= 3 mV over a million mismatch seeds, in ngspice. `chip/debate.py` runs the agents itself
through the Anthropic API, one bash shell each: debaters A and B design, attack each other's design with failing
operating points or seeds, revise and attack again, while solo designers work alone for comparison.

```
debaterl chip debate --out runs/chip1 [--solo 2]  # needs ANTHROPIC_API_KEY
debaterl chip judge design.json [--attack a.json]
debaterl chip sim design.json --process SS --temp 125 --vdd 1.7
```

The agents simulate with `python -m debaterl.chip.sim`. The judge builds a one-million-seed mismatch table in
`DEBATERL_CACHE` on first use.

## Files

```
__init__.py        one thread per process (the judges run processes in parallel)
__main__.py        the debaterl command; ssh runner
helpers.py         settings, design build cache, B-rep helpers, gait curves, parallel cached runs
cad/               the hardware debates
  fem.py           linear FEA: gmsh tet10 mesh, CalculiX unit cases, superposition, rigid seat fit
  debate.py        the YES/NO protocol: rules, round messages, judge steps, ledger
  hip_geom.py      bracket geometry claims, stop angles, mass
  hip_body.py      MuJoCo wearer and hip exo
  hip_drive.py     ngspice motor drive
  hip_motor.py     actuator model: five numbers to R, L, inertia, gearing, mass
  hip_judge.py     bracket judge
  hip_actuator.py  actuator judge
  hip_contact.py   push-only belt clamp check
  hip_bracket.py   the final bracket (design file the actuator judge uses)
chip/              the op-amp contest
  sim.py           op-amp netlist and device model: one operating point, or the offset of one mismatch seed
  judge.py         105-point PVT grid, million-seed offset screen, attacks
  debate.py        agent harness (Anthropic API, bash) and the debate: two debaters plus solo designers
```

The wearer's motion is prescribed, so falls and impacts are not simulated, and the material and actuator values are
assumptions; see `not_covered` in the claims file.

## License

CC BY-NC 4.0 (`LICENSE`): free to use and adapt for non-commercial purposes, with attribution to Hassana Labs.
Commercial use is not permitted.
