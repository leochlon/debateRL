<div align="center">

<h1>resips</h1>

<i>res ipsa loquitur</i>: the thing speaks for itself

<b>Two AI agents argue over a hardware design. The judge reads no arguments: it builds the part and runs the physics.</b>

by Hassana Labs

<h3>

[How it works](#how-it-works) | [Install](#install) | [Problems](#problems) | [Meter](#meter) | [Citation](#citation)

</h3>

[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Judge](https://img.shields.io/badge/judge-OpenCASCADE%20%7C%20MuJoCo%20%7C%20ngspice%20%7C%20CalculiX-2ea44f)](#how-it-works)

</div>

<table>
<tr>
<td width="50%"><img src="assets/cad_bracket.gif" alt="Hip-exoskeleton bracket improving over 24 designs"></td>
<td width="50%"><img src="assets/chip_ldo.gif" alt="SKY130 low-dropout regulator improving over 20 designs"></td>
</tr>
<tr>
<td align="center"><b>CAD:</b> hip-exoskeleton bracket, 3 rounds, 24 designs</td>
<td align="center"><b>Chip:</b> SKY130 low-dropout regulator, same method, 20 designs</td>
</tr>
</table>

---

## How it works

- **YES** changes the design. **NO** attacks the change with up to 5 scenarios inside agreed ranges.
- The judge rebuilds the part and runs every scenario through real simulators, so a failure either reproduces or it doesn't.
- A change is accepted only if it passes every claim on the judge's grid, its motive is two-sided (a claim instance that
  fails on the baseline and holds on the change), and none of NO's scenarios reproduces a failure.

| stage | tool | what it settles |
|---|---|---|
| geometry | build123d / OpenCASCADE | one valid solid, printer fit, keep-outs, interfaces, hard stops, mass |
| motion | MuJoCo | the loads on the part while the wearer walks or stands up from a chair, across body sizes |
| drive | ngspice | the torque the motor drive actually delivers (battery sag, current loop, back-EMF) |
| stress | gmsh + CalculiX | six unit load cases on quadratic tets, superposed over the load history |

## Install

```sh
apt install calculix-ccx ngspice     # ccx and ngspice on the PATH
pip install -e .
resips validate                      # FEA against closed-form answers
```

The repository root is the `resips` package, so it runs once installed.

## Problems

### Hip exoskeleton (`cad/`)

Two judges: `hip`, a hip-exoskeleton bracket as build123d code, and `actuator`, the hip actuator as five numbers on the
final bracket. `cad/hip_claims.json` is the contract: every claim and threshold, the scenario ranges, the model settings
a challenger may change, the judge's grid, and what is not covered.

`resips debate` builds each round's messages only from protocol fields (the baseline, the judge's verdicts, the motive
check, judge-confirmed attacks) and snapshots both sides' files before judging them. Launching the agents is up to you.

### Op-amp (`chip/`)

Size a fixed two-stage Miller op-amp (15 numbers) for the smallest area that meets a PVT spec and keeps |offset| <= 3 mV
over a million mismatch seeds, in ngspice. Debaters A and B design, attack each other's design with failing operating
points or seeds, revise and attack again, while solo designers work alone for comparison. With `ANTHROPIC_API_KEY` set,
`chip/debate.py` runs the agents itself through the API, one bash shell each. Without it, whoever launched the debate
runs them (as subagents, say): each turn's message is written to the designer's folder as `turnK.md`, and the turn ends
when the reply is saved next to it as `turnK.reply`.

```sh
resips chip debate --out runs/chip1 [--solo 2]
resips chip judge design.json [--attack a.json]
resips chip sim design.json --process SS --temp 125 --vdd 1.7
```

The judge builds the one-million-seed mismatch table in `RESIPS_CACHE` on first use.

## Meter

Puts agents on a simulation budget. Each agent gets a token and, for the whole run, a budget per kind. A judge command
costs what it asks to simulate, a command that would overdraw a budget is refused whole, a command that fails is not
charged, and every call is logged with its files to `STATE/log/NAME.jsonl`.

| command | costs |
|---|---|
| `hip check`, `actuator check` | 36 scenarios |
| `hip run`, `actuator run` | one scenario per scenario |
| `hip motive`, `actuator motive` | 2 scenarios for a scenario motive, otherwise nothing |
| `hip contact` | one scenario per move and variant |
| `hip geometry` | nothing |
| `chip sim` | 1 perf (an operating point) or 1 offset (a mismatch seed) |
| `chip judge` | 105 perf and 1009 offset; an attack, its points and seeds |

```sh
resips meter serve RUN/meter --names yes,no --budget scenarios=300 &   # the hip and actuator debates pick it up
resips chip debate --out runs/chip1 --budget perf=1500,offset=1500     # starts its own meter
RESIPS_METER=$(cat RUN/meter/url.txt) RESIPS_TOKEN=... resips hip check design.py
resips meter budget
```

## Layout

```
__main__.py        the resips command; ssh runner
helpers.py         settings, design build cache, B-rep helpers, gait curves, parallel cached runs
meter.py           simulation budgets: tokens, a budget per kind, the call log; judge commands go through it
cad/
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
  hip_claims.json  the contract
chip/
  sim.py           op-amp netlist and device model: one operating point, or the offset of one mismatch seed
  judge.py         105-point PVT grid, million-seed offset screen, attacks
  debate.py        the debate, two debaters plus solo designers: agents through the API, or run by the launcher
```

## Citation

```bibtex
@software{resips2026,
  author = {{Hassana Labs}},
  title  = {resips: physics judges for design debates between AI agents},
  year   = {2026},
  url    = {https://github.com/leochlon/resips}
}
```

## License

CC BY-NC 4.0 (`LICENSE`): free to use and adapt for non-commercial purposes, with attribution to Hassana Labs.
Commercial use is not permitted.
