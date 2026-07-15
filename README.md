# PuzzleNumber-AI — Solving the 3×3 Number Puzzle with Value Iteration

A Reinforcement Learning project: an agent that solves the 3×3 sliding number
puzzle (the **8-puzzle**) using **tabular Value Iteration** based on the Bellman
optimality equation. After the value function is computed, the agent follows it
**greedily** and reaches the goal from any solvable start state, in the minimum
number of moves.

- **Environment:** 3×3 board, eight numbered tiles + one blank; a tile adjacent to
  the blank can slide into it.
- **Algorithm:** tabular Value Iteration (dynamic programming), discount `γ = 0.95`.
- **Interface:** a pygame window that shows the agent solving the puzzle, plus a
  human-play mode.

---

## Attribution
The environment/framework — `Puzzle`, `State`, `Action`, `Graphics`, `Game`,
`Human_Agent`, `constants` — was provided as **starter code by the course
lecturer, Gilad Markman**. This repository is a fork of his project.

**Original starter repository:** https://github.com/MarkmanGilad/PuzzleNumber-AI

---

## What I changed and created

The starter framework was left intact except for the one file I was asked to
complete. Everything else here is additional documentation and analysis.

### Changed
| File | What I changed | Purpose |
|---|---|---|
| `AI_Agent.py` | Implemented the `Value_Iteration` method (it was an empty stub), and made `get_V` return a `0` default for unseen states. | This is the core of the project — building the value table `V(s)` that solves the puzzle. |
| `.gitignore` | Added rules to ignore the large generated value tables (`Data/V.pth`, `V_2.pth`, `V_3.pth`). | Keep the repository lightweight (the tables are rebuilt by the code). |

### Created
| File | Purpose |
|---|---|
| `README.md` | This document. |
| `experiments/run_experiments.py` | A standalone script that reproduces the three figures used in the project book (see [Experiments](#experiments)). |
| `experiments/graph1_convergence.png` | Value-Iteration convergence graph. |
| `experiments/graph2_gamma.png` | Effect of the discount factor γ. |
| `experiments/graph3_distribution.png` | Distribution of solution lengths + shuffle-depth effect. |
| `experiments/requirements.txt` | The experiments' only extra dependency (`matplotlib`). |

---

## How the algorithm works

- **State:** the 3×3 board as a NumPy array, where `0` is the empty square.
- **Actions:** slide the blank **up / down / left / right** (the legal actions depend
  on the blank's position).
- **Reward:** `1` on reaching the goal, otherwise `0` (a *sparse* reward). Discount `γ = 0.95`.
- **Value Iteration** repeatedly sweeps over all states and updates each with the
  Bellman optimality rule until the values stabilise:

  ```
  V(s) = max over legal actions a of [ R(s, a) + γ · V(s') ]
  ```

  where `s'` is the state reached by action `a`. We stop when the largest change in a
  sweep, `Δ`, drops below `θ = 1e-4`. Because the only reward is at the goal, the
  values settle to roughly `γ^(distance-to-goal)`, so states nearer the goal have
  higher values.
- **Greedy policy:** at each step `get_Action` performs a one-step look-ahead and
  picks the action maximising `R + γ·V(next)` — which always moves the agent one step
  closer to the goal.

---

## Project structure

```
PuzzleNumber-AI/
├── AI_Agent.py        # the agent — Value_Iteration + greedy get_Action  (MY WORK)
├── Puzzle.py          # the environment: legal actions, transitions, reward   (lecturer)
├── State.py           # board state (NumPy array) + blank position            (lecturer)
├── Action.py          # UP / DOWN / LEFT / RIGHT enum                         (lecturer)
├── Graphics.py        # pygame drawing of the board                          (lecturer)
├── Human_Agent.py     # keyboard control for a human player                  (lecturer)
├── Game.py            # main loop; use_ai switches AI / human                (lecturer)
├── constants.py       # board size, colours, FPS                            (lecturer)
├── requirements.txt   # pygame, numpy, torch                                 (lecturer)
├── Data/              # saved value tables (.pth)
└── experiments/       # my analysis scripts + graphs   (MY WORK)
```

---

## Run

```bash
pip install -r requirements.txt   # pygame, numpy, torch
python Game.py
```
In `Game.py`, the `use_ai` flag switches between the AI agent (`True`) and a human
player (`False`). In AI mode the program first runs Value Iteration, then solves the
puzzle step by step.

---

## Experiments

The scripts and graphs live in [`experiments/`](experiments/). To reproduce them:

```bash
cd experiments
pip install -r requirements.txt   # matplotlib
python run_experiments.py
```

> Note: `run_experiments.py` is a **fast, standalone reproduction** of the algorithm
> (a pure-Python model of the puzzle, no pygame). It mirrors the same Bellman /
> Value-Iteration logic used in `AI_Agent.py`, so it can run headless and produce the
> plots in a few seconds.

### 1. Convergence of Value Iteration
![Convergence](experiments/graph1_convergence.png)

The maximum change `Δ` per sweep, on a logarithmic axis. `Δ` starts at `1.0` and
decreases geometrically (≈ ×γ each sweep) as the goal value propagates one "ring" of
states outward per sweep. After **32 sweeps** every reachable state — the puzzle's
diameter is **31 moves** — has its final value, and `Δ` drops to `0`. The straight
log-scale decline confirms the contraction (guaranteed) convergence of Value Iteration.

### 2. Effect of the discount factor γ
![Gamma effect](experiments/graph2_gamma.png)

Across `γ ∈ {0.5, 0.9, 0.95, 0.99}` the **average solution length is identical —
about 22 steps — and optimal**. The greedy policy depends only on the *ordering* of
state values, and discounting preserves that ordering for any `0 < γ < 1`. What γ
*does* change is the numeric dynamic range (right panel): the value of the farthest
states, `γ^30`, collapses from about `0.21` at `γ = 0.95` to about `9 × 10⁻¹⁰` at
`γ = 0.5`. At this board size that is still representable, so the policy stays optimal;
a moderate `γ = 0.95` keeps a healthy value gradient with reasonable convergence.

### 3. Distribution of solution lengths & shuffle depth
![Distribution](experiments/graph3_distribution.png)

Over **3000 random solvable start states** the agent solved every one. Solution length
is bell-shaped with a **mean of ≈ 22 steps**, ranging from **8 to 31** — matching the
known 31-move diameter of the 8-puzzle. The right panel shows the effect of shuffle
depth: a deeper shuffle produces harder (more distant) start states, with the average
solution length rising and then flattening (a 100-move random shuffle averages ≈ 18
steps, below the uniform-random average because a random walk revisits states).
