"""Experiments for the PuzzleNumber-AI project book (Value Iteration on the 8-puzzle).
This version faithfully replicates the PRODUCTION algorithm in AI_Agent.Value_Iteration:
in-place (Gauss-Seidel) updates, over ALL 9! = 362,880 permutations, in the same order
(itertools.permutations), with the theta = 1e-4 stopping rule."""
import os, random, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import permutations

random.seed(42)
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- states enumerated in the production order (itertools.permutations) ----
PERMS = list(permutations(range(9)))          # 362,880 tuples
IDX = {s: i for i, s in enumerate(PERMS)}
N = len(PERMS)
GOAL = tuple(range(9)); GOAL_I = IDX[GOAL]

def neighbor_tuples(s):
    p = s.index(0); r, c = divmod(p, 3); out = []
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            q = nr*3+nc; t = list(s); t[p], t[q] = t[q], t[p]; out.append(tuple(t))
    return out

ADJ = [[IDX[t] for t in neighbor_tuples(s)] for s in PERMS]

# ---- BFS from goal: solvable component + optimal distances ----
dist = {GOAL_I: 0}; dq = collections.deque([GOAL_I])
while dq:
    u = dq.popleft()
    for v in ADJ[u]:
        if v not in dist:
            dist[v] = dist[u] + 1; dq.append(v)
SOLV = list(dist)
MAXD = max(dist.values()); AVGD = sum(dist.values())/len(dist)
print(f"[model] total={N}  solvable={len(SOLV)}  unsolvable={N-len(SOLV)}  maxD={MAXD}  avgD={AVGD:.4f}")

# ---- production-faithful Value Iteration (Gauss-Seidel, all states, theta stop) ----
def value_iteration(gamma, theta=1e-4):
    V = [0.0]*N; deltas = []
    while True:
        delta = 0.0
        for i in range(N):
            if i == GOAL_I:
                continue
            best = None
            for v in ADJ[i]:
                val = (1.0 if v == GOAL_I else 0.0) + gamma*V[v]   # in-place read = Gauss-Seidel
                if best is None or val > best:
                    best = val
            d = V[i] - best
            if d < 0: d = -d
            if d > delta: delta = d
            V[i] = best
        deltas.append(delta)
        if delta < theta:
            break
    return V, deltas

def solve_len(start, V, gamma, cap=200):
    i = start; steps = 0
    while i != GOAL_I and steps < cap:
        best = None; bi = None
        for v in ADJ[i]:
            val = (1.0 if v == GOAL_I else 0.0) + gamma*V[v]
            if best is None or val > best:
                best = val; bi = v
        i = bi; steps += 1
    return steps if i == GOAL_I else None

# ===== GRAPH 1: production convergence (gamma = 0.95) =====
GAMMA = 0.95
V95, deltas = value_iteration(GAMMA)
print(f"[VI prod g=0.95] sweeps={len(deltas)} delta_1={deltas[0]:.4g} delta_last={deltas[-1]:.4g}")
chk = random.sample(SOLV, 300)
opt_ok = all(solve_len(s, V95, GAMMA) == dist[s] for s in chk)
print("[VI prod] greedy optimal on 300 solvable sample:", opt_ok)

plt.figure(figsize=(7.2,4.3))
plt.semilogy(range(1, len(deltas)+1), deltas, marker="o", ms=4, color="#2563eb")
plt.axhline(1e-4, ls="--", color="#dc2626", lw=1, label=r"$\theta=10^{-4}$")
plt.xlabel("Sweep number  k"); plt.ylabel(r"$\Delta$  (max value change)  [log]")
plt.title(f"Convergence — production Gauss-Seidel, all {N:,} states  ($\\gamma$={GAMMA})")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/graph1_convergence.png", dpi=150); plt.close()

# ===== GRAPH 2: real theta-based VI per gamma =====
GAMMAS = [0.5, 0.9, 0.95, 0.99]
test = random.sample(SOLV, 1000)
succ = {}; sweeps_g = {}; avglen = {}
for g in GAMMAS:
    Vg, dl = value_iteration(g)
    res = [solve_len(s, Vg, g) for s in test]
    ok = [r for r in res if r is not None]
    succ[g] = 100.0*len(ok)/len(test); sweeps_g[g] = len(dl)
    avglen[g] = sum(ok)/len(ok) if ok else 0
    print(f"[g={g}] sweeps={len(dl)} success={succ[g]:.1f}% avg_len={avglen[g]:.2f} failed={len(test)-len(ok)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4.2))
ax1.bar([str(g) for g in GAMMAS], [succ[g] for g in GAMMAS], color="#2563eb", width=0.6)
ax1.set_ylim(0,108); ax1.set_ylabel("states solved (%)"); ax1.set_xlabel(r"discount factor $\gamma$")
ax1.set_title("Success rate vs $\\gamma$  (real $\\theta$-based VI)")
for i,g in enumerate(GAMMAS): ax1.text(i, succ[g]+2, f"{succ[g]:.0f}%", ha="center", fontsize=9)
ax2.bar([str(g) for g in GAMMAS], [sweeps_g[g] for g in GAMMAS], color="#dc2626", width=0.6)
ax2.set_ylabel(r"sweeps until $\Delta<\theta$"); ax2.set_xlabel(r"discount factor $\gamma$")
ax2.set_title("Sweeps before stopping vs $\\gamma$")
for i,g in enumerate(GAMMAS): ax2.text(i, sweeps_g[g]+0.4, str(sweeps_g[g]), ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/graph2_gamma.png", dpi=150); plt.close()

# ===== GRAPH 3: distribution + shuffle depth (converged gamma=0.95) =====
rand_states = random.sample(SOLV, 3000)
sol_lens = [dist[s] for s in rand_states]
def shuffle_idx(depth):
    i = GOAL_I
    for _ in range(depth): i = random.choice(ADJ[i])
    return i
DEPTHS = [1,2,3,5,8,10,15,20,30,50,100]
depth_avg = [sum(dist[shuffle_idx(d)] for _ in range(800))/800 for d in DEPTHS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4.2))
ax1.hist(sol_lens, bins=range(0, MAXD+2), color="#2563eb", edgecolor="white", align="left")
ax1.axvline(sum(sol_lens)/len(sol_lens), color="#dc2626", ls="--",
            label=f"mean={sum(sol_lens)/len(sol_lens):.1f}")
ax1.set_xlabel("solution length (steps to goal)"); ax1.set_ylabel("number of start states")
ax1.set_title("Distribution of solution length\n(3000 random solvable states)"); ax1.legend()
ax2.plot(DEPTHS, depth_avg, marker="o", color="#2563eb")
ax2.axhline(AVGD, color="#6b7280", ls="--", label=f"overall avg dist={AVGD:.1f}")
ax2.set_xlabel("shuffle depth (random moves)"); ax2.set_ylabel("avg solution length")
ax2.set_title("Effect of shuffle depth"); ax2.grid(True, ls=":", alpha=0.5); ax2.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/graph3_distribution.png", dpi=150); plt.close()
print(f"[graph3] mean={sum(sol_lens)/len(sol_lens):.3f} min={min(sol_lens)} max={max(sol_lens)}")
print("DONE")
