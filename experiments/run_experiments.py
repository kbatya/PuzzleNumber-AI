"""Experiments for the PuzzleNumber-AI project book (Value Iteration on the 8-puzzle).
Pure-Python model (tuples) + matplotlib. Mirrors AI_Agent.Value_Iteration (Bellman
optimality, gamma-discounted, reward=1 at goal). Outputs 3 PNGs + printed numbers."""
import random, collections, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
OUT = r"C:/Users/plane/AppData/Local/Temp/claude/C--Users-plane-OneDrive-Documents-marvad-371/8ddba47e-c178-4619-aeee-81128fea0f2b/scratchpad"
GOAL = tuple(range(9))          # (0,1,2,3,4,5,6,7,8), 0 = blank

def neighbors(s):
    p = s.index(0); r, c = divmod(p, 3); out = []
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            q = nr*3+nc; t = list(s); t[p], t[q] = t[q], t[p]; out.append(tuple(t))
    return out

# ---- 1. BFS from goal: distances + the solvable component ----
dist = {GOAL: 0}; dq = collections.deque([GOAL])
while dq:
    s = dq.popleft()
    for n in neighbors(s):
        if n not in dist:
            dist[n] = dist[s] + 1; dq.append(n)
STATES = list(dist)
MAXD = max(dist.values()); AVGD = sum(dist.values())/len(dist)
print(f"[model] solvable states={len(STATES)}  max dist={MAXD}  avg dist={AVGD:.3f}")

# ---- 2. Value Iteration (Jacobi sweeps), logging Delta per sweep ----
def value_iteration(gamma, theta=1e-4):
    V = {s: 0.0 for s in STATES}
    deltas = []
    while True:
        newV = {}; delta = 0.0
        for s in STATES:
            if s == GOAL:
                newV[s] = 0.0; continue
            best = max((1.0 if n == GOAL else 0.0) + gamma*V[n] for n in neighbors(s))
            newV[s] = best; delta = max(delta, abs(best - V[s]))
        V = newV; deltas.append(delta)
        if delta < theta:
            break
    return V, deltas

GAMMA = 0.95
V, deltas = value_iteration(GAMMA)
print(f"[VI] gamma={GAMMA}  sweeps={len(deltas)}  final V matches gamma^(d-1): "
      f"{all(abs(V[s]-(GAMMA**(dist[s]-1) if s!=GOAL else 0))<1e-6 for s in STATES)}")

# ---- greedy policy solver (uses V) ----
def solve_len(start, V, gamma, cap=200):
    s = start; steps = 0
    while s != GOAL and steps < cap:
        s = max(neighbors(s), key=lambda n: (1.0 if n == GOAL else 0.0) + gamma*V[n])
        steps += 1
    return steps if s == GOAL else None

# sanity: greedy length == optimal distance
sample = random.sample(STATES, 300)
assert all(solve_len(s, V, GAMMA) == dist[s] for s in sample), "greedy != optimal!"
print("[VI] greedy policy is optimal on 300-state sample: OK")

# ================= GRAPH 1: convergence =================
plt.figure(figsize=(7,4.3))
plt.semilogy(range(1, len(deltas)+1), deltas, marker="o", ms=4, color="#2563eb")
plt.axhline(1e-4, ls="--", color="#dc2626", lw=1, label=r"$\theta=10^{-4}$")
plt.xlabel("Sweep number  k"); plt.ylabel(r"$\Delta$  (max value change)  [log]")
plt.title(f"Value Iteration convergence  ($\\gamma$={GAMMA})")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/graph1_convergence.png", dpi=150); plt.close()
print(f"[graph1] sweeps={len(deltas)}  delta_1={deltas[0]:.4g}  delta_last={deltas[-1]:.4g}")

# ================= GRAPH 2: effect of gamma =================
GAMMAS = [0.5, 0.9, 0.95, 0.99]
test_states = random.sample(STATES, 1000)
avg_len = {}; min_val = {}
for g in GAMMAS:
    Vg = {s: (0.0 if s == GOAL else g**(dist[s]-1)) for s in STATES}   # = VI fixed point
    lens = [solve_len(s, Vg, g) for s in test_states]
    avg_len[g] = sum(lens)/len(lens)
    min_val[g] = g**(MAXD-1)
    print(f"[gamma={g}] avg solution length={avg_len[g]:.3f}  min state value gamma^{MAXD-1}={min_val[g]:.3e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4.2))
ax1.bar([str(g) for g in GAMMAS], [avg_len[g] for g in GAMMAS], color="#2563eb", width=0.6)
ax1.set_xlabel(r"discount factor  $\gamma$"); ax1.set_ylabel("avg solution length (steps)")
ax1.set_title("Solution length vs $\\gamma$  (identical = optimal)")
ax1.set_ylim(0, AVGD*1.4)
for i,g in enumerate(GAMMAS): ax1.text(i, avg_len[g]+0.3, f"{avg_len[g]:.1f}", ha="center", fontsize=9)
ax2.semilogy(range(len(GAMMAS)), [min_val[g] for g in GAMMAS], marker="s", color="#dc2626")
ax2.set_xticks(range(len(GAMMAS))); ax2.set_xticklabels([str(g) for g in GAMMAS])
ax2.set_xlabel(r"discount factor  $\gamma$"); ax2.set_ylabel(r"min state value  $\gamma^{\,d_{max}}$  [log]")
ax2.set_title("Value dynamic range vs $\\gamma$"); ax2.grid(True, which="both", ls=":", alpha=0.5)
plt.tight_layout(); plt.savefig(f"{OUT}/graph2_gamma.png", dpi=150); plt.close()

# ================= GRAPH 3: solution-length distribution + shuffle depth =================
rand_states = random.sample(STATES, 3000)
sol_lens = [dist[s] for s in rand_states]   # greedy == optimal (verified)
def shuffle_state(depth):
    s = GOAL
    for _ in range(depth):
        s = random.choice(neighbors(s))
    return s
DEPTHS = [1,2,3,5,8,10,15,20,30,50,100]
depth_avg = []
for d in DEPTHS:
    ds = [dist[shuffle_state(d)] for _ in range(800)]
    depth_avg.append(sum(ds)/len(ds))

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

print(f"[graph3] mean solution length={sum(sol_lens)/len(sol_lens):.3f}  "
      f"min={min(sol_lens)} max={max(sol_lens)}")
print("[depth] " + "  ".join(f"{d}:{a:.1f}" for d,a in zip(DEPTHS, depth_avg)))
print("DONE")
