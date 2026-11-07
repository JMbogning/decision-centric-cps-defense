#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_iet_final.py  --  reference implementation of
"A Decision-Centric Defense Framework for Cyber-Physical Systems"

What this file does
-------------------
1. Physical layer: a *mode-dependent (Markov-jump) zero-sum LQ game*.  The mode process has transition
   matrix lambda_ij(sigma_1, sigma_2) induced by the cyber strategies, so J*(s) = T_P(sigma)
   depends on the cyber layer.  Separate input channels B (control) and C (disturbance); the saddle-point
   conditions  R_u + B'ZB > 0  and  R_v - C'ZC > 0 (Schur form) are CHECKED at every step (error otherwise).
2. Cyber layer: discounted zero-sum stochastic game (Shapley), defender MINIMISES cost, solved by value
   iteration + linear programming; fixed point sigma = T_C(T_P(sigma)) by damped iteration.
3. Baseline: same game with a constant false-negative penalty kappa instead of the physical feedback.
4. Evaluation of ANY policy in the SAME physical game against the SAME set of attackers
   (27 stationary pure attacker policies = worst case, plus the uniform attacker).
5. Minimax-Q learning that samples real transitions (model-free on the cyber layer only).
6. Experiments: exact solution, Pareto front, sensitivity study, Q-learning with 20 seeds + statistics.

Usage
-----
    python decision_iet_final.py exact pareto sensitivity qlearn      # (default: all stages)
Results go to ./results/*.json .  `build_paper.py` turns them into figures, LaTeX tables and text.

Requirements: numpy, scipy  (matplotlib only for build_paper.py).  Everything is deterministic (fixed seeds).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import stats
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# =====================================================================================
# 1. MODEL
# =====================================================================================
STATES = ["Normal", "Alert", "Degraded", "Out of Service"]
DEF_ACTIONS = ["None", "Library1", "Library2", "Full"]
ATT_ACTIONS = ["Attack1", "Attack2", "Attack3"]

# Detection sets (indices of attacks neutralised by each IDS configuration), per instance.
#  "A" nested libraries (used in the paper)       : Lib1 -> {1}, Lib2 -> {1,2}, Full -> {1,2,3}
#  "B" complementary libraries (robustness check) : Lib1 -> {1}, Lib2 -> {2},   Full -> {1,2,3}
DETECT_SETS = {
    "A": ((), (0,), (0, 1), (0, 1, 2)),
    "B": ((), (0,), (1,), (0, 1, 2)),
}

# Next-state distributions over (N, A, D, H).
#   U[s][a2]  : attack a2 NOT detected      DT[s][a2] : attack a2 detected
# Six entries are taken VERBATIM from Table 2 of the manuscript (listed in TABLE2_ROWS);
# the remaining entries are the authors' completion, monotone in attack severity (illustrative).
U = {
    0: [[.3, .4, .3, 0], [0, .4, .4, .2], [0, .2, .5, .3]],
    1: [[.1, .4, .4, .1], [0, .2, .5, .3], [0, 0, .7, .3]],
    2: [[0, .1, .6, .3], [0, 0, .4, .6], [0, 0, .2, .8]],
}
DT = {
    0: [[.5, .3, .2, 0], [.4, .3, .3, 0], [.3, .3, .3, .1]],
    1: [[.6, 0, .4, 0], [.5, .1, .4, 0], [.4, .1, .4, .1]],
    2: [[.5, .1, .4, 0], [.6, 0, 0, .4], [.4, 0, .3, .3]],
}
TABLE2_ROWS = [  # (state, defender action, attacker action) taken verbatim from Table 2
    ("Normal", "Library1", "Attack1"), ("Normal", "None", "Attack2"), ("Alert", "Library2", "Attack1"),
    ("Alert", "None", "Attack3"), ("Degraded", "Full", "Attack2"), ("Degraded", "None", "Attack3")]


def build_transitions(U=U, DT=DT, detect=DETECT_SETS["A"]) -> np.ndarray:
    """T[s, a1, a2, s'] ; Out-of-Service is absorbing."""
    T = np.zeros((4, 4, 3, 4))
    for s in range(3):
        for a1 in range(4):
            for a2 in range(3):
                T[s, a1, a2] = DT[s][a2] if a2 in detect[a1] else U[s][a2]
    T[3, :, :, 3] = 1.0
    return T


def build_undetected(U=U) -> np.ndarray:
    """Ua[s, a2, s'] : next-state distribution if attack a2 is not detected."""
    Ua = np.zeros((4, 3, 4))
    for s in range(3):
        for a2 in range(3):
            Ua[s, a2] = U[s][a2]
    Ua[3, :, 3] = 1.0
    return Ua


def detected_mask(detect) -> np.ndarray:
    D = np.zeros((4, 3))
    for a1 in range(4):
        for a2 in detect[a1]:
            D[a1, a2] = 1.0
    return D


@dataclass(frozen=True)
class Params:
    instance: str = "A"
    # ---- cyber layer ----
    cost: tuple = (0.0, 2.0, 3.0, 5.0)         # C(F_i)                 (Table 1 of the manuscript)
    att_cost: tuple = (0.0, 1.0, 2.5)          # attacker effort c(a2)  (enters the payoff as  - c(a2))
    beta: float = 0.5                           # discount factor of the cyber game
    Lam: float = 20.0                           # damage scale  Lambda
    eta: float = 0.05                           # damage steepness (called eta in the manuscript)
    T: np.ndarray = field(default_factory=build_transitions, compare=False)
    Uarr: np.ndarray = field(default_factory=build_undetected, compare=False)
    # ---- physical layer ----
    Ts: float = 0.1                             # sampling step (Euler discretisation of the error chain)
    K: int = 20                                 # horizon (steps)
    q_scale: float = 1.0
    ru_scale: float = 1.0
    rv: float = 100.0                           # attacker cost R_v (must exceed C'ZC for the saddle point)
    x0: tuple = (0.1, 0.1, 0.1, 0.1)            # initial platoon error [position, speed, acceleration, jerk]
    # ---- evaluation ----
    beta_eval: float = 0.9                      # occupancy discount used for the average IDS spend

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    @property
    def detect(self):
        return DETECT_SETS[self.instance]


def make_params(instance: str = "A", **kw) -> Params:
    return Params(instance=instance, T=build_transitions(detect=DETECT_SETS[instance]), **kw)


# ---- physical matrices ----------------------------------------------------------------
A_NORMAL = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], float)   # error chain
Q_MODE = [np.diag([1, 3, 5, 7.]), np.diag([2, 5, 8, 11.]), np.diag([3.5, 7.5, 11.5, 15.]), np.diag([5, 10, 15, 20.])]
RU_MODE = [1.0, 3.0, 6.0, 10.0]
B_GAIN = [1.0, 0.9, 0.7, 0.0]                  # actuation authority g_s of the control channel in each mode
B_VEC = np.array([[0], [0], [0], [1.0]])       # commanded acceleration enters the last stage of the chain
C_VEC = np.array([[0], [0], [1.0], [0]])       # manipulated sensor / V2X data enters the acceleration channel


def physical_matrices(p: Params):
    Ad = [np.eye(4) + p.Ts * A_NORMAL for _ in range(4)]
    Bd = [p.Ts * g * B_VEC for g in B_GAIN]
    Cd = p.Ts * C_VEC
    return Ad, Bd, Cd, [p.q_scale * q for q in Q_MODE], [p.ru_scale * r for r in RU_MODE], [p.rv] * 4


# =====================================================================================
# 2. SOLVERS
# =====================================================================================
def mode_matrix(p: Params, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """lambda_ij(sigma1, sigma2) = sum_{a1,a2} s1[i,a1] s2[i,a2] T[i,a1,a2,j]."""
    return np.einsum("ia,ib,iabj->ij", s1, s2, p.T)


def physical_values(p: Params, lam: np.ndarray, check: bool = True) -> np.ndarray:
    """J*(s) = x0' P_{0,s} x0 : value of the Markov-jump zero-sum LQ game started in mode s.

    Backward DP, V_k(x,i) = min_u max_v [ x'Q_i x + R_u,i u^2 - R_v,i v^2 + sum_j lam_ij V_{k+1}(x',j) ],
    x' = A x + B_i u + C v.  With M_i = [B_i C], Z = sum_j lam_ij P_j and H = diag(R_u,-R_v) + M'ZM the saddle point
    exists iff H_11 > 0 and the Schur complement H_22 - H_12^2/H_11 < 0 (checked at every step).
    """
    Ad, Bd, Cd, Q, Ru, Rv = physical_matrices(p)
    Ms = [np.hstack([Bd[i], Cd]) for i in range(4)]
    P = [q.copy() for q in Q]
    for _ in range(p.K):
        Pn = []
        for i in range(4):
            Z = sum(lam[i, j] * P[j] for j in range(4))
            M = Ms[i]
            H = np.diag([Ru[i], -Rv[i]]) + M.T @ Z @ M
            if check:
                a = H[0, 0]
                schur = H[1, 1] - H[0, 1] ** 2 / a
                if not (a > 0 and schur < 0):
                    raise ValueError(f"saddle-point condition violated (mode {i}): "
                                     f"R_u+B'ZB={a:.3g}, Schur={schur:.3g}")
            L = np.linalg.solve(H, M.T @ Z @ Ad[i])
            Pn.append(Q[i] + Ad[i].T @ Z @ Ad[i] - Ad[i].T @ Z @ M @ L)
        P = Pn
    x0 = np.array(p.x0)
    return np.array([x0 @ P[i] @ x0 for i in range(4)])


def solve_matrix_game(G: np.ndarray):
    """Row player MINIMISES, column player MAXIMISES.  Returns (value, row strategy, column strategy)."""
    m, n = G.shape
    c = np.zeros(m + 1); c[-1] = 1.0
    A_ub = np.hstack([G.T, -np.ones((n, 1))])
    A_eq = np.zeros((1, m + 1)); A_eq[0, :m] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = np.clip(res.x[:m], 0, None); x /= x.sum()
    y = np.clip(-res.ineqlin.marginals, 0, None)
    y = y / y.sum() if y.sum() > 0 else np.ones(n) / n
    return res.x[-1], x, y


def stage_cost(p: Params, damage) -> np.ndarray:
    """omega[s,a1,a2] = C(a1) + 1[a2 undetected by a1] * damage[s,a2] - c(a2).
    `damage` : scalar (constant penalty = baseline), vector over states, or array [s,a2]."""
    D = detected_mask(p.detect)
    d = np.asarray(damage, float)
    if d.ndim == 0:
        d = np.full((4, 3), float(d))
    elif d.ndim == 1:
        d = np.repeat(d[:, None], 3, axis=1)
    return (np.array(p.cost)[None, :, None] + (1 - D)[None, :, :] * d[:, None, :]
            - np.array(p.att_cost)[None, None, :])


def shapley(p: Params, omega: np.ndarray, v0=None, tol=1e-10, maxit=500, history=False):
    v = np.zeros(4) if v0 is None else v0.copy()
    hist = [v.copy()]
    for _ in range(maxit):
        vn = np.zeros(4); S1 = np.zeros((4, 4)); S2 = np.zeros((4, 3))
        for s in range(4):
            vn[s], S1[s], S2[s] = solve_matrix_game(omega[s] + p.beta * p.T[s] @ v)
        done = np.max(np.abs(vn - v)) < tol
        v = vn; hist.append(v.copy())
        if done:
            break
    return (v, S1, S2, np.array(hist)) if history else (v, S1, S2)


def damage_from_J(p: Params, J: np.ndarray) -> np.ndarray:
    """d[s,a2] = Lam * (1 - exp(-eta * [E_U J*(s') - J*(s)]^+)):  bounded, increasing in the physical-cost
    INCREASE caused by an undetected attack a2 in state s."""
    dJ = np.einsum("sat,t->sa", p.Uarr, J) - J[:, None]
    return p.Lam * (1.0 - np.exp(-p.eta * np.clip(dJ, 0.0, None)))


def solve_proposed(p: Params, damp=0.5, tol=1e-7, maxit=300, history=False):
    """Fixed point sigma = T_C(T_P(sigma)) by damped iteration."""
    s1 = np.ones((4, 4)) / 4; s2 = np.ones((4, 3)) / 3; v = None; hist = []
    for it in range(maxit):
        J = physical_values(p, mode_matrix(p, s1, s2))
        v, t1, t2 = shapley(p, stage_cost(p, damage_from_J(p, J)), v0=v)
        err = max(np.abs(t1 - s1).max(), np.abs(t2 - s2).max()); hist.append(float(err))
        s1 = (1 - damp) * s1 + damp * t1; s2 = (1 - damp) * s2 + damp * t2
        if err < tol:
            break
    J = physical_values(p, mode_matrix(p, s1, s2))
    v, t1, t2 = shapley(p, stage_cost(p, damage_from_J(p, J)), v0=v)
    out = dict(s1=s1, s2=s2, v=v, J=J, iters=it + 1, converged=bool(err < tol),
               residual=float(max(np.abs(t1 - s1).max(), np.abs(t2 - s2).max())))
    if history:
        out["fp_history"] = hist
    return out


def solve_baseline(p: Params, kappa: float):
    v, s1, s2 = shapley(p, stage_cost(p, float(kappa)))
    return dict(s1=s1, s2=s2, v=v)


def off_when_absorbed(s1: np.ndarray) -> np.ndarray:
    """Variant 'off-in-H' of a cyber-only policy: IDS switched off in the absorbing Out-of-Service state."""
    s = s1.copy(); s[3] = [1, 0, 0, 0]; return s


# =====================================================================================
# 3. EVALUATION OF A POLICY IN THE COMMON PHYSICAL GAME
# =====================================================================================
def evaluate(p: Params, s1: np.ndarray, s2: np.ndarray, start: int = 0):
    lam = mode_matrix(p, s1, s2)
    J = physical_values(p, lam)
    be = p.beta_eval
    e = np.zeros(4); e[start] = 1.0
    mu = (1 - be) * e @ np.linalg.inv(np.eye(4) - be * lam)       # discounted state occupancy
    return dict(J_N=float(J[start]), EC=float(mu @ (s1 @ np.array(p.cost))))


def attacker_policies():
    """27 stationary pure attacker policies on the three non-absorbing states."""
    for combo in itertools.product(range(3), repeat=3):
        s2 = np.zeros((4, 3))
        for s, a in enumerate(combo):
            s2[s, a] = 1.0
        s2[3, 0] = 1.0
        yield combo, s2


def nominal_cost(p: Params, start: int = 0) -> float:
    """J*_nom : minimax cost if the mode never leaves `start` (no cyber-induced degradation)."""
    return float(physical_values(p, np.eye(4))[start])


def metrics(p: Params, s1: np.ndarray, Jn: float):
    """Physical degradation  Delta_J = J*(N) - J*_nom  and average IDS spend  EC, under
    (w) the worst of the 27 pure attackers, (u) the uniform attacker."""
    worst = None
    for combo, s2 in attacker_policies():
        e = evaluate(p, s1, s2)
        if worst is None or e["J_N"] > worst[1]:
            worst = (combo, e["J_N"], e["EC"])
    u = evaluate(p, s1, np.ones((4, 3)) / 3)
    return dict(EC_w=worst[2], exc_w=worst[1] - Jn, EC_u=u["EC"], exc_u=u["J_N"] - Jn, worst=list(worst[0]))


# =====================================================================================
# 4. MINIMAX-Q LEARNING (model-free on the cyber layer; physical LQ game solved exactly)
# =====================================================================================
def run_qlearning(p: Params, mode="proposed", kappa=10.0, seed=0, episodes=1000, steps=12, alpha=0.05,
                  eps=0.05, refresh_J=100, snap_every=600, ref=None):
    rng = np.random.default_rng(seed)
    Q = np.zeros((4, 4, 3)); dirty = np.ones(4, bool)
    V = np.zeros(4); P1 = np.ones((4, 4)) / 4; P2 = np.ones((4, 3)) / 3

    def refresh(s):
        if dirty[s]:
            V[s], P1[s], P2[s] = solve_matrix_game(Q[s]); dirty[s] = False

    def build_omega():
        for s in range(4):
            refresh(s)
        if mode == "proposed":
            return stage_cost(p, damage_from_J(p, physical_values(p, mode_matrix(p, P1, P2))))
        return stage_cost(p, float(kappa))

    omega = build_omega(); trace = []; n = 0
    for _ in range(episodes):
        s = int(rng.integers(0, 4))
        for _ in range(steps):
            refresh(s)
            a1 = int(rng.integers(0, 4)) if rng.random() < eps else int(rng.choice(4, p=P1[s]))
            a2 = int(rng.integers(0, 3)) if rng.random() < eps else int(rng.choice(3, p=P2[s]))
            sn = int(rng.choice(4, p=p.T[s, a1, a2]))
            refresh(sn)
            Q[s, a1, a2] += alpha * (omega[s, a1, a2] + p.beta * V[sn] - Q[s, a1, a2])
            dirty[s] = True; s = sn; n += 1
            if n % refresh_J == 0:
                omega = build_omega()
            if ref is not None and n % snap_every == 0:
                for k in range(4):
                    refresh(k)
                trace.append((n, float(np.abs(P1[:3] - ref[0][:3]).sum(axis=1).mean()), float(V[0])))
    for k in range(4):
        refresh(k)
    return dict(s1=P1.copy(), s2=P2.copy(), v=V.copy(), trace=np.array(trace))


# =====================================================================================
# 5. EXPERIMENTS
# =====================================================================================
P0 = make_params("A")
KAPPAS = np.round(np.geomspace(1.0, 200.0, 36), 3)
LAMS = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100]
N_SEEDS = 20
N_EPISODES = 3000      # minimax-Q episodes (12 updates each)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o))


def dump(obj, name):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def baseline_curve(p, Jn, kappas=KAPPAS):
    out = []
    for k in kappas:
        b = solve_baseline(p, float(k))
        out.append(dict(kappa=float(k), keep=metrics(p, b["s1"], Jn), off=metrics(p, off_when_absorbed(b["s1"]), Jn),
                        s1=b["s1"]))
    return out


def gap_vs_baseline(prop_m, curve, key="off", att="w"):
    """Excess(best baseline whose spend <= proposed spend) - excess(proposed).  > 0 : proposed is better."""
    ec, ex = prop_m["EC_" + att], prop_m["exc_" + att]
    cands = [c[key]["exc_" + att] for c in curve if c[key]["EC_" + att] <= ec + 1e-9]
    return (min(cands) if cands else float("nan")) - ex


def stage_exact():
    p = P0; Jn = nominal_cost(p)
    r = solve_proposed(p, history=True)
    out = dict(J_nom=Jn, converged=r["converged"], iters=r["iters"], residual=r["residual"], s1=r["s1"], s2=r["s2"],
               v=r["v"], J=r["J"], damage=damage_from_J(p, r["J"]), fp_history=r["fp_history"],
               metrics=metrics(p, r["s1"], Jn),
               none=metrics(p, np.tile([1., 0, 0, 0], (4, 1)), Jn), full=metrics(p, np.tile([0, 0, 0, 1.], (4, 1)), Jn))
    out["baselines"] = {}
    for k in (1, 5, 10, 20):
        b = solve_baseline(p, float(k))
        out["baselines"][str(k)] = dict(s1=b["s1"], v=b["v"], keep=metrics(p, b["s1"], Jn),
                                        off=metrics(p, off_when_absorbed(b["s1"]), Jn))
    omega = stage_cost(p, damage_from_J(p, r["J"]))
    out["vi_history"] = shapley(p, omega, history=True)[3]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    out["beta_grid"] = betas; out["v_vs_beta"] = []; out["full_vs_beta"] = []
    for b_ in betas:
        rr = solve_proposed(p.with_(beta=b_)); out["v_vs_beta"].append(rr["v"]); out["full_vs_beta"].append(rr["s1"][:, 3])
    dump(out, "exact.json")
    print("exact: J_nom=%.3f  fixed point converged=%s in %d it  EC_w=%.3f  exc_w=%.3f" %
          (Jn, r["converged"], r["iters"], out["metrics"]["EC_w"], out["metrics"]["exc_w"]))


def stage_pareto():
    p = P0; Jn = nominal_cost(p)
    curve = baseline_curve(p, Jn)
    prop = []
    for L in LAMS:
        pl = p.with_(Lam=float(L)); r = solve_proposed(pl)
        prop.append(dict(Lam=L, converged=r["converged"], s1=r["s1"], metrics=metrics(pl, r["s1"], Jn)))
    base = [x for x in prop if x["Lam"] == P0.Lam][0]["metrics"]
    j = int(np.argmin([abs(c["off"]["EC_w"] - base["EC_w"]) for c in curve]))
    out = dict(J_nom=Jn, baseline=curve, proposed=prop, matched_kappa=curve[j]["kappa"],
               gap_w_off=gap_vs_baseline(base, curve, "off", "w"), gap_u_off=gap_vs_baseline(base, curve, "off", "u"),
               gap_w_keep=gap_vs_baseline(base, curve, "keep", "w"), gap_u_keep=gap_vs_baseline(base, curve, "keep", "u"))
    dump(out, "pareto.json")
    print("pareto: matched kappa=%.2f | gap(off): worst %.3f unif %.3f | gap(keep): worst %.3f unif %.3f" %
          (out["matched_kappa"], out["gap_w_off"], out["gap_u_off"], out["gap_w_keep"], out["gap_u_keep"]))


# ---- sensitivity ------------------------------------------------------------------------
def _perturb_transitions(level, seed, detect):
    rng = np.random.default_rng(seed)

    def pert(vec):
        v = np.array(vec, float); m = v > 0
        v[m] = v[m] * (1 + level * rng.uniform(-1, 1, m.sum()))      # independent noise per entry
        return (v / v.sum()).tolist()
    U2 = {s: [pert(x) for x in U[s]] for s in U}
    D2 = {s: [pert(x) for x in DT[s]] for s in DT}
    return build_transitions(U2, D2, detect), build_undetected(U2)


def sens_cases():
    c = []
    for lvl in (0.2, 0.5):
        for d in range(10):
            c.append(("Transition probabilities", f"+/-{int(lvl * 100)}% (draw {d})", dict(_trans=(lvl, 1000 * int(lvl * 10) + d))))
    for sc in (0.8, 1.2):
        c.append(("IDS costs C(F)", f"x{sc} (all)", dict(cost=tuple(np.array(P0.cost) * sc))))
    for d in range(6):
        c.append(("IDS costs C(F)", f"+/-20% individually (draw {d})", dict(_cost=d)))
    for sc in (0.8, 1.2): c.append(("State cost Q(s)", f"x{sc}", dict(q_scale=sc)))
    for sc in (0.8, 1.2): c.append(("Control cost R_u(s)", f"x{sc}", dict(ru_scale=sc)))
    for rv in (60., 200.): c.append(("Attacker cost R_v", f"{rv:g}", dict(rv=rv)))
    for k in (10, 40): c.append(("Horizon K", f"{k}", dict(K=k)))
    for ts in (0.05, 0.2): c.append(("Sampling step T_s", f"{ts}", dict(Ts=ts)))
    for b in (0.3, 0.7, 0.9): c.append(("Discount beta", f"{b}", dict(beta=b)))
    for L in (10., 40.): c.append(("Damage scale Lambda", f"{L:g}", dict(Lam=L)))
    for g in (0.025, 0.1): c.append(("Damage steepness eta", f"{g}", dict(eta=g)))
    for sc in (0.5, 1.5): c.append(("Attacker effort c(a)", f"x{sc}", dict(att_cost=tuple(np.array(P0.att_cost) * sc))))
    c.append(("Detection structure", "complementary libraries (instance B)", dict(_instance="B")))
    return c


def run_case(case):
    fam, label, kw = case
    kw = dict(kw)
    try:
        inst = kw.pop("_instance", "A")
        p = make_params(inst)
        if "_trans" in kw:
            lvl, seed = kw.pop("_trans"); T, Ua = _perturb_transitions(lvl, seed, DETECT_SETS[inst]); p = p.with_(T=T, Uarr=Ua)
        if "_cost" in kw:
            d = kw.pop("_cost"); rng = np.random.default_rng(500 + d)
            c = np.array(P0.cost); c[1:] = c[1:] * (1 + 0.2 * rng.uniform(-1, 1, 3)); kw["cost"] = tuple(c)
        p = p.with_(**kw)
        Jn = nominal_cost(p); r = solve_proposed(p); m = metrics(p, r["s1"], Jn)
        curve = baseline_curve(p, Jn, kappas=np.round(np.geomspace(1.0, 200.0, 22), 3))
        return dict(family=fam, label=label, ok=True, converged=r["converged"], s1=r["s1"], EC_w=m["EC_w"], exc_w=m["exc_w"],
                    EC_u=m["EC_u"], exc_u=m["exc_u"], none_exc_w=metrics(p, np.tile([1., 0, 0, 0], (4, 1)), Jn)["exc_w"],
                    gap_w=gap_vs_baseline(m, curve, "off", "w"), gap_u=gap_vs_baseline(m, curve, "off", "u"),
                    gap_w_keep=gap_vs_baseline(m, curve, "keep", "w"))
    except Exception as e:  # e.g. saddle-point condition violated
        return dict(family=fam, label=label, ok=False, err=str(e)[:160])


def stage_sensitivity():
    cases = sens_cases(); t = time.time()
    with ProcessPoolExecutor(min(4, os.cpu_count() or 2)) as ex:
        res = list(ex.map(run_case, cases))
    dump(res, "sensitivity.json")
    print("sensitivity: %d/%d cases solved in %.0fs" % (sum(r["ok"] for r in res), len(res), time.time() - t))


# ---- Q-learning with statistics ---------------------------------------------------------
def _ql_job(args):
    mode, seed, kappa = args
    p = P0; Jn = nominal_cost(p); ex = load("exact.json")
    q = run_qlearning(p, mode, kappa=kappa, seed=seed, episodes=N_EPISODES, snap_every=1800,
                      ref=(np.array(ex["s1"]), np.array(ex["s2"])))
    return dict(mode=mode, seed=seed, s1=q["s1"], s2=q["s2"], v=q["v"], trace=q["trace"],
                keep=metrics(p, q["s1"], Jn), off=metrics(p, off_when_absorbed(q["s1"]), Jn),
                dist_exact=float(np.abs(q["s1"][:3] - np.array(ex["s1"])[:3]).sum(axis=1).mean()))


def _ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def stage_qlearn():
    kappa = load("pareto.json")["matched_kappa"]
    jobs = [("proposed", s, 0.0) for s in range(N_SEEDS)] + [("baseline", s, kappa) for s in range(N_SEEDS)]
    t = time.time()
    with ProcessPoolExecutor(min(4, os.cpu_count() or 2)) as ex:
        res = list(ex.map(_ql_job, jobs))
    prop = [r for r in res if r["mode"] == "proposed"]; base = [r for r in res if r["mode"] == "baseline"]
    summ = {}
    for var in ("off", "keep"):
        for key in ("exc_w", "EC_w", "exc_u", "EC_u"):
            a = np.array([r[var][key] for r in prop]); b = np.array([r[var][key] for r in base])
            try:
                pval = float(stats.wilcoxon(a, b).pvalue)
            except ValueError:
                pval = float("nan")
            summ[f"{var}_{key}"] = dict(prop_mean=a.mean(), prop_std=a.std(ddof=1), prop_ci=_ci95(a), base_mean=b.mean(),
                                       base_std=b.std(ddof=1), base_ci=_ci95(b), wilcoxon_p=pval)
    summ["dist_exact_mean"] = float(np.mean([r["dist_exact"] for r in prop]))
    summ["dist_exact_ci"] = _ci95([r["dist_exact"] for r in prop])
    dump(dict(kappa=kappa, summary=summ, runs=res), "qlearn.json")
    print("qlearn: %d runs in %.0fs ; mean L1 distance to exact policy (proposed) = %.3f" %
          (len(res), time.time() - t, summ["dist_exact_mean"]))


if __name__ == "__main__":
    stages = sys.argv[1:] or ["exact", "pareto", "sensitivity", "qlearn"]
    for s in stages:
        globals()["stage_" + s]()