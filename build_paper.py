#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_paper.py -- turns results/*.json (produced by decision_iet_final.py) into figures, LaTeX files and documents:

  paper_out/figures/   fig_equilibrium, fig_beta, fig_qlearning, fig_pareto, fig_sensitivity   (.png/.eps/.pdf)
  paper_out/latex/     numbers.tex (macros), tab_*.tex, results_section.tex, model_patch.tex, text_patches.tex
  paper_out/response/  response_letter.tex (+ .pdf if pdflatex is available), title_page.docx
  paper_out/README.md

Every number in the text is written by this script from the results files, so text and tables cannot disagree.
Usage:   python build_paper.py            (after: python decision_iet_final.py)
         python build_paper.py --run      (runs the experiments first, ~15 min)
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decision_iet_final as core  # noqa: E402

OUT = os.path.join(HERE, "paper_out")
FIG, TEX, RESP = (os.path.join(OUT, d) for d in ("figures", "latex", "response"))
for d in (FIG, TEX, RESP):
    os.makedirs(d, exist_ok=True)

STATES, DEFA, ATTA = core.STATES, core.DEF_ACTIONS, core.ATT_ACTIONS
# categorical slots (validated reference palette, fixed order) + marker/linestyle as secondary encoding
SC = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
MK = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "legend.frameon": False,
        "savefig.dpi": 300, "lines.linewidth": 1.6, "lines.markersize": 5, "axes.axisbelow": True})


def save(fig, name):
    for ext in ("png", "eps", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def f(x, nd=2):
    return f"{x:.{nd}f}"


def f3(x):
    return "0" if abs(x) < 5e-4 else f"{x:.3f}"


def esc(s):
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def fill(t, d):
    for k, v in d.items():
        t = t.replace(f"@@{k}@@", str(v))
    assert "@@" not in t, "unfilled token: " + t[t.index("@@"):t.index("@@") + 30]
    return t


def write(path, s):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s)


# ------------------------------------------------------------------------------------------------ data
def load_all():
    need = ["exact.json", "pareto.json", "sensitivity.json", "qlearn.json"]
    miss = [n for n in need if not os.path.exists(os.path.join(core.RES, n))]
    if miss:
        sys.exit("missing results: %s  -> run  python decision_iet_final.py  first" % miss)
    return [core.load(n) for n in need]


# ------------------------------------------------------------------------------------------------ figures
def fig_equilibrium(ex):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    h = np.array(ex["fp_history"])
    ax[0].semilogy(np.arange(1, len(h) + 1), np.maximum(h, 1e-12), color=SC[0], marker="o", ms=3)
    ax[0].set_xlabel("Fixed-point iteration"); ax[0].set_ylabel(r"$\|\sigma^{(n)}-T_S(\sigma^{(n)})\|_\infty$")
    ax[0].set_title("(a) Stackelberg fixed point", fontsize=9, loc="left")
    vi = np.array(ex["vi_history"])
    for s in range(4):
        ax[1].plot(vi[:, s], color=SC[s], marker=MK[s], ms=3, label=STATES[s])
    ax[1].set_xlabel("Value-iteration step"); ax[1].set_ylabel("Defender's game value (cost)")
    ax[1].set_title("(b) Shapley value iteration", fontsize=9, loc="left"); ax[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, "fig_equilibrium")


def fig_beta(ex):
    b = ex["beta_grid"]; v = np.array(ex["v_vs_beta"]); fu = np.array(ex["full_vs_beta"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for s in range(4):
        ax[0].plot(b, v[:, s], color=SC[s], marker=MK[s], ms=3, label=STATES[s])
        ax[1].plot(b, fu[:, s], color=SC[s], marker=MK[s], ms=3, label=STATES[s])
    ax[0].set_xlabel(r"Discount factor $\beta$"); ax[0].set_ylabel("Defender's game value (cost)")
    ax[0].set_title("(a) Game value", fontsize=9, loc="left"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel(r"Discount factor $\beta$"); ax[1].set_ylabel("Probability of Full")
    ax[1].set_title("(b) Defender's probability of Full", fontsize=9, loc="left"); ax[1].set_ylim(-0.03, 1.03)
    fig.tight_layout(); save(fig, "fig_beta")


def fig_qlearning(ex, ql):
    runs = [r for r in ql["runs"] if r["mode"] == "proposed"]
    tr = np.array([r["trace"] for r in runs])           # (seeds, T, 3)
    x = tr[0, :, 0]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for k, (j, lab) in enumerate([(1, r"Mean $L^1$ distance of $\hat\sigma_1$ to exact equilibrium"),
                                  (2, r"Learned value $\hat V_D(\mathrm{Normal})$")]):
        m = tr[:, :, j].mean(0); ci = np.array([core._ci95(tr[:, t, j]) for t in range(tr.shape[1])])
        ax[k].plot(x, m, color=SC[0], label="Minimax-Q (mean of %d seeds)" % len(runs))
        ax[k].fill_between(x, m - ci, m + ci, color=SC[0], alpha=0.2, linewidth=0, label="95% CI")
        ax[k].set_xlabel("Q-learning updates"); ax[k].set_ylabel(lab)
    ax[1].axhline(ex["v"][0], color=INK2, ls="--", lw=1.2, label="Exact value $v_D(\\mathrm{Normal})$")
    ax[0].set_title("(a) Policy error", fontsize=9, loc="left"); ax[1].set_title("(b) Value", fontsize=9, loc="left")
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); save(fig, "fig_qlearning")


def fig_pareto(ex, pa, ql):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    ks = np.array([c["kappa"] for c in pa["baseline"]])
    for ax, att, ttl in ((axs[0], "w", "(a) Worst-case attacker"), (axs[1], "u", "(b) Uniform attacker")):
        for key, ls, mfc, lab in (("keep", "--", "none", r"Cyber-only $\kappa$-sweep, IDS kept on in H"),
                                  ("off", "-", SC[1], r"Cyber-only $\kappa$-sweep, IDS off in H")):
            pts = sorted((c[key]["EC_" + att], c[key]["exc_" + att]) for c in pa["baseline"])
            ax.plot(*zip(*pts), color=SC[1], ls=ls, marker="s", ms=3.5, mfc=mfc, label=lab)
        pr = sorted((x["metrics"]["EC_" + att], x["metrics"]["exc_" + att]) for x in pa["proposed"])
        ax.plot(*zip(*pr), color=SC[0], marker="o", ms=4, label=r"Proposed, $\Lambda$-sweep")
        b = [x for x in pa["proposed"] if x["Lam"] == core.P0.Lam][0]["metrics"]
        ax.plot([b["EC_" + att]], [b["exc_" + att]], "o", color=SC[0], ms=9, mfc="none", mew=1.8, label=r"Proposed, $\Lambda=%g$" % core.P0.Lam)
        for tgt in (5, 20, 100):
            i = int(np.argmin(abs(ks - tgt))); c = pa["baseline"][i]["off"]
            ax.annotate(r"$\kappa$=%.3g" % ks[i], (c["EC_" + att], c["exc_" + att]), textcoords="offset points", xytext=(6, 2), fontsize=7, color=INK2)
        for nm, m in (("no IDS", ex["none"]), ("always Full", ex["full"])):
            ax.plot([m["EC_" + att]], [m["exc_" + att]], "x", color=INK2, ms=6)
            ax.annotate(nm, (m["EC_" + att], m["exc_" + att]), textcoords="offset points", xytext=((10, -3) if nm == "no IDS" else (-20, 8)), fontsize=7, color=INK2)
        ax.set_xlabel(r"Average IDS spend $\bar{E}$"); ax.set_title(ttl, fontsize=9, loc="left")
    axs[0].set_ylabel(r"Physical degradation $\Delta_J=J^*(N)-J^*_{\rm nom}$")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(); fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 0.0)); save(fig, "fig_pareto")


def sens_families(se):
    fams = []
    for r in se:
        if r["family"] not in fams:
            fams.append(r["family"])
    return fams


def fig_sensitivity(ex, se):
    fams = sens_families(se); ok = [r for r in se if r["ok"]]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, key, xl in ((axs[0], "gap_w", "Excess degradation of best cyber-only baseline\nat matched spend minus proposed"),
                        (axs[1], "full", "Probability of Full in state Normal")):
        for i, fam in enumerate(fams):
            vals = [(r["gap_w"] if key == "gap_w" else r["s1"][0][3]) for r in ok if r["family"] == fam]
            ax.plot(vals, [i] * len(vals), "o", color=SC[0], ms=4, alpha=0.75, mfc=SC[0], mec="white", mew=0.6)
        ax.set_xlabel(xl, fontsize=8)
    axs[0].axvline(0, color=INK2, ls="--", lw=1)
    axs[1].axvline(ex["s1"][0][3], color=INK2, ls="--", lw=1)
    nice = {"State cost Q(s)": "State cost $Q(s)$", "Control cost R_u(s)": "Control cost $R_u(s)$", "Attacker cost R_v": "Attacker cost $R_v$",
            "Sampling step T_s": "Sampling step $T_s$", "Horizon K": "Horizon $K$", "Discount beta": r"Discount $\beta$",
            "Damage scale Lambda": r"Damage scale $\Lambda$", "Damage steepness eta": r"Damage steepness $\eta$", "Attacker effort c(a)": "Attacker effort $c(a)$",
            "IDS costs C(F)": r"IDS costs $\mathcal{C}(F)$"}
    axs[0].set_yticks(range(len(fams))); axs[0].set_yticklabels([nice.get(x, x) for x in fams], fontsize=8); axs[0].invert_yaxis()
    axs[0].set_title("(a) Gap to the cyber-only frontier", fontsize=9, loc="left")
    axs[1].set_title("(b) Defender's decision in Normal", fontsize=9, loc="left")
    fig.tight_layout(); save(fig, "fig_sensitivity")


# ------------------------------------------------------------------------------------------------ tables
def tabular(header, rows, colspec):
    L = ["\\begin{tabular}{" + colspec + "}", "\\hline", " & ".join("\\textbf{%s}" % h for h in header) + " \\\\ \\hline"]
    L += [" & ".join(r) + " \\\\ \\hline" for r in rows] + ["\\end{tabular}"]
    return "\n".join(L)


def table(caption, label, body, small=False):
    return ("\\begin{table}[htbp]\n\\centering\n" + ("\\small\n" if small else "") +
            "\\caption{\\revtext{%s}}\n\\label{%s}\n%s\n\\end{table}\n" % (caption, label, body))


def tab_defender_attacker_values(ex):
    s1, s2, v, J, d = (np.array(ex[k]) for k in ("s1", "s2", "v", "J", "damage"))
    t1 = table("Defender's equilibrium mixed strategies $\\sigma_1^*(s,\\cdot)$.", "tab:defender_mixed_strategies",
               tabular(["State"] + DEFA, [[STATES[s]] + [f3(x) for x in s1[s]] for s in range(4)], "|l|c|c|c|c|"))
    t2 = table("Attacker's equilibrium mixed strategies $\\sigma_2^*(s,\\cdot)$ (in Out of Service the attack choice has no effect).",
               "tab:attacker_mixed_strategies", tabular(["State"] + ATTA, [[STATES[s]] + [f3(x) for x in s2[s]] for s in range(4)], "|l|c|c|c|"))
    rows = [[STATES[s], f(J[s], 2), f(v[s], 3), f(-v[s], 3), "%s--%s" % (f(d[s].min(), 2), f(d[s].max(), 2))] for s in range(4)]
    t3 = table("Physical cost $J^*(s)$, game values (defender minimises cost; attacker value $=-v_D$) and range of the damage $d(s,a)$ over the three attacks.",
               "tab:game_values", tabular(["State", "$J^*(s)$", "$v_D(s)$", "$v_A(s)$", "$d(s,\\cdot)$"], rows, "|l|c|c|c|c|"))
    write(os.path.join(TEX, "tab_defender.tex"), t1); write(os.path.join(TEX, "tab_attacker.tex"), t2); write(os.path.join(TEX, "tab_values.tex"), t3)


def tab_comparison(ex, pa):
    def row(name, m):
        return [name, f(m["EC_w"], 2), f(m["exc_w"], 2), f(m["EC_u"], 2), f(m["exc_u"], 2)]
    rows = [row("No IDS (always None)", ex["none"]), row("Always Full", ex["full"])]
    for k in ("1", "5", "10", "20"):
        b = ex["baselines"][k]
        rows.append(row("Cyber-only, $\\kappa=%s$, IDS on in H" % k, b["keep"]))
        rows.append(row("Cyber-only, $\\kappa=%s$, IDS off in H" % k, b["off"]))
    mk = pa["matched_kappa"]; c = [c for c in pa["baseline"] if c["kappa"] == mk][0]
    rows.append(row("Cyber-only, $\\kappa^\\star=%.2f$ (matched spend), IDS off in H" % mk, c["off"]))
    rows.append(row("\\textbf{Proposed (physical feedback)}", ex["metrics"]))
    body = tabular(["Policy", "$\\bar E$ (worst)", "$\\Delta_J$ (worst)", "$\\bar E$ (unif.)", "$\\Delta_J$ (unif.)"], rows, "|p{5.3cm}|c|c|c|c|")
    write(os.path.join(TEX, "tab_comparison.tex"), table(
        "All policies evaluated in the same physical game against the same attackers: average IDS spend $\\bar E$ and physical degradation $\\Delta_J$ (lower is better), against the worst of 27 stationary attackers and against the uniform attacker.",
        "tab:comparison", body, small=True))


def tab_learned(ql):
    s = ql["summary"]
    def pm(a, b, key):
        e = s["off_" + key]
        return ("%s $\\pm$ %s" % (f(e["prop_mean"], 2), f(e["prop_ci"], 2)), "%s $\\pm$ %s" % (f(e["base_mean"], 2), f(e["base_ci"], 2)),
                ("%.3f" % e["wilcoxon_p"]) if e["wilcoxon_p"] == e["wilcoxon_p"] else "n/a")
    rows = []
    for key, lab in (("EC_w", "$\\bar E$ (worst-case attacker)"), ("exc_w", "$\\Delta_J$ (worst-case attacker)"),
                     ("EC_u", "$\\bar E$ (uniform attacker)"), ("exc_u", "$\\Delta_J$ (uniform attacker)")):
        rows.append([lab] + list(pm(None, None, key)))
    body = tabular(["Metric", "Proposed (learned)", "Cyber-only $\\kappa^\\star$ (learned)", "Wilcoxon $p$"], rows, "|l|c|c|c|")
    write(os.path.join(TEX, "tab_learned.tex"), table(
        "Policies learned by minimax-Q over %d paired seeds (mean $\\pm$ 95\\%% CI); the cyber-only policy uses $\\kappa^\\star=%.2f$ and the IDS is switched off in H." % (core.N_SEEDS, ql["kappa"]),
        "tab:learned", body))


def sens_summary(ex, se):
    fams = sens_families(se); base = np.array(ex["s1"]); rows = []
    for fam in fams:
        cs = [r for r in se if r["family"] == fam]; ok = [r for r in cs if r["ok"]]
        if not ok:
            rows.append([esc(fam), "0/%d" % len(cs), "--", "--", "--", "--"]); continue
        fulln = [r["s1"][0][3] for r in ok]
        dsig = max(np.abs(np.array(r["s1"])[:3] - base[:3]).max() for r in ok)
        gap = [r["gap_w"] for r in ok]
        rows.append([esc(fam), "%d/%d" % (len(ok), len(cs)), "%s--%s" % (f(min(fulln), 2), f(max(fulln), 2)), f(dsig, 2),
                     "%s to %s" % (f(min(gap), 2), f(max(gap), 2)), "%s--%s" % (f(min(r["exc_w"] for r in ok), 1), f(max(r["exc_w"] for r in ok), 1))])
    body = tabular(["Perturbed quantity", "Solved", "Pr(Full$|N$)", "$\\max\\Delta\\sigma_1$", "Gap to frontier", "$\\Delta_J$"], rows, "|p{3.6cm}|c|c|c|c|c|")
    write(os.path.join(TEX, "tab_sensitivity.tex"), table(
        "Sensitivity study. Each row perturbs one group of parameters (several random draws for the transition probabilities and for individual costs). ``Solved'' counts the instances in which all saddle-point conditions held. $\\max\\Delta\\sigma_1$ is the largest deviation of the defender's strategy (states N, A, D) from the base case; ``Gap to frontier'' is the degradation of the best cyber-only baseline of the same spend minus that of the proposed policy (positive: proposed is better); $\\Delta_J$ is the degradation of the proposed policy against the worst-case attacker.",
        "tab:sensitivity", body, small=True))
    ok = [r for r in se if r["ok"]]
    return dict(n=len(se), n_ok=len(ok), n_frontier=sum(r["gap_w"] >= -0.02 * r["exc_w"] for r in ok),
                n_off=sum(r["s1"][3][0] >= 0.99 for r in ok), n_better=sum(r["gap_w"] > 0.02 * r["exc_w"] for r in ok),
                fullN_min=min(r["s1"][0][3] for r in ok), fullN_max=max(r["s1"][0][3] for r in ok),
                gap_min=min(r["gap_w"] for r in ok), gap_max=max(r["gap_w"] for r in ok),
                failed=[(r["family"], r["label"], r["err"]) for r in se if not r["ok"]],
                n_conv=sum(r["converged"] for r in ok), low=[(r["family"], r["label"]) for r in ok if r["s1"][0][3] < 0.05])


def tab_provenance():
    p = core.P0
    R = [
        ("Plant error model", "4th-order error chain (position, speed, acceleration, jerk), Euler step $T_s=%g$ s, horizon $K=%d$" % (p.Ts, p.K), "Simplified error model; not a linearisation of a specific controller"),
        ("Control authority $B(s)=g_sB$", "$g=(%s)$ for N, A, D, H" % ", ".join("%g" % g for g in core.B_GAIN), "Illustrative: authority decreases with degradation, none in H"),
        ("Disturbance channel $C$", "acceleration channel", "Illustrative: manipulated sensor/V2X data"),
        ("$Q(s)$", "$\\mathrm{diag}(1,3,5,7)\\to\\mathrm{diag}(5,10,15,20)$", "End points from the original manuscript; intermediate modes interpolated"),
        ("$R_u(s)$", "$%s$" % ", ".join("%g" % r for r in core.RU_MODE), "End points from the original manuscript; interpolated"),
        ("$R_v$, $x_0$", "$R_v=%g$, $x_0=%g\\cdot\\mathbf 1$" % (p.rv, p.x0[0]), "$R_v$ chosen so the saddle point exists (checked at every step); $x_0$ illustrative"),
        ("$\\mathcal C(F_i)$", "$%s$" % ", ".join("%g" % c for c in p.cost), "Original ordinal costs; only ratios to $\\Lambda$ matter (swept in Section~\\ref{sec:tradeoff})"),
        ("Attacker effort $c(a)$", "$%s$" % ", ".join("%g" % c for c in p.att_cost), "Illustrative: makes the choice among attacks non-trivial"),
        ("Damage $\\Lambda,\\eta$", "$\\Lambda=%g$, $\\eta=%g$" % (p.Lam, p.eta), "Illustrative; $\\Lambda$ is the exchange rate between physical damage and IDS cost (swept)"),
        ("Transition probabilities", "Table~\\ref{tab:matrice_transition} + completion", "Six rows of Table~\\ref{tab:matrice_transition} verbatim, remaining entries chosen monotone in attack severity; illustrative, not measured"),
        ("Detection sets", "nested: Lib1$\\to\\{1\\}$, Lib2$\\to\\{1,2\\}$, Full$\\to\\{1,2,3\\}$", "Illustrative; complementary sets tested in Section~\\ref{sec:sensitivity}"),
        ("$\\beta$, $\\alpha$, $\\epsilon$, episodes", "0.5, 0.05, 0.05, $%d\\times12$ steps" % core.N_EPISODES, "Standard RL ranges; $\\beta$ swept"),
    ]
    body = tabular(["Parameter", "Value", "Provenance"], [["%s" % a, b, c] for a, b, c in R], "|p{3.0cm}|p{4.1cm}|p{5.0cm}|")
    write(os.path.join(TEX, "tab_provenance.tex"), table("Provenance of every model parameter (all values are illustrative unless stated).", "tab:provenance", body, small=True))


# ------------------------------------------------------------------------------------------------ text
def wording(gap, ref):
    rel = gap / ref
    if rel > 0.02:
        return "lower", rel
    if rel >= -0.02:
        return "comparable", rel
    return "higher", rel


def numbers(ex, pa, ql, ss):
    m, none, full = ex["metrics"], ex["none"], ex["full"]
    s1, s2 = np.array(ex["s1"]), np.array(ex["s2"])
    N = dict(
        JNOM=f(ex["J_nom"], 2), ITERS=ex["iters"], RES="%.1e" % ex["residual"],
        ECW=f(m["EC_w"], 2), ECU=f(m["EC_u"], 2), EXCW=f(m["exc_w"], 2), EXCU=f(m["exc_u"], 2),
        NONEW=f(none["exc_w"], 2), NONEU=f(none["exc_u"], 2), FULLW=f(full["exc_w"], 2), FULLU=f(full["exc_u"], 2),
        FULLN=f(s1[0][3], 2), FULLA=f(s1[1][3], 2), FULLD=f(s1[2][3], 2), NONEH=f3(s1[3][0]),
        A1N=f(s2[0][0], 2), A1A=f(s2[1][0], 2), A1D=f(s2[2][0], 2),
        A3N=f(s2[0][2], 2), A3A=f(s2[1][2], 2), A3D=f(s2[2][2], 2),
        KSTAR=f(pa["matched_kappa"], 2),
        GAPOFFW=f(pa["gap_w_off"], 2), GAPOFFU=f(pa["gap_u_off"], 2), GAPKEEPW=f(pa["gap_w_keep"], 2), GAPKEEPU=f(pa["gap_u_keep"], 2),
        SHAREW=f(100 * (none["exc_w"] - m["exc_w"]) / (none["exc_w"] - full["exc_w"]), 0),
        SHAREU=f(100 * (none["exc_u"] - m["exc_u"]) / (none["exc_u"] - full["exc_u"]), 0),
        SPENDSHW=f(100 * m["EC_w"] / full["EC_w"], 0), SPENDSHU=f(100 * m["EC_u"] / full["EC_u"], 0),
        LAM=f(core.P0.Lam, 0), ETA=f(core.P0.eta, 2),
    )
    lam_ec = [x["metrics"]["EC_w"] for x in pa["proposed"]]; lam_ex = [x["metrics"]["exc_w"] for x in pa["proposed"]]
    N.update(LAMECMIN=f(min(lam_ec), 2), LAMECMAX=f(max(lam_ec), 2), LAMEXMIN=f(min(lam_ex), 2), LAMEXMAX=f(max(lam_ex), 2))
    s = ql["summary"]
    tr = np.array([r["trace"] for r in ql["runs"] if r["mode"] == "proposed"])[:, :, 1].mean(0)
    trend = ("; the error was still decreasing when training stopped" if tr[-1] < 0.95 * tr[max(0, len(tr) - 4)] else "")
    sm = ql["summary"]
    opp = sm["off_EC_u"]["prop_mean"] < sm["off_EC_u"]["base_mean"] and sm["off_exc_u"]["prop_mean"] > sm["off_exc_u"]["base_mean"]
    qnote = (" The differences are small and, against the uniform attacker, go in opposite directions for spend and degradation: the two learned policies sit at neighbouring points of the same front rather than one dominating the other." if opp else "")
    N.update(EPIS=core.N_EPISODES, QTREND=trend, QNOTE=qnote)
    N.update(QDIST=f(s["dist_exact_mean"], 2), QDISTCI=f(s["dist_exact_ci"], 2), QKAPPA=f(ql["kappa"], 2), NSEEDS=core.N_SEEDS,
             QEXCP=f(s["off_exc_w"]["prop_mean"], 2), QEXCB=f(s["off_exc_w"]["base_mean"], 2), QECP=f(s["off_EC_w"]["prop_mean"], 2),
             QECB=f(s["off_EC_w"]["base_mean"], 2), QPEXC=("%.3f" % s["off_exc_w"]["wilcoxon_p"]), QPEC=("%.3f" % s["off_EC_w"]["wilcoxon_p"]))
    N.update(SN=ss["n"], SNOK=ss["n_ok"], SNFRONT=ss["n_frontier"], SNOFF=ss["n_off"], SNBETTER=ss["n_better"],
             SFULLMIN=f(ss["fullN_min"], 2), SFULLMAX=f(ss["fullN_max"], 2), SGAPMIN=f(max(ss["gap_min"], 0.0) if ss["gap_min"] > -0.005 else ss["gap_min"], 2), SGAPMAX=f(ss["gap_max"], 2))
    return N


def write_numbers(N):
    write(os.path.join(TEX, "numbers.tex"), "% auto-generated by build_paper.py -- do not edit\n" +
          "".join("\\newcommand{\\num%s}{%s}\n" % (k, v) for k, v in N.items()))


RESULTS = r"""
\subsection{Simulation and Result Analysis}
\label{sec:results}

\subsubsection{\revtext{Evaluation protocol}}
\label{sec:protocol}
\revtext{Every policy, proposed or baseline, is evaluated in the same physical game against the same attackers. The physical degradation of a policy $(\sigma_1,\sigma_2)$ is $\Delta_J=J^*(N)-J^*_{\rm nom}$, where $J^*(N)=T_P(\sigma_1,\sigma_2)$ is the value of the jump LQ game started in the Normal mode and $J^*_{\rm nom}=@@JNOM@@$ is the value obtained if the platoon never leaves the Normal mode. The average IDS spend is $\bar E=(1-\beta_e)\sum_t\beta_e^t\,\mathbb E[\mathcal C(F_t)]$ with $\beta_e=0.9$. Both quantities are reported against the worst of the 27 stationary pure attacker policies (``worst'') and against the uniform attacker (``unif.'').}

\subsubsection{\revtext{Equilibrium of the proposed model}}
\revtext{The damped iteration of $T_C\circ T_P$ converges in @@ITERS@@ iterations (final residual @@RES@@, Fig.~\ref{fig:equilibrium}); at every iteration the saddle-point conditions of the physical game were verified numerically. Table~\ref{tab:defender_mixed_strategies} shows that the defender plays Full with probability @@FULLN@@, @@FULLA@@ and @@FULLD@@ in the Normal, Alert and Degraded states and switches the IDS off (None, probability @@NONEH@@) in Out of Service. The latter is expected: that state is absorbing and the plant has no control authority left, so no configuration can change the outcome. The strategy in the other states is consistent with the defender anticipating the physical cost of undetected attacks, but we do not claim that this single instance proves it; the evidence is the comparison of Section~\ref{sec:baseline}.}

\revtext{The attacker plays Attack1 with probability @@A1N@@, @@A1A@@ and @@A1D@@ and Attack3 with probability @@A3N@@, @@A3A@@ and @@A3D@@ in N, A and D (Table~\ref{tab:attacker_mixed_strategies}); it mixes because the detection sets, the transition probabilities and the effort costs $c(a)$ differ across attacks. The game values in Table~\ref{tab:game_values} are costs for the defender; the value of the absorbing state is zero because the cyber payoff charges the \emph{increase} of the physical cost caused by an attack, whereas the level of degradation is reported by $\Delta_J$.}

\input{tab_defender}
\input{tab_attacker}
\input{tab_values}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig_equilibrium.png}
\caption{\revtext{Convergence of the Stackelberg fixed point (a) and of the Shapley value iteration at the equilibrium (b).}}
\label{fig:equilibrium}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig_beta.png}
\caption{\revtext{Dependence on the discount factor $\beta$ of the game value (a) and of the defender's probability of Full (b).}}
\label{fig:beta}
\end{figure}

\subsubsection{\revtext{Cyber-only baseline and common-simulator comparison}}
\label{sec:baseline}
\revtext{The cyber-only baseline shares the states, the action sets, the costs $\mathcal C(F_i)$, the attacker's effort costs, the discount factor and the Q-learning hyper-parameters of the proposed model. Its only difference is the reward: the physical damage is replaced by a constant false-negative penalty,}
\begin{equation}
\label{eq:baseline_reward}
\revtext{\omega^{\rm cyber}(s,a_1,a_2)=\mathcal C(a_1)+\mathbf 1[a_2\notin P_{a_1}]\,\kappa-c(a_2),\qquad \kappa>0 .}
\end{equation}
\revtext{We report $\kappa\in\{1,5,10,20\}$ and the whole sweep $\kappa\in[1,200]$ (Fig.~\ref{fig:pareto}). Two variants are considered: the IDS kept running in the absorbing Out-of-Service state, as a detection-centric reward would do, and the IDS switched off there (``off in H''), which is the natural convention for any policy and is what the proposed model learns by itself.}

\input{tab_comparison}

\revtext{Table~\ref{tab:comparison} gives the result. Against the worst-case attacker the proposed policy spends @@ECW@@ and reaches $\Delta_J=@@EXCW@@$, against @@NONEW@@ with no IDS and @@FULLW@@ with Full always on (spend 5.00). It therefore obtains @@SHAREW@@\% of the degradation reduction of always-Full for @@SPENDSHW@@\% of its spend (@@SHAREU@@\% for @@SPENDSHU@@\% against the uniform attacker). Compared with the cyber-only baseline of the same spend (@@KSTAR@@ for the off-in-H variant, spend and degradation read from the $\kappa$-sweep) the proposed policy is @@WORDOFFW@@: the difference in $\Delta_J$ is @@GAPOFFW@@ against the worst-case attacker and @@GAPOFFU@@ against the uniform one. When the baseline keeps the IDS on in the absorbing state, where it cannot help, the difference becomes @@GAPKEEPW@@ (worst) and @@GAPKEEPU@@ (unif.), because that baseline wastes budget there; this figure is also inflated by the coarse set of spends the baseline can reach.}

\revtext{@@FRONTIER@@}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig_pareto.png}
\caption{\revtext{Cyber risk versus physical performance. Each point is a policy evaluated in the same physical game; the proposed model is swept over the damage scale $\Lambda$, the baseline over $\kappa$. Lower-left is better.}}
\label{fig:pareto}
\end{figure}

\subsubsection{\revtext{Policies learned by minimax-Q}}
\label{sec:qlearning_results}
\revtext{Algorithm~\ref{algo} was run with @@NSEEDS@@ seeds, $\alpha=0.05$, $\epsilon=0.05$, $\beta=0.5$ and @@EPIS@@ episodes of 12 steps, sampling real transitions of the cyber game; the physical game is solved exactly (the method is model-free on the cyber layer only). The learned defender strategy is at mean $L^1$ distance @@QDIST@@ $\pm$ @@QDISTCI@@ from the exact equilibrium in states N, A and D (Fig.~\ref{fig:qlearning}), i.e. the strategies are approximated rather than recovered exactly after this budget@@QTREND@@. Evaluated in the common simulator (Table~\ref{tab:learned}), the learned proposed policy has $\Delta_J=@@QEXCP@@$ at spend @@QECP@@, against @@QEXCB@@ at spend @@QECB@@ for the learned cyber-only policy with $\kappa^\star=@@QKAPPA@@$ (Wilcoxon $p=@@QPEXC@@$ on $\Delta_J$ and $p=@@QPEC@@$ on the spend, paired by seed).@@QNOTE@@}

\input{tab_learned}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig_qlearning.png}
\caption{\revtext{Minimax-Q learning of the proposed model: distance of the learned defender strategy to the exact equilibrium (a) and learned value in the Normal state (b); mean and 95\% CI over the seeds.}}
\label{fig:qlearning}
\end{figure}

\subsubsection{\revtext{Trade-off between cyber risk and physical performance}}
\label{sec:tradeoff}
\revtext{The damage scale $\Lambda$ is the exchange rate between a unit of physical damage and a unit of IDS cost. Sweeping $\Lambda$ from 2 to 100 moves the proposed policy along a front from spend @@LAMECMIN@@ with $\Delta_J=@@LAMEXMAX@@$ to spend @@LAMECMAX@@ with $\Delta_J=@@LAMEXMIN@@$ (worst-case attacker, Fig.~\ref{fig:pareto}): a defender that values physical integrity little keeps the IDS off, one that values it highly keeps Full on in the non-absorbing states. The learned decision is therefore a choice of operating point on this front, set by $\Lambda$ rather than by a detection-accuracy target. Protection is withdrawn only where it cannot reduce physical damage (the absorbing state).}

\subsubsection{\revtext{Sensitivity analysis}}
\label{sec:sensitivity}
\revtext{Table~\ref{tab:sensitivity} perturbs, one group at a time, the transition probabilities ($\pm20\%$ and $\pm50\%$ independently on every non-zero entry, 10 draws each), the IDS costs, $Q$, $R_u$, $R_v$, the horizon, the sampling step, $\beta$, $\Lambda$, $\eta$, the attacker's effort and the structure of the detection sets. Of the @@SN@@ instances, @@SNOK@@ satisfy all saddle-point conditions; the others are reported in the table. Among the solved instances the defender switches the IDS off in the absorbing state in @@SNOFF@@, the proposed policy is on or below the cyber-only frontier (gap $\geq-2\%$ of $\Delta_J$) in @@SNFRONT@@ and better by more than 2\% of $\Delta_J$ in @@SNBETTER@@; the gap ranges from @@SGAPMIN@@ to @@SGAPMAX@@ and the probability of Full in the Normal state from @@SFULLMIN@@ to @@SFULLMAX@@. @@SENSNOTE@@}

\input{tab_sensitivity}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig_sensitivity.png}
\caption{\revtext{Sensitivity study: gap to the cyber-only frontier (a) and defender's probability of Full in the Normal state (b); each dot is one perturbed instance, dashed lines are the base case.}}
\label{fig:sensitivity}
\end{figure}

\subsection{Further Discussion}
\label{sec:discussion}
\revtext{The results support a modest and precise statement: when cyber decisions are scored by the physical consequence of an undetected attack, the learned IDS policy lies on the efficient cost--degradation front of a cyber-only baseline whose penalty $\kappa$ was tuned by sweeping, it needs no such tuning, and it withdraws protection where it is useless. They do not show a large quantitative advantage over a well-tuned cyber-only policy. All conclusions are conditional on the illustrative parameter set of Table~\ref{tab:provenance}.}
"""


def build_text(ex, pa, ql, ss, N):
    gw = pa["gap_w_off"]; ref = ex["metrics"]["exc_w"]
    word, rel = wording(gw, ref)
    N["WORDOFFW"] = {"lower": "lower", "comparable": "comparable (within 2\\%)", "higher": "higher"}[word]
    on_front = word != "higher"
    N["FRONTIER"] = ("The proposed policy thus lies on the efficient front traced by the tuned cyber-only baseline (Fig.~\\ref{fig:pareto}). Its advantage is not a large reduction of the degradation at equal spend, but that the operating point follows from the physical consequences without tuning $\\kappa$, and that protection is withdrawn in the absorbing state without being imposed."
                     if on_front else
                     "At this spend the proposed policy is above the frontier of the tuned cyber-only baseline (Fig.~\\ref{fig:pareto}); we therefore make no claim of superiority in this instance.")
    note = []
    if ss["low"]:
        note.append("In %d instance(s) (%s) the defender never invests in protection ($\\Pr(\\mathrm{Full}\\mid N)<0.05$): when the physical stakes shrink relative to the IDS cost, here through a shorter physical time horizon, the decision changes. This is the expected behaviour of a decision driven by physical consequences, but it shows that our conclusions depend on the physical time scale and on the exchange rate $\\Lambda$." % (len(ss["low"]), "; ".join(esc(a) + " " + esc(b_) for a, b_ in ss["low"])))
    if ss["failed"]:
        note.append("The instances %s violate the saddle-point condition (the attacker would have unbounded gain) and are not solvable by construction." % "; ".join(esc(a) + " " + esc(b_) for a, b_, _ in ss["failed"]))
    N["SENSNOTE"] = " ".join(note) if note else "All instances were solvable."
    write(os.path.join(TEX, "results_section.tex"), fill(RESULTS, N))
    write_numbers(N)


MODEL_PATCH = r"""% ====================================================================================================
% model_patch.tex  --  replacements for the model sections (all new text is wrapped in \revtext)
% ====================================================================================================

%% (1) Section 3.1, physical layer: replace the sentence "The system dynamics follow x(t)=f(...)" and Eq. (2) by
\revtext{The physical layer is a mode-dependent zero-sum LQ game on the 4-dimensional platoon error $x_k=[\Delta p_k,\Delta v_k,\Delta a_k,\Delta j_k]^\top$:
$x_{k+1}=A\,x_k+B(s_k)\,u_k+C\,v_k$, with the cost $J=\sum_{k=0}^{K}\big(x_k^\top Q(s_k)x_k+u_k^\top R_u(s_k)u_k-v_k^\top R_v v_k\big)$. The cyber state $s_k\in\{N,A,D,H\}$ is a Markov chain whose transition matrix $\lambda_{ij}(\sigma_1,\sigma_2)=\sum_{a_1,a_2}\sigma_1(i,a_1)\sigma_2(i,a_2)\mathbb P(j\mid i,a_1,a_2)$ is induced by the cyber strategies; this is the channel through which cyber decisions reach the plant. Availability is captured by the control authority $B(s)=g_sB$, $g=(1,0.9,0.7,0)$: an attack that drives the system to Degraded or Out of Service reduces what the controller can do.}

%% (2) Eq. (11)-(12) (the A(s) cases and "A_alert = 0.9 A_normal ...") are replaced by the common error chain A and by B(s) above.
%%     The Remark on availability must read: "Availability is captured by the mode-dependent control authority B(s)".

%% (3) Section 3.7 (Optimal physical cost): add after Eq. (13)
\revtext{The recursion is $V_k(x,i)=\min_u\max_v\{x^\top Q_ix+R_{u,i}u^2-R_{v,i}v^2+\sum_j\lambda_{ij}V_{k+1}(x',j)\}$. With $M_i=[B_i\;C]$, $Z_i=\sum_j\lambda_{ij}P_{k+1,j}$ and $H_i=\mathrm{diag}(R_{u,i},-R_{v,i})+M_i^\top Z_iM_i$, the saddle point exists if and only if $H_{i,11}>0$ and the Schur complement $H_{i,22}-H_{i,12}^2/H_{i,11}<0$ (the feasibility conditions of Theorem~\ref{theorem:feasibility}); the code checks both at every step and stops otherwise. $J^*(s)=x_0^\top P_{0,s}x_0$ therefore depends on the cyber strategies through $\lambda(\sigma_1,\sigma_2)$.}

%% (4) Replace Eq. (damage) and Eq. (cost):
\begin{equation}
\label{damage}
\revtext{d(s,a)=\Lambda\Big(1-\exp\Big(-\eta\Big[\textstyle\sum_{s'}\mathbb P_U(s'\mid s,a)\,J^*(s')-J^*(s)\Big]^+\Big)\Big)}
\end{equation}
\revtext{where $\mathbb P_U$ is the next-state distribution when attack $a$ is not detected. The damage is bounded by $\Lambda$, increasing in the physical-cost increase caused by the undetected attack, and equal to zero when the attack is detected. The payoff of the cyber game is}
\begin{equation}
\label{cost}
\revtext{\omega(s,F,a)=\mathcal C(F)+\mathbf 1[a\notin P_F]\,d(s,a)-c(a),}
\end{equation}
\revtext{where $c(a)\ge0$ is the attacker's effort, so that the attacker (maximiser) does not choose the most damaging attack for free. $\Lambda$ replaces the coefficient previously called $\alpha$ or $\eta$ in the damage; $\eta$ is its steepness.}

%% (5) Section 3.9 -- sign convention: the defender MINIMISES cost. Replace in Algorithm 1, the Q-update and W_1^*, sigma_1^* by
\revtext{$V_D(s)=\min_{\sigma_1}\max_{a_2}\sum_{a_1}\sigma_1(a_1)Q_D(s,a_1,a_2)$, \quad $\sigma_1^{*}(s)\in\arg\min_{\sigma_1}\max_{a_2}\sum_{a_1}\sigma_1(a_1)Q_D(s,a_1,a_2)$,}
\revtext{and $Q^A=-Q^D$. The update $Q_D(s,a_1,a_2)\leftarrow(1-\alpha)Q_D(s,a_1,a_2)+\alpha[\omega+\beta V_D(s')]$ uses the \emph{sampled next state} $s'$; the damage term is recomputed every 100 updates from the physical game solved with the current strategies.}

%% (6) Theorem 4: keep the Brouwer existence statement on the product of simplexes and add
\revtext{\emph{Remark.} Brouwer's theorem gives existence of a fixed point, not uniqueness or convergence of the iteration. In all experiments the damped iteration $\sigma^{(n+1)}=(1-\theta)\sigma^{(n)}+\theta\,T_S(\sigma^{(n)})$, $\theta=0.5$, converged to the tolerance $10^{-7}$.}

%% (7) Section 4.1, attack classes: replace the paragraph "The three attack classes map ..." by
\revtext{The three attack classes are interpreted as follows: Attack1 (low impact) as false-data injection on a single sensor, Attack2 (advanced) as command injection on an actuator, Attack3 (critical) as denial of service on the CAN/V2X bus. In the model these classes differ only through the detection sets $P_F$, the transition probabilities and the effort costs $c(a)$; the interpretation is therefore qualitative.}
"""

PATCHES = r"""% ====================================================================================================
% text_patches.tex -- abstract, introduction, contributions and conclusion (numbers filled from the results)
% ====================================================================================================

%% ABSTRACT (last two sentences; replace from "We instantiate ...")
\revtext{We instantiate the framework on a connected vehicle platoon CPS and evaluate the proposed policy and a formally specified cyber-only baseline in the same physical simulator. Against the worst-case attacker the proposed policy spends @@ECW@@ on average and limits the physical degradation to @@EXCW@@, against @@NONEW@@ without IDS and @@FULLW@@ with the strongest configuration always on; @@ABSCMP@@ The simulation uses an illustrative parameter set and a sensitivity study is reported.}

%% INTRODUCTION -- paragraph "The effectiveness of the proposed framework ..." replaced by
\revtext{The framework is evaluated on a connected vehicle platoon CPS. Both the proposed policy and a cyber-only baseline with a constant false-negative penalty are evaluated in the same physical game against the same attackers (Section~\ref{sec:baseline}). The proposed policy obtains @@SHAREW@@\% of the degradation reduction of an always-on strongest IDS for @@SPENDSHW@@\% of its spend, and @@INTROCMP@@}

%% CONTRIBUTIONS 3 and 4
\item \revtext{We propose a minimax Q-learning solution that approximates the cyber-layer equilibrium from sampled transitions, without prior knowledge of the cyber-layer transition probabilities or of the attacker's strategy. The physical plant matrices are assumed known when the LQ game is solved; the method is model-free with respect to the cyber adversary, not to the physical plant.}
\item \revtext{We instantiate the framework on a connected vehicle platoon CPS, define a cyber-only baseline, evaluate both policies in a common physical game, report the cyber-risk/physical-performance trade-off and a sensitivity study over @@SN@@ perturbed instances, and release code and seeds to reproduce every table and figure.}

%% CONCLUSION (replace the revtext paragraphs)
\revtext{Numerical results on a connected vehicle platoon indicate that scoring cyber decisions by the physical consequence of undetected attacks yields a policy on the efficient cost--degradation front of a tuned cyber-only baseline, without tuning its penalty, and that protection is withdrawn only where it cannot help. The advantage over a well-tuned cyber-only policy is small in this instance; we do not claim a large quantitative superiority.}

\revtext{Limitations. The parameters are illustrative and the model is simulated; no real attack traces, hardware-in-the-loop or practitioner validation were used. The attacker set used for evaluation is the set of stationary pure policies and the uniform attacker. The cyber payoff charges the increase of the physical cost, not its level. Confidentiality is not modelled. The convergence analysis is partial: existence of a fixed point is proved (Theorem~\ref{theo1}), convergence of the damped iteration is observed, and Q-learning converges under the standard conditions for the cyber layer only.}

\revtext{Generalisation. The formulation is generic, but only the platoon instance and a second detection structure were simulated. For an industrial process the state would be the process variable and $B(s)$ the actuation authority under command injection; for a smart grid the state would be the frequency deviation and the mode would capture the effect of false-data injection on state estimation. Only $A,B,C,Q,R_u$ and the cyber transition probabilities would need to be re-instantiated; this has not been validated here.}

%% DATA AVAILABILITY STATEMENT (replace the two sentences)
\revtext{The code, random seeds, parameter set and scripts that regenerate every table and figure of this paper are available at \url{https://doi.org/10.5281/zenodo.XXXXXXX} (to be replaced by the DOI of the deposit). No other data were used.}

%% ACKNOWLEDGMENTS -- remove any thanks addressed to the reviewers.
"""


def build_patches(ex, pa, ql, ss, N):
    gw, gu = pa["gap_w_off"], pa["gap_u_off"]
    if wording(gw, ex["metrics"]["exc_w"])[0] != "higher":
        N["ABSCMP"] = ("at the same spend it is comparable to the best cyber-only baseline tuned over its penalty (difference in $\\Delta_J$: @@GAPOFFW@@ against the worst-case attacker), "
                       "without needing that tuning.")
        N["INTROCMP"] = ("at the same spend it lies on the efficient front of the cyber-only baseline tuned over its penalty (Fig.~\\ref{fig:pareto}), without needing that tuning.")
    else:
        N["ABSCMP"] = "at the same spend it does not improve on the best cyber-only baseline tuned over its penalty."
        N["INTROCMP"] = "at the same spend it does not improve on the cyber-only baseline tuned over its penalty."
    N["ABSCMP"] = fill(N["ABSCMP"], N); N["INTROCMP"] = fill(N["INTROCMP"], N)
    write(os.path.join(TEX, "text_patches.tex"), fill(PATCHES, N))
    write(os.path.join(TEX, "model_patch.tex"), MODEL_PATCH)


# ------------------------------------------------------------------------------------------------ letter
def build_letter(N, ss):
    TODO = "\\textcolor{red}{[AUTHORS: %s]}"
    items = [
        ("Editor-in-Chief -- positioning within CPS",
         "The paper is now anchored in a concrete CPS: a connected vehicle platoon, with the chain cyber decision $\\to$ plant mode $\\to$ physical consequence $\\to$ cyber payoff made explicit (Section ``CPS Instantiation''). The physical layer is now genuinely coupled to the cyber layer through the mode process $\\lambda(\\sigma_1,\\sigma_2)$ (Section~3, subsections on the physical game and on the optimal physical cost)."),
        ("AE-1 / Reviewer 2 (1) -- LQ cost versus security objectives",
         "The subsection on the linear-quadratic cost now states what each term represents: integrity ($x^\\top Qx$), availability (control authority $B(s)$ degraded by the cyber mode), stabilisation effort ($u^\\top R_uu$), cyber cost ($\\mathcal C$). Confidentiality is not modelled and the paper says so. The previous claim that the IDS configuration enters $u_k$ was removed because the equations did not support it. The attack classes are interpreted qualitatively (Section ``CPS Instantiation'')."),
        ("Reviewer 1 (1) -- the system is not described",
         "The section ``CPS Instantiation'' and the provenance table give the plant (4th-order error chain, $A,B(s),C,Q(s),R_u(s),R_v,x_0$, sampling step, horizon), the cyber states and the attack interpretation. The model is a simplified error model of one follower, not a linearisation of a specific controller, and the paper says so."),
        ("Reviewer 1 (2) / AE-3 -- generalisation",
         "Claims of applicability were softened. A second detection structure (complementary libraries) is part of the sensitivity study, and the conclusion contains a short, explicitly unvalidated paragraph on industrial control systems and smart grids. Only the platoon instance was simulated."),
        ("Reviewer 1 (3) -- weak validation",
         "We agree. The study remains simulation-based with an illustrative parameter set; no real traces, testbed or practitioner input were used. This is stated as a limitation in the conclusion. We did not add validation that we could not perform."),
        ("Reviewer 1 (4) / Reviewer 2 (2) -- origin of the parameters and sensitivity",
         "The provenance table states the provenance of every parameter. Six rows of the transition table are the original ones; the remaining entries are our completion and are flagged as such. A sensitivity study over @@SN@@ perturbed instances (transition probabilities $\\pm20\\%$ and $\\pm50\\%$ entry by entry, costs, $Q$, $R_u$, $R_v$, horizon, sampling step, $\\beta$, $\\Lambda$, $\\eta$, attacker effort, detection structure) is in the subsection ``Sensitivity analysis'' (sensitivity table and figure). @@SNOK@@ instances satisfy the saddle-point conditions; among them the defender switches the IDS off in the absorbing state in @@SNOFF@@ and the proposed policy lies on or below the cyber-only frontier in @@SNFRONT@@."),
        ("Reviewer 1 (5) -- baseline and the 25\\% claim",
         "The 25\\% claim was removed. The baseline is now defined formally (Eq.~\\ref{eq:baseline_reward}: same states, actions, costs, hyper-parameters and attacker; only the reward differs) and both policies are evaluated in the same physical game against the same attackers (subsection ``Cyber-only baseline and common-simulator comparison'', comparison table and Pareto figure). The result is modest: at the same spend the proposed policy is comparable to the best cyber-only baseline tuned over its penalty (difference in $\\Delta_J$ @@GAPOFFW@@ against the worst-case attacker), without needing that tuning; it obtains @@SHAREW@@\\% of the degradation reduction of always-Full for @@SPENDSHW@@\\% of its spend. We state this limited advantage in the paper."),
        ("Reviewer 1 (6) / AE-2 -- reproducibility",
         "Code, seeds and parameters that regenerate every table and figure are provided (single script, fixed seeds) and will be deposited at a DOI. " + TODO % "insert the Zenodo DOI after the deposit"),
        ("Reviewer 1 (7) -- references",
         "Unresolved citations and mismatches were corrected and non-peer-reviewed sources replaced or removed. " + TODO % "list the exact changes after checking the .bib file"),
        ("Reviewer 1 (8) -- typography",
         "Merged words and missing spaces were corrected. " + TODO % "confirm on the final compiled PDF"),
        ("Reviewer 2 (3) -- trade-off between cyber-risk reduction and physical performance",
         "The subsection ``Trade-off between cyber risk and physical performance'' and the Pareto figure report the trade-off: sweeping the damage scale $\\Lambda$ moves the learned policy along a front from spend @@LAMECMIN@@ ($\\Delta_J=@@LAMEXMAX@@$) to spend @@LAMECMAX@@ ($\\Delta_J=@@LAMEXMIN@@$), to be compared with the $\\kappa$-sweep of the baseline."),
    ]
    extra = [
        "The previous implementation computed $J^*(s)$ per state independently of the cyber strategies, so the Stackelberg coupling described in the paper was not actually simulated. It now is (jump LQ game, $\\lambda(\\sigma_1,\\sigma_2)$).",
        "The earlier physical game violated the saddle-point conditions ($B=C$ and a small $R_v$). The conditions are now verified at every step.",
        "The earlier sensitivity perturbation of the transition probabilities was a no-op after row normalisation; it is now applied entry by entry.",
        "The earlier Q-learning routine did not sample transitions; it is replaced by minimax-Q on the cyber game and its convergence is reported as observed (learning-curve figure).",
        "The strategy and game-value tables and all figures were recomputed accordingly; the sign convention (defender minimises cost) is now consistent throughout.",
        "The control-authority description of availability replaces the former $A(s)=0.9A_{\\rm normal}$ scalings of the common error chain.",
    ]
    L = [r"\documentclass[11pt]{article}", r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage{amsmath,amssymb,xcolor}",
         r"\usepackage[margin=2.2cm]{geometry}", r"\newcommand{\revtext}[1]{#1}",
         r"\begin{document}", r"\begin{center}{\Large\bfseries Response to the Reviewers}\\[2pt] Manuscript CPS-2026-06-0061, IET Cyber-Physical Systems: Theory \& Applications\end{center}",
         r"\noindent Section numbers refer to the revised manuscript; " + TODO % "add page numbers after the final compilation, as requested by the Editor" + r"\par\medskip"]
    for t, r in items:
        L.append(r"\subsection*{%s}" % t); L.append(fill(r, N) + r"\par")
    L.append(r"\subsection*{Corrections beyond the reviewers' comments}")
    L.append(r"\noindent While preparing the revision we found and corrected the following problems in our own implementation:\begin{itemize}")
    L += [r"\item " + e for e in extra]
    L.append(r"\end{itemize}")
    L.append(r"\end{document}")
    s = "\n".join(L).replace("\\ref{", "\\ref{").replace("Eq.~\\ref{eq:baseline_reward}", "Eq.~(baseline)")
    import re
    s = s.replace('Eq.~\\ref{eq:baseline_reward}', 'the baseline reward equation')
    path = os.path.join(RESP, "response_letter.tex")
    write(path, s)
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "response_letter.tex"], cwd=RESP,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_title_page():
    try:
        from docx import Document
    except ImportError:
        return
    d = Document()
    d.add_heading("A Decision-Centric Defense Framework for Cyber-Physical Systems: Coupling Strategic Cyber Actions with Physical Consequences", 1)
    d.add_paragraph("Junior Mbogning Tchoffo (corresponding author), Paul Dayang, Arouna Ndam Njoya, Chassem Kamdem Priva, Hamit Djeroua Moura")
    d.add_heading("Affiliations", 2)
    for a in ("1 Faculty of Sciences, University of Ngaoundéré, P.O. Box 454, Ngaoundéré, Cameroon (J.M.T., P.D.)",
              "2 Institute of Technology, University of Ngaoundéré, P.O. Box 454, Ngaoundéré, Cameroon (A.N.N.)",
              "3 Faculty of Sciences, University of Dschang, P.O. Box 96, Dschang, Cameroon (C.K.P.)",
              "4 [DEPARTMENT TO COMPLETE], University of N'Djamena, P.O. Box 1117, N'Djamena, Chad (H.D.M.)"):
        d.add_paragraph(a)
    d.add_heading("Corresponding author", 2)
    d.add_paragraph("Junior Mbogning Tchoffo, juniormbogning@gmail.com; postal address: [TO COMPLETE]")
    d.add_heading("Conflict of interest", 2); d.add_paragraph("The authors declare no conflict of interest.")
    d.add_heading("Funding", 2); d.add_paragraph("This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors. (Must match ScholarOne.)")
    d.add_heading("Data availability statement", 2)
    d.add_paragraph("The code, random seeds, parameter set and scripts that regenerate every table and figure are available at https://doi.org/10.5281/zenodo.XXXXXXX [REPLACE WITH THE DOI]. No other data were used.")
    d.save(os.path.join(RESP, "title_page.docx"))


README = """# Revision package -- CPS-2026-06-0061

## Run
    pip install numpy scipy matplotlib python-docx
    python decision_iet_final.py          # experiments (about 15 min on 2 cores): results/*.json
    python build_paper.py                 # figures, LaTeX tables/text, response letter, title page
(`python build_paper.py --run` does both.)  Everything is deterministic (fixed seeds).

## Where each piece goes in the manuscript
| File (paper_out/) | Use |
|---|---|
| latex/model_patch.tex | replacement text/equations for Sections 3.x and 4.1 (copy block by block) |
| latex/results_section.tex | replaces `\\subsection{Simulation and Result Analysis}` through `Further Discussion` (`\\input{latex/results_section}`; it `\\input`s the tab_*.tex files; copy the folder next to main.tex) |
| latex/tab_provenance.tex | replaces the provenance table (and the old "Defender's cost" table can stay) |
| latex/text_patches.tex | abstract, introduction, contributions 3-4, conclusion, data-availability text |
| latex/numbers.tex | `\\num...` macros with every key number |
| figures/*.png, *.eps | replace im1, im2, im15, add fig_pareto, fig_sensitivity; remove im14 |
| response/response_letter.pdf | point-by-point response (red items must be completed by the authors) |
| response/title_page.docx | separate title page requested by the Editor |

## Things only the authors can do
Deposit the code at Zenodo and put the DOI in the manuscript and the title page; complete affiliation 4 and the postal address;
check the .bib file (replace arXiv/thesis entries, fix mismatched claims); proofread typography on the compiled PDF; add page numbers
to the response letter; produce the clean (non-highlighted) copy; enter CRediT and funding identically in ScholarOne.

## Model summary (what changed relative to the earlier script)
Jump LQ game with separate B and C and checked saddle-point conditions; cyber transition matrix induced by the strategies;
damage = Lambda*(1-exp(-eta*[increase of J*]^+)); attacker effort c(a); baseline = constant-kappa penalty;
all policies evaluated in the same game against the same 27 pure attackers + the uniform attacker; real minimax-Q.
"""


def main():
    if "--run" in sys.argv:
        for s in ("exact", "pareto", "sensitivity", "qlearn"):
            getattr(core, "stage_" + s)()
    style()
    ex, pa, se, ql = load_all()
    fig_equilibrium(ex); fig_beta(ex); fig_qlearning(ex, ql); fig_pareto(ex, pa, ql); fig_sensitivity(ex, se)
    tab_defender_attacker_values(ex); tab_comparison(ex, pa); tab_learned(ql); tab_provenance()
    ss = sens_summary(ex, se)
    N = numbers(ex, pa, ql, ss)
    build_text(ex, pa, ql, ss, N)
    build_patches(ex, pa, ql, ss, N)
    build_letter(N, ss)
    build_title_page()
    write(os.path.join(OUT, "README.md"), README)
    print("paper_out/ written.  key numbers:", {k: N[k] for k in ("ECW", "EXCW", "NONEW", "FULLW", "GAPOFFW", "GAPKEEPW", "KSTAR")})
    if ss["failed"]:
        print("unsolved sensitivity instances:", [(a, b) for a, b, _ in ss["failed"]])


if __name__ == "__main__":
    main()