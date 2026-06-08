# ─────────────────────────────────────────────────────────────────────────────
# ADD-ON: General M vs N Scaling — Publication Plots  (M = 1 … 40)
# Run this cell AFTER the main script in Colab.
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Gather full M = 1…40 dataset at T_LF_target = 12 min ────────────────────
scaling = general_MN_analysis(M_range=range(1, 41), T_LF_target=12)

M_vals   = np.array([r["M"]     for r in scaling])
N_vals   = np.array([r["N_min"] for r in scaling])
rho_vals = np.array([r["rho"]   if r["rho"]  is not None else np.nan for r in scaling])
wq_vals  = np.array([r["E_Wq"]  if r["E_Wq"] is not None else np.nan for r in scaling])
tat_vals = np.array([r["TAT"]   if r["TAT"]  is not None else np.nan for r in scaling])
e_vals   = np.array([r["E_kWh_T"] for r in scaling])
feas     = np.array([r["feasible"] for r in scaling])

# ── Colour scheme ─────────────────────────────────────────────────────────────
C_BLUE   = "#1B4F72"
C_ORANGE = "#D35400"
C_GREEN  = "#1E8449"
C_RED    = "#C0392B"
C_GREY   = "#707070"
C_LIGHT  = "#AED6F1"
C_TEAL   = "#148F77"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F7F9FC",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "grid.color"        : "white",
    "grid.linewidth"    : 0.9,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 8.5,
    "legend.framealpha" : 0.9,
})

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2 × 2 overview dashboard
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "General M + N Scaling Analysis  —  T_LF = 12 min  |  T_IF = 165 min  |  M = 1 … 40",
    fontsize=13, fontweight="bold", y=1.01
)

# ── Panel A: N_min required vs M (step chart) ─────────────────────────────────
ax = axes[0, 0]
# Colour bars by feasibility
bar_colours = [C_GREEN if f else C_RED for f in feas]
ax.bar(M_vals, N_vals, color=bar_colours, edgecolor="white", lw=0.6, zorder=3)

# Step overlay line
ax.step(M_vals, N_vals, where="mid", color=C_BLUE, lw=2.0, zorder=4,
        label="N_min (step)")

# Annotate N transitions
prev_N = None
for m, n in zip(M_vals, N_vals):
    if n != prev_N:
        ax.axvline(m - 0.5, color=C_GREY, lw=0.8, ls=":", alpha=0.7)
        ax.text(m + 0.3, n + 0.05, f"N={n}", fontsize=8,
                color=C_BLUE, fontweight="bold")
        prev_N = n

ax.set_xlabel("Number of Induction Furnaces  M")
ax.set_ylabel("Minimum LFs Required  N_min")
ax.set_title("A  —  Minimum LF Count  vs  Plant Size")
ax.set_xlim(0, 41); ax.set_ylim(0, N_vals.max() + 1)
ax.set_xticks(range(0, 41, 5))
ax.grid(axis="y", lw=0.6, zorder=0)
feas_patch   = mpatches.Patch(color=C_GREEN, label="Feasible (TAT ≤ 40 min)")
infeas_patch = mpatches.Patch(color=C_RED,   label="Infeasible (TAT > 40 min)")
ax.legend(handles=[feas_patch, infeas_patch], loc="upper left")

# ── Panel B: TAT vs M with 40-min limit ───────────────────────────────────────
ax = axes[0, 1]
ax.axhspan(40, tat_vals[np.isfinite(tat_vals)].max() + 2,
           alpha=0.08, color=C_RED, label="Infeasible zone")
ax.axhline(40, color=C_RED, lw=1.8, ls="--", zorder=4, label="40 min TAT limit")

# Plot feasible and infeasible separately
f_idx = feas.astype(bool)
ax.scatter(M_vals[f_idx],  tat_vals[f_idx],  color=C_GREEN,  s=55, zorder=5,
           label="Feasible")
ax.scatter(M_vals[~f_idx], tat_vals[~f_idx], color=C_RED,    s=55, zorder=5,
           marker="X", label="Infeasible")
ax.plot(M_vals, tat_vals, color=C_GREY, lw=1.2, alpha=0.5, zorder=3)

# Highlight the two study cases
for M_c, label_c, col_c in [(8, "Case 1\n8 IF+2LF", C_BLUE),
                              (4, "Case 2\n4 IF+1LF", C_ORANGE)]:
    idx = M_c - 1
    ax.scatter([M_vals[idx]], [tat_vals[idx]], color=col_c,
               s=120, zorder=6, edgecolors="white", lw=1.5)
    ax.annotate(label_c,
                xy=(M_vals[idx], tat_vals[idx]),
                xytext=(M_vals[idx] + 1.2, tat_vals[idx] + 0.5),
                fontsize=8.5, color=col_c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=col_c, lw=1.0))

ax.set_xlabel("Number of Induction Furnaces  M")
ax.set_ylabel("Total TAT  (min)")
ax.set_title("B  —  Ladle Turnaround Time  vs  Plant Size")
ax.set_xlim(0, 41)
ax.set_xticks(range(0, 41, 5))
ax.legend(loc="upper left"); ax.grid(lw=0.5)

# ── Panel C: Server utilisation ρ vs M ───────────────────────────────────────
ax = axes[1, 0]
ax.axhspan(0.40, 1.05, alpha=0.07, color=C_ORANGE, label="Advisory ρ > 0.40")
ax.axhspan(0.0,  0.40, alpha=0.05, color=C_GREEN,  label="Comfortable ρ < 0.40")
ax.axhline(0.40, color=C_ORANGE, lw=1.5, ls="--", zorder=4)
ax.axhline(1.00, color=C_RED,    lw=1.5, ls=":",  zorder=4, label="ρ = 1 (unstable)")

scatter = ax.scatter(M_vals, rho_vals, c=rho_vals, cmap="RdYlGn_r",
                     vmin=0, vmax=1, s=60, zorder=5, edgecolors="white", lw=0.5)
ax.plot(M_vals, rho_vals, color=C_GREY, lw=1.0, alpha=0.4, zorder=3)
plt.colorbar(scatter, ax=ax, label="ρ utilisation", pad=0.02)

for M_c, col_c in [(8, C_BLUE), (4, C_ORANGE)]:
    idx = M_c - 1
    ax.scatter([M_vals[idx]], [rho_vals[idx]], color=col_c,
               s=130, zorder=6, edgecolors="white", lw=1.5)

ax.set_xlabel("Number of Induction Furnaces  M")
ax.set_ylabel("Server Utilisation  ρ")
ax.set_title("C  —  LF Utilisation  vs  Plant Size")
ax.set_xlim(0, 41); ax.set_ylim(0, 1.05)
ax.set_xticks(range(0, 41, 5))
ax.legend(loc="upper left"); ax.grid(lw=0.5)

# ── Panel D: E[Wq] queue wait vs M ───────────────────────────────────────────
ax = axes[1, 1]
ax.axhspan(15, wq_vals[np.isfinite(wq_vals)].max() + 2,
           alpha=0.08, color=C_RED, label="Hold limit exceeded")
ax.axhline(15, color=C_RED, lw=1.8, ls="--", zorder=4, label="15 min hold limit")

ax.fill_between(M_vals, 0, np.where(np.isfinite(wq_vals), wq_vals, 0),
                alpha=0.18, color=C_TEAL)
ax.plot(M_vals, wq_vals, color=C_TEAL, lw=2.0, zorder=4, label="E[Wq] (min)")
ax.scatter(M_vals[f_idx],  wq_vals[f_idx],  color=C_GREEN, s=45, zorder=5)
ax.scatter(M_vals[~f_idx], wq_vals[~f_idx], color=C_RED,   s=45, zorder=5, marker="X")

for M_c, col_c in [(8, C_BLUE), (4, C_ORANGE)]:
    idx = M_c - 1
    ax.scatter([M_vals[idx]], [wq_vals[idx]], color=col_c,
               s=130, zorder=6, edgecolors="white", lw=1.5)

ax.set_xlabel("Number of Induction Furnaces  M")
ax.set_ylabel("Expected Queue Wait  E[Wq]  (min)")
ax.set_title("D  —  Queue Wait  vs  Plant Size")
ax.set_xlim(0, 41)
ax.set_xticks(range(0, 41, 5))
ax.legend(loc="upper left"); ax.grid(lw=0.5)

plt.tight_layout()
plt.savefig("plots/08_mn_scaling_dashboard.png", dpi=200, bbox_inches="tight")
plt.show()
print("  Saved: plots/08_mn_scaling_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — N vs M feasibility heatmap  (all M×N combinations)
# ═════════════════════════════════════════════════════════════════════════════

# Build a grid: M = 1…40, N = 1…8  — compute TAT for every cell
M_grid = np.arange(1, 41)
N_grid = np.arange(1, 9)

TAT_grid  = np.full((len(N_grid), len(M_grid)), np.nan)
RHO_grid  = np.full((len(N_grid), len(M_grid)), np.nan)

for j, M in enumerate(M_grid):
    for i, N in enumerate(N_grid):
        qs = queue_stats(M, N, 12)
        if qs["stable"]:
            E_Wq = qs["E_Wq_analytical"]
            prof = thermal_profile(12, min(E_Wq, 15))
            TAT  = PARAMS["T_tap_op"] + PARAMS["T_tr1"] + E_Wq + 12 + PARAMS["T_tr2"]
            TAT_grid[i, j]  = TAT
            RHO_grid[i, j]  = qs["rho"]

# Feasibility mask: TAT ≤ 40 AND ρ < 1
feasibility = np.where(TAT_grid <= 40, 1.0, 0.0)
feasibility[np.isnan(TAT_grid)] = np.nan

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
fig2.suptitle(
    "M × N Feasibility Space  (T_LF = 12 min)  —  Full Grid M = 1…40, N = 1…8",
    fontsize=13, fontweight="bold"
)

# ── Left: TAT heatmap ────────────────────────────────────────────────────────
ax = axes2[0]
masked_TAT = np.ma.masked_invalid(TAT_grid)
im = ax.pcolormesh(M_grid - 0.5, N_grid - 0.5, masked_TAT,
                   cmap="RdYlGn_r", vmin=28, vmax=55, shading="auto")
# 40-min contour
cs = ax.contour(M_grid, N_grid, TAT_grid,
                levels=[40], colors=["white"], linewidths=[2.5], linestyles=["--"])
ax.clabel(cs, fmt={40: "TAT = 40 min"}, fontsize=9, colors="white")

# Mark the two study cases
ax.scatter([8], [2], color="white", s=200, zorder=5, edgecolors=C_BLUE, lw=2.5)
ax.text(8.3, 2.0, "Case 1\n(8,2)", color=C_BLUE, fontsize=8.5, fontweight="bold", va="center")
ax.scatter([4], [1], color="white", s=200, zorder=5, edgecolors=C_ORANGE, lw=2.5)
ax.text(4.3, 1.0, "Case 2\n(4,1)", color=C_ORANGE, fontsize=8.5, fontweight="bold", va="center")

plt.colorbar(im, ax=ax, label="TAT  (min)", pad=0.02)
ax.set_xlabel("Number of Induction Furnaces  M")
ax.set_ylabel("Number of Ladle Furnaces  N")
ax.set_title("TAT Heatmap  (green = feasible ≤ 40 min)")
ax.set_xticks(range(0, 41, 5))
ax.set_yticks(N_grid)
ax.set_xlim(0.5, 40.5); ax.set_ylim(0.5, 8.5)

# ── Right: utilisation heatmap ───────────────────────────────────────────────
ax = axes2[1]
masked_RHO = np.ma.masked_invalid(RHO_grid)
im2 = ax.pcolormesh(M_grid - 0.5, N_grid - 0.5, masked_RHO,
                    cmap="RdYlGn_r", vmin=0, vmax=1.0, shading="auto")

# Advisory ρ = 0.40 contour
cs2 = ax.contour(M_grid, N_grid, RHO_grid,
                 levels=[0.40], colors=["white"], linewidths=[2.0], linestyles=["--"])
ax.clabel(cs2, fmt={0.40: "ρ = 0.40"}, fontsize=9, colors="white")

# ρ = 1 (instability boundary)
cs3 = ax.contour(M_grid, N_grid, RHO_grid,
                 levels=[1.0], colors=["black"], linewidths=[2.0], linestyles=[":"])

ax.scatter([8], [2], color="white", s=200, zorder=5, edgecolors=C_BLUE,   lw=2.5)
ax.scatter([4], [1], color="white", s=200, zorder=5, edgecolors=C_ORANGE, lw=2.5)

plt.colorbar(im2, ax=ax, label="Server Utilisation  ρ", pad=0.02)
ax.set_xlabel("Number of Induction Furnaces  M")
ax.set_ylabel("Number of Ladle Furnaces  N")
ax.set_title("LF Utilisation Heatmap  (green = ρ < 0.40 advisory)")
ax.set_xticks(range(0, 41, 5))
ax.set_yticks(N_grid)
ax.set_xlim(0.5, 40.5); ax.set_ylim(0.5, 8.5)

plt.tight_layout()
plt.savefig("plots/09_mn_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()
print("  Saved: plots/09_mn_heatmap.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — N_min step curve with efficiency bands (wide format)
# ═════════════════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(15, 5))

# Background bands per N tier
N_max_shown = int(N_vals.max()) + 1
band_colours = ["#EBF5FB", "#D5F5E3", "#FEF9E7", "#FDEDEC", "#F4ECF7",
                "#E8F8F5", "#FDF2E9", "#F2F3F4"]
n_breaks = [0] + [int(M_vals[N_vals == n].max()) for n in range(1, N_max_shown)
                  if np.any(N_vals == n)] + [40]

for n in range(1, N_max_shown):
    mask = N_vals == n
    if not np.any(mask):
        continue
    x_lo = M_vals[mask].min() - 0.5
    x_hi = M_vals[mask].max() + 0.5
    col  = band_colours[(n - 1) % len(band_colours)]
    ax.axvspan(x_lo, x_hi, alpha=0.45, color=col, zorder=1)
    ax.text((x_lo + x_hi) / 2, N_max_shown - 0.25,
            f"N = {n}", ha="center", fontsize=9.5,
            color=C_GREY, fontweight="bold")

# Feasibility shading on N step line
ax.fill_between(M_vals, 0, N_vals,
                where=feas,  alpha=0.30, color=C_GREEN,  step="mid",
                label="Feasible (TAT ≤ 40 min)")
ax.fill_between(M_vals, 0, N_vals,
                where=~feas, alpha=0.30, color=C_RED,    step="mid",
                label="Infeasible (TAT > 40 min)")

# Step line
ax.step(M_vals, N_vals, where="mid", color=C_BLUE, lw=3.0, zorder=5,
        label="N_min required")

# Dot per M, coloured by feasibility
ax.scatter(M_vals[f_idx],  N_vals[f_idx],  color=C_GREEN, s=55, zorder=6, edgecolors="white", lw=0.8)
ax.scatter(M_vals[~f_idx], N_vals[~f_idx], color=C_RED,   s=55, zorder=6, edgecolors="white", lw=0.8, marker="X")

# Highlight study cases
for M_c, lbl, col_c in [(8, "Case 1\n(8 IF+2LF)\nFeasible ✓", C_BLUE),
                         (4, "Case 2\n(4 IF+1LF)\nInfeasible ✗", C_ORANGE)]:
    idx = M_c - 1
    ax.scatter([M_vals[idx]], [N_vals[idx]], color=col_c, s=180, zorder=7,
               edgecolors="white", lw=2.0)
    ax.annotate(lbl, xy=(M_vals[idx], N_vals[idx]),
                xytext=(M_vals[idx] + 1.5, N_vals[idx] + 0.35),
                fontsize=9, color=col_c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=col_c, lw=1.3),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col_c, alpha=0.85))

ax.set_xlabel("Number of Induction Furnaces  M", fontsize=11)
ax.set_ylabel("Minimum LFs  N_min", fontsize=11)
ax.set_title(
    "LF Scaling Rule  —  N_min vs M  (T_LF = 12 min, T_IF = 165 min)\n"
    "Shaded bands = N tier  |  Green = feasible  |  Red = TAT > 40 min",
    fontsize=11, fontweight="bold"
)
ax.set_xlim(0, 41); ax.set_ylim(0, N_max_shown)
ax.set_xticks(range(0, 41, 2))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", lw=0.6, zorder=0)

plt.tight_layout()
plt.savefig("plots/10_mn_scaling_rule.png", dpi=200, bbox_inches="tight")
plt.show()
print("  Saved: plots/10_mn_scaling_rule.png")

print("\n  ✓  All 3 M-vs-N scaling figures generated.")