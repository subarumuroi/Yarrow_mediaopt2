"""
Visualisation of LHS design space for Delft-centred media optimisation.
Mirrors the compound range plot style from the original media matrix.
Supports overlaying BO suggestions for subsequent iterations.

Usage:
    # Initial LHS visualisation
    python visualise_design_space.py

    # After BO suggests next points, pass the suggestions CSV:
    python visualise_design_space.py --bo_suggestions bo_round1.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────

LHS_CSV = "lhs_unique_conditions.csv"

# Delft 1x final concentrations (g/L in final well)
DELFT_FINAL = {
    "KH2PO4":    13.68,
    "NH4_2SO4":   7.13,
    "MgSO4":      0.475,
    "Glucose":   20.00,
    "Trace metals (mult)": 1.0,
    "Vitamins (mult)":     1.0,
}

# Actual delivered value columns in the CSV (post-rounding back-calculation)
MULT_COLS = {
    "KH2PO4":              "KH2PO4_conc_g_per_L",
    "NH4_2SO4":            "NH4_2SO4_conc_g_per_L",
    "MgSO4":               "MgSO4_conc_g_per_L",
    "Glucose":             "Glucose_conc_g_per_L",
    "Trace metals (mult)": "trace_actual_mult",
    "Vitamins (mult)":     "vitamin_actual_mult",
}

# Bounds
LHS_LO, LHS_HI = 0.5, 2.0


# ── Data preparation ───────────────────────────────────────────────────────────

def load_lhs(path=LHS_CSV):
    df = pd.read_csv(path)
    records = []
    for label, col in MULT_COLS.items():
        delft_val = DELFT_FINAL[label]
        # Columns are already in absolute units (g/L or multiplier) — no conversion needed
        values = df[col].values
        records.append({
            "Compound":      label,
            "Delft":         delft_val,
            "Min":           values.min(),
            "Max":           values.max(),
            "Median":        np.median(values),
            "Mean":          values.mean(),
            "Bound_Lo":      LHS_LO * delft_val,
            "Bound_Hi":      LHS_HI * delft_val,
            "All_values":    values,
        })
    return pd.DataFrame(records)


def load_bo_suggestions(path, df_design):
    """
    Load BO-suggested points. Expects same column names as lhs_unique_conditions.csv
    (conc_g_per_L for macronutrients, actual_mult for stocks).
    """
    df_bo = pd.read_csv(path)
    suggestions = {}
    for label, col in MULT_COLS.items():
        if col in df_bo.columns:
            suggestions[label] = df_bo[col].values
    return suggestions


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_design_space(df_design, bo_suggestions=None, save_path=None):
    n = len(df_design)
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, row in enumerate(df_design.itertuples()):
        y = i

        # ── Prescribed bounds (0.5x–2.0x) ──
        ax.plot(
            [row.Bound_Lo, row.Bound_Hi], [y, y],
            color="skyblue", lw=8, alpha=0.35,
            label="LHS bounds (0.5x–2.0x)" if i == 0 else ""
        )

        # ── Actual LHS sampled range ──
        ax.plot(
            [row.Min, row.Max], [y, y],
            color="steelblue", lw=2,
            label="Sampled range (LHS)" if i == 0 else ""
        )

        # ── Individual LHS points (jittered for visibility) ──
        jitter = np.random.uniform(-0.15, 0.15, size=len(row.All_values))
        ax.scatter(
            row.All_values, np.full(len(row.All_values), y) + jitter,
            color="steelblue", alpha=0.3, s=12, zorder=2,
            label="LHS samples" if i == 0 else ""
        )

        # ── Delft 1x reference ──
        ax.axvline(x=row.Delft, color="grey", lw=0.5, linestyle=":", alpha=0.4)
        ax.plot(
            row.Delft, y, "D",
            color="black", markersize=7, zorder=4,
            label="Delft 1x" if i == 0 else ""
        )

        # ── Median and mean ──
        ax.plot(
            row.Median, y, "o",
            color="green", markersize=7, zorder=5,
            label="Median" if i == 0 else ""
        )
        ax.plot(
            row.Mean, y, "x",
            color="orange", markersize=8, markeredgewidth=2, zorder=5,
            label="Mean" if i == 0 else ""
        )

        # ── BO suggestions overlay ──
        if bo_suggestions and row.Compound in bo_suggestions:
            bo_vals = bo_suggestions[row.Compound]
            ax.scatter(
                bo_vals, np.full(len(bo_vals), y),
                color="red", marker="^", s=60, zorder=6,
                label="BO suggestion" if i == 0 else ""
            )

    # ── Axes ──
    ax.set_yticks(range(n))
    ax.set_yticklabels(df_design["Compound"], fontsize=11)
    ax.set_xlabel("Concentration (g/L) or multiplier", fontsize=11)
    ax.set_title(
        "Delft-Centred Media Optimisation — LHS Design Space\n"
        "Y. lipolytica growth rate, n=64 conditions",
        fontsize=12
    )
    ax.set_xscale("linear")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ── Discrete level markers for trace/vitamin ──
    discrete_labels = ["Trace metals (mult)", "Vitamins (mult)"]
    for label in discrete_labels:
        idx = df_design[df_design["Compound"] == label].index[0]
        delft = DELFT_FINAL[label]
        for level in [0.5, 1.0, 2.0]:
            ax.axvline(
                x=level * delft, ymin=(idx - 0.4) / n, ymax=(idx + 0.4) / n,
                color="purple", lw=1.5, alpha=0.5, linestyle="--"
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise LHS design space")
    parser.add_argument("--lhs_csv",       default=LHS_CSV,  help="Path to LHS unique conditions CSV")
    parser.add_argument("--bo_suggestions", default=None,     help="Path to BO suggestions CSV (optional)")
    parser.add_argument("--save",           default=None,     help="Save figure to this path instead of showing")
    args = parser.parse_args()

    np.random.seed(42)
    df_design = load_lhs(args.lhs_csv)

    bo_suggestions = None
    if args.bo_suggestions:
        bo_suggestions = load_bo_suggestions(args.bo_suggestions, df_design)
        print(f"Loaded BO suggestions from {args.bo_suggestions}")

    plot_design_space(df_design, bo_suggestions=bo_suggestions, save_path=args.save)