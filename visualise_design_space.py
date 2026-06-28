"""
Visualise media design-space from either:
1) New matrix workbook format (Main sheet; e.g. media_matrix_delft_24_3plate.xlsx), or
2) Legacy lhs_unique_conditions.csv format.

Examples:
    python visualise_design_space.py
    python visualise_design_space.py --matrix_xlsx media_matrix_delft_24_3plate.xlsx --save design_space.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Legacy defaults
LHS_CSV_DEFAULT = "lhs_unique_conditions.csv"
MATRIX_XLSX_DEFAULT = "media_matrix_delft_24_3plate.xlsx"

LHS_LO, LHS_HI = 0.5, 2.0
SUMMARY_COLUMNS = {"Compound", "ShorthandName", "PubChemCID"}

# Legacy CSV column mapping
DELFT_FINAL_LEGACY = {
    "KH2PO4": 13.68,
    "NH4_2SO4": 7.13,
    "MgSO4": 0.475,
    "Glucose": 20.00,
    "Trace metals (mult)": 1.0,
    "Vitamins (mult)": 1.0,
}

MULT_COLS_LEGACY = {
    "KH2PO4": "KH2PO4_conc_g_per_L",
    "NH4_2SO4": "NH4_2SO4_conc_g_per_L",
    "MgSO4": "MgSO4_conc_g_per_L",
    "Glucose": "Glucose_conc_g_per_L",
    "Trace metals (mult)": "trace_actual_mult",
    "Vitamins (mult)": "vitamin_actual_mult",
}


def _to_numeric_condition_columns(df: pd.DataFrame) -> list[str]:
    cond_cols = []
    for col in df.columns:
        if col in SUMMARY_COLUMNS:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            cond_cols.append(col)
    return cond_cols


def load_from_matrix_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Main")
    if "Compound" not in df.columns:
        raise ValueError("Main sheet must contain a 'Compound' column")

    cond_cols = _to_numeric_condition_columns(df)
    if len(cond_cols) < 2:
        raise ValueError("Could not identify condition columns in Main sheet")

    # Keep the first numeric condition as Delft control and the rest as sampled values.
    control_col = cond_cols[0]
    sampled_cols = cond_cols[1:]

    records = []
    for _, row in df.iterrows():
        compound = str(row["Compound"])
        delft = float(pd.to_numeric(row[control_col], errors="coerce"))
        values = pd.to_numeric(row[sampled_cols], errors="coerce").dropna().to_numpy(dtype=float)

        if values.size == 0 or np.isnan(delft):
            continue

        records.append(
            {
                "Compound": compound,
                "Delft": delft,
                "Min": values.min(),
                "Max": values.max(),
                "Median": np.median(values),
                "Mean": values.mean(),
                "Bound_Lo": LHS_LO * delft,
                "Bound_Hi": LHS_HI * delft,
                "All_values": values,
            }
        )

    if not records:
        raise ValueError("No valid compound rows found in Main sheet")

    return pd.DataFrame(records)


def load_from_legacy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in MULT_COLS_LEGACY.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Legacy CSV missing expected columns: {missing}")

    records = []
    for label, col in MULT_COLS_LEGACY.items():
        delft_val = DELFT_FINAL_LEGACY[label]
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        records.append(
            {
                "Compound": label,
                "Delft": delft_val,
                "Min": values.min(),
                "Max": values.max(),
                "Median": np.median(values),
                "Mean": values.mean(),
                "Bound_Lo": LHS_LO * delft_val,
                "Bound_Hi": LHS_HI * delft_val,
                "All_values": values,
            }
        )
    return pd.DataFrame(records)


def load_design_data(matrix_xlsx: str | None, lhs_csv: str | None) -> tuple[pd.DataFrame, str]:
    if matrix_xlsx and Path(matrix_xlsx).exists():
        return load_from_matrix_xlsx(matrix_xlsx), "matrix_xlsx"

    if lhs_csv and Path(lhs_csv).exists():
        return load_from_legacy_csv(lhs_csv), "legacy_csv"

    tried = [p for p in [matrix_xlsx, lhs_csv] if p]
    raise FileNotFoundError(f"No valid input file found. Tried: {tried}")


def load_bo_suggestions(path: str, df_design: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Load BO suggestions by matching columns to compound labels.
    If exact name does not exist, tries a normalized-name match.
    """
    df_bo = pd.read_csv(path)
    suggestions: dict[str, np.ndarray] = {}

    def normalize(s: str) -> str:
        return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    bo_norm_to_col = {normalize(c): c for c in df_bo.columns}
    for label in df_design["Compound"].tolist():
        if label in df_bo.columns:
            vals = pd.to_numeric(df_bo[label], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                suggestions[label] = vals
            continue

        norm = normalize(label)
        if norm in bo_norm_to_col:
            vals = pd.to_numeric(df_bo[bo_norm_to_col[norm]], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size:
                suggestions[label] = vals

    return suggestions


def transform_plot_space(df_design: pd.DataFrame, x_mode: str) -> tuple[pd.DataFrame, str, float]:
    """
    Transform plotting coordinates.
    - absolute: use g/L values directly
    - fold: normalize each compound by its control (Delft) value
    """
    if x_mode not in {"absolute", "fold"}:
        raise ValueError(f"Unsupported x_mode: {x_mode}")

    df_plot = df_design.copy(deep=True)
    if x_mode == "absolute":
        return df_plot, "Concentration (g/L)", 1.0

    for idx, row in df_plot.iterrows():
        delft = float(row["Delft"])
        df_plot.at[idx, "Delft"] = 1.0
        df_plot.at[idx, "Min"] = float(row["Min"]) / delft
        df_plot.at[idx, "Max"] = float(row["Max"]) / delft
        df_plot.at[idx, "Median"] = float(row["Median"]) / delft
        df_plot.at[idx, "Mean"] = float(row["Mean"]) / delft
        df_plot.at[idx, "Bound_Lo"] = float(row["Bound_Lo"]) / delft
        df_plot.at[idx, "Bound_Hi"] = float(row["Bound_Hi"]) / delft
        df_plot.at[idx, "All_values"] = np.asarray(row["All_values"], dtype=float) / delft

    return df_plot, "Fold change vs control (x)", 1.0


def resolve_axis_scale(df_plot: pd.DataFrame, axis_scale: str) -> str:
    if axis_scale in {"linear", "log"}:
        return axis_scale

    all_vals = []
    for row in df_plot.itertuples():
        vals = np.asarray(row.All_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size:
            all_vals.append(vals)

    if not all_vals:
        return "linear"

    combined = np.concatenate(all_vals)
    dynamic_range = combined.max() / combined.min()
    return "log" if dynamic_range >= 20 else "linear"


def plot_design_space(df_design: pd.DataFrame, bo_suggestions=None, save_path=None, x_mode="absolute", axis_scale="auto"):
    df_plot, x_label, _ = transform_plot_space(df_design, x_mode=x_mode)
    final_axis_scale = resolve_axis_scale(df_plot, axis_scale)

    n = len(df_design)
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, row in enumerate(df_plot.itertuples()):
        y = i

        ax.plot(
            [row.Bound_Lo, row.Bound_Hi],
            [y, y],
            color="skyblue",
            lw=8,
            alpha=0.35,
            label="Bounds (0.5x-2.0x)" if i == 0 else "",
        )
        ax.plot(
            [row.Min, row.Max],
            [y, y],
            color="steelblue",
            lw=2,
            label="Sampled range" if i == 0 else "",
        )

        jitter = np.random.uniform(-0.15, 0.15, size=len(row.All_values))
        ax.scatter(
            row.All_values,
            np.full(len(row.All_values), y) + jitter,
            color="steelblue",
            alpha=0.3,
            s=12,
            zorder=2,
            label="Samples" if i == 0 else "",
        )

        ax.axvline(x=row.Delft, color="grey", lw=0.5, linestyle=":", alpha=0.4)
        ax.plot(
            row.Delft,
            y,
            "D",
            color="black",
            markersize=7,
            zorder=4,
            label="Control" if i == 0 else "",
        )

        ax.plot(
            row.Median,
            y,
            "o",
            color="green",
            markersize=7,
            zorder=5,
            label="Median" if i == 0 else "",
        )
        ax.plot(
            row.Mean,
            y,
            "x",
            color="orange",
            markersize=8,
            markeredgewidth=2,
            zorder=5,
            label="Mean" if i == 0 else "",
        )

        if bo_suggestions and row.Compound in bo_suggestions:
            bo_vals = np.asarray(bo_suggestions[row.Compound], dtype=float)
            if x_mode == "fold":
                bo_vals = bo_vals / float(df_design.iloc[i]["Delft"])
            ax.scatter(
                bo_vals,
                np.full(len(bo_vals), y),
                color="red",
                marker="^",
                s=60,
                zorder=6,
                label="BO suggestion" if i == 0 else "",
            )

    if final_axis_scale == "log":
        ax.set_xscale("log")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df_plot["Compound"], fontsize=11)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(f"Media Optimisation - Design Space ({x_mode}, {final_axis_scale} x-axis)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise media design space")
    parser.add_argument(
        "--matrix_xlsx",
        default=MATRIX_XLSX_DEFAULT,
        help="Path to matrix workbook with Main sheet",
    )
    parser.add_argument(
        "--lhs_csv",
        default=LHS_CSV_DEFAULT,
        help="Path to legacy lhs_unique_conditions.csv",
    )
    parser.add_argument(
        "--bo_suggestions",
        default=None,
        help="Path to BO suggestions CSV (optional)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save figure to this path instead of showing",
    )
    parser.add_argument(
        "--x_mode",
        choices=["absolute", "fold"],
        default="fold",
        help="Plot x values in absolute concentration or fold-change vs control",
    )
    parser.add_argument(
        "--axis_scale",
        choices=["linear", "log", "auto"],
        default="auto",
        help="X-axis scaling. auto uses log when dynamic range is large.",
    )
    args = parser.parse_args()

    np.random.seed(42)
    df_design, source = load_design_data(args.matrix_xlsx, args.lhs_csv)
    print(f"Loaded design data from: {source}")

    bo_suggestions = None
    if args.bo_suggestions:
        bo_suggestions = load_bo_suggestions(args.bo_suggestions, df_design)
        print(f"Loaded BO suggestions from {args.bo_suggestions}")

    print("Sampling in generate_media_matrix.py uses linear-space LHS (qmc.scale on linear bounds).")
    plot_design_space(
        df_design,
        bo_suggestions=bo_suggestions,
        save_path=args.save,
        x_mode=args.x_mode,
        axis_scale=args.axis_scale,
    )