"""
LHS matrix generator for Delft-centred media optimisation
Y. lipolytica growth rate experiment

Design:
- 6 variables: 4 continuous macronutrients (LHS), 2 discrete stock multipliers (snapped)
- 64 unique conditions, triplicates = 192 wells across 2x 96-well plates
- 300 µL media volume per well + 50 µL inoculum = 350 µL total
- Assembly order: macronutrients → pH → trace metals → vitamins → MilliQ top-up → inoculum

NOTE: Stock concentrations below are placeholders.
      Validate against solubility limits and volume budget before use.
      Run this script to check volume feasibility before committing to stocks.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ── Configuration ─────────────────────────────────────────────────────────────

MEDIA_VOL_UL = 300          # µL media per well (inoculum added separately)
N_UNIQUE     = 64           # unique LHS conditions
N_REPLICATES = 3
SEED         = 42
PIPETTE_FLOOR_UL = 1.0      # µL, FlowBot ONE minimum reliable volume

# Delft final concentrations (g/L in media)
DELFT_FINAL = {
    "KH2PO4":   13.68,
    "NH4_2SO4":  7.13,
    "MgSO4":     0.475,
    "Glucose":  20.00,
}

# LHS bounds as multipliers of Delft (0.5x to 2.0x)
LHS_BOUNDS = (0.5, 2.0)

# Stock concentrations (g/L) — ADJUST BASED ON SOLUBILITY AND VOLUME BUDGET
# Validate using check_volume_budget() below before finalising
STOCK_CONC = {
    "KH2PO4":   80.0,    # ~solubility limit at RT; confirm with wetlab
    "NH4_2SO4": 107.0,   # well within solubility (~706 g/L)
    "MgSO4":    7.125,   # well within solubility
    "Glucose":  300.0,   # well within solubility (~909 g/L)
}

# Trace metal and vitamin stock: prepared at 1/10th Delft recipe concentration
# giving 6 µL addition at 1x in 300 µL well
TRACE_METAL_1X_VOL_UL = 6.0   # µL at 1x Delft in 300 µL well
VITAMIN_1X_VOL_UL      = 6.0   # µL at 1x Delft in 300 µL well

# Discrete levels for trace metals and vitamins
DISCRETE_LEVELS = np.array([0.5, 1.0, 2.0])


# ── Volume budget check ────────────────────────────────────────────────────────

def check_volume_budget(stock_conc=STOCK_CONC, verbose=True):
    """
    Validates that all addition volumes are:
    - Above the pipetting floor at 0.5x
    - Non-negative MilliQ top-up at 2x (worst case all variables at max)
    """
    results = {}
    for compound, final_g_per_l in DELFT_FINAL.items():
        mass_ug = final_g_per_l * MEDIA_VOL_UL          # µg in one well at 1x
        vol_1x  = mass_ug / stock_conc[compound]         # µL at 1x
        vol_05x = vol_1x * 0.5
        vol_2x  = vol_1x * 2.0
        results[compound] = {"0.5x": vol_05x, "1x": vol_1x, "2x": vol_2x}

    trace_2x   = TRACE_METAL_1X_VOL_UL * 2.0
    vitamin_2x = VITAMIN_1X_VOL_UL * 2.0

    total_2x = sum(r["2x"] for r in results.values()) + trace_2x + vitamin_2x
    milliQ_at_2x = MEDIA_VOL_UL - total_2x

    total_1x = sum(r["1x"] for r in results.values()) + TRACE_METAL_1X_VOL_UL + VITAMIN_1X_VOL_UL
    milliQ_at_1x = MEDIA_VOL_UL - total_1x

    if verbose:
        print("=" * 60)
        print("VOLUME BUDGET CHECK")
        print("=" * 60)
        print(f"{'Component':<20} {'0.5x (µL)':>10} {'1x (µL)':>10} {'2x (µL)':>10}")
        print("-" * 55)
        for compound, vols in results.items():
            flag_05x = " ⚠ BELOW FLOOR" if vols["0.5x"] < PIPETTE_FLOOR_UL else ""
            print(f"{compound:<20} {vols['0.5x']:>10.2f} {vols['1x']:>10.2f} {vols['2x']:>10.2f}{flag_05x}")
        print(f"{'Trace metals':<20} {TRACE_METAL_1X_VOL_UL*0.5:>10.2f} {TRACE_METAL_1X_VOL_UL:>10.2f} {trace_2x:>10.2f}")
        print(f"{'Vitamins':<20} {VITAMIN_1X_VOL_UL*0.5:>10.2f} {VITAMIN_1X_VOL_UL:>10.2f} {vitamin_2x:>10.2f}")
        print("-" * 55)
        print(f"{'MilliQ top-up':<20} {'':>10} {milliQ_at_1x:>10.2f} {milliQ_at_2x:>10.2f}")
        print()
        if milliQ_at_2x < 0:
            print("❌ FAIL: MilliQ volume negative at 2x — stock concentrations too low")
        else:
            print(f"✓  MilliQ headroom at 2x (worst case): {milliQ_at_2x:.1f} µL")
        below_floor = [c for c, v in results.items() if v["0.5x"] < PIPETTE_FLOOR_UL]
        if below_floor:
            print(f"❌ FAIL: Below pipetting floor at 0.5x: {below_floor}")
        else:
            print("✓  All 0.5x additions above pipetting floor")
        print("=" * 60)

    return results, milliQ_at_2x


# ── LHS generation ─────────────────────────────────────────────────────────────

def snap_to_levels(values, levels=DISCRETE_LEVELS):
    """Snap continuous LHS values to nearest discrete level."""
    levels = np.array(levels)
    return levels[np.argmin(np.abs(values[:, None] - levels[None, :]), axis=1)]


def generate_lhs_matrix(stock_conc=STOCK_CONC, verbose=True):
    """
    Generate LHS design matrix.
    Returns DataFrame with:
    - multiplier columns (dimensionless, relative to Delft 1x)
    - addition volume columns (µL per 300 µL well)
    - MilliQ top-up volume
    - plate and well assignments
    """
    sampler = qmc.LatinHypercube(d=6, seed=SEED)
    sample  = sampler.random(n=N_UNIQUE)

    # Scale to 0.5x–2.0x bounds
    lo, hi = LHS_BOUNDS
    scaled = qmc.scale(sample, l_bounds=[lo]*6, u_bounds=[hi]*6)

    # Columns: KH2PO4, NH4_2SO4, MgSO4, Glucose (continuous), trace, vitamin (discrete)
    macros  = scaled[:, :4]
    trace   = snap_to_levels(scaled[:, 4])
    vitamin = snap_to_levels(scaled[:, 5])

    compounds = list(DELFT_FINAL.keys())

    # Compute addition volumes (µL)
    vol_records = []
    for i in range(N_UNIQUE):
        row = {}
        total_vol = 0.0
        for j, compound in enumerate(compounds):
            multiplier    = macros[i, j]
            final_g_per_l = DELFT_FINAL[compound]
            mass_ug       = final_g_per_l * MEDIA_VOL_UL * multiplier
            vol_ul_raw    = mass_ug / stock_conc[compound]

            # Round to nearest pipettable unit (1 µL floor)
            vol_ul_rounded = max(PIPETTE_FLOOR_UL, round(vol_ul_raw))

            # Back-calculate actual delivered concentration from rounded volume
            actual_conc = (vol_ul_rounded * stock_conc[compound]) / MEDIA_VOL_UL

            row[f"{compound}_lhs_mult"]     = round(multiplier, 4)   # traceability only
            row[f"{compound}_vol_uL"]       = vol_ul_rounded          # what gets pipetted
            row[f"{compound}_conc_g_per_L"] = round(actual_conc, 4)  # BO input
            total_vol += vol_ul_rounded

        row["trace_mult"]   = trace[i]
        row["vitamin_mult"] = vitamin[i]
        trace_vol   = max(PIPETTE_FLOOR_UL, round(TRACE_METAL_1X_VOL_UL * trace[i]))
        vitamin_vol = max(PIPETTE_FLOOR_UL, round(VITAMIN_1X_VOL_UL * vitamin[i]))

        # Back-calculate actual multiplier delivered after rounding
        row["trace_vol_uL"]        = trace_vol
        row["trace_actual_mult"]   = round(trace_vol / TRACE_METAL_1X_VOL_UL, 4)
        row["vitamin_vol_uL"]      = vitamin_vol
        row["vitamin_actual_mult"] = round(vitamin_vol / VITAMIN_1X_VOL_UL, 4)
        total_vol += trace_vol + vitamin_vol

        # MilliQ rounded to 1 µL as well
        milliQ = round(MEDIA_VOL_UL - total_vol)
        row["milliQ_vol_uL"]  = milliQ
        row["total_media_uL"] = MEDIA_VOL_UL
        vol_records.append(row)

    df = pd.DataFrame(vol_records)
    df.insert(0, "condition_id", range(1, N_UNIQUE + 1))

    # Validate no negative MilliQ
    neg_milliQ = df[df["milliQ_vol_uL"] < 0]
    if not neg_milliQ.empty:
        print(f"❌ WARNING: {len(neg_milliQ)} conditions have negative MilliQ volume.")
        print("   Increase stock concentrations or reduce upper bound.")

    below_floor = {}
    for compound in compounds:
        col = f"{compound}_vol_uL"
        bad = df[df[col] < PIPETTE_FLOOR_UL]
        if not bad.empty:
            below_floor[compound] = len(bad)
    if below_floor:
        print(f"❌ WARNING: Conditions below pipetting floor: {below_floor}")

    return df


def assign_plates_and_wells(df, n_replicates=N_REPLICATES):
    """
    Expand unique conditions to triplicates and assign plate/well positions.
    Randomises well order to avoid positional bias.
    """
    expanded = pd.concat([df] * n_replicates, ignore_index=True)
    expanded["replicate"] = np.tile(np.arange(1, n_replicates + 1), len(df))
    expanded = expanded.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Assign plate and well
    rows_96 = list("ABCDEFGH")
    cols_96 = list(range(1, 13))
    wells = [f"{r}{c}" for r in rows_96 for c in cols_96]

    expanded["plate"] = [(i // 96) + 1 for i in range(len(expanded))]
    expanded["well"]  = [wells[i % 96] for i in range(len(expanded))]

    return expanded


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nStep 1: Volume budget validation")
    _, milliQ_headroom = check_volume_budget()

    if milliQ_headroom < 0:
        print("\nHalt: Fix stock concentrations before generating matrix.")
    else:
        print("\nStep 2: Generating LHS matrix")
        df_unique = generate_lhs_matrix()
        print(f"Generated {len(df_unique)} unique conditions")

        print("\nStep 3: Assigning plates and wells")
        df_full = assign_plates_and_wells(df_unique)
        print(f"Total wells: {len(df_full)} ({N_UNIQUE} conditions × {N_REPLICATES} replicates)")
        print(f"Plates required: {df_full['plate'].max()}")

        # Save outputs
        df_unique.to_csv("lhs_unique_conditions.csv", index=False)
        df_full.to_csv("lhs_full_plate_layout.csv", index=False)
        print("\nSaved:")
        print("  lhs_unique_conditions.csv  — 64 unique conditions with volumes")
        print("  lhs_full_plate_layout.csv  — 192 wells with plate/well assignments")

        print("\nSample (first 5 unique conditions):")
        preview_cols = ["condition_id",
                        "KH2PO4_conc_g_per_L", "NH4_2SO4_conc_g_per_L", "MgSO4_conc_g_per_L", "Glucose_conc_g_per_L",
                        "trace_actual_mult", "vitamin_actual_mult", "milliQ_vol_uL"]
        print(df_unique[preview_cols].head().to_string(index=False))