"""
LHS matrix generator for Delft-centred media optimisation
Y. lipolytica growth rate experiment

Design:
- 6 variables: 4 continuous macronutrients (LHS), 2 discrete stock multipliers (snapped)
- 64 unique conditions, triplicates = 192 wells across 2x 96-well plates
- 300 µL media volume per well + 50 µL inoculum = 350 µL total
- Assembly order: macronutrients → pH → trace metals → vitamins → MilliQ top-up → inoculum

NOTE ON pH: pH adjustment strategy across 192 wells is unresolved pending wetlab input.
KH2PO4 concentration varies across wells, so KOH requirement differs per well.
Per-well titration at plate scale is impractical. Options: pre-adjust stock pH,
or accept ~±0.2 pH variation relying on KH2PO4 buffering capacity.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ── Configuration ─────────────────────────────────────────────────────────────

MEDIA_VOL_UL     = 300      # µL media per well (inoculum added separately)
N_UNIQUE         = 64       # unique LHS conditions
N_REPLICATES     = 3
SEED             = 42
PIPETTE_FLOOR_UL = 1.0      # µL, FlowBot ONE minimum reliable volume

# Delft final concentrations (g/L in media)
DELFT_FINAL = {
    "KH2PO4":   13.68,
    "NH4_2SO4":  7.13,
    "MgSO4":     0.475,
    "Glucose":  20.00,
}

# Per-variable bounds (g/L in final well)
# Where bounds deviate from simple 0.5x-2.0x Delft, reason is noted.
BOUNDS = {
    "KH2PO4": (
        1.00,   # Below 0.5x Delft (6.84). Literature min is 1.0 g/L across 11 Yarrowia studies.
        27.36,  # 2x Delft. Literature max is 15.0 g/L so upper bound extends beyond literature.
    ),
    "NH4_2SO4": (
        3.56,   # 0.5x Delft.
        14.25,  # 2x Delft. Literature max is 8.0 g/L so upper bound extends beyond literature.
    ),
    "MgSO4": (
        0.238,  # 0.5x Delft.
        2.50,   # Above 2x Delft (0.950). Literature max is 2.5 g/L across 11 Yarrowia studies.
    ),
    "Glucose": (
        10.00,  # 0.5x Delft.
        40.00,  # 2x Delft. Literature goes to 80 g/L but 40 g/L chosen as biological ceiling
                # to avoid overflow metabolism.
    ),
}

# Stock concentrations (g/L)
# Chosen so addition volumes are comfortably above the 1 µL FlowBot floor
# across the full range defined in BOUNDS above.
STOCK_CONC = {
    "KH2PO4":   80.0,   # Solubility ~83 g/L at RT — confirm with wetlab before preparation
    "NH4_2SO4": 107.0,  # Well within solubility (~706 g/L)
    "MgSO4":    25.0,   # Increased from Delft stock (7.125 g/L) to accommodate upper bound of 2.5 g/L
                        # while keeping addition volumes pipettable (2.86-30 µL range)
    "Glucose":  300.0,  # Well within solubility (~909 g/L)
}

# Trace metal and vitamin stocks prepared at 1/10th Delft recipe concentration
# so that the 1x addition volume is 6 µL (pipettable), giving a 0.5x-2x range of 3-12 µL
TRACE_METAL_1X_VOL_UL = 6.0
VITAMIN_1X_VOL_UL      = 6.0

# Discrete levels for trace metals and vitamins
DISCRETE_LEVELS = np.array([0.5, 1.0, 2.0])


# ── Volume budget check ────────────────────────────────────────────────────────

def check_volume_budget(verbose=True):
    """
    Validates that all addition volumes are above the pipetting floor
    at the lower bound, and MilliQ top-up is non-negative at the upper bound
    (worst case: all variables at max simultaneously).
    """
    results = {}
    for compound, final_g_per_l in DELFT_FINAL.items():
        lo, hi = BOUNDS[compound]
        vol_lo = (lo * MEDIA_VOL_UL) / STOCK_CONC[compound]
        vol_1x = (final_g_per_l * MEDIA_VOL_UL) / STOCK_CONC[compound]
        vol_hi = (hi * MEDIA_VOL_UL) / STOCK_CONC[compound]
        results[compound] = {"lo": vol_lo, "1x": vol_1x, "hi": vol_hi,
                             "bound_lo": lo, "bound_hi": hi}

    trace_hi   = TRACE_METAL_1X_VOL_UL * 2.0
    vitamin_hi = VITAMIN_1X_VOL_UL * 2.0
    total_hi   = sum(r["hi"] for r in results.values()) + trace_hi + vitamin_hi
    milliQ_hi  = MEDIA_VOL_UL - total_hi

    total_1x  = sum(r["1x"] for r in results.values()) + TRACE_METAL_1X_VOL_UL + VITAMIN_1X_VOL_UL
    milliQ_1x = MEDIA_VOL_UL - total_1x

    if verbose:
        print("=" * 72)
        print("VOLUME BUDGET CHECK")
        print("=" * 72)
        print(f"{'Component':<20} {'Bound lo':>10} {'Vol lo (µL)':>12} {'Vol 1x (µL)':>12} {'Vol hi (µL)':>12} {'Bound hi':>10}")
        print("-" * 72)
        for compound, r in results.items():
            flag = " ⚠ BELOW FLOOR" if r["lo"] < PIPETTE_FLOOR_UL else ""
            print(f"{compound:<20} {r['bound_lo']:>10.3f} {r['lo']:>12.2f} {r['1x']:>12.2f} {r['hi']:>12.2f} {r['bound_hi']:>10.3f}{flag}")
        print(f"{'Trace metals':<20} {'0.5x':>10} {TRACE_METAL_1X_VOL_UL*0.5:>12.2f} {TRACE_METAL_1X_VOL_UL:>12.2f} {trace_hi:>12.2f} {'2x':>10}")
        print(f"{'Vitamins':<20} {'0.5x':>10} {VITAMIN_1X_VOL_UL*0.5:>12.2f} {VITAMIN_1X_VOL_UL:>12.2f} {vitamin_hi:>12.2f} {'2x':>10}")
        print("-" * 72)
        print(f"{'MilliQ top-up':<20} {'':>10} {'':>12} {milliQ_1x:>12.2f} {milliQ_hi:>12.2f}")
        print()
        if milliQ_hi < 0:
            print(f"❌ FAIL: MilliQ volume negative at upper bounds ({milliQ_hi:.1f} µL) — increase stock concentrations")
        else:
            print(f"✓  MilliQ headroom at upper bounds (worst case): {milliQ_hi:.1f} µL")
        below = [c for c, r in results.items() if r["lo"] < PIPETTE_FLOOR_UL]
        if below:
            print(f"❌ FAIL: Below pipetting floor at lower bound: {below}")
        else:
            print("✓  All lower bound additions above pipetting floor")
        print("=" * 72)

    return results, milliQ_hi


# ── LHS generation ─────────────────────────────────────────────────────────────

def snap_to_levels(values, levels=DISCRETE_LEVELS):
    levels = np.array(levels)
    return levels[np.argmin(np.abs(values[:, None] - levels[None, :]), axis=1)]


def generate_lhs_matrix(verbose=True):
    """
    Generate LHS design matrix using per-variable bounds from BOUNDS dict.
    Returns DataFrame with:
    - LHS multiplier columns (traceability)
    - addition volume columns (µL per well, rounded to nearest 1 µL)
    - back-calculated actual concentration (BO input)
    - MilliQ top-up volume
    """
    compounds = list(DELFT_FINAL.keys())
    lo_bounds = [BOUNDS[c][0] for c in compounds] + [0.5, 0.5]  # stocks use multiplier bounds
    hi_bounds = [BOUNDS[c][1] for c in compounds] + [2.0, 2.0]

    sampler = qmc.LatinHypercube(d=6, seed=SEED)
    sample  = sampler.random(n=N_UNIQUE)
    scaled  = qmc.scale(sample, l_bounds=lo_bounds, u_bounds=hi_bounds)

    macros       = scaled[:, :4]   # g/L directly (bounds are in g/L)
    trace_mults  = snap_to_levels(scaled[:, 4])
    vitamin_mults = snap_to_levels(scaled[:, 5])

    vol_records = []
    for i in range(N_UNIQUE):
        row = {}
        total_vol = 0.0
        for j, compound in enumerate(compounds):
            target_conc   = macros[i, j]           # g/L in final well
            vol_ul_raw    = (target_conc * MEDIA_VOL_UL) / STOCK_CONC[compound]
            vol_ul_rounded = max(PIPETTE_FLOOR_UL, round(vol_ul_raw))
            actual_conc   = (vol_ul_rounded * STOCK_CONC[compound]) / MEDIA_VOL_UL

            row[f"{compound}_target_g_per_L"]  = round(target_conc, 4)   # LHS suggestion
            row[f"{compound}_vol_uL"]          = vol_ul_rounded           # what gets pipetted
            row[f"{compound}_conc_g_per_L"]    = round(actual_conc, 4)   # BO input
            total_vol += vol_ul_rounded

        row["trace_mult"]        = trace_mults[i]
        row["vitamin_mult"]      = vitamin_mults[i]
        trace_vol   = max(PIPETTE_FLOOR_UL, round(TRACE_METAL_1X_VOL_UL * trace_mults[i]))
        vitamin_vol = max(PIPETTE_FLOOR_UL, round(VITAMIN_1X_VOL_UL * vitamin_mults[i]))
        row["trace_vol_uL"]      = trace_vol
        row["trace_actual_mult"] = round(trace_vol / TRACE_METAL_1X_VOL_UL, 4)
        row["vitamin_vol_uL"]      = vitamin_vol
        row["vitamin_actual_mult"] = round(vitamin_vol / VITAMIN_1X_VOL_UL, 4)
        total_vol += trace_vol + vitamin_vol

        row["milliQ_vol_uL"]  = round(MEDIA_VOL_UL - total_vol)
        row["total_media_uL"] = MEDIA_VOL_UL
        vol_records.append(row)

    df = pd.DataFrame(vol_records)
    df.insert(0, "condition_id", range(1, N_UNIQUE + 1))

    neg = df[df["milliQ_vol_uL"] < 0]
    if not neg.empty:
        print(f"❌ WARNING: {len(neg)} conditions have negative MilliQ volume.")

    return df


def assign_plates_and_wells(df, n_replicates=N_REPLICATES):
    expanded = pd.concat([df] * n_replicates, ignore_index=True)
    expanded["replicate"] = np.tile(np.arange(1, n_replicates + 1), len(df))
    expanded = expanded.sample(frac=1, random_state=SEED).reset_index(drop=True)

    rows_96 = list("ABCDEFGH")
    cols_96 = list(range(1, 13))
    wells   = [f"{r}{c}" for r in rows_96 for c in cols_96]

    expanded["plate"] = [(i // 96) + 1 for i in range(len(expanded))]
    expanded["well"]  = [wells[i % 96] for i in range(len(expanded))]
    return expanded


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nStep 1: Volume budget validation")
    _, milliQ_headroom = check_volume_budget()

    if milliQ_headroom < 0:
        print("\nHalt: Fix stock concentrations or bounds before generating matrix.")
    else:
        print("\nStep 2: Generating LHS matrix")
        df_unique = generate_lhs_matrix()

        # Check uniqueness after rounding
        conc_cols = [f"{c}_conc_g_per_L" for c in DELFT_FINAL] + ["trace_actual_mult", "vitamin_actual_mult"]
        n_unique_actual = df_unique[conc_cols].drop_duplicates().shape[0]
        print(f"Generated {len(df_unique)} conditions, {n_unique_actual} unique after rounding")
        for col in conc_cols:
            vals = df_unique[col].values
            print(f"  {col}: {len(set(vals))} unique values, range {vals.min():.3f}–{vals.max():.3f}")

        print("\nStep 3: Assigning plates and wells")
        df_full = assign_plates_and_wells(df_unique)
        print(f"Total wells: {len(df_full)} ({N_UNIQUE} conditions × {N_REPLICATES} replicates)")
        print(f"Plates required: {df_full['plate'].max()}")

        df_unique.to_csv("lhs_unique_conditions.csv", index=False)
        df_full.to_csv("lhs_full_plate_layout.csv", index=False)
        print("\nSaved: lhs_unique_conditions.csv, lhs_full_plate_layout.csv")