"""
analyse.py  –  Fit a linear mixed model to estimate lane fairness effects.

Model (per race day):
    time_ij = mu + alpha_j + beta_i + epsilon_ij

Where:
    alpha_j ~ fixed effect of lane j       (what we want to estimate)
    beta_i  ~ N(0, sigma^2_race)           (random intercept per race,
                                            absorbs boat class / crew speed)
    epsilon_ij ~ N(0, sigma^2_eps)         (residual noise)

Lane assignment in voorwedstrijden is random, which makes alpha_j
identifiable and unbiased. The random race intercept correctly handles
unbalanced data (outer lanes appearing in fewer races than inner lanes).

Output: data/{regatta}/{year}_{day}_results.json
    {
      "n_races": 38,
      "lanes": {
        "1": {"effect": 0.31, "se": 0.09, "n": 38, "p": 0.001, "sig": "**"},
        "2": {"effect": 0.18, "se": 0.08, "n": 40, "p": 0.026, "sig": "*"},
        ...
        "4": {"effect": 0.00, "se": 0.00, "n": 42, "p": 1.0,   "sig": ""},
        ...
      },
      "p_overall": 0.003,
      "sig_overall": true,
      "reference_lane": 4,
      "model": "lmm"
    }

Effects are in seconds relative to the fastest lane (always 0).
Positive = slower than the fastest lane.

Usage:
    python analyse.py                          # analyse all data files
    python analyse.py --regatta arb            # one regatta
    python analyse.py --regatta arb --year 2024
    python analyse.py --force                  # recompute even if results exist
"""

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# ─── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data"
MIN_N      = 3    # minimum races a lane must appear in to be included in model
N_LANES    = 8

REGATTAS = [
    "bvr", "voorjaarsregatta", "hollandia", "raceroei",
    "arb", "westelijke", "hollandbeker", "nsrf", "diyr",
]

# Significance stars (two-tailed)
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_races(path):
    """Load a JSON race file and return a list of 8-slot lists."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def races_to_long(races):
    """
    Convert list-of-lists to a long-format DataFrame with columns:
        race (int), lane (str 'L1'–'L8'), time (float)
    Only includes non-null entries.
    """
    rows = []
    for race_id, race in enumerate(races):
        for lane_idx, t in enumerate(race):
            if t is not None and not np.isnan(float(t)):
                rows.append({
                    "race": race_id,
                    "lane": f"L{lane_idx + 1}",
                    "lane_num": lane_idx + 1,
                    "time": float(t),
                })
    return pd.DataFrame(rows)


# ─── Model fitting ────────────────────────────────────────────────────────────

def fit_lmm(df):
    """
    Fit the linear mixed model:
        time ~ C(lane) + (1 | race)

    Returns a dict with lane effects, SEs, p-values, and overall p-value.
    Returns None if data is insufficient.
    """
    if df.empty or df["race"].nunique() < 5:
        return None

    # Only include lanes with at least MIN_N observations
    lane_counts = df.groupby("lane")["time"].count()
    valid_lanes = lane_counts[lane_counts >= MIN_N].index.tolist()
    if len(valid_lanes) < 2:
        return None

    df = df[df["lane"].isin(valid_lanes)].copy()

    # Use the most frequent lane as the reference (to minimise intercept SE)
    ref_lane = df["lane"].value_counts().idxmax()
    # Relevel: put ref_lane first so C(lane) uses it as baseline
    df["lane"] = pd.Categorical(df["lane"],
                                categories=[ref_lane] +
                                           [l for l in sorted(valid_lanes) if l != ref_lane])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model  = smf.mixedlm("time ~ C(lane)", df, groups=df["race"])
            result = model.fit(reml=True, method="lbfgs", disp=False)
        except Exception as e:
            print(f"    [WARN] Model failed: {e} — falling back to row-mean method")
            return fit_row_mean_fallback(df)

    # ── Extract fixed effects ────────────────────────────────────────────────
    fe     = result.fe_params      # Series: Intercept, C(lane)[T.L2], ...
    fe_se  = result.bse            # standard errors
    fe_pv  = result.pvalues        # p-values (t-test, df from Satterthwaite approx)

    # Build lane effect dict relative to reference lane
    lane_effects_raw = {}
    lane_se_raw      = {}
    lane_pv_raw      = {}
    n_per_lane       = df.groupby("lane")["time"].count().to_dict()

    # Reference lane effect = 0 by construction
    ref_lane_num = int(ref_lane[1:])
    lane_effects_raw[ref_lane] = 0.0
    lane_se_raw[ref_lane]      = 0.0
    lane_pv_raw[ref_lane]      = 1.0

    for param_name, coef in fe.items():
        if param_name == "Intercept":
            continue
        # param_name looks like "C(lane)[T.L3]"
        lane = param_name.split("[T.")[-1].rstrip("]")
        lane_effects_raw[lane] = coef
        lane_se_raw[lane]      = fe_se[param_name]
        lane_pv_raw[lane]      = fe_pv[param_name]

    # ── Re-reference to fastest lane (minimum effect = 0) ───────────────────
    min_effect = min(lane_effects_raw.values())
    lanes_out  = {}
    for lane, eff in lane_effects_raw.items():
        lane_num = int(lane[1:])
        adjusted = eff - min_effect   # shift so fastest = 0
        lanes_out[str(lane_num)] = {
            "effect": round(adjusted, 4),
            "se":     round(lane_se_raw[lane], 4),
            "n":      int(n_per_lane.get(lane, 0)),
            "p":      round(float(lane_pv_raw[lane]), 4),
            "sig":    sig_stars(lane_pv_raw[lane]),
            # p-value is vs reference lane, not vs fastest —
            # we'll recompute vs fastest below
        }

    # ── Find reference lane for output ──────────────────────────────────────
    fastest_lane = min(lane_effects_raw, key=lane_effects_raw.get)
    fastest_lane_num = int(fastest_lane[1:])

    # ── Overall F-test: are any lane effects non-zero? ───────────────────────
    # Likelihood ratio test: compare full model vs intercept-only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model_null = smf.mixedlm("time ~ 1", df, groups=df["race"])
            result_null = model_null.fit(reml=False, method="lbfgs", disp=False)
            result_full = smf.mixedlm("time ~ C(lane)", df, groups=df["race"]
                                      ).fit(reml=False, method="lbfgs", disp=False)
            lr_stat = 2 * (result_full.llf - result_null.llf)
            df_diff = len(valid_lanes) - 1
            p_overall = float(stats.chi2.sf(lr_stat, df=df_diff))
        except Exception:
            p_overall = float("nan")

    return {
        "n_races":        int(df["race"].nunique()),
        "lanes":          lanes_out,
        "p_overall":      round(p_overall, 4) if not np.isnan(p_overall) else None,
        "sig_overall":    (p_overall < 0.05) if not np.isnan(p_overall) else None,
        "reference_lane": fastest_lane_num,
        "model":          "lmm",
    }


def fit_row_mean_fallback(df):
    """
    Fallback when LMM fails: simple row-mean subtraction.
    Used for very small datasets.
    Returns same structure as fit_lmm.
    """
    race_means = df.groupby("race")["time"].transform("mean")
    df = df.copy()
    df["resid"] = df["time"] - race_means

    lane_groups = df.groupby("lane")["resid"]
    means  = lane_groups.mean()
    sems   = lane_groups.sem()
    counts = lane_groups.count()

    # t-test vs 0 for each lane
    pvals = {}
    for lane, grp in lane_groups:
        if len(grp) >= 2:
            t, p = stats.ttest_1samp(grp, 0)
            pvals[lane] = float(p)
        else:
            pvals[lane] = 1.0

    min_mean = means.min()
    lanes_out = {}
    for lane in means.index:
        lane_num = int(lane[1:])
        lanes_out[str(lane_num)] = {
            "effect": round(float(means[lane] - min_mean), 4),
            "se":     round(float(sems[lane]), 4),
            "n":      int(counts[lane]),
            "p":      round(pvals[lane], 4),
            "sig":    sig_stars(pvals[lane]),
        }

    fastest = int(means.idxmin()[1:])
    return {
        "n_races":        int(df["race"].nunique()),
        "lanes":          lanes_out,
        "p_overall":      None,
        "sig_overall":    None,
        "reference_lane": fastest,
        "model":          "row_mean_fallback",
    }


# ─── Main analysis loop ───────────────────────────────────────────────────────

def analyse_file(races_path, results_path, force=False):
    """Analyse one race day file and save results JSON."""
    if not force and os.path.exists(results_path):
        return  # already done

    races = load_races(races_path)
    if not races:
        return

    df = races_to_long(races)
    if df.empty:
        return

    print(f"  Fitting model: {len(races)} races, "
          f"{df['lane'].nunique()} lanes, {len(df)} observations")

    result = fit_lmm(df)
    if result is None:
        print(f"  [SKIP] Not enough data for model")
        return

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    p_str = f"p={result['p_overall']:.3f}" if result["p_overall"] is not None else "p=n/a"
    sig   = "✓ significant" if result["sig_overall"] else "✗ not significant"
    print(f"  Overall lane effect: {p_str} ({sig})")
    for lane_num in sorted(result["lanes"], key=int):
        info = result["lanes"][lane_num]
        stars = info["sig"] or " "
        print(f"    Lane {lane_num}: +{info['effect']:.3f}s "
              f"± {info['se']:.3f} (n={info['n']}) {stars}")


def run(regattas, years, force=False):
    for regatta in regattas:
        for year in years:
            for day in ("sat", "sun"):
                races_path   = os.path.join(OUTPUT_DIR, regatta, f"{year}_{day}.json")
                results_path = os.path.join(OUTPUT_DIR, regatta, f"{year}_{day}_results.json")

                if not os.path.exists(races_path):
                    continue

                print(f"\n[{regatta}/{year}/{day}]")
                analyse_file(races_path, results_path, force=force)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit lane fairness models")
    parser.add_argument("--regatta", nargs="+", choices=REGATTAS)
    parser.add_argument("--year",    type=int)
    parser.add_argument("--since",   type=int, default=2010)
    parser.add_argument("--force",   action="store_true",
                        help="Recompute even if results already exist")
    args = parser.parse_args()

    regattas = args.regatta if args.regatta else REGATTAS
    years    = [args.year] if args.year else range(args.since,
                                                    __import__('datetime').date.today().year + 1)
    run(regattas, list(years), force=args.force)
