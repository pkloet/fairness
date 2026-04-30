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
    Fit the linear mixed model with sum-to-zero (deviation) coding:
        time ~ C(lane, Sum) + (1 | race)

    Sum-to-zero coding means every lane gets its own coefficient and SE —
    no lane is silently chosen as a zero-SE reference. The constraint is
    sum(alpha_j) = 0, so each alpha_j is that lane's deviation from the
    grand mean. We then shift all effects so the fastest lane = 0 for
    display, but every lane — including the fastest — retains its honest SE.

    The underlying model fit, likelihood, and overall p-value are identical
    to treatment coding; only the parameterisation changes.
    """
    if df.empty or df["race"].nunique() < 5:
        return None

    # Only include lanes with at least MIN_N observations
    lane_counts = df.groupby("lane")["time"].count()
    valid_lanes = sorted(lane_counts[lane_counts >= MIN_N].index.tolist())
    if len(valid_lanes) < 2:
        return None

    df = df[df["lane"].isin(valid_lanes)].copy()
    n_per_lane = df.groupby("lane")["time"].count().to_dict()

    # ── Build sum-to-zero contrast matrix manually ───────────────────────────
    # For k lanes, create k-1 orthogonal sum-to-zero contrasts.
    # Each column c_j satisfies sum(c_j) = 0 and c_j[j] = 1, c_j[k] = -1.
    # This is the standard Helmert/sum coding approach.
    # We add these as numeric columns so statsmodels sees plain regression.
    k = len(valid_lanes)

    # Contrast matrix: shape (k, k-1)
    # Column j: +1 for lane j, -1 for last lane, 0 elsewhere
    contrasts = np.zeros((k, k - 1))
    for j in range(k - 1):
        contrasts[j, j]  =  1.0
        contrasts[k-1, j] = -1.0

    lane_to_idx = {lane: i for i, lane in enumerate(valid_lanes)}

    # Add contrast columns to df
    for j in range(k - 1):
        col = f"z{j}"
        df[col] = df["lane"].map(lambda l: contrasts[lane_to_idx[l], j])

    formula = "time ~ " + " + ".join(f"z{j}" for j in range(k - 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model  = smf.mixedlm(formula, df, groups=df["race"])
            result = model.fit(reml=True, method="lbfgs", disp=False)
        except Exception as e:
            print(f"    [WARN] Model failed: {e} — falling back to row-mean method")
            return fit_row_mean_fallback(df)

    fe    = result.fe_params   # Intercept, z0, z1, ...
    fe_se = result.bse
    mu    = fe["Intercept"]    # grand mean

    # ── Recover per-lane effects: alpha = C @ beta_contrasts ─────────────────
    beta = np.array([fe.get(f"z{j}", 0.0) for j in range(k - 1)])
    V_beta = result.cov_params().loc[
        [f"z{j}" for j in range(k - 1)],
        [f"z{j}" for j in range(k - 1)]
    ].values  # (k-1) x (k-1) covariance matrix

    # Lane effects (deviations from grand mean)
    alpha     = contrasts @ beta                          # shape (k,)
    var_alpha = np.array([contrasts[i] @ V_beta @ contrasts[i] for i in range(k)])
    se_alpha  = np.sqrt(np.maximum(var_alpha, 0))

    # ── Shift so fastest lane = 0 ────────────────────────────────────────────
    min_alpha        = alpha.min()
    fastest_idx      = int(np.argmin(alpha))
    fastest_lane     = valid_lanes[fastest_idx]
    fastest_lane_num = int(fastest_lane[1:])

    # Full covariance matrix of lane effects: V_alpha = C @ V_beta @ C^T
    V_alpha = contrasts @ V_beta @ contrasts.T   # shape (k, k)

    # ── Per-lane p-value vs fastest lane ────────────────────────────────────
    # For each lane j, test H0: alpha_j - alpha_fastest = 0
    # Var(alpha_j - alpha_f) = Var(alpha_j) + Var(alpha_f) - 2*Cov(alpha_j, alpha_f)
    # The fastest lane itself gets p=1 by definition.
    f = fastest_idx
    pvals = np.ones(k)
    for i in range(k):
        if i == f:
            continue
        diff    = alpha[i] - alpha[f]
        var_diff = V_alpha[i, i] + V_alpha[f, f] - 2 * V_alpha[i, f]
        if var_diff > 0:
            t_stat  = diff / np.sqrt(var_diff)
            pvals[i] = float(2 * stats.norm.sf(abs(t_stat)))
        else:
            pvals[i] = 1.0

    lanes_out = {}
    for i, lane in enumerate(valid_lanes):
        lane_num = int(lane[1:])
        lanes_out[str(lane_num)] = {
            "effect": round(float(alpha[i] - min_alpha), 4),
            "se":     round(float(np.sqrt(V_alpha[i, i])), 4),
            "n":      int(n_per_lane.get(lane, 0)),
            "p":      round(float(pvals[i]), 4),
            "sig":    sig_stars(float(pvals[i])),
        }

    # ── Overall likelihood ratio test ────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result_null = smf.mixedlm("time ~ 1", df, groups=df["race"]
                                      ).fit(reml=False, method="lbfgs", disp=False)
            result_full = smf.mixedlm(formula, df, groups=df["race"]
                                      ).fit(reml=False, method="lbfgs", disp=False)
            lr_stat   = 2 * (result_full.llf - result_null.llf)
            p_overall = float(stats.chi2.sf(lr_stat, df=k - 1))
        except Exception:
            p_overall = float("nan")

    return {
        "n_races":        int(df["race"].nunique()),
        "lanes":          lanes_out,
        "p_overall":      round(p_overall, 4) if not np.isnan(p_overall) else None,
        "sig_overall":    (p_overall < 0.05) if not np.isnan(p_overall) else None,
        "reference_lane": fastest_lane_num,
        "model":          "lmm_sum_coding",
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
