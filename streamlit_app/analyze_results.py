#!/usr/bin/env python3
"""
DIAGONAL Human Evaluation — Reproducible Analysis Script
=========================================================
Computes all statistics reported in Section 5.4 of the paper:
  - Per-model win rates and tie rate
  - Per-model mean Likert scores (Narrative Compliance)
  - Spearman rho: PTM ranking vs human Likert ranking
  - Fleiss' kappa: inter-annotator agreement on Likert ratings

Usage:
    python analyze_results.py [--results PATH] [--eval PATH]
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats as sp_stats
from pathlib import Path

MODELS = ["storydiff", "echoshot", "vgot", "vic"]
MLABEL = {"storydiff": "StoryDiffusion", "echoshot": "EchoShot",
          "vgot": "VGoT", "vic": "VIC"}


def fleiss_kappa(mat):
    """Fleiss' kappa from (n_items × n_categories) count matrix."""
    n_items, n_cats = mat.shape
    n_raters = int(mat.sum(axis=1)[0])
    if n_raters < 2:
        return float("nan")
    p_j = mat.sum(axis=0) / (n_items * n_raters)
    Pi = ((mat ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = Pi.mean()
    Pe = (p_j ** 2).sum()
    if abs(1 - Pe) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0
    return float((P_bar - Pe) / (1 - Pe))


def analyze(results_path, eval_path):
    with open(results_path) as f:
        results = json.load(f)
    rdf = pd.DataFrame(results)

    print("=" * 60)
    print("DIAGONAL Human Evaluation Analysis")
    print("=" * 60)
    print(f"Total evaluations: {len(rdf)}")
    print(f"Unique evaluators: {rdf['evaluator'].nunique()}")
    print(f"Unique scenarios:  {rdf['vid'].nunique()}")
    print()

    # ── Win rates ──
    print("── Model Win Rates ──")
    wins = defaultdict(lambda: {"W": 0, "L": 0, "T": 0})
    for _, r in rdf.iterrows():
        ma, mb = r["model_a"], r["model_b"]
        if r["preference"] == "A":
            wins[ma]["W"] += 1; wins[mb]["L"] += 1
        elif r["preference"] == "B":
            wins[mb]["W"] += 1; wins[ma]["L"] += 1
        else:
            wins[ma]["T"] += 1; wins[mb]["T"] += 1

    for m in MODELS:
        total = wins[m]["W"] + wins[m]["L"] + wins[m]["T"]
        wr = 100 * wins[m]["W"] / max(total, 1)
        print(f"  {MLABEL[m]:18s}  W={wins[m]['W']:3d}  L={wins[m]['L']:3d}  "
              f"T={wins[m]['T']:3d}  WinRate={wr:.1f}%")

    total_comps = len(rdf)
    total_ties = (rdf["preference"] == "Tie").sum()
    tie_rate = 100 * total_ties / max(total_comps, 1)
    print(f"\n  Global Tie Rate: {tie_rate:.1f}%")
    print()

    # ── Likert scores ──
    if "likert_a" in rdf.columns:
        print("── Likert Scores (Narrative Compliance, 1-5) ──")
        likert = defaultdict(list)
        for _, r in rdf.iterrows():
            likert[r["model_a"]].append(r["likert_a"])
            likert[r["model_b"]].append(r["likert_b"])

        human_means = {}
        for m in MODELS:
            mu = np.mean(likert[m]) if likert[m] else float("nan")
            sd = np.std(likert[m]) if likert[m] else float("nan")
            human_means[m] = mu
            print(f"  {MLABEL[m]:18s}  Mean={mu:.2f}  Std={sd:.2f}  N={len(likert[m])}")
        print()

        # ── Spearman ρ: PTM vs Human Likert ──
        if eval_path and Path(eval_path).exists():
            print("── Spearman Correlation (PTM ↔ Human Likert) ──")
            with open(eval_path) as f:
                edata = json.load(f)
            edf = pd.DataFrame(edata)
            ptm_means = edf.groupby("method")["match_prescribed"].mean().reindex(MODELS)
            human_arr = np.array([human_means[m] for m in MODELS])
            ptm_arr = ptm_means.values

            rho, pval = sp_stats.spearmanr(ptm_arr, human_arr)
            print(f"  Spearman ρ = {rho:.4f}  (p = {pval:.4f})")
            print(f"  PTM  ranking: {list(ptm_arr.round(3))}")
            print(f"  Human ranking: {list(np.round(human_arr, 2))}")
            print()

        # ── Fleiss' κ ──
        n_evaluators = rdf["evaluator"].nunique()
        if n_evaluators >= 2:
            print("── Fleiss' Kappa (Inter-Annotator Agreement) ──")
            item_ratings = defaultdict(list)
            for _, r in rdf.iterrows():
                item_ratings[(r["vid"], r["model_a"], r["model_b"], "A")].append(r["likert_a"])
                item_ratings[(r["vid"], r["model_a"], r["model_b"], "B")].append(r["likert_b"])

            rater_counts = Counter(len(v) for v in item_ratings.values())
            target_n = rater_counts.most_common(1)[0][0]
            filtered = {k: v for k, v in item_ratings.items() if len(v) == target_n}

            if target_n >= 2 and filtered:
                n_items = len(filtered)
                mat = np.zeros((n_items, 5), dtype=int)
                for i, (k, ratings) in enumerate(filtered.items()):
                    for r in ratings:
                        mat[i, r - 1] += 1
                kappa = fleiss_kappa(mat)
                print(f"  Fleiss' κ = {kappa:.3f}")
                print(f"  Items = {n_items}, Raters per item = {target_n}")
                if kappa > 0.8:
                    print("  Interpretation: Almost perfect agreement")
                elif kappa > 0.6:
                    print("  Interpretation: Substantial agreement")
                elif kappa > 0.4:
                    print("  Interpretation: Moderate agreement")
                else:
                    print("  Interpretation: Fair or poor agreement")
            else:
                print(f"  Insufficient overlap (max {target_n} raters per item)")
        else:
            print("── Fleiss' κ: requires ≥2 evaluators ──")

    print()
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIAGONAL human eval analysis")
    parser.add_argument("--results", default="data/human_eval_results.json",
                        help="Path to human_eval_results.json")
    parser.add_argument("--eval", default="data/vlm_fullscale_merged.json",
                        help="Path to vlm_fullscale_merged.json (for PTM correlation)")
    args = parser.parse_args()
    analyze(args.results, args.eval)
