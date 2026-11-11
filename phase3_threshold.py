import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from common import (RESULTS_DIR, SEEDS, COST_RATIOS, RECALL_MIN, PRECISION_MIN, URL_FEATURES,
                    load_clean, raw_scores, apply_platt, label_metrics, cost_metrics,
                    search_threshold, summarize)

pd.set_option("display.width", 200)
preds = pd.read_csv(f"{RESULTS_DIR}/phase2_predictions.csv")
preds = preds[preds["model"] != "lsd_hard"]
models = sorted(preds["model"].unique())

print("PHASE 3: COST BASED THRESHOLD SELECTION")
print(f"Cost of a false alarm C_FP = 1, cost of a missed phishing site C_FN = ratio")
print(f"Ratios: {COST_RATIOS}, recall floor: {RECALL_MIN}, precision floor: {PRECISION_MIN}\n")

rows = []
for seed in SEEDS:
    for name in models:
        sub = preds[(preds["seed"] == seed) & (preds["model"] == name)]
        val, test = sub[sub["split"] == "val"], sub[sub["split"] == "test"]
        y_va, p_va = val["y"].values, val["p_cal"].values
        y_te, p_te = test["y"].values, test["p_cal"].values
        for ratio in COST_RATIOS:
            policies = {
                "default_0.5": 0.5,
                "analytic_tau_star": 1 / (1 + ratio),
                "searched_cost_only": search_threshold(y_va, p_va, ratio),
                "searched_with_recall_floor": search_threshold(y_va, p_va, ratio, recall_min=RECALL_MIN),
                "searched_with_recall_and_precision_floor": search_threshold(y_va, p_va, ratio, recall_min=RECALL_MIN, precision_min=PRECISION_MIN),
            }
            for policy, tau in policies.items():
                row = {"seed": seed, "model": name, "cost_ratio": ratio, "policy": policy}
                if tau is None:
                    row.update({"tau": np.nan, "feasible": 0})
                else:
                    pred = (p_te >= tau).astype(int)
                    row.update({"tau": tau, "feasible": 1})
                    row.update(label_metrics(y_te, pred))
                    row.update(cost_metrics(y_te, pred, ratio))
                rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(f"{RESULTS_DIR}/phase3_thresholds_per_seed.csv", index=False)

feas = res.groupby(["model", "cost_ratio", "policy"])["feasible"].sum().reset_index(name="feasible_seeds_out_of_5")
ok = res[res["feasible"] == 1]
cols = ["tau", "cost_per_1000", "missed_phishing_per_1000", "precision", "recall", "fpr", "accuracy"]
summary = summarize(ok, ["model", "cost_ratio", "policy"], cols)
summary = feas.merge(summary, on=["model", "cost_ratio", "policy"], how="left")
summary.to_csv(f"{RESULTS_DIR}/phase3_summary.csv", index=False)

infeasible = feas[feas["feasible_seeds_out_of_5"] < 5]
infeasible.to_csv(f"{RESULTS_DIR}/phase3_infeasible.csv", index=False)

show_cols = ["model", "policy", "feasible_seeds_out_of_5", "tau_mean", "cost_per_1000_mean", "cost_per_1000_std",
             "missed_phishing_per_1000_mean", "precision_mean", "recall_mean", "fpr_mean"]
for ratio in COST_RATIOS:
    print(f"=== Cost ratio C_FN/C_FP = {ratio} (analytic tau* = {1 / (1 + ratio):.3f}) ===")
    part = summary[summary["cost_ratio"] == ratio].sort_values(["policy", "cost_per_1000_mean"])
    print(part[show_cols].round(4).to_string(index=False))
    print()

acc = pd.read_csv(f"{RESULTS_DIR}/phase2_summary.csv")[["model", "accuracy_mean"]]
rank_rows = []
for ratio in COST_RATIOS:
    c = summary[(summary["cost_ratio"] == ratio) & (summary["policy"] == "searched_with_recall_floor")][["model", "cost_per_1000_mean"]]
    m = acc.merge(c, on="model")
    m["rank_by_accuracy"] = m["accuracy_mean"].rank(ascending=False).astype(int)
    m["rank_by_cost"] = m["cost_per_1000_mean"].rank(ascending=True, method="min")
    m["cost_ratio"] = ratio
    rank_rows.append(m)
ranks = pd.concat(rank_rows)
ranks.to_csv(f"{RESULTS_DIR}/phase3_accuracy_vs_cost_ranking.csv", index=False)
print("Is the most accurate model also the cheapest? (policy: searched with recall floor)")
for ratio in COST_RATIOS:
    r = ranks[ranks["cost_ratio"] == ratio].sort_values("rank_by_cost")
    print(f"  ratio {ratio}: most accurate = {acc.iloc[acc['accuracy_mean'].idxmax()]['model']}, cheapest = {r.iloc[0]['model']}")

df, features = load_clean()
main_model = "random_forest"
main_ratio = 10
fig, ax = plt.subplots(figsize=(6, 5))
sub = preds[(preds["seed"] == 0) & (preds["model"] == main_model) & (preds["split"] == "test")]
fpr, tpr, _ = roc_curve(sub["y"], sub["p_cal"])
ax.plot(fpr, tpr, label=f"{main_model} (seed 0, test)")
for ratio, marker in zip(COST_RATIOS, ["o", "s", "^"]):
    r = res[(res["seed"] == 0) & (res["model"] == main_model) & (res["cost_ratio"] == ratio) & (res["policy"] == "searched_with_recall_floor")].iloc[0]
    if r["feasible"] == 1:
        ax.plot(r["fpr"], r["recall"], marker, markersize=9, label=f"ratio {ratio}, tau={r['tau']:.3f}")
r = res[(res["seed"] == 0) & (res["model"] == main_model) & (res["cost_ratio"] == 2) & (res["policy"] == "default_0.5")].iloc[0]
ax.plot(r["fpr"], r["recall"], "kx", markersize=10, label="default tau=0.5")
ax.set_xlim(0, 0.3)
ax.set_ylim(0.7, 1.0)
ax.set_xlabel("False positive rate")
ax.set_ylabel("Recall (true positive rate)")
ax.set_title("ROC curve with cost based operating points")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/phase3_roc_operating_points.png", dpi=150)

cheap_changes = {"Symbol@": 1, "Redirecting//": 1, "ShortURL": 1, "LongURL": 1, "PrefixSuffix-": 1, "HTTPSDomainURL": 1}
sens_rows = []
err_rows = []
for seed in SEEDS:
    test = df[df[f"split_{seed}"] == "test"]
    val = df[df[f"split_{seed}"] == "val"]
    phish = test[test["label"] == 1]
    for name in ["random_forest", "gradient_boosting", "lsd_soft", "logistic_regression"]:
        bundle = joblib.load(f"{RESULTS_DIR}/models/seed{seed}_{name}.joblib")
        m, platt = bundle["model"], bundle["platt"]
        p_va = apply_platt(platt, raw_scores(m, val[features].values))
        tau = search_threshold(val["label"].values, p_va, main_ratio, recall_min=RECALL_MIN)
        if tau is None:
            continue
        base = (apply_platt(platt, raw_scores(m, phish[features].values)) >= tau).mean()
        sens_rows.append({"seed": seed, "model": name, "change": "none", "tau": tau, "recall_on_test_phishing": base})
        for feat, legit_value in cheap_changes.items():
            X = phish[features].copy()
            X[feat] = legit_value
            rec = (apply_platt(platt, raw_scores(m, X.values)) >= tau).mean()
            sens_rows.append({"seed": seed, "model": name, "change": f"set {feat} to legit", "tau": tau, "recall_on_test_phishing": rec})
        X = phish[features].copy()
        for feat, legit_value in cheap_changes.items():
            X[feat] = legit_value
        rec = (apply_platt(platt, raw_scores(m, X.values)) >= tau).mean()
        sens_rows.append({"seed": seed, "model": name, "change": "all cheap URL changes together", "tau": tau, "recall_on_test_phishing": rec})
        if name == main_model:
            p_ph = apply_platt(platt, raw_scores(m, phish[features].values))
            missed = phish[p_ph < tau]
            caught = phish[p_ph >= tau]
            for feat in features:
                err_rows.append({"seed": seed, "feature": feat,
                                 "missed_share_legit_value": (missed[feat] == 1).mean() if len(missed) else np.nan,
                                 "caught_share_legit_value": (caught[feat] == 1).mean(),
                                 "n_missed": len(missed)})

sens = pd.DataFrame(sens_rows)
sens.to_csv(f"{RESULTS_DIR}/phase3_sensitivity_per_seed.csv", index=False)
sens_sum = summarize(sens, ["model", "change"], ["recall_on_test_phishing"])
base = sens_sum[sens_sum["change"] == "none"][["model", "recall_on_test_phishing_mean"]].rename(columns={"recall_on_test_phishing_mean": "base"})
sens_sum = sens_sum.merge(base, on="model")
sens_sum["recall_drop"] = sens_sum["base"] - sens_sum["recall_on_test_phishing_mean"]
sens_sum.to_csv(f"{RESULTS_DIR}/phase3_sensitivity_summary.csv", index=False)
print(f"\nSensitivity: recall on test phishing sites after cheap URL changes (ratio {main_ratio}, recall floor {RECALL_MIN})")
print(sens_sum.sort_values(["model", "recall_drop"])[["model", "change", "recall_on_test_phishing_mean", "recall_drop"]].round(4).to_string(index=False))

err = pd.DataFrame(err_rows)
err_sum = err.groupby("feature")[["missed_share_legit_value", "caught_share_legit_value", "n_missed"]].mean()
err_sum["difference"] = err_sum["missed_share_legit_value"] - err_sum["caught_share_legit_value"]
err_sum = err_sum.sort_values("difference", ascending=False)
err_sum.to_csv(f"{RESULTS_DIR}/phase3_error_analysis.csv")
print(f"\nError analysis ({main_model}): share of phishing sites showing the legitimate value (1) per feature, missed vs caught")
print(err_sum.head(10).round(3).to_string())
print("\nSaved results/phase3_*.csv and results/phase3_roc_operating_points.png")
