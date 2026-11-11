import numpy as np
import pandas as pd
from common import (DATA_DIR, RESULTS_DIR, SEEDS, COST_RATIOS, RECALL_MIN, URL_FEATURES,
                    load_clean, make_split, train_all, raw_scores, fit_platt, apply_platt,
                    label_metrics, score_metrics, cost_metrics, search_threshold, summarize)
from url_features import extract, registered_domain

pd.set_option("display.width", 200)
MODELS = ["logistic_regression", "naive_bayes", "knn", "decision_tree", "random_forest",
          "gradient_boosting", "svc", "mlp", "lsd_soft"]

print("PHASE 4: GENERALIZATION TO THE MENDELEY DATASET (URL ONLY FEATURES)")

orig, _ = load_clean()
print(f"Original data: {len(orig)} unique websites, using {len(URL_FEATURES)} URL only features: {URL_FEATURES}")

mend = pd.read_csv(f"{DATA_DIR}/dataset_B_05_2020.csv", usecols=["url", "status"])
mend = mend.drop_duplicates(subset="url").reset_index(drop=True)
mend["label"] = (mend["status"] == "phishing").astype(int)
feats = pd.DataFrame([extract(u) for u in mend["url"]])
mend = pd.concat([mend, feats], axis=1)
mend["domain"] = [registered_domain(u) for u in mend["url"]]
mend["group"] = pd.factorize(mend["domain"])[0]
mend[["url", "label", "domain"] + URL_FEATURES].to_csv(f"{DATA_DIR}/mendeley_url_features.csv", index=False)
print(f"Mendeley data: {len(mend)} URLs (phishing {int(mend['label'].sum())}, legitimate {int((mend['label'] == 0).sum())}), {mend['domain'].nunique()} domains")

dist_rows = []
for feat in URL_FEATURES:
    for value in [-1, 0, 1]:
        for name, d in [("original", orig), ("mendeley", mend)]:
            m = d[feat] == value
            dist_rows.append({"feature": feat, "value": value, "dataset": name,
                              "share_of_rows": m.mean(),
                              "phishing_rate": d.loc[m, "label"].mean() if m.any() else np.nan})
dist = pd.DataFrame(dist_rows)
dist = dist[dist.groupby(["feature", "value"])["share_of_rows"].transform("sum") > 0]
dist_wide = dist.pivot_table(index=["feature", "value"], columns="dataset", values=["share_of_rows", "phishing_rate"])
dist_wide.columns = [f"{a}_{b}" for a, b in dist_wide.columns]
dist_wide = dist_wide.reset_index()
dist_wide.to_csv(f"{RESULTS_DIR}/phase4_feature_distribution.csv", index=False)
print("\nFeature values: share of rows and phishing rate in each dataset")
print(dist_wide.round(3).to_string(index=False))

rows = []
for seed in SEEDS:
    print(f"\nSeed {seed}: training URL only models on the original data...")
    s = orig[f"split_{seed}"]
    tr, va, te = orig[s == "train"], orig[s == "val"], orig[s == "test"]
    models, _ = train_all(tr[URL_FEATURES].values, tr["label"].values, tr["group"].values, seed)

    y_m = mend["label"].values
    m_tr, m_va, m_te = make_split(y_m, mend["group"].values, seed)
    ref_tr, ref_va, ref_te = mend.iloc[m_tr], mend.iloc[m_va], mend.iloc[m_te]
    ref_models, _ = train_all(ref_tr[URL_FEATURES].values, ref_tr["label"].values, ref_tr["group"].values, seed)

    settings = [
        ("original_internal_test", models, va, te),
        ("mendeley_full_external", models, va, mend),
        ("mendeley_within_reference", ref_models, ref_va, ref_te),
    ]
    for setting, mset, val, test in settings:
        for name in MODELS:
            m = mset[name]
            p_va_raw = raw_scores(m, val[URL_FEATURES].values)
            platt = fit_platt(p_va_raw, val["label"].values)
            p_va = apply_platt(platt, p_va_raw)
            p_te = apply_platt(platt, raw_scores(m, test[URL_FEATURES].values))
            y_te = test["label"].values
            base = {"seed": seed, "setting": setting, "model": name}
            base.update(score_metrics(y_te, p_te))
            for ratio in COST_RATIOS:
                taus = {
                    "analytic_tau_star": 1 / (1 + ratio),
                    "searched_with_recall_floor": search_threshold(val["label"].values, p_va, ratio, recall_min=RECALL_MIN),
                }
                for policy, tau in taus.items():
                    row = dict(base, cost_ratio=ratio, policy=policy)
                    if tau is None:
                        row.update({"tau": np.nan, "feasible": 0})
                    else:
                        pred = (p_te >= tau).astype(int)
                        row.update({"tau": tau, "feasible": 1})
                        row.update(label_metrics(y_te, pred))
                        row.update(cost_metrics(y_te, pred, ratio))
                    rows.append(row)
        print(f"  {setting:28s} done")

res = pd.DataFrame(rows)
res.to_csv(f"{RESULTS_DIR}/phase4_per_seed.csv", index=False)

cols = ["roc_auc", "pr_auc", "brier", "ece", "tau", "cost_per_1000", "missed_phishing_per_1000", "precision", "recall", "fpr", "accuracy"]
feas = res.groupby(["setting", "model", "cost_ratio", "policy"])["feasible"].sum().reset_index(name="feasible_seeds_out_of_5")
summary = summarize(res[res["feasible"] == 1], ["setting", "model", "cost_ratio", "policy"], cols)
summary = feas.merge(summary, on=["setting", "model", "cost_ratio", "policy"], how="left")
summary.to_csv(f"{RESULTS_DIR}/phase4_summary.csv", index=False)

auc = summarize(res.drop_duplicates(["seed", "setting", "model"]), ["setting", "model"], ["roc_auc", "pr_auc", "brier", "ece"])
auc_wide = auc.pivot(index="model", columns="setting", values="roc_auc_mean")
auc_wide["gap_internal_minus_external"] = auc_wide["original_internal_test"] - auc_wide["mendeley_full_external"]
auc_wide.to_csv(f"{RESULTS_DIR}/phase4_roc_auc_gap.csv")
print("\nROC AUC, mean over 5 seeds (threshold free)")
print(auc_wide.round(4).to_string())

main = summary[(summary["cost_ratio"] == 10) & (summary["policy"] == "searched_with_recall_floor")]
gap_cols = ["cost_per_1000_mean", "missed_phishing_per_1000_mean", "recall_mean", "precision_mean", "fpr_mean"]
gap = main.pivot(index="model", columns="setting", values=gap_cols)
gap.columns = [f"{a.replace('_mean', '')}__{b}" for a, b in gap.columns]
gap.to_csv(f"{RESULTS_DIR}/phase4_gap_ratio10.csv")
print(f"\nAt the threshold chosen on the ORIGINAL validation set (ratio 10, recall floor {RECALL_MIN}), applied without retuning:")
for setting in ["original_internal_test", "mendeley_full_external", "mendeley_within_reference"]:
    part = main[main["setting"] == setting][["model", "feasible_seeds_out_of_5", "tau_mean", "cost_per_1000_mean",
                                             "missed_phishing_per_1000_mean", "recall_mean", "precision_mean", "fpr_mean"]]
    print(f"\n--- {setting} ---")
    print(part.sort_values("cost_per_1000_mean").round(4).to_string(index=False))
print("\nSaved data/mendeley_url_features.csv and results/phase4_*.csv")
