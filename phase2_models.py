import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from common import (RESULTS_DIR, SEEDS, MODEL_NAMES, load_clean, train_all, raw_scores,
                    fit_platt, apply_platt, label_metrics, score_metrics, summarize)

drop_ambiguous = "--drop-ambiguous" in sys.argv
tag = "_no_ambiguous" if drop_ambiguous else ""
os.makedirs(f"{RESULTS_DIR}/models{tag}", exist_ok=True)

df, features = load_clean()
if drop_ambiguous:
    df = df[df["ambiguous"] == 0].reset_index(drop=True)

print(f"PHASE 2: REBUILD MODELS{' (ambiguous rows removed)' if drop_ambiguous else ''}")
print(f"Rows used: {len(df)}")

metric_rows = []
pred_rows = []
params = {}
for seed in SEEDS:
    print(f"\nSeed {seed}: tuning with 5 fold cross validation inside the training set...")
    s = df[f"split_{seed}"]
    tr, va, te = df[s == "train"], df[s == "val"], df[s == "test"]
    X_tr, y_tr, g_tr = tr[features].values, tr["label"].values, tr["group"].values
    X_va, y_va = va[features].values, va["label"].values
    X_te, y_te = te[features].values, te["label"].values

    models, best = train_all(X_tr, y_tr, g_tr, seed)
    params[seed] = {k: {a: str(b) for a, b in v.items()} for k, v in best.items()}

    for name in MODEL_NAMES:
        m = models[name]
        p_va_raw, p_te_raw = raw_scores(m, X_va), raw_scores(m, X_te)
        if name == "lsd_hard":
            platt = None
            p_va_cal, p_te_cal = p_va_raw, p_te_raw
        else:
            platt = fit_platt(p_va_raw, y_va)
            p_va_cal, p_te_cal = apply_platt(platt, p_va_raw), apply_platt(platt, p_te_raw)
        joblib.dump({"model": m, "platt": platt, "features": features}, f"{RESULTS_DIR}/models{tag}/seed{seed}_{name}.joblib")

        row = {"seed": seed, "model": name}
        row.update(label_metrics(y_te, (p_te_cal >= 0.5).astype(int)))
        if name != "lsd_hard":
            raw_m = score_metrics(y_te, p_te_raw)
            cal_m = score_metrics(y_te, p_te_cal)
            row.update({"roc_auc": cal_m["roc_auc"], "pr_auc": cal_m["pr_auc"],
                        "brier_raw": raw_m["brier"], "brier_calibrated": cal_m["brier"],
                        "ece_raw": raw_m["ece"], "ece_calibrated": cal_m["ece"]})
        metric_rows.append(row)

        for split_name, yy, pr, pc in [("val", y_va, p_va_raw, p_va_cal), ("test", y_te, p_te_raw, p_te_cal)]:
            pred_rows.append(pd.DataFrame({"seed": seed, "model": name, "split": split_name,
                                           "y": yy, "p_raw": pr, "p_cal": pc}))
        print(f"  {name:20s} accuracy {row['accuracy']:.4f}  recall {row['recall']:.4f}  precision {row['precision']:.4f}")

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(f"{RESULTS_DIR}/phase2_metrics_per_seed{tag}.csv", index=False)
pd.concat(pred_rows).to_csv(f"{RESULTS_DIR}/phase2_predictions{tag}.csv", index=False)
with open(f"{RESULTS_DIR}/phase2_best_params{tag}.json", "w") as f:
    json.dump(params, f, indent=2)

cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
        "brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated"]
summary = summarize(metrics, ["model"], cols).sort_values("accuracy_mean", ascending=False)
summary.to_csv(f"{RESULTS_DIR}/phase2_summary{tag}.csv", index=False)

pd.set_option("display.width", 200)
show = summary[["model", "accuracy_mean", "accuracy_std", "precision_mean", "recall_mean",
                "specificity_mean", "f1_mean", "roc_auc_mean", "ece_raw_mean", "ece_calibrated_mean"]]
print("\nTest results at threshold 0.5, mean over 5 seeds (std for accuracy):")
print(show.round(4).to_string(index=False))
print(f"\nSaved results/phase2_summary{tag}.csv, phase2_metrics_per_seed{tag}.csv, phase2_predictions{tag}.csv")
