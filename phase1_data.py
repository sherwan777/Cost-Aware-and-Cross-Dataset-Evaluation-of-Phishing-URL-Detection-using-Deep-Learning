import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import DATA_DIR, RESULTS_DIR, SEEDS, make_split

os.makedirs(RESULTS_DIR, exist_ok=True)

raw = pd.read_csv(f"{DATA_DIR}/phishing.csv")
raw = raw.drop(columns=["Index"])
features = [c for c in raw.columns if c != "class"]

print("PHASE 1: DATA CLEANING AND SPLITS")
print(f"Raw rows: {len(raw)}, features: {len(features)}")
print(f"Missing values: {int(raw.isna().sum().sum())}")
print(f"Class counts in file: {raw['class'].value_counts().to_dict()}")

raw["label"] = (raw["class"] == -1).astype(int)
raw = raw.drop(columns=["class"])
print(f"After recoding: phishing (label 1) = {int(raw['label'].sum())}, legitimate (label 0) = {int((raw['label'] == 0).sum())}")

leak_rows = []
for seed in SEEDS:
    tr, te = train_test_split(raw, test_size=0.3, stratify=raw["label"], random_state=seed)
    train_keys = set(map(tuple, tr[features].values))
    share = np.mean([tuple(r) in train_keys for r in te[features].values])
    leak_rows.append({"seed": seed, "method": "paper_style_random_70_30_on_raw_file", "test_rows_with_twin_in_train": share})

n_dup = int(raw.duplicated().sum())
df = raw.drop_duplicates().reset_index(drop=True)
print(f"Exact duplicate rows removed: {n_dup} ({n_dup / len(raw):.1%})")
print(f"Unique rows left: {len(df)} (phishing {int(df['label'].sum())}, legitimate {int((df['label'] == 0).sum())})")

pattern = df[features].astype(str).agg("|".join, axis=1)
df["group"] = pd.factorize(pattern)[0]
label_count = df.groupby("group")["label"].transform("nunique")
df["ambiguous"] = (label_count > 1).astype(int)
print(f"Feature patterns with both labels: {df.loc[df['ambiguous'] == 1, 'group'].nunique()} ({int(df['ambiguous'].sum())} rows), kept and flagged")

y = df["label"].values
groups = df["group"].values
split_rows = []
for seed in SEEDS:
    tr, va, te = make_split(y, groups, seed)
    col = np.empty(len(df), dtype=object)
    col[tr], col[va], col[te] = "train", "val", "test"
    df[f"split_{seed}"] = col
    train_keys = set(map(tuple, df.loc[tr, features].values))
    share = np.mean([tuple(r) in train_keys for r in df.loc[te, features].values])
    leak_rows.append({"seed": seed, "method": "our_dedup_grouped_split", "test_rows_with_twin_in_train": share})
    for name, idx in [("train", tr), ("val", va), ("test", te)]:
        split_rows.append({"seed": seed, "split": name, "rows": len(idx), "phishing_share": y[idx].mean()})

df.to_csv(f"{DATA_DIR}/clean.csv", index=False)

leak = pd.DataFrame(leak_rows)
splits = pd.DataFrame(split_rows)
leak.to_csv(f"{RESULTS_DIR}/phase1_leakage_check.csv", index=False)
splits.to_csv(f"{RESULTS_DIR}/phase1_split_sizes.csv", index=False)

summary = pd.DataFrame([
    {"item": "raw_rows", "value": len(raw)},
    {"item": "duplicate_rows_removed", "value": n_dup},
    {"item": "unique_rows", "value": len(df)},
    {"item": "unique_phishing", "value": int(df["label"].sum())},
    {"item": "unique_legitimate", "value": int((df["label"] == 0).sum())},
    {"item": "ambiguous_patterns", "value": int(df.loc[df["ambiguous"] == 1, "group"].nunique())},
    {"item": "ambiguous_rows", "value": int(df["ambiguous"].sum())},
])
summary.to_csv(f"{RESULTS_DIR}/phase1_summary.csv", index=False)

print("\nLeakage check (share of test rows that also appear in training):")
print(leak.groupby("method")["test_rows_with_twin_in_train"].agg(["mean", "min", "max"]).round(3))
print("\nSplit sizes (seed 0):")
print(splits[splits["seed"] == 0].round(3).to_string(index=False))
print(f"\nSaved {DATA_DIR}/clean.csv and results/phase1_*.csv")
