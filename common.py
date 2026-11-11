import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

DATA_DIR = "data"
RESULTS_DIR = "results"
SEEDS = [0, 1, 2, 3, 4]
COST_RATIOS = [2, 10, 50]
RECALL_MIN = 0.95
PRECISION_MIN = 0.90

URL_FEATURES = ["UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
                "PrefixSuffix-", "SubDomains", "HTTPSDomainURL"]

MODEL_NAMES = ["logistic_regression", "naive_bayes", "knn", "decision_tree", "random_forest",
               "gradient_boosting", "svc", "mlp", "lsd_soft", "lsd_hard"]


def load_clean():
    df = pd.read_csv(f"{DATA_DIR}/clean.csv")
    features = [c for c in df.columns if c not in ["label", "group", "ambiguous"] and not c.startswith("split_")]
    return df, features


def make_split(y, groups, seed):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr, va = next(inner.split(np.zeros(len(rest)), y[rest], groups[rest]))
    return rest[tr], rest[va], test


def base_models(seed):
    return {
        "logistic_regression": (LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1, 10]}),
        "naive_bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 9, 15, 25]}),
        "decision_tree": (DecisionTreeClassifier(random_state=seed), {"max_depth": [3, 5, 10, 20, None]}),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1), {"max_depth": [10, 20, None]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=seed), {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
        "svc": (SVC(random_state=seed), {"C": [0.1, 1, 10]}),
        "mlp": (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=seed), {"alpha": [1e-4, 1e-3, 1e-2]}),
    }


def train_all(X_tr, y_tr, g_tr, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fitted = {}
    best = {}
    for name, (model, grid) in base_models(seed).items():
        search = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_tr, y_tr, groups=g_tr)
        best[name] = search.best_params_
        model = search.best_estimator_
        if name == "svc":
            model = SVC(probability=True, random_state=seed, **search.best_params_).fit(X_tr, y_tr)
        fitted[name] = model
    members = [("lr", fitted["logistic_regression"]), ("svc", fitted["svc"]), ("dt", fitted["decision_tree"])]
    fitted["lsd_soft"] = VotingClassifier(members, voting="soft").fit(X_tr, y_tr)
    fitted["lsd_hard"] = VotingClassifier(members, voting="hard").fit(X_tr, y_tr)
    return fitted, best


def raw_scores(model, X):
    if isinstance(model, VotingClassifier) and model.voting == "hard":
        return model.predict(X).astype(float)
    return model.predict_proba(X)[:, 1]


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(p_val, y_val):
    return LogisticRegression(C=1e6, max_iter=1000).fit(to_logit(p_val), y_val)


def apply_platt(platt, p):
    return platt.predict_proba(to_logit(p))[:, 1]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


def counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def label_metrics(y, pred):
    tp, fp, fn, tn = counts(y, pred)
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def score_metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
    }


def cost_metrics(y, pred, ratio):
    tp, fp, fn, tn = counts(y, pred)
    n = len(y)
    return {
        "cost_per_1000": (ratio * fn + fp) / n * 1000,
        "missed_phishing_per_1000": fn / n * 1000,
    }


def search_threshold(y_val, p_val, ratio, recall_min=None, precision_min=None):
    candidates = np.unique(np.concatenate([np.linspace(0, 1, 1001), p_val]))
    best_tau, best_cost = None, None
    for tau in candidates:
        pred = (p_val >= tau).astype(int)
        tp, fp, fn, tn = counts(y_val, pred)
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        if recall_min is not None and rec < recall_min:
            continue
        if precision_min is not None and prec < precision_min:
            continue
        cost = ratio * fn + fp
        if best_cost is None or cost < best_cost or (cost == best_cost and tau > best_tau):
            best_tau, best_cost = tau, cost
    return best_tau


def summarize(df, group_cols, value_cols):
    agg = df.groupby(group_cols)[value_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
