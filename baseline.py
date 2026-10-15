"""
Baseline replication of "LendingClub Loan Default and Profitability Prediction"
(Li & Han, Stanford).  Run:  python baseline.py --data loan.csv

Pipeline: filter 2012-2015 finalized loans -> labels (clf + NAR) -> preprocessing
(mean/zero/max-set imputation, one-hot, days-since-epoch, standardise) -> 70/30 split
-> 3 classifiers + 4 regressors -> loan selection strategy -> metrics.json + plots.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                             roc_auc_score, mean_squared_error, r2_score)

SEED = 42
GOOD, BAD = ["Fully Paid"], ["Charged Off", "Default"]

# Columns only known AFTER the loan is issued (label leakage) or useless identifiers/text.
LEAK = ["id", "member_id", "url", "desc", "title", "emp_title", "funded_amnt", "funded_amnt_inv",
        "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
        "last_fico_range_high", "last_fico_range_low", "pymnt_plan", "policy_code",
        "debt_settlement_flag", "loan_status", "issue_d", "grade", "sub_grade_dummy"]
LEAK_PREFIXES = ("hardship", "settlement", "orig_projected", "debt_settlement", "deferral")
DATE_COLS = ["earliest_cr_line"]
MAX_SET_PREFIX = "mths_since"     # "never happened" -> large value, not zero
MEAN_SET = ["dti", "revol_util"]  # zero would wrongly reward the borrower


def load_and_label(path, sample=None):
    df = pd.read_csv(path, low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df[df["issue_dt"].dt.year.between(2012, 2015)]
    df = df[df["loan_status"].isin(GOOD + BAD)].copy()
    if sample:
        df = df.sample(min(sample, len(df)), random_state=SEED)
    y_clf = df["loan_status"].isin(GOOD).astype(int).values

    # Net annualised return: (total_payment / loan_amount) ** (365 / D) - 1
    # NOTE: the paper prints the exponent as 1/(365/D); the standard LendingClub NAR
    # annualises with 365/D (D = days from issue to last payment).
    last = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    D = (last - df["issue_dt"]).dt.days.clip(lower=30).fillna(30)  # month granularity
    nar = (df["total_pymnt"] / df["loan_amnt"]) ** (365.0 / D) - 1
    nar = nar.replace([np.inf, -np.inf], np.nan)
    keep = nar.notna().values
    df, y_clf, nar = df[keep], y_clf[keep], nar[keep].values
    print(f"Loans: {len(df):,} | positive (Fully Paid): {y_clf.mean():.1%}")
    return df, y_clf, nar


def build_features(df, miss_thresh=0.8):
    drop = [c for c in df.columns if c in LEAK or c.startswith(LEAK_PREFIXES) or c == "issue_dt"]
    X = df.drop(columns=drop)
    # drop mostly-empty and constant columns
    X = X.loc[:, X.isna().mean() < miss_thresh]
    X = X.loc[:, X.nunique(dropna=False) > 1]
    # percent strings -> numbers
    for c in X.columns:
        if X[c].dtype == object and X[c].dropna().astype(str).str.endswith("%").mean() > 0.9:
            X[c] = pd.to_numeric(X[c].astype(str).str.rstrip("%"), errors="coerce")
    # dates -> days since epoch
    for c in DATE_COLS:
        if c in X:
            X[c] = (pd.to_datetime(X[c], format="%b-%Y", errors="coerce")
                    - pd.Timestamp("1970-01-01")).dt.days
    cat = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=cat, dtype=np.float32)
    print(f"Feature count after preprocessing: {X.shape[1]} (paper: 1097)")
    return X


def impute_scale(Xtr, Xte, factor=1.0):
    """Impute using TRAIN statistics only, then standardise."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if Xtr[c].isna().any() or Xte[c].isna().any():
            if c in MEAN_SET:
                fill = Xtr[c].mean()
            elif c.startswith(MAX_SET_PREFIX):
                fill = factor * Xtr[c].max()
            else:
                fill = 0.0
            Xtr[c], Xte[c] = Xtr[c].fillna(fill), Xte[c].fillna(fill)
    sc = StandardScaler()
    return (sc.fit_transform(Xtr).astype(np.float32),
            sc.transform(Xte).astype(np.float32))


def clf_metrics(name, model, Xtr, ytr, Xte, yte, results):
    t = time.time(); model.fit(Xtr, ytr)
    out = {"train_time_s": round(time.time() - t, 1)}
    for split, X, y in [("train", Xtr, ytr), ("test", Xte, yte)]:
        p = model.predict(X)
        out[split] = {"confusion": confusion_matrix(y, p).tolist(),
                      "report": classification_report(y, p, output_dict=True,
                                                      target_names=["Default", "Paid Off"])}
    print(f"\n=== {name} (test) ===")
    print(classification_report(yte, model.predict(Xte), target_names=["Default", "Paid Off"]))
    results[name] = out
    return model


def reg_metrics(name, model, Xtr, ytr, Xte, yte, results):
    model.fit(Xtr, ytr)
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    r = {"train_mse": mean_squared_error(ytr, ptr), "test_mse": mean_squared_error(yte, pte),
         "train_r2": r2_score(ytr, ptr), "test_r2": r2_score(yte, pte)}
    print(f"{name:10s} " + " ".join(f"{k}={v:.4g}" for k, v in r.items()))
    results[name] = {k: float(v) for k, v in r.items()}
    return model, ptr, pte


def strategy_curve(pred, true):
    order = np.argsort(-pred)
    cum = np.cumsum(true[order]) / np.arange(1, len(true) + 1)
    return order, cum


def main(a):
    os.makedirs(a.out, exist_ok=True)
    df, y_clf, nar = load_and_label(a.data, a.sample)
    X = build_features(df)
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED)
    Xtr, Xte = impute_scale(X.iloc[tr], X.iloc[te])
    ytr, yte, ntr, nte = y_clf[tr], y_clf[te], nar[tr], nar[te]
    res = {"n_samples": len(X), "n_features": int(X.shape[1])}

    # ---------------- Classification ----------------
    cres = {}
    clf_metrics("LogReg", LogisticRegression(class_weight="balanced", max_iter=1000),
                Xtr, ytr, Xte, yte, cres)
    clf_metrics("NeuralNet", MLPClassifier(hidden_layer_sizes=(10, 10, 5, 3), activation="logistic",
                batch_size=200, learning_rate_init=0.001, alpha=1e-4, max_iter=a.nn_iter,
                random_state=SEED), Xtr, ytr, Xte, yte, cres)
    rf = clf_metrics("RandomForest", RandomForestClassifier(n_estimators=200, max_features=50,
                     criterion="gini", n_jobs=-1, random_state=SEED), Xtr, ytr, Xte, yte, cres)
    res["classification"] = cres
    fpr, tpr, _ = roc_curve(yte, rf.predict_proba(Xte)[:, 1])
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]); res["rf_auc"] = float(auc)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}"); plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - Random Forest"); plt.legend(); plt.savefig(f"{a.out}/roc_rf.png", dpi=150)

    # ---------------- Regression ----------------
    print("\n=== Regression (target = NAR) ===")
    rres = {}
    reg_metrics("LinearReg", LinearRegression(), Xtr, ntr, Xte, nte, rres)
    reg_metrics("RidgeReg", Ridge(alpha=1.0), Xtr, ntr, Xte, nte, rres)
    reg_metrics("NeuralNet", MLPRegressor(hidden_layer_sizes=(20, 10, 5, 3), activation="relu",
                solver="adam", batch_size=200, alpha=1e-4, max_iter=a.nn_iter, random_state=SEED),
                Xtr, ntr, Xte, nte, rres)
    best = None
    for d in (4, 8, 10):
        m, ptr, pte = reg_metrics(f"RF{d}", RandomForestRegressor(n_estimators=a.rf_trees,
                                  max_depth=d, max_features=50, n_jobs=-1, random_state=SEED),
                                  Xtr, ntr, Xte, nte, rres)
        if d == 10: best = (ptr, pte)
    res["regression"] = rres

    # ---------------- Loan selection strategy ----------------
    ptr, pte = best
    M = a.threshold
    strat = {}
    plt.figure(figsize=(6, 7))
    for i, (nm, p, t) in enumerate([("train", ptr, ntr), ("test", pte, nte)]):
        order, cum = strategy_curve(p, t)
        sel = p > M
        strat[nm] = {"threshold": M, "pct_loans_selected": float(sel.mean() * 100),
                     "avg_true_nar_selected": float(t[sel].mean()) if sel.any() else None}
        plt.subplot(2, 1, i + 1)
        plt.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100)
        plt.title(f"Investment Return Trend ({nm})"); plt.ylabel("annualised return (%)")
        plt.xlabel("% of loans invested")
    plt.tight_layout(); plt.savefig(f"{a.out}/investment_return_trend.png", dpi=150)
    res["strategy"] = strat
    print("\n=== Strategy (paper: 15% return with 1.76% of loans at M=0.132) ===")
    print(json.dumps(strat, indent=2))

    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved metrics + plots to {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to LendingClub loan.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=None, help="subsample N loans for quick runs")
    ap.add_argument("--threshold", type=float, default=0.132)
    ap.add_argument("--rf-trees", dest="rf_trees", type=int, default=100)
    ap.add_argument("--nn-iter", dest="nn_iter", type=int, default=50)
    main(ap.parse_args())
