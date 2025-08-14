#!/usr/bin/env python3
"""
significance_testing.py
-----------------------

Paired comparison of two pipelines across repeated stratified 5-fold CV.
- RF on the "base" (no extracted features) dataset
- XGB on the "full" (all features) dataset

Usage:
    python significance_testing.py \
        --full-file data_full.csv \
        --base-file data_base.csv \
        --target-col Progressive \
        --id-col "research ID" \
        --n-repeats 80 \
        --random-state 42
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ranksums
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score


# ----------- Custom Feature Selector -----------
class WilcoxonCollinearPruner(BaseEstimator, TransformerMixin):
    """Top-k by Wilcoxon p-value, prune correlated features first."""

    def __init__(self, k=10, corr_thresh=0.9):
        self.k = k
        self.corr_thresh = corr_thresh

    def fit(self, X, y):
        df = pd.DataFrame(
            X, columns=getattr(X, "columns", [f"f{i}" for i in range(X.shape[1])])
        )
        yarr = np.asarray(y)

        # p-values per feature (binary labels 0/1 assumed)
        pvals = []
        for c in df.columns:
            v0, v1 = df.loc[yarr == 0, c], df.loc[yarr == 1, c]
            if v0.std() == 0 and v1.std() == 0:
                pvals.append(1.0)
            else:
                _, p = ranksums(v0, v1)
                pvals.append(p)
        self.pvals_ = np.array(pvals)

        # prune one of each highly correlated pair (keep lower p-value)
        corr = df.corr().abs().values
        to_drop = set()
        idxs = np.arange(df.shape[1])
        for i, j in zip(*np.triu_indices(len(idxs), k=1)):
            if corr[i, j] > self.corr_thresh:
                drop = i if self.pvals_[i] > self.pvals_[j] else j
                to_drop.add(drop)

        kept = [i for i in idxs if i not in to_drop]
        kept = sorted(kept, key=lambda i: self.pvals_[i])[: self.k]
        self.selected_ = kept
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_]


def _build_argparser():
    p = argparse.ArgumentParser(description="Paired RF vs XGB significance test.")
    p.add_argument("--full-file", type=str, required=True, help="CSV with full features.")
    p.add_argument("--base-file", type=str, required=True, help="CSV with base/no-extracted features.")
    p.add_argument("--target-col", type=str, default="Progressive")
    p.add_argument("--id-col", type=str, default="research ID")
    p.add_argument("--n-repeats", type=int, default=80)  
    p.add_argument("--random-state", type=int, default=42)
    return p


# ----------- Main 100-Run Comparison (well… 5x80 = 400 splits) -----------
if __name__ == "__main__":
    args = _build_argparser().parse_args()

    # Load datasets 
    df_full = pd.read_csv(args.full_file)
    df_base = pd.read_csv(args.base_file)

    # Drop target + id from X; keep target as y 
    X_full = df_full.drop(columns=[args.target_col, args.id_col], errors="ignore")
    y_full = df_full[args.target_col]
    X_base = df_base.drop(columns=[args.target_col, args.id_col], errors="ignore")
    y_base = df_base[args.target_col]

    # Pipelines with fixed params (these mirror my "best" from a prior run)
    pipe_rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-2)),
            ("select", WilcoxonCollinearPruner(k=8)),
            ("scale", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    bootstrap=False,
                    max_depth=5,
                    max_features="log2",
                    min_samples_leaf=1,
                    min_samples_split=6,
                    n_estimators=add,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe_xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-2)),
            ("select", WilcoxonCollinearPruner(k=13)),
            ("scale", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    eval_metric="auc",
                    use_label_encoder=False,
                    random_state=42,
                    colsample_bytree= add,
                    gamma=add,
                    learning_rate=add,
                    max_depth=add,
                    n_estimators=add,
                    reg_alpha=add,
                    reg_lambda=add,
                    subsample=add,
                ),
            ),
        ]
    )

    # Repeated 5-fold (same splits used for both models)
    rkf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=args.n_repeats, random_state=args.random_state
    )

    aucs_rf, aucs_xgb = [], []
    # tiny sanity counter I don't really use later:
    _split_counter = 0  # <- I think this is correct now

    for train_idx, test_idx in rkf.split(X_full, y_full):
        # NOTE: We use the same splits for both:
        #   RF indexes into base (no extracted features)
        #   XGB indexes into full (all features)
        Xb_train, yb_train = X_base.iloc[train_idx], y_base.iloc[train_idx]
        Xb_test, yb_test = X_base.iloc[test_idx], y_base.iloc[test_idx]

        Xf_train, yf_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
        Xf_test, yf_test = X_full.iloc[test_idx], y_full.iloc[test_idx]

        # fit & predict RF
        pipe_rf.fit(Xb_train, yb_train)
        prob_rf = pipe_rf.predict_proba(Xb_test)[:, 1]
        aucs_rf.append(roc_auc_score(yb_test, prob_rf))

        # fit & predict XGB
        pipe_xgb.fit(Xf_train, yf_train)
        prob_xgb = pipe_xgb.predict_proba(Xf_test)[:, 1]
        aucs_xgb.append(roc_auc_score(yf_test, prob_xgb))

        _split_counter += 1  # keeping track just in case

    aucs_rf = np.array(aucs_rf)
    aucs_xgb = np.array(aucs_xgb)

    # Paired t-test (same as before)
    t_stat, p_val = ttest_rel(aucs_xgb, aucs_rf)

    # Report (keeping the same label text style as my earlier notes)
    print(f"RF  (100 runs) AUC = {aucs_rf.mean():.3f} ± {aucs_rf.std():.3f}")
    print(f"XGB (100 runs) AUC = {aucs_xgb.mean():.3f} ± {aucs_xgb.std():.3f}")
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
