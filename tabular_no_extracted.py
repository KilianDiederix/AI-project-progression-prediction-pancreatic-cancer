#!/usr/bin/env python3
"""
tabular_no_extracted.py
-----------------------

Runs LR / RF / XGB with a Wilcoxon-based feature selector
(no segmentation-based extracted features in this dataset),
nested CV + random search, and a final soft-voting ensemble.

Usage:
    python tabular_no_extracted.py \
        --data-file data.csv \
        --target-col Progressive \
        --id-col "research ID" \
        --log-dir logs_no_segsv2 \
        --random-state 42 \
        --n-iter 400
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

from scipy.stats import ranksums, randint, uniform, loguniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
)
from xgboost import XGBClassifier



# ----------- Custom Feature Selector -----------
class WilcoxonCollinearPruner(BaseEstimator, TransformerMixin):
    """Select top-k features by Wilcoxon p-value, dropping highly correlated ones."""

    def __init__(self, k=10, corr_thresh=0.9):
        self.k = k
        self.corr_thresh = corr_thresh

    def fit(self, X, y):
        df = pd.DataFrame(
            X, columns=getattr(X, "columns", [f"f{i}" for i in range(X.shape[1])])
        )
        yarr = np.asarray(y)

        # calculate p-values for each feature
        pvals = []
        for c in df.columns:
            v0, v1 = df.loc[yarr == 0, c], df.loc[yarr == 1, c]
            if v0.std() == 0 and v1.std() == 0:
                pvals.append(1.0)  # no variance -> neutral p-value
            else:
                _, p = ranksums(v0, v1)
                pvals.append(p)
        self.pvals_ = np.array(pvals)

        # drop one of any highly correlated pair (keep lower p-value feature)
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

    def get_support(self, indices=False):
        if indices:
            return self.selected_
        mask = np.zeros(self.n_features_in_, bool)
        mask[self.selected_] = True
        return mask


# ----------- Main evaluation -----------
def evaluate_models(
    df,
    target_col="Progressive",
    id_col="research ID",
    random_state=42,
    log_dir="logs_no_segsv2",
    n_iter=400,
):
    """Run nested CV + random search for LR/RF/XGB, then evaluate a voting ensemble."""
    os.makedirs(log_dir, exist_ok=True)

    # logger setup
    logger = logging.getLogger("nested_no_segsv2")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "nested.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # --- Data prep ---
    dfc = df.drop(columns=["Local_PD"], errors="ignore").dropna(axis=1, how="all")
    y = dfc[target_col]
    X = dfc.drop(columns=[target_col, id_col], errors="ignore")
    feat_names = X.columns
    n_feats = X.shape[1]

    # CV definitions
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    scoring = dict(roc_auc="roc_auc", recall="recall", precision="precision", f1="f1")

    # Pipelines & param grids
    pipes, params = {}, {}

    # LR
    pipes["LR"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-2)),
            ("select", WilcoxonCollinearPruner(k=min(16, n_feats))),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]
    )
    params["LR"] = {
        "select__k": randint(8, min(16, n_feats) + 1),
        "clf__penalty": ["l1", "l2"],
        "clf__C": loguniform(1e-2, 1e1),
        "clf__solver": ["liblinear", "saga"],
        "clf__class_weight": ["balanced", None],
    }

    # RF
    pipes["RF"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-2)),
            ("select", WilcoxonCollinearPruner(k=min(16, n_feats))),
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=random_state)),
        ]
    )
    params["RF"] = {
        "select__k": randint(8, min(16, n_feats) + 1),
        "clf__n_estimators": randint(100, 501),
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": randint(2, 11),
        "clf__min_samples_leaf": randint(1, 6),
        "clf__max_features": ["sqrt", "log2", None],
        "clf__bootstrap": [True, False],
    }

    # XGB
    pipes["XGB"] = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(1e-2)),
            ("select", WilcoxonCollinearPruner(k=min(16, n_feats))),
            ("scale", StandardScaler()),
            ("clf", XGBClassifier(eval_metric="auc", random_state=random_state)),
        ]
    )
    params["XGB"] = {
        "select__k": randint(8, min(16, n_feats) + 1),
        "clf__n_estimators": randint(100, 501),
        "clf__learning_rate": loguniform(1e-4, 1.0),
        "clf__max_depth": randint(3, 12),
        "clf__subsample": uniform(0.3, 0.7),
        "clf__colsample_bytree": uniform(0.3, 0.7),
        "clf__gamma": uniform(0, 10),
        "clf__reg_alpha": loguniform(1e-6, 1),
        "clf__reg_lambda": loguniform(1e-6, 1),
    }

    all_metrics = {m: {mdl: [] for mdl in pipes} for m in scoring}
    final_pipelines = {}

    # ---- Nested CV + RandomSearchCV ----
    for name, pipe in pipes.items():
        print(f"\n>>> Starting nested CV + RandomSearch for {name}")
        logger.info(f"=== {name} START nested CV + RandomSearch (n_iter={n_iter}) ===")

        search = RandomizedSearchCV(
            pipe,
            params[name],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            random_state=random_state,
        )

        res = cross_validate(
            search, X, y, cv=outer_cv, scoring=scoring, return_estimator=True, n_jobs=-1
        )

        # record metrics
        for m in scoring:
            all_metrics[m][name].extend(res[f"test_{m}"])

        # log per‐fold best params
        fold_params = [est.best_params_ for est in res["estimator"]]
        logger.info(f"{name} per‐fold best_params:\n{json.dumps(fold_params, indent=2)}")

        # final refit on full data
        logger.info(f"=== {name} FINAL refit on full data ===")
        final_search = RandomizedSearchCV(
            pipe,
            params[name],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            random_state=random_state,
        )
        final_search.fit(X, y)

        best_p = final_search.best_params_
        logger.info(f"{name}: best_params_full = {best_p}")
        with open(os.path.join(log_dir, f"{name}_best_params_no_segsv2.json"), "w") as f:
            json.dump(best_p, f, indent=2)

        sel_idx = final_search.best_estimator_.named_steps["select"].get_support(
            indices=True
        )
        sel_feats = list(feat_names[sel_idx])
        logger.info(f"{name}: selected_feats_no_segsv2 = {sel_feats}")
        with open(
            os.path.join(log_dir, f"{name}_selected_feats_no_segsv2.txt"), "w"
        ) as f:
            f.write("\n".join(sel_feats))

        # store tuned pipeline (unfitted for now)
        tuned_pipe = pipe.set_params(**best_p)
        final_pipelines[name] = tuned_pipe

        logger.info(f"=== {name} DONE ===")

    # ---- Summarize results ----
    rows = []
    for mdl in pipes:
        r = {"model": mdl}
        for m in scoring:
            arr = np.array(all_metrics[m][mdl])
            r[f"{m}_mean"] = round(arr.mean(), 3)
            r[f"{m}_std"] = round(arr.std(), 3)
            r[f"{m}_5pct"] = round(np.percentile(arr, 5), 3)
            r[f"{m}_95pct"] = round(np.percentile(arr, 95), 3)
            r[f"n_splits"] = len(arr)
        rows.append(r)

    summary_df = pd.DataFrame(rows).set_index("model")
    print("\n=== NESTED CV RESULTS ===")
    print(summary_df)
    summary_df.to_csv(os.path.join(log_dir, "summary_no_segsv2.csv"))
    logger.info("Saved summary_no_segsv2.csv")

    # ---- Soft Voting Ensemble ----
    voting = VotingClassifier(
        estimators=[(n, final_pipelines[n]) for n in pipes], voting="soft", n_jobs=-1
    )
    vote_res = cross_validate(voting, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1)

    vr = {"model": "Voting"}
    for m in scoring:
        arr = vote_res[f"test_{m}"]
        vr[f"{m}_mean"] = round(arr.mean(), 3)
        vr[f"{m}_std"] = round(arr.std(), 3)
        vr[f"{m}_5pct"] = round(np.percentile(arr, 5), 3)
        vr[f"{m}_95pct"] = round(np.percentile(arr, 95), 3)
        vr[f"n_splits"] = len(arr)

    vote_df = pd.DataFrame([vr]).set_index("model")
    print("\n=== VOTING ENSEMBLE ===")
    print(vote_df)
    vote_df.to_csv(os.path.join(log_dir, "voting_summary_no_segsv2.csv"))
    logger.info("Saved voting_summary_no_segsv2.csv")

    return summary_df, final_pipelines, vote_df


def _build_argparser():
    p = argparse.ArgumentParser(description="Run nested CV (no extracted features).")
    p.add_argument("--data-file", type=str, required=True)
    p.add_argument("--target-col", type=str, default="Progressive")
    p.add_argument("--id-col", type=str, default="research ID")
    p.add_argument("--log-dir", type=str, default="logs_no_segsv2")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-iter", type=int, default=400)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    df = pd.read_csv(args.data_file)
    # print("Data shape:", df.shape)  # quick check if needed
    evaluate_models(
        df=df,
        target_col=args.target_col,
        id_col=args.id_col,
        random_state=args.random_state,
        log_dir=args.log_dir,
        n_iter=args.n_iter,
    )
