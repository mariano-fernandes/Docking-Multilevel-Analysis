from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

try:
    import cupy as cp
except ImportError:
    cp = None


def cuda_available() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def create_xgboost(XGBRegressor, *, n_estimators: int, random_state: int, n_jobs: int,
                   max_bin: int, verbosity: int):
    uses_cuda = cuda_available()
    model = XGBRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        device="cuda" if uses_cuda else "cpu",
        max_bin=max_bin,
        verbosity=verbosity,
    )
    return model, uses_cuda


def load_ml_workbook(xlsx_path, required_columns):
    with pd.ExcelFile(xlsx_path) as workbook:
        sheet_name = "ML Analysis" if "ML Analysis" in workbook.sheet_names else workbook.sheet_names[0]
        df = pd.read_excel(workbook, sheet_name=sheet_name)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required column(s) missing from the selected worksheet: {missing_columns}")

    print(f"[info] worksheet used: '{sheet_name}'")
    return df, sheet_name


def build_validation_plan(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None,
                               *, max_splits: int, random_state: int):
    n_groups = groups.nunique() if groups is not None else 0
    if groups is not None and n_groups >= 2:
        n_splits = min(max_splits, n_groups)
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
        return splits, f"GroupKFold (n_splits={n_splits}, grouped by ligand)", None

    warning = None
    if groups is not None and n_groups < 2:
        warning = (f"the ligand column has only {n_groups} distinct value(s) -- "
                   "cross-validation cannot be grouped by ligand. Using standard KFold.")
    n_splits = min(max_splits, len(X))
    if n_splits < 2:
        return [], "Cross-validation was not run: insufficient data.", warning
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y))
    return splits, "Standard KFold (WITHOUT ligand grouping -- risk of optimistic R^2)", warning


def evaluate_model_by_fold(model, X: pd.DataFrame, y: pd.Series, splits):
    results = []
    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_model = clone(model)
        uses_cuda = fold_model.get_params().get("device") == "cuda"
        if uses_cuda:
            X_train = cp.asarray(X.iloc[train_idx].to_numpy())
            y_train = cp.asarray(y.iloc[train_idx].to_numpy())
            X_test = cp.asarray(X.iloc[test_idx].to_numpy())
            fold_model.fit(X_train, y_train)
            y_pred = cp.asnumpy(fold_model.predict(X_test))
        else:
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = fold_model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        results.append({
            "fold": fold_number,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "r2": r2_score(y_test, y_pred),
            "mae_kcal_mol": mean_absolute_error(y_test, y_pred),
        })
    return pd.DataFrame(results)


def assign_folds_to_poses(pose_ids: pd.Series, splits) -> pd.DataFrame:
    records = []
    for fold_number, (_, test_idx) in enumerate(splits, start=1):
        records.extend({"PoseGlobalID": pose_ids.iloc[index], "fold": fold_number}
                       for index in test_idx)
    return pd.DataFrame(records)
