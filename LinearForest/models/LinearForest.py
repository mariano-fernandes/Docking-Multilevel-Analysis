import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from cv_metrics import (
    assign_folds_to_poses,
    build_validation_plan,
    create_xgboost,
    evaluate_model_by_fold,
    load_ml_workbook,
)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


COL_POSE = "Pose"
COL_RESINT = "Res-Int"
COL_ENERGIA = "Energy (kcal/mol)"
COL_LIGANTE_CANDIDATOS = ["Ligand", "Compound", "Compound/Ligand", "Ligante", "Composto", "Composto/Ligante"]
MIN_FREQ = 0.02
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 500
MAX_CV_SPLITS = 3


def selecionar_arquivo_via_janela() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    caminho = filedialog.askopenfilename(
        title="Select the workbook (.xlsx) containing the poses",
        filetypes=[("Excel workbooks", "*.xlsx"), ("All files", "*.*")],
    )
    root.destroy()
    if not caminho:
        raise SystemExit("No file selected. Exiting.")
    return Path(caminho)


def carregar_matriz(xlsx_path: Path, min_freq: float):
    df, _ = load_ml_workbook(xlsx_path, (COL_POSE, COL_RESINT, COL_ENERGIA))

    col_ligante = next((coluna for coluna in COL_LIGANTE_CANDIDATOS if coluna in df.columns), None)
    df = df.copy()
    df["PoseGlobalID"] = df[COL_POSE]
    fingerprint = (pd.crosstab(df["PoseGlobalID"], df[COL_RESINT]) > 0).astype(int)
    meta_cols = ["PoseGlobalID", COL_ENERGIA] + ([col_ligante] if col_ligante else [])
    meta = df[meta_cols].drop_duplicates(subset="PoseGlobalID").set_index("PoseGlobalID")
    matriz = meta.join(fingerprint, how="left").fillna(0).reset_index()
    feat_cols = [coluna for coluna in matriz.columns if coluna not in ("PoseGlobalID", COL_ENERGIA, col_ligante)]
    mantidas = matriz[feat_cols].mean(axis=0)
    feat_cols = mantidas[mantidas >= min_freq].index.tolist()
    print(f"[filter] {len(feat_cols)} interactions retained (frequency >= {min_freq:.0%})")
    if not feat_cols:
        raise ValueError("No interactions remained after frequency filtering.")
    return matriz, feat_cols, col_ligante


def salvar_relatorio(modelo, X, y, splits, cabecalho, out_dir: Path, nome_modelo: str, planilha: str):
    identificador = "rf" if nome_modelo == "Random Forest" else "xgboost"
    caminho = out_dir / f"cv_scores_linearforest_{identificador}.txt"
    resultados = evaluate_model_by_fold(modelo, X, y, splits)
    resultados.insert(0, "workbook", planilha)
    resultados.insert(1, "input_type", "Binary Res-Int")
    resultados.insert(2, "model", nome_modelo)
    r2 = resultados["r2"]
    mae = resultados["mae_kcal_mol"]
    texto = (
        f"Input workbook: {planilha}\nInput type: Binary Res-Int\nModel: {nome_modelo}\n"
        f"{cabecalho}\nR^2 by fold: {np.round(r2, 3).tolist()}\n"
        f"Mean R^2: {r2.mean():.3f} (+/- {r2.std():.3f})\n"
        f"MAE by fold (kcal/mol): {np.round(mae, 3).tolist()}\n"
        f"Mean MAE (kcal/mol): {mae.mean():.3f} (+/- {mae.std():.3f})\n"
    )
    print(f"[cross-validation - {nome_modelo}]\n{texto}")
    caminho.write_text(texto, encoding="utf-8")
    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Res-Int comparison without SHAP")
    parser.add_argument("xlsx_path", nargs="?", default=None)
    parser.add_argument("--min-freq", type=float, default=MIN_FREQ)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else selecionar_arquivo_via_janela()
    out_dir = Path(args.out_dir) if args.out_dir else xlsx_path.parent / f"Model Comparison - {xlsx_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    matriz, feat_cols, col_ligante = carregar_matriz(xlsx_path, args.min_freq)
    X = matriz[feat_cols]
    y = matriz[COL_ENERGIA]
    grupos = matriz[col_ligante] if col_ligante else None
    splits, cabecalho, aviso = build_validation_plan(
        X, y, grupos, max_splits=MAX_CV_SPLITS, random_state=RANDOM_STATE
    )
    if aviso:
        print(f"[warning] {aviso}")
    if not splits:
        raise ValueError("Cross-validation was not run: insufficient data.")

    folds = assign_folds_to_poses(matriz["PoseGlobalID"], splits)
    folds.insert(0, "workbook", xlsx_path.name)
    folds.insert(1, "input_type", "Binary Res-Int")
    folds.to_csv(out_dir / "folds_validation_linearforest.csv", index=False, encoding="utf-8-sig")

    metricas = [salvar_relatorio(
        RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
        X, y, splits, cabecalho, out_dir, "Random Forest", xlsx_path.name
    )]
    if XGBRegressor is None:
        print("[warning] xgboost is not installed -- XGBoost results were not calculated.")
    else:
        xgb, usa_cuda = create_xgboost(
            XGBRegressor,
            n_estimators=XGB_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_bin=64,
            verbosity=1,
        )
        print("[XGBoost] Using CUDA GPU." if usa_cuda else "[XGBoost] CUDA/CuPy unavailable; using CPU.")
        metricas.append(salvar_relatorio(
            xgb,
            X, y, splits, cabecalho, out_dir, "XGBoost", xlsx_path.name
        ))
    pd.concat(metricas, ignore_index=True).to_csv(
        out_dir / "cv_metrics_linearforest_by_fold.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
