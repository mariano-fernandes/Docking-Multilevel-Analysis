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


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
COL_POSE = "Pose"
COL_RESINT = "Res-Int"
COL_ENERGIA = "Energy (kcal/mol)"
COL_LIGANTE_CANDIDATOS = ["Ligand", "Compound", "Compound/Ligand", "Ligante", "Composto", "Composto/Ligante"]

COL_NUMERO_INTERACOES = "Interaction_Count"
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 500
MAX_CV_SPLITS = 3


def carregar_dados(xlsx_path: Path) -> tuple[pd.DataFrame, str | None]:
    df, _ = load_ml_workbook(xlsx_path, (COL_POSE, COL_RESINT, COL_ENERGIA))

    col_ligante = next(
        (coluna for coluna in COL_LIGANTE_CANDIDATOS if coluna in df.columns),
        None,
    )
    if col_ligante:
        print(
            f"[info] ligand column detected: '{col_ligante}' -> "
            "grouped cross-validation will be used."
        )
    else:
        print(
            "[warning] no Ligand/Compound column was found. Cross-validation will use "
            "standard KFold without grouping -> R^2 may be optimistic if poses from the same "
            "compound fall into different folds."
        )

    return df, col_ligante


def detectar_poses(df: pd.DataFrame, col_ligante: str | None) -> pd.DataFrame:
    """Use Pose as the global unique identifier, as in the main script."""
    df = df.copy()
    df["PoseGlobalID"] = df[COL_POSE]

    n_energias = df.groupby("PoseGlobalID")[COL_ENERGIA].nunique()
    inconsistentes = n_energias[n_energias > 1]
    if not inconsistentes.empty:
        print(
            f"[warning] {len(inconsistentes)} pose(s) have more than one associated Energy value "
            "-- check the workbook for possible typing or duplication errors: "
            f"{inconsistentes.index.tolist()[:10]}"
        )

    if col_ligante:
        n_ligantes = df.groupby("PoseGlobalID")[col_ligante].nunique()
        inconsistentes_ligante = n_ligantes[n_ligantes > 1]
        if not inconsistentes_ligante.empty:
            print(
                f"[warning] {len(inconsistentes_ligante)} pose(s) are associated with more than one ligand "
                "-- check for possible copy/paste errors between blocks: "
                f"{inconsistentes_ligante.index.tolist()[:10]}"
            )

    return df


def montar_tabela_baseline(df: pd.DataFrame, col_ligante: str | None) -> pd.DataFrame:
    """Create one row per pose and count every recorded contact."""
    contagens = df.groupby("PoseGlobalID").size().rename(COL_NUMERO_INTERACOES)

    meta_cols = ["PoseGlobalID", COL_ENERGIA] + ([col_ligante] if col_ligante else [])
    meta = df[meta_cols].drop_duplicates(subset="PoseGlobalID").set_index("PoseGlobalID")

    return meta.join(contagens, how="left").reset_index()


def validar_modelo(
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    cabecalho: str,
    out_dir: Path,
    nome_modelo: str,
    planilha: str,
    tipo_entrada: str,
) -> pd.DataFrame:
    """Calculate and save fold-level R^2 and MAE using the main script's rule."""
    nome_arquivo = nome_modelo.lower().replace(" ", "_")
    caminho_scores = out_dir / f"cv_scores_{nome_arquivo}_baseline.txt"

    if not splits:
        texto = (
            f"Input workbook: {planilha}\n"
            f"Input type: {tipo_entrada}\n"
            f"Model: {nome_modelo}\n"
            "Cross-validation was not run: insufficient data.\n"
        )
        print(f"[warning] {texto.strip()}")
        caminho_scores.write_text(texto, encoding="utf-8")
        return pd.DataFrame()

    resultados = evaluate_model_by_fold(modelo, X, y, splits)
    resultados.insert(0, "workbook", planilha)
    resultados.insert(1, "input_type", tipo_entrada)
    resultados.insert(2, "model", nome_modelo)
    r2_scores = resultados["r2"]
    mae_scores = resultados["mae_kcal_mol"]

    texto = (
        f"Input workbook: {planilha}\n"
        f"Input type: {tipo_entrada}\n"
        f"Model: {nome_modelo}\n"
        f"{cabecalho}\n"
        f"R^2 by fold: {np.round(r2_scores, 3).tolist()}\n"
        f"Mean R^2: {r2_scores.mean():.3f} (+/- {r2_scores.std():.3f})\n"
        f"MAE by fold (kcal/mol): {np.round(mae_scores, 3).tolist()}\n"
        f"Mean MAE (kcal/mol): {mae_scores.mean():.3f} (+/- {mae_scores.std():.3f})\n"
    )
    print(f"[cross-validation - {nome_modelo} baseline]\n" + texto)
    caminho_scores.write_text(texto, encoding="utf-8")
    return resultados


def selecionar_arquivo_via_janela() -> Path:
    """Open the file selector while preserving the main script's usage mode."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as erro:
        raise RuntimeError(
            "tkinter is unavailable in this Python installation. "
            "Run the script by passing the path manually."
        ) from erro

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Forest baseline: total contacts per pose to predict energy"
    )
    parser.add_argument(
        "xlsx_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the .xlsx workbook. If omitted, a file-selection window opens.",
    )
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else selecionar_arquivo_via_janela()
    out_dir = xlsx_path.parent / f"Baseline - {xlsx_path.stem}"
    out_dir.mkdir(exist_ok=True)

    print(f"[1/4] Loading {xlsx_path.name} ...")
    df, col_ligante = carregar_dados(xlsx_path)

    print("[2/4] Detecting poses ...")
    df = detectar_poses(df, col_ligante)
    print(f"   -> {df['PoseGlobalID'].nunique()} poses detected in total.")

    print("[3/4] Counting contacts per pose ...")
    tabela = montar_tabela_baseline(df, col_ligante)
    tabela.to_excel(out_dir / "baseline_pose_data.xlsx", index=False)

    X = tabela[[COL_NUMERO_INTERACOES]]
    y = tabela[COL_ENERGIA]
    grupos = tabela[col_ligante] if col_ligante else None
    splits, cabecalho_cv, aviso_cv = build_validation_plan(
        X, y, grupos, max_splits=MAX_CV_SPLITS, random_state=RANDOM_STATE
    )
    if aviso_cv:
        print(f"[warning] {aviso_cv}")

    tipo_entrada = "Total interaction count"
    planilha = xlsx_path.name
    folds = assign_folds_to_poses(tabela["PoseGlobalID"], splits)
    folds.insert(0, "workbook", planilha)
    folds.insert(1, "input_type", tipo_entrada)
    folds.to_csv(out_dir / "folds_validation.csv", index=False, encoding="utf-8-sig")
    metricas_por_modelo = []

    print("[4/4] Calculating cross-validation R^2 ...")
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    metricas_por_modelo.append(
        validar_modelo(
            rf, X, y, splits, cabecalho_cv, out_dir, "Random Forest", planilha, tipo_entrada
        )
    )

    if XGBRegressor is None:
        print(
            "[warning] xgboost is not installed -- baseline XGBoost R^2 was not calculated. "
            "Install it with: pip install xgboost"
        )
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
        metricas_por_modelo.append(
            validar_modelo(
                xgb, X, y, splits, cabecalho_cv, out_dir, "XGBoost", planilha, tipo_entrada
            )
        )

    colunas_metricas = [
        "workbook", "input_type", "model", "fold", "n_train", "n_test", "r2", "mae_kcal_mol"
    ]
    metricas = [tabela_modelo for tabela_modelo in metricas_por_modelo if not tabela_modelo.empty]
    if metricas:
        pd.concat(metricas, ignore_index=True).to_csv(
            out_dir / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=colunas_metricas).to_csv(
            out_dir / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\nFinished. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
