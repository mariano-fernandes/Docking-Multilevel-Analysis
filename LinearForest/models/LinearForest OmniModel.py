import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, KFold

from cv_metrics import (
    assign_folds_to_poses,
    build_validation_plan,
    create_xgboost,
    evaluate_model_by_fold,
    load_ml_workbook,
)

try:
    import shap
except ImportError:
    shap = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
COL_POSE = "Pose"
COL_RESINT = "Res-Int"
COL_ENERGIA = "Energy (kcal/mol)"
COL_LIGANTE_CANDIDATOS = ["Ligand", "Compound", "Compound/Ligand", "Ligante", "Composto", "Composto/Ligante"]

MIN_FREQ = 0.02
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 500
MAX_CV_SPLITS = 3
RF_SHAP_SAMPLE_SIZE = None       # None retains all poses for final interpretation.
XGB_SHAP_SAMPLE_SIZE = None      # Use all poses for comparative interpretation.
TOP_N_DEPENDENCE = 6      # Number of SHAP dependence plots to generate (RF and XGBoost).
TOP_N_SINTESE = 15        # Size of the top list used for comparison between approaches.
PNG_SUMMARY_DPI = 1000


# ----------------------------------------------------------------------
# 2. DATA LOADING AND POSES
# ----------------------------------------------------------------------
def carregar_dados(xlsx_path: Path) -> tuple[pd.DataFrame, str | None]:
    df, _ = load_ml_workbook(xlsx_path, (COL_POSE, COL_RESINT, COL_ENERGIA))

    col_ligante = next((c for c in COL_LIGANTE_CANDIDATOS if c in df.columns), None)
    if col_ligante:
        print(f"[info] ligand column detected: '{col_ligante}' -> grouped cross-validation will be used.")
    else:
        print("[warning] no Ligand/Compound column was found. Cross-validation will use "
              "standard KFold without grouping -> R^2 may be optimistic if poses from the same "
              "compound fall into different folds. Consider adding this column in the future.")

    return df, col_ligante


def detectar_poses(df: pd.DataFrame, col_ligante: str | None) -> pd.DataFrame:
    """Use the Pose column itself as the global, sequential pose identifier."""
    df = df.copy()
    df["PoseGlobalID"] = df[COL_POSE]

    n_energias = df.groupby("PoseGlobalID")[COL_ENERGIA].nunique()
    inconsistentes = n_energias[n_energias > 1]
    if len(inconsistentes) > 0:
        print(f"[warning] {len(inconsistentes)} pose(s) have more than one associated Energy value "
              f"-- check the workbook for possible typing or duplication errors: "
              f"{inconsistentes.index.tolist()[:10]}")

    if col_ligante:
        n_ligantes = df.groupby("PoseGlobalID")[col_ligante].nunique()
        inconsistentes_lig = n_ligantes[n_ligantes > 1]
        if len(inconsistentes_lig) > 0:
            print(f"[warning] {len(inconsistentes_lig)} pose(s) are associated with more than one ligand "
                  f"-- check for possible copy/paste errors between blocks: "
                  f"{inconsistentes_lig.index.tolist()[:10]}")

    return df


# ----------------------------------------------------------------------
# 3. FEATURE MATRIX
# ----------------------------------------------------------------------
def montar_matriz(df: pd.DataFrame, col_ligante: str | None) -> pd.DataFrame:
    fingerprint = pd.crosstab(df["PoseGlobalID"], df[COL_RESINT])
    fingerprint = (fingerprint > 0).astype(int)

    meta_cols = ["PoseGlobalID", COL_ENERGIA] + ([col_ligante] if col_ligante else [])
    meta = df[meta_cols].drop_duplicates(subset="PoseGlobalID").set_index("PoseGlobalID")

    matriz = meta.join(fingerprint, how="left").fillna(0).reset_index()
    return matriz


def filtrar_features_raras(matriz: pd.DataFrame, feat_cols: list[str], min_freq: float) -> list[str]:
    freq = matriz[feat_cols].mean(axis=0)
    mantidas = freq[freq >= min_freq].index.tolist()
    print(f"[filter] {len(mantidas)} of {len(feat_cols)} interactions retained "
          f"(frequency >= {min_freq:.0%})")
    return mantidas


# ----------------------------------------------------------------------
# 4. GENERIC CROSS-VALIDATION (used by all models)
# ----------------------------------------------------------------------
def validar_modelo(modelo, X: pd.DataFrame, y: pd.Series, splits, cabecalho: str,
                    out_dir: Path, nome_modelo: str, planilha: str,
                    tipo_entrada: str) -> pd.DataFrame:
    """Calculate R^2 and MAE from test-set predictions for each fold."""
    nome_arquivo = nome_modelo.lower().replace(" ", "_")
    caminho_scores = out_dir / f"cv_scores_{nome_arquivo}.txt"
    if not splits:
        texto = (f"Input workbook: {planilha}\nInput type: {tipo_entrada}\n"
                 f"Model: {nome_modelo}\nCross-validation was not run: insufficient data.\n")
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
        f"Input workbook: {planilha}\nInput type: {tipo_entrada}\nModel: {nome_modelo}\n"
        f"{cabecalho}\n"
        f"R^2 by fold: {np.round(r2_scores, 3).tolist()}\n"
        f"Mean R^2: {r2_scores.mean():.3f} (+/- {r2_scores.std():.3f})\n"
        f"MAE by fold (kcal/mol): {np.round(mae_scores, 3).tolist()}\n"
        f"Mean MAE (kcal/mol): {mae_scores.mean():.3f} (+/- {mae_scores.std():.3f})\n"
    )
    print(f"[cross-validation - {nome_modelo}]\n" + texto)
    caminho_scores.write_text(texto, encoding="utf-8")
    return resultados


# ----------------------------------------------------------------------
# 5. APPROACH A - NONLINEAR MODELS (Random Forest and XGBoost) + SHAP
# ----------------------------------------------------------------------
def salvar_grafico_em_png_e_svg(caminho_base: Path, dpi_png: int = 150) -> None:
    """Save the current figure in viewing and vector-editing formats."""
    plt.savefig(f"{caminho_base}.png", dpi=dpi_png)
    plt.savefig(f"{caminho_base}.svg", format="svg")


def selecionar_amostra_shap(X: pd.DataFrame, tamanho_amostra: int | None) -> pd.DataFrame:
    """Select a reproducible SHAP sample without affecting full-model training.

    ``None`` retains all poses for the final Random Forest result. XGBoost can
    use a smaller sample only to speed up the comparison.
    """
    if tamanho_amostra is None or len(X) <= tamanho_amostra:
        return X
    return X.sample(n=tamanho_amostra, random_state=RANDOM_STATE)


def treinar_modelo_arvore_shap(nome: str, modelo, X: pd.DataFrame, y: pd.Series,
                                out_dir: Path, shap_sample_size: int | None = None) -> pd.Series | None:
    """Train a tree model (RF or XGBoost), calculate SHAP (or fall back to
    feature_importances_ when SHAP is unavailable), and save summary and
    dependence plots. Return the sorted importance series (mean |SHAP|), or
    None if the model is unavailable."""
    if modelo is None:
        print(f"[warning] {nome} is not installed -- skipping this Approach A step.")
        return None

    sub_dir = out_dir / nome.lower()
    sub_dir.mkdir(exist_ok=True)

    modelo.fit(X, y)

    if shap is None:
        print(f"\n[warning] 'shap' is not installed. Using {nome} feature_importances_ instead.")
        importancia = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
        importancia.name = "fallback_importance"
        importancia.to_excel(sub_dir / f"importance_{nome.lower()}_fallback.xlsx", header=True)
        print(importancia.head(15))
        return importancia

    X_shap = selecionar_amostra_shap(X, shap_sample_size)
    n_poses_total = len(X)
    n_poses_shap = len(X_shap)
    print(f"[SHAP - {nome}] Calculating values for {n_poses_shap:,} of "
          f"{n_poses_total:,} poses...".replace(",", "."))
    (sub_dir / "shap_sample_info.txt").write_text(
        "Poses used for training: " + str(n_poses_total) + "\n"
        "Poses used for SHAP: " + str(n_poses_shap) + "\n"
        "Sampling: " + ("not applied" if n_poses_shap == n_poses_total else "random, without replacement") + "\n"
        "Configured SHAP limit: " + ("all poses" if shap_sample_size is None else str(shap_sample_size)) + "\n"
        "Random state: " + str(RANDOM_STATE) + "\n",
        encoding="utf-8",
    )

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_shap)

    importancia = pd.Series(np.abs(shap_values).mean(axis=0), index=X_shap.columns).sort_values(ascending=False)
    importancia.name = "mean_abs_shap"
    importancia.to_excel(sub_dir / f"shap_importance_{nome.lower()}.xlsx", header=True)
    print(f"\n[SHAP - {nome}] top 15 residues/interactions most relevant to Energy:")
    print(importancia.head(15))

    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    caminho_base = sub_dir / f"shap_summary_{nome.lower()}"
    salvar_grafico_em_png_e_svg(caminho_base, dpi_png=PNG_SUMMARY_DPI)
    plt.close()

    for feat in importancia.head(TOP_N_DEPENDENCE).index:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_shap, show=False)
        plt.tight_layout()
        nome_arquivo = feat.replace("/", "-").replace(" ", "_")
        caminho_base = sub_dir / f"shap_dependence_{nome.lower()}_{nome_arquivo}"
        salvar_grafico_em_png_e_svg(caminho_base)
        plt.close()

    return importancia


# ----------------------------------------------------------------------
# 6. APPROACH B - LINEAR MODELS (PLS with VIP, LASSO, ElasticNet)
# ----------------------------------------------------------------------
def calcular_vip(pls: PLSRegression, X: np.ndarray) -> np.ndarray:
    """VIP (Variable Importance in Projection) for a fitted PLS model."""
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape

    s = np.diag(t.T @ t @ q.T @ q)   # 1D, length h -- sum of squares explained by component
    total_s = np.sum(s)

    vips = np.zeros(p)
    for i in range(p):
        peso = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * float(peso @ s) / total_s)
    return vips


def treinar_pls(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> pd.Series:
    sub_dir = out_dir / "pls"
    sub_dir.mkdir(exist_ok=True)

    n_componentes = max(1, min(10, X.shape[1] - 1, X.shape[0] - 1))
    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    pls = PLSRegression(n_components=n_componentes)
    pls.fit(X_esc, y)

    vip = calcular_vip(pls, X_esc)
    importancia = pd.Series(vip, index=X.columns).sort_values(ascending=False)
    importancia.name = "VIP_score"
    importancia.to_excel(sub_dir / "vip_scores_pls.xlsx", header=True)
    print(f"\n[PLS] {n_componentes} component(s). Top 15 by VIP score (VIP >= 1 is commonly considered relevant):")
    print(importancia.head(15))

    return importancia, pls


def treinar_linear_esparso(nome: str, modelo_cv, X: pd.DataFrame, y: pd.Series,
                            grupos: pd.Series | None, out_dir: Path) -> pd.Series:
    """LASSO or ElasticNet with CV-selected alpha (grouped where possible).
    Return a sorted |coefficient| series and also save its sign (+/-)."""
    sub_dir = out_dir / nome.lower()
    sub_dir.mkdir(exist_ok=True)

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    # Generate folds manually rather than passing `groups` to .fit, avoiding
    # dependency on metadata routing, which varies across scikit-learn versions.
    # This makes ligand grouping behave consistently in every supported version.
    n_grupos = grupos.nunique() if grupos is not None else 0
    if grupos is not None and n_grupos >= 2:
        splitter = GroupKFold(n_splits=min(MAX_CV_SPLITS, n_grupos))
        cv_splits = list(splitter.split(X_esc, y, groups=grupos))
    else:
        splitter = KFold(n_splits=min(MAX_CV_SPLITS, len(X)), shuffle=True, random_state=RANDOM_STATE)
        cv_splits = list(splitter.split(X_esc, y))

    modelo_cv.set_params(cv=cv_splits)
    modelo_cv.fit(X_esc, y)

    coefs = pd.Series(modelo_cv.coef_, index=X.columns)
    tabela = pd.DataFrame({
        "coefficient": coefs,
        "abs_coefficient": coefs.abs(),
        "direction": np.where(coefs > 0, "+", np.where(coefs < 0, "-", "0")),
    }).sort_values("abs_coefficient", ascending=False)
    tabela.to_excel(sub_dir / f"coefficients_{nome.lower()}.xlsx")

    alpha_txt = f"alpha selected by CV: {modelo_cv.alpha_:.5f}\n"
    (sub_dir / "alpha.txt").write_text(alpha_txt, encoding="utf-8")

    print(f"\n[{nome}] {alpha_txt.strip()}")
    print(f"[{nome}] top 15 by |coefficient| (sign indicates the direction of the Energy effect):")
    print(tabela.head(15))

    return tabela["abs_coefficient"].sort_values(ascending=False)


# ----------------------------------------------------------------------
# 7. SYNTHESIS ACROSS APPROACHES
# ----------------------------------------------------------------------
def sintetizar_rankings(rankings: dict[str, pd.Series], out_dir: Path,
                         modelos_frente_a: list[str], modelos_frente_b: list[str],
                         top_n: int = TOP_N_SINTESE) -> None:
    """Receive {model_name: sorted_descending_importance_series} and build:
       - a comparative rank table by model
       - a classification of each residue as high-confidence / nonlinear-only / linear-only
    """
    sub_dir = out_dir / "synthesis"
    sub_dir.mkdir(exist_ok=True)

    rankings = {k: v for k, v in rankings.items() if v is not None}
    if not rankings:
        print("[warning] no model is available for synthesis.")
        return

    tops = {nome: set(serie.head(top_n).index) for nome, serie in rankings.items()}
    todos_residuos = sorted(set().union(*tops.values()))

    tabela_rank = pd.DataFrame(index=todos_residuos)
    for nome, serie in rankings.items():
        rank_series = pd.Series(range(1, len(serie) + 1), index=serie.index)
        tabela_rank[f"rank_{nome}"] = rank_series.reindex(todos_residuos)

    modelos_a_disponiveis = [m for m in modelos_frente_a if m in tops]
    modelos_b_disponiveis = [m for m in modelos_frente_b if m in tops]

    def n_frentes_presentes(residuo: str) -> tuple[bool, bool]:
        em_a = any(residuo in tops[m] for m in modelos_a_disponiveis)
        em_b = any(residuo in tops[m] for m in modelos_b_disponiveis)
        return em_a, em_b

    classificacoes = []
    for residuo in todos_residuos:
        em_a, em_b = n_frentes_presentes(residuo)
        n_modelos_total = sum(residuo in tops[m] for m in tops)
        if em_a and em_b:
            classe = "High confidence (Approach A + Approach B)"
        elif em_a and not em_b:
            classe = "Nonlinear synergy (Approach A only -- see dependence plot)"
        elif em_b and not em_a:
            classe = "Pure linear effect (Approach B only)"
        else:
            classe = "Undefined"
        classificacoes.append((residuo, classe, n_modelos_total))

    tabela_rank["classification"] = [c[1] for c in classificacoes]
    tabela_rank["n_models_in_top"] = [c[2] for c in classificacoes]
    tabela_rank = tabela_rank.sort_values(
        ["n_models_in_top"], ascending=False
    )
    tabela_rank.to_excel(sub_dir / "ranking_comparison.xlsx")

    resumo_txt_linhas = [f"Synthesis across approaches (top {top_n} from each model)\n"]
    for classe in [
        "High confidence (Approach A + Approach B)",
        "Nonlinear synergy (Approach A only -- see dependence plot)",
        "Pure linear effect (Approach B only)",
    ]:
        residuos_classe = [r for r, c, _ in classificacoes if c == classe]
        resumo_txt_linhas.append(f"\n{classe} ({len(residuos_classe)} residue(s)):")
        resumo_txt_linhas.extend(f"  - {r}" for r in residuos_classe)

    resumo_txt = "\n".join(resumo_txt_linhas)
    (sub_dir / "synthesis_summary.txt").write_text(resumo_txt, encoding="utf-8")
    print("\n[SYNTHESIS]\n" + resumo_txt)


# ----------------------------------------------------------------------
# 8. LIGAND SUMMARY
# ----------------------------------------------------------------------
def resumo_por_ligante(matriz: pd.DataFrame, col_ligante: str, out_dir: Path) -> None:
    resumo = matriz.groupby(col_ligante)[COL_ENERGIA].agg(
        n_poses="count", mean_energy="mean", best_energy="min"
    ).sort_values("best_energy")
    resumo.to_excel(out_dir / "ligand_summary.xlsx")
    print("\n[ligand summary] (sorted by best energy)")
    print(resumo)


def selecionar_arquivo_via_janela() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError(
            "tkinter is unavailable in this Python installation. "
            "Run the script by passing the path manually:\n"
            "    python 'LinearForest OmniModel.py' \"path\\to\\workbook.xlsx\""
        )

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


# ----------------------------------------------------------------------
# 9. MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Random Forest + XGBoost + PLS + LASSO/ElasticNet across multi-ligand poses"
    )
    parser.add_argument("xlsx_path", type=str, nargs="?", default=None,
                         help="Path to the .xlsx workbook. If omitted, a file-selection window opens.")
    parser.add_argument("--min-freq", type=float, default=MIN_FREQ)
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else selecionar_arquivo_via_janela()
    out_dir = xlsx_path.parent / f"LinearForest - {xlsx_path.stem}"
    out_dir.mkdir(exist_ok=True)

    print(f"[1/6] Loading {xlsx_path.name} ...")
    df, col_ligante = carregar_dados(xlsx_path)

    print("[2/6] Detecting pose blocks ...")
    df = detectar_poses(df, col_ligante)
    print(f"   -> {df['PoseGlobalID'].nunique()} poses detected in total.")

    print("[3/6] Building feature matrix ...")
    matriz = montar_matriz(df, col_ligante)
    matriz.to_excel(out_dir / "feature_matrix.xlsx", index=False)

    feat_cols_todas = [c for c in matriz.columns if c not in ("PoseGlobalID", COL_ENERGIA, col_ligante)]
    feat_cols = filtrar_features_raras(matriz, feat_cols_todas, args.min_freq)

    X = matriz[feat_cols]
    y = matriz[COL_ENERGIA]
    grupos = matriz[col_ligante] if col_ligante else None
    splits, cabecalho_cv, aviso_cv = build_validation_plan(
        X, y, grupos, max_splits=MAX_CV_SPLITS, random_state=RANDOM_STATE
    )
    if aviso_cv:
        print(f"[warning] {aviso_cv}")
    planilha = xlsx_path.name
    tipo_entrada = "Binary Res-Int"
    folds = assign_folds_to_poses(matriz["PoseGlobalID"], splits)
    folds.insert(0, "workbook", planilha)
    folds.insert(1, "input_type", tipo_entrada)
    folds.to_csv(out_dir / "folds_validation.csv", index=False, encoding="utf-8-sig")
    metricas_por_modelo = []

    rankings: dict[str, pd.Series] = {}

    print("\n[4/6] Approach A (nonlinear): Random Forest and XGBoost + SHAP ...")

    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    metricas_por_modelo.append(
        validar_modelo(rf, X, y, splits, cabecalho_cv, out_dir, "Random Forest", planilha, tipo_entrada)
    )
    rankings["RandomForest"] = treinar_modelo_arvore_shap(
        "RandomForest", rf, X, y, out_dir,
        shap_sample_size=RF_SHAP_SAMPLE_SIZE,
    )

    if XGBRegressor is not None:
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
            validar_modelo(xgb, X, y, splits, cabecalho_cv, out_dir, "XGBoost", planilha, tipo_entrada)
        )
        rankings["XGBoost"] = treinar_modelo_arvore_shap(
            "XGBoost", xgb, X, y, out_dir,
            shap_sample_size=XGB_SHAP_SAMPLE_SIZE,
        )
    else:
        print("[warning] xgboost is not installed -- skipping XGBoost. Install it with: "
              "pip install xgboost --break-system-packages")
        rankings["XGBoost"] = None

    print("\n[5/6] Approach B (linear): PLS (VIP) and LASSO/ElasticNet ...")

    importancia_pls, pls_modelo = treinar_pls(X, y, out_dir)
    rankings["PLS"] = importancia_pls

    lasso_cv = LassoCV(random_state=RANDOM_STATE, n_jobs=-1, max_iter=10000)
    rankings["LASSO"] = treinar_linear_esparso("LASSO", lasso_cv, X, y, grupos, out_dir)

    elasticnet_cv = ElasticNetCV(random_state=RANDOM_STATE, n_jobs=-1, max_iter=10000, l1_ratio=[.1, .5, .7, .9, .95, 1])
    rankings["ElasticNet"] = treinar_linear_esparso("ElasticNet", elasticnet_cv, X, y, grupos, out_dir)

    print("\n[6/6] Synthesizing rankings across approaches ...")
    sintetizar_rankings(
        rankings, out_dir,
        modelos_frente_a=["RandomForest", "XGBoost"],
        modelos_frente_b=["PLS", "LASSO", "ElasticNet"],
    )

    if col_ligante:
        resumo_por_ligante(matriz, col_ligante, out_dir)

    colunas_metricas = [
        "workbook", "input_type", "model", "fold", "n_train", "n_test", "r2", "mae_kcal_mol"
    ]
    metricas = [tabela for tabela in metricas_por_modelo if not tabela.empty]
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
