# LinearForest

**LinearForest** is the companion machine-learning workflow for the CALS project. It evaluates the association between residue-interaction pairs (`Res-Int`) and AutoDock Vina estimated binding energy, then compares this interaction-profile approach with a baseline based only on the total number of interactions per pose.

> **Project status:** Research workflow. Treat its outputs as analytical support and assess molecular structures, docking settings, data size, validation strategy, and scientific conclusions before use.

## Citation

**Please cite:**

> Mariano-Fernandes, L.; Marinho, M. M.; Marinho, E. S. Multilevel interaction networks and machine learning identify target-specific interaction patterns of lapachone derivatives with potential antileishmanial activity. DOI:

## Main features

- Reads CALS workbooks directly from the `ML Analysis` worksheet.
- Converts pose-level `Res-Int` records into a binary residue-interaction feature matrix.
- Runs Random Forest and, when installed, XGBoost models for the interaction-profile workflow and the interaction-count baseline.
- Provides a complete workflow with SHAP, PLS, LASSO, ElasticNet, ranking synthesis, and ligand summaries.
- Provides a fast comparison workflow without SHAP or figure generation.
- Uses ligand-aware cross-validation when an appropriate ligand/group column is available.
- Exports tables, validation metrics, and figures next to the input workbook.

## Requirements

- Windows with a graphical environment when a workbook path is not provided, because the entry points can open a file-selection dialog.
- Python **3.14.5**, matching the CALS documented environment.
- A CALS-generated Excel workbook containing the `ML Analysis` worksheet.

For the integrated CALS and LinearForest installation, use the root [`../requirements.txt`](../requirements.txt). It contains the shared dependencies with compatible minimum versions plus the LinearForest packages: matplotlib, SHAP, and XGBoost.

[`requirements.txt`](requirements.txt) remains available for a standalone LinearForest installation. CUDA and CuPy are optional and are intentionally not included because the appropriate CuPy package depends on the installed CUDA version.

## Installation

Install Python 3.14 for Windows and ensure that the `py` launcher is available. Open PowerShell in the CALS project root, then run:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This root installation supports both CALS and LinearForest. If the environment already exists, run only one of the execution commands below.

### Optional GPU support

When a compatible CUDA installation and CuPy are available, the code can use GPU processing for XGBoost fold data. Otherwise, it automatically uses the CPU. Install the CuPy build that matches the local CUDA version, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install cupy-cuda12x
```

## Input workbook

Use a CALS-generated `.xlsx` workbook. LinearForest preferentially reads the worksheet named exactly `ML Analysis`; if that worksheet is absent, it reads the first worksheet. The selected worksheet must contain the following columns:

| Column | Required | Purpose |
| --- | --- | --- |
| `Pose` | Yes | Unique pose identifier. |
| `Res-Int` | Yes | Residue-interaction pair identifier; each row represents one detected interaction. |
| `Energy (kcal/mol)` | Yes | AutoDock Vina estimated binding energy; response variable. |
| `Ligand`, `Compound`, or `Compound/Ligand` | Recommended | Group used for ligand-aware cross-validation. |

The legacy group-column names `Ligante`, `Composto`, and `Composto/Ligante` are also accepted for compatibility. `CALS - General.xlsx` is the recommended input for multi-ligand analyses because it includes the `Ligand` column and global pose labels.

`ML Analysis` contains only poses with detected interactions. Poses without interactions are not exported to that worksheet and consequently are not modeled by LinearForest.

## Complete workflow

Run the complete `Res-Int` workflow and the interaction-count baseline from the project root:

```powershell
.\.venv\Scripts\python.exe .\LinearForest\OmniModel.py "C:\path\to\CALS - General.xlsx"
```

If no workbook path is supplied, a file-selection dialog opens:

```powershell
.\.venv\Scripts\python.exe .\LinearForest\OmniModel.py
```

The `--min-freq` option sets the minimum frequency retained for `Res-Int` features; its default is `0.02`:

```powershell
.\.venv\Scripts\python.exe .\LinearForest\OmniModel.py "C:\path\to\workbook.xlsx" --min-freq 0.02
```

## Fast model comparison

Run Random Forest and XGBoost for the `Res-Int` and interaction-count inputs without SHAP or plots:

```powershell
.\.venv\Scripts\python.exe .\LinearForest\modelComparison.py "C:\path\to\CALS - General.xlsx"
```

This command also accepts `--min-freq` and opens a file-selection dialog if the workbook path is omitted.

## Outputs

The complete workflow creates folders beside the selected workbook:

| Folder | Content |
| --- | --- |
| `LinearForest - <workbook>` | Feature matrix; cross-validation reports; Random Forest and XGBoost results; SHAP tables and figures; PLS, LASSO, and ElasticNet results; ranking synthesis; and ligand summaries when a ligand column is available. |
| `Baseline - <workbook>` | Equivalent Random Forest and XGBoost validation using only the total interaction count per pose. |

The complete workflow produces PNG and SVG figures for the Random Forest and XGBoost SHAP analyses. The baseline does not calculate SHAP or produce figures.

The fast comparison creates `Model Comparison - <workbook>` with, among other files:

- `cv_scores_linearforest_rf.txt` and `cv_scores_linearforest_xgboost.txt`
- `cv_scores_baseline_rf.txt` and `cv_scores_baseline_xgboost.txt`
- `cv_metrics_linearforest_by_fold.csv` and `cv_metrics_baseline_by_fold.csv`
- `folds_validation_linearforest.csv` and `folds_validation_baseline.csv`

## Validation

The interaction-profile model uses a binary `Res-Int` matrix. The baseline uses `Interaction_Count`, which counts every interaction row for a pose, including repeated records of the same `Res-Int` value.

When at least two ligand groups are available, validation uses `GroupKFold` so that all poses from the same ligand stay in one group. Otherwise, it uses `KFold` with `random_state=42`; this may yield more optimistic performance estimates. The exported fold CSV files make the assigned splits auditable.

Reports include fold-level and aggregate R² and MAE. R² measures the variation in Vina energy explained by a model. MAE is the mean absolute prediction error, expressed in kcal/mol; lower values indicate closer predictions.

## Limitations

- LinearForest evaluates associations in the supplied CALS interaction data; it does not establish causality or replace structural inspection or experimental validation.
- Results depend on docking quality, interaction detection, feature frequency filtering, sample size, and the validation split.
- Inputs without a suitable ligand/group column use ordinary K-fold validation rather than ligand-aware validation.
- Poses without detected interactions cannot be analyzed because they do not appear in CALS `ML Analysis`.
- XGBoost is skipped when its optional dependency cannot be imported; the remaining supported analyses continue.

## Project structure

```text
LinearForest/
├── OmniModel.py                         # Complete workflow entry point
├── modelComparison.py                   # Fast comparison entry point
├── requirements.txt                     # Standalone LinearForest dependencies
├── README.md                            # This documentation
└── models/
    ├── LinearForest OmniModel.py        # Complete Res-Int modeling workflow
    ├── Baseline - Interaction Count.py  # Complete interaction-count baseline
    ├── LinearForest.py                  # Fast Res-Int comparison
    ├── Baseline.py                      # Fast interaction-count baseline
    └── cv_metrics.py                    # Shared validation metric helpers
```
