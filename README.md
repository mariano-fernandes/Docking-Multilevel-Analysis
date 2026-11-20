# Cluster Analysis and Lead Selection (CALS)

**CALS** is a local Python tool for organizing and analyzing molecular docking poses. It reads AutoDock Vina results, classifies poses by pocket and site, calculates conformational metrics, identifies protein-ligand interactions, and produces consolidated Excel workbooks for inspection and subsequent analysis.

> **Project status:** Alpha release. Use the results as analytical support and review structures, parameters, and conclusions before any scientific or decision-making use.

## Citation

**Please cite:**

> Mariano-Fernandes, L.; Marinho, M. M.; Marinho, E. S. Multilevel interaction networks and machine learning identify target-specific interaction patterns of lapachone derivatives with potential antileishmanial activity. DOI:

## Main features

- Batch processing of multiple ligands, with one Vina output folder per ligand.
- Reading of multiple AutoDock Vina output files (`.pdbqt`) in each ligand folder.
- Pose classification by pockets and sites, including clustering refinement.
- Conformational and positional metrics by pose, site, and pocket.
- Protein-ligand interaction identification, including hydrogen bonds, hydrophobic contacts, pi interactions, salt bridges, and halogen bonds when applicable.
- BP-selection ranking, incidence tables for networks, and a machine-learning input dataset.
- Export of an Excel workbook containing all poses, including audit records for poses without interactions.

## Requirements

- Windows with a graphical environment because file selection uses Tkinter dialogs.
- Python **3.14.5**, the version used by the current project virtual environment.
- Access to the structural files and docking results described below.

Python dependencies are listed in [`requirements.txt`](requirements.txt): NumPy, pandas, scikit-learn, SciPy, openpyxl, and RDKit.

## Installation

Install Python 3.14 for Windows and ensure the `py` launcher is available. Then open PowerShell in the folder that contains `CALS.py` and run:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the virtual environment and dependencies already exist, run only the execution command in the next section.

## Input files

| Input | Accepted format | Purpose in CALS |
| --- | --- | --- |
| Pockets | One or more `.pdb` files | Define the pockets used to classify poses. |
| Protein | One `.pdb` file | Receptor structure used for contact and interaction analysis. |
| Docking result folders | One or more folders | Each folder corresponds to one ligand and directly contains its `.pdbqt` files. |
| Reference ligands | `.pdb`, `.mol`, `.sdf`, or `.mol2` | One structure per Vina folder, used as the structural and connectivity reference. |
| SMILES | Required `.txt` | One file per Vina folder. Its first non-empty line provides the SMILES string. |

Atoms and connectivity in the reference structure must be compatible with the docking poses for RMSD and interaction calculations to run correctly.

## Running CALS

```powershell
.\.venv\Scripts\python.exe .\CALS.py
```

The application opens selection dialogs in this order:

1. Select one or more pocket PDB files.
2. Select the protein/receptor PDB file.
3. In the tree selector, use Ctrl or Shift to select all required Vina folders.
4. Select all reference structures.
5. Select all SMILES TXT files.
6. Select the single destination folder for Excel workbooks.

The Vina folder, reference, and SMILES file must have matching names without case sensitivity. For example: `LAP1`, `LAP1.pdb`, and `LAP1.txt`. CALS validates every pair before starting. If any ligand fails during processing, the batch stops and completed workbooks are preserved.

## Output files

CALS creates one Excel workbook per ligand in the selected destination folder:

```text
CALS - <Vina-folder-name>.xlsx
```

If a workbook with the same name already exists, CALS creates a numbered variant without overwriting the previous result. The batch-level workbook is named `CALS - General.xlsx`.

## Excel workbook sheets

### Classification and summaries

| Sheet | Content |
| --- | --- |
| `General Information` | Per-pose identity, pocket, site, docking energy, Vina RMSD, and centroid coordinates. |
| `Pockets` | Residues that compose each supplied pocket. |
| `Pocket Summary` | Aggregate statistics by pocket. |
| `Site Summary` | Aggregate statistics by site within each pocket. |
| `Centroids` | Centroids of supplied pockets and classified sites. |

### Interactions

| Sheet | Content |
| --- | --- |
| `Interactions by Type` | Identified interactions for each pose, including residues, atoms, and geometric parameters when available. |
| `Interactions by Pocket` | Residue interaction frequency aggregated by pocket. |
| `Interactions by Site` | Residue interaction frequency aggregated by site. |

A pose without detected molecular interactions appears in `Interactions by Type` as **No interactions detected**, with Pocket and Site set to `N/A`. This record is not a chemical interaction and the pose does not enter downstream analyses.

### Conformational analysis and ranking

| Sheet | Content |
| --- | --- |
| `Conformational Analysis` | Per-pose energy, `Structural Reproducibility (%)`, `Functional Reproducibility (%)`, Score BP, and retained population fields. |
| `Ranking BP Selection` | Poses ranked by local population, energy window, `Structural Reproducibility (%)`, `Functional Reproducibility (%)`, and Score BP. |
| `Score BP` | Audit of energy decay, `Structural Reproducibility (%)`, `Functional Reproducibility (%)`, and Score BP. |

`Benchmarking` is an optional worksheet in this group. Set `INCLUDE_BENCHMARKING_SHEET = True` in `modules/engine.py` to include it after `Score BP`.

Within each pocket-site combination, poses up to 1.0 kcal/mol above the best local energy are considered energetically competitive. The ranking uses **Score BP**, the geometric mean of Energy (exp), Structural Reproducibility (%), and Functional Reproducibility (%). Energy (exp) is calculated as `exp(-(ΔG - ΔGminimum)/2)` within the same local scope.

### Networks and machine learning

| Sheet | Content |
| --- | --- |
| `Structural` | Pose-residue incidence matrix. |
| `Functional` | Pose-residue/interaction incidence matrix. |
| `Edges Structural` and `Nodes Structural` | Structural-network edges and nodes. |
| `Edges Functional` and `Nodes Functional` | Functional-network edges and nodes. |
| `Structural by Site` | Structural residues and edges with local ligand/pocket/site frequency, centrality, and weights. |
| `Functional by Site` | Functional residue-interaction entries and edges with local ligand/pocket/site frequency, centrality, and weights. |
| `ML Analysis` | Pose-residue/interaction-energy dataset for external machine-learning analysis. |

These sheets contain only poses with real interactions. Poses without detected interactions do not generate nodes, edges, or ML Analysis rows.

### Coordinates

| Sheet | Content |
| --- | --- |
| `Pose Coordinates` | Pocket/site identification and centroid coordinates for all poses. |

## Isolated poses and poses without interactions

All read poses are retained for auditing. CALS distinguishes two cases:

- **Singleton group with a real interaction:** the pose enters summaries, Conformational Analysis, Ranking BP Selection, and the network analyses.
- **Pose without a detected interaction:** the pose is assigned neither a Pocket nor a Site; both fields are `N/A`. It appears only in `General Information`, `Interactions by Type`, and `Pose Coordinates`; it does not participate in clustering, summaries, ranking, centroids, networks, or ML Analysis.

## Project structure

```text
CALS.py               # Interface, batch selection, and session lifecycle
modules/               # Reading, classification, RMSD, interactions, ranking, and export
requirements.txt       # Python dependencies
README.md              # Documentation
```

## Limitations and troubleshooting

| Situation | Recommended action |
| --- | --- |
| The selection window does not open | Run CALS in a Windows session with a graphical interface; do not run it in a purely headless environment. |
| RDKit or another dependency installation fails | Confirm Python 3.14.5 and recreate the virtual environment before repeating `pip install -r requirements.txt`. |
| Ligand/pose compatibility error | Confirm that the reference ligand represents the same molecule and has the same atom correspondence as the PDBQT poses. |
| SMILES was not selected or does not match the Vina folder | The batch does not start; select a matching TXT file for every ligand. |
| Network or ML Analysis sheets have no rows | Check `Interactions by Type`. Poses without detected interactions are intentionally excluded from these sheets. |
| Pocket or Site is `N/A` | The pose had no detected interaction, is retained only for auditing, and does not participate in downstream analyses. |

## Notes

- CALS does not replace structural inspection, experimental validation, or critical evaluation of docking parameters.
