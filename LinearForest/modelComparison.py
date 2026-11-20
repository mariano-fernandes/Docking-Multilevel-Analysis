import argparse
from pathlib import Path
import subprocess
import sys


MIN_FREQ = 0.02
MODELS_DIR = Path(__file__).resolve().parent / "models"


def select_file_with_dialog() -> Path:
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


def run_step(name: str, command: list[str]) -> None:
    print(f"\n{'=' * 70}\nStarting: {name}\n{'=' * 70}")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Res-Int and baseline without SHAP")
    parser.add_argument("xlsx_path", nargs="?", default=None, help="Path to the .xlsx workbook")
    parser.add_argument("--min-freq", type=float, default=MIN_FREQ,
                        help="Minimum frequency applied only to Res-Int")
    args = parser.parse_args()
    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else select_file_with_dialog()
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")
    out_dir = xlsx_path.parent / f"Model Comparison - {xlsx_path.stem}"
    out_dir.mkdir(exist_ok=True)

    run_step(
        "simplified Res-Int model",
        [sys.executable, str(MODELS_DIR / "LinearForest.py"), str(xlsx_path),
         "--min-freq", str(args.min_freq), "--out-dir", str(out_dir)],
    )
    run_step(
        "simplified baseline",
        [sys.executable, str(MODELS_DIR / "Baseline.py"), str(xlsx_path),
         "--out-dir", str(out_dir)],
    )
    print(f"\nComparison finished. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
