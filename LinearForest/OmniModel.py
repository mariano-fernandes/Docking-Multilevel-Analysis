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
    parser = argparse.ArgumentParser(description="Run the complete Res-Int analysis and baseline")
    parser.add_argument("xlsx_path", nargs="?", default=None, help="Path to the .xlsx workbook")
    parser.add_argument("--min-freq", type=float, default=MIN_FREQ,
                        help="Minimum frequency applied only to the Res-Int model")
    args = parser.parse_args()
    xlsx_path = Path(args.xlsx_path) if args.xlsx_path else select_file_with_dialog()
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    run_step(
        "complete Res-Int model",
        [sys.executable, str(MODELS_DIR / "LinearForest OmniModel.py"), str(xlsx_path),
         "--min-freq", str(args.min_freq)],
    )
    run_step(
        "interaction-count baseline",
        [sys.executable, str(MODELS_DIR / "Baseline - Interaction Count.py"), str(xlsx_path)],
    )
    print("\nComplete workflow finished.")


if __name__ == "__main__":
    main()
