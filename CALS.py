"""CALS interface for batch ligand analyses."""

import os
import shutil
import string
from pathlib import Path
from tkinter import Tk, TclError, Toplevel, filedialog, messagebox, ttk

from modules.engine import BatchAnalysisError, run_batch_analysis


REFERENCE_SUFFIXES = {".pdb", ".mol", ".sdf", ".mol2"}


class BatchValidationError(ValueError):
    """Raised when the selected files do not form a valid batch."""


def natural_sort_key(value):
    """Sort names such as LAP1, LAP1a, and LAP2 predictably."""

    import re

    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    ]


class MultiDirectorySelector:
    """Tk directory selector with multiple selection and lazy tree loading."""

    def __init__(self, parent):
        self.parent = parent
        self.result = None
        self.paths_by_item = {}

        # The main root remains hidden during the session. Making this window
        # non-transient prevents Windows from hiding it with its hidden owner.
        self.window = Toplevel()
        self.window.title("Select AutoDock Vina output folders")
        self.window.resizable(True, True)
        self.window.minsize(620, 420)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        instructions = ttk.Label(
            self.window,
            text=(
                "Use Ctrl or Shift to select all ligand folders. Expand "
                "nodes to browse to them."
            ),
            wraplength=580,
            justify="left",
        )
        instructions.pack(fill="x", padx=12, pady=(12, 6))

        content = ttk.Frame(self.window)
        content.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.tree = ttk.Treeview(content, selectmode="extended", show="tree")
        scrollbar = ttk.Scrollbar(
            content, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewOpen>>", self.populate_opened_item)

        buttons = ttk.Frame(self.window)
        buttons.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(
            side="right"
        )
        ttk.Button(buttons, text="Confirm selection", command=self.confirm).pack(
            side="right", padx=(0, 8)
        )

        self.populate_roots()

    @staticmethod
    def available_roots():
        """Return available Windows drives without assuming drive letters."""

        return tuple(
            candidate
            for letter in string.ascii_uppercase
            if (candidate := Path(f"{letter}:/")).is_dir()
        )

    def add_item(self, parent, path):
        item = self.tree.insert(parent, "end", text=str(path), open=False)
        self.paths_by_item[item] = path
        # Delay reading contents until the user expands a node so slow network
        # drives do not delay opening the selector.
        self.tree.insert(item, "end", text="Loading...")
        return item

    def populate_roots(self):
        for root in self.available_roots():
            self.add_item("", root)

    def populate_opened_item(self, _event):
        item = self.tree.focus()
        path = self.paths_by_item.get(item)
        if path is None:
            return

        children = self.tree.get_children(item)
        if not children or children[0] in self.paths_by_item:
            return

        for child in children:
            self.tree.delete(child)

        try:
            directories = sorted(
                (child for child in path.iterdir() if child.is_dir()),
                key=lambda child: natural_sort_key(child.name),
            )
        except OSError as exc:
            messagebox.showwarning(
                "Folder unavailable",
                f"Could not list the folder:\n{path}\n\n{exc}",
                parent=self.window,
            )
            return

        for directory in directories:
            self.add_item(item, directory)

    def confirm(self):
        selected_paths = {
            self.paths_by_item[item]
            for item in self.tree.selection()
            if item in self.paths_by_item
        }
        if not selected_paths:
            messagebox.showwarning(
                "No folder selected",
                "Select at least one Vina results folder.",
                parent=self.window,
            )
            return

        self.result = tuple(
            str(path)
            for path in sorted(
                selected_paths, key=lambda path: natural_sort_key(path.name)
            )
        )
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()

    def show(self):
        self.window.deiconify()
        self.window.update_idletasks()
        self.window.lift()
        try:
            self.window.focus_force()
        except TclError:
            pass
        self.window.grab_set()
        self.parent.wait_window(self.window)
        return self.result


def select_target_files():
    """Select persistent target files for the current session."""

    pocket_files = filedialog.askopenfilenames(
        title="Select pocket files",
        filetypes=[("PDB files (.pdb)", "*.pdb")],
    )
    if not pocket_files:
        return None

    protein_file = filedialog.askopenfilename(
        title="Select the protein file without residues",
        filetypes=[("PDB files (.pdb)", "*.pdb")],
    )
    if not protein_file:
        return None

    return tuple(pocket_files), protein_file


def select_batch_files(root):
    """Select all ligand-specific inputs once."""

    vina_folders = MultiDirectorySelector(root).show()
    if not vina_folders:
        return None

    reference_files = filedialog.askopenfilenames(
        title="Select all ligand reference poses",
        filetypes=[
            ("Ligand structures", "*.pdb *.mol *.sdf *.mol2"),
            ("PDB (.pdb)", "*.pdb"),
            ("Mol (.mol)", "*.mol"),
            ("SDF (.sdf)", "*.sdf"),
            ("Mol2 (.mol2)", "*.mol2"),
        ],
    )
    if not reference_files:
        return None

    smiles_files = filedialog.askopenfilenames(
        title="Select all ligand SMILES files",
        filetypes=[("Text files (.txt)", "*.txt")],
    )
    if not smiles_files:
        return None

    output_folder = filedialog.askdirectory(
        title="Select the destination folder for Excel files"
    )
    if not output_folder:
        return None

    return vina_folders, tuple(reference_files), tuple(smiles_files), output_folder


def index_paths(paths, name_for_path):
    """Index paths by ligand identifier and identify duplicates."""

    indexed = {}
    duplicates = {}

    for raw_path in paths:
        path = Path(raw_path)
        key = name_for_path(path).casefold()
        if key in indexed:
            duplicates.setdefault(key, [indexed[key]]).append(path)
        else:
            indexed[key] = path

    return indexed, duplicates


def direct_vina_files(folder):
    """List PDBQTs only at the first level of a ligand folder."""

    try:
        return tuple(
            sorted(
                (
                    child
                    for child in folder.iterdir()
                    if child.is_file() and child.suffix.casefold() == ".pdbqt"
                ),
                key=lambda child: natural_sort_key(child.name),
            )
        )
    except OSError as exc:
        raise BatchValidationError(
            f"Could not read Vina folder '{folder}': {exc}"
        ) from exc


def validate_batch(vina_folders, reference_files, smiles_files):
    """Validate and build folder/reference/SMILES pairs before analysis."""

    folders, folder_duplicates = index_paths(vina_folders, lambda path: path.name)
    references, reference_duplicates = index_paths(
        reference_files, lambda path: path.stem
    )
    smiles, smiles_duplicates = index_paths(smiles_files, lambda path: path.stem)

    errors = []
    for label, duplicates in (
        ("Vina folders", folder_duplicates),
        ("references", reference_duplicates),
        ("SMILES files", smiles_duplicates),
    ):
        for name, paths in sorted(
            duplicates.items(), key=lambda item: natural_sort_key(item[0])
        ):
            errors.append(
                f"Duplicate name in {label} for '{name}': "
                + "; ".join(str(path) for path in paths)
            )

    for folder in folders.values():
        if not folder.is_dir():
            errors.append(f"Vina folder does not exist or is not a directory: '{folder}'.")
    for reference in references.values():
        if not reference.is_file() or reference.suffix.casefold() not in REFERENCE_SUFFIXES:
            errors.append(f"Invalid reference: '{reference}'.")
    for smile in smiles.values():
        if not smile.is_file() or smile.suffix.casefold() != ".txt":
            errors.append(f"Invalid SMILES file: '{smile}'.")

    folder_keys = set(folders)
    reference_keys = set(references)
    smiles_keys = set(smiles)

    for key in sorted(folder_keys - reference_keys, key=natural_sort_key):
        errors.append(f"Missing reference for ligand '{folders[key].name}'.")
    for key in sorted(folder_keys - smiles_keys, key=natural_sort_key):
        errors.append(f"Missing SMILES for ligand '{folders[key].name}'.")
    for key in sorted(reference_keys - folder_keys, key=natural_sort_key):
        errors.append(f"Reference without a matching Vina folder: '{references[key]}'.")
    for key in sorted(smiles_keys - folder_keys, key=natural_sort_key):
        errors.append(f"SMILES without a matching Vina folder: '{smiles[key]}'.")

    vina_by_folder = {}
    for key, folder in folders.items():
        if not folder.is_dir():
            continue
        vina_files = direct_vina_files(folder)
        if not vina_files:
            errors.append(
                f"Vina folder '{folder}' contains no .pdbqt files at its top level."
            )
        vina_by_folder[key] = vina_files

    if errors:
        raise BatchValidationError("\n".join(f"• {error}" for error in errors))

    return tuple(
        {
            "ligand_name": folders[key].name,
            "vina_folder": folders[key],
            "vina_files": tuple(str(path) for path in vina_by_folder[key]),
            "reference_file": str(references[key]),
            "smiles_file": str(smiles[key]),
        }
        for key in sorted(
            folder_keys, key=lambda name: natural_sort_key(folders[name].name)
        )
    )


def build_destination_path(output_folder, ligand_name):
    """Create an unused destination name without overwriting workbooks."""

    output_folder = Path(output_folder)
    base_name = f"CALS - {ligand_name}"
    candidate = output_folder / f"{base_name}.xlsx"
    index = 1

    while candidate.exists():
        candidate = output_folder / f"{base_name} ({index}).xlsx"
        index += 1

    return candidate


def move_excel_to_destination(source_file, output_folder, ligand_name):
    """Move an individual workbook without changing engine export logic."""

    source = Path(source_file)
    destination_folder = Path(output_folder)
    if source.parent.resolve() == destination_folder.resolve():
        return str(source)

    destination = build_destination_path(destination_folder, ligand_name)
    shutil.move(str(source), str(destination))
    return str(destination)


def close_tk_root(root):
    """Close the Tkinter root only once, including after cancellation."""

    if root is None:
        return

    try:
        if root.winfo_exists():
            root.quit()
            root.destroy()
    except TclError:
        pass


def run_session():
    """Run a validated batch and stop at the first analysis error."""

    root = Tk()
    root.withdraw()
    pid = os.getpid()
    exit_reason = "session completed"
    completed_files = []
    print(f"CALS: session started (PID={pid}).")

    try:
        target_selection = select_target_files()
        if target_selection is None:
            exit_reason = "target selection canceled"
            return

        batch_selection = select_batch_files(root)
        if batch_selection is None:
            exit_reason = "batch selection canceled"
            return

        pocket_files, protein_file = target_selection
        vina_folders, reference_files, smiles_files, output_folder = batch_selection

        try:
            batches = validate_batch(vina_folders, reference_files, smiles_files)
        except BatchValidationError as exc:
            exit_reason = "invalid batch"
            messagebox.showerror(
                "Invalid batch",
                "No analysis was started. Correct the items below:\n\n"
                f"{exc}",
                parent=root,
            )
            return

        def save_batch_excel(batch, generated_excel):
            ligand_name = batch["ligand_name"]
            final_excel = move_excel_to_destination(
                generated_excel, output_folder, ligand_name
            )
            completed_files.append(final_excel)
            print(f"CALS: workbook saved to: {final_excel}")
            return final_excel

        def save_general_excel(generated_excel):
            final_excel = move_excel_to_destination(
                generated_excel, output_folder, "General"
            )
            completed_files.append(final_excel)
            print(
                "CALS: general workbook saved to: "
                f"{final_excel}"
            )
            return final_excel

        try:
            run_batch_analysis(
                pocket_files,
                protein_file,
                batches,
                on_excel_created=save_batch_excel,
                on_general_excel_created=save_general_excel,
            )
        except BatchAnalysisError as exc:
            ligand_name = exc.ligand_name
            exit_reason = f"analysis error for {ligand_name}"
            completed_text = (
                "\n".join(completed_files)
                if completed_files
                else "No workbook was completed."
            )
            messagebox.showerror(
                "Batch interrupted",
                f"Analysis for ligand '{ligand_name}' failed:\n\n{exc.cause}"
                f"\n\nCompleted workbooks:\n{completed_text}",
                parent=root,
            )
            print(
                f"CALS: batch interrupted at {ligand_name}: "
                f"{exc.cause}"
            )
            return

        messagebox.showinfo(
            "Batch completed",
            "Analysis completed successfully.\n\nGenerated workbooks:\n"
            + "\n".join(completed_files),
            parent=root,
        )
    finally:
        print(
            f"CALS: closing session "
            f"(PID={pid}; reason={exit_reason})."
        )
        close_tk_root(root)
        print(f"CALS: process ready to exit (PID={pid}).")


if __name__ == "__main__":
    run_session()
    raise SystemExit(0)
