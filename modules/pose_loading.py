from pathlib import Path
import math
import re
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolAlign, ChemicalFeatures
from rdkit.Geometry import Point3D

"""Unicode-safe reading of CALS reference ligands."""

SUPPORTED_FORMATS = {".mol", ".sdf", ".mol2", ".pdb"}

def _read_ligand_text(path: Path) -> str:
    """Read molecular blocks without passing a Unicode path to RDKit."""

    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Older molecular files often use ANSI encoding; latin-1 preserves
        # every byte and avoids changing the block coordinates.
        return data.decode("latin-1")

def load_ligand_template(ligand_file: str | Path):
    """Load the first supported molecule without relying on an RDKit path.

    RDKit C++ readers may fail on Windows when paths contain Unicode
    characters. The ``*Block`` equivalents receive the content in memory and
    preserve the same chemistry, including ``removeHs=True``.
    """

    path = Path(ligand_file)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        raise ValueError("Supported formats: MOL, SDF, MOL2, or PDB.")
    if not path.is_file():
        raise FileNotFoundError(f"Ligand file not found: {path}")

    try:
        block = _read_ligand_text(path)
    except OSError as exc:
        raise OSError(f"Could not read ligand: {path}") from exc

    if extension == ".sdf":
        # CALS uses only the first molecule in the SDF.
        block = block.split("$$$$", 1)[0]

    if extension in {".mol", ".sdf"}:
        molecule = Chem.MolFromMolBlock(block, removeHs=True)
    elif extension == ".mol2":
        molecule = Chem.MolFromMol2Block(block, removeHs=True)
    else:
        molecule = Chem.MolFromPDBBlock(block, removeHs=True)

    if molecule is None:
        raise ValueError(f"Could not read ligand: {path}")
    return molecule

def read_vina_poses(file_path):
    poses = []
    current_pose = None
    current_h_coords = []

    with open(file_path, "r") as f:

        for line in f:

            line = line.strip()

            # START OF A NEW POSE
            if line.startswith("MODEL"):

                current_pose = {
                    "pose": int(line.split(" ")[1]),
                    "energy": None,
                    "rmsd_lb": None,
                    "rmsd_ub": None,
                    "atoms": []
                }
                current_h_coords = []

            # ENERGY AND RMSDs
            elif line.startswith("REMARK VINA RESULT:"):

                parts = line.split()

                current_pose["energy"] = float(parts[3])
                current_pose["rmsd_lb"] = float(parts[4])
                current_pose["rmsd_ub"] = float(parts[5])

            # ATOM COORDINATES
            elif line.startswith("ATOM") or line.startswith("HETATM"):

                atom_name = line[12:16].strip()
                element = line[77:78].strip()

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # Vina/AutoDock preserves only polar hydrogens (bonded to
                # N/O/S) in PDBQT. Keep their positions for hydrogen-bond
                # classification, but do not include them in RMSD heavy atoms.
                if element.upper() == "H":

                    current_h_coords.append(
                        np.array([x, y, z], dtype=float)
                    )
                    continue

                atom_name = line[12:16].strip()
                atom_type = line[77:].strip()

                current_pose["atoms"].append({

                    "name": atom_name,
                    "type": atom_type,
                    "element": element,
                    "coord": np.array([x, y, z], dtype=float)

                })

            # END OF POSE
            elif line.startswith("ENDMDL"):

                current_pose["polar_h"] = assign_polar_hydrogens(
                    current_h_coords,
                    current_pose["atoms"]
                )

                poses.append(current_pose)
                current_pose = None
    return poses

def assign_polar_hydrogens(h_coords, heavy_atoms, bond_cutoff=1.3):
    """Associate each polar PDBQT hydrogen with its nearest N/O/S heavy atom.

    That atom is its donor and is used for the donor-H-acceptor angle in
    hydrogen-bond classification.
    """

    polar_h = []

    for h_coord in h_coords:

        best_idx = None
        best_dist = None

        for idx, atom in enumerate(heavy_atoms):

            if atom["element"].upper() not in ("N", "O", "S"):
                continue

            dist = np.linalg.norm(h_coord - atom["coord"])

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None and best_dist <= bond_cutoff:

            polar_h.append({
                "coord": h_coord,
                "parent_idx": best_idx,
            })

    return polar_h

def pose_to_template_mol(pose_atoms, template):

    mol = Chem.Mol(template)

    conf = Chem.Conformer(mol.GetNumAtoms())

    if len(pose_atoms) != mol.GetNumAtoms():
        raise ValueError(
            f"Different atom counts: "
            f"template={mol.GetNumAtoms()} "
            f"pose={len(pose_atoms)}"
        )

    for i, atom in enumerate(pose_atoms):

        x, y, z = atom["coord"]

        conf.SetAtomPosition(
            i,
            Point3D(float(x), float(y), float(z))
        )

    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    return mol

def canonicalize_pose_atom_order(pose, template):
    """Align each pose to the template by PDB name, never by a silent assumption."""
    template_names = []
    for atom in template.GetAtoms():
        info = atom.GetPDBResidueInfo()
        template_names.append(info.GetName().strip() if info else "")
    pose_names = [atom["name"].strip() for atom in pose["atoms"]]
    if (not all(template_names) or len(set(template_names)) != len(template_names)
            or len(set(pose_names)) != len(pose_names)
            or set(template_names) != set(pose_names)):
        raise ValueError(
            "Ambiguous template--pose mapping: PDB atom names must be unique "
            "and match exactly between the ligand and pose."
        )
    original_index = {name: index for index, name in enumerate(pose_names)}
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(original_index[name] for name in template_names)}
    pose["atoms"] = [pose["atoms"][original_index[name]] for name in template_names]
    for hydrogen in pose.get("polar_h", []):
        hydrogen["parent_idx"] = old_to_new[hydrogen["parent_idx"]]

def read_pocket(pocket_file):

    pocket_atoms = []

    with open(pocket_file, "r") as f:

        for line in f:

            if line.startswith("ATOM"):

                atom = {
                    "atom_name": line[12:16].strip(),
                    "residue": line[17:20].strip(),
                    "chain": line[21:22].strip(),
                    "resnum": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54])
                }

                pocket_atoms.append(atom)

    return  pocket_atoms

def read_protein(protein_file, select_altloc=False):

    protein_atoms = []

    if select_altloc:
        atom_lines = _selected_protein_atom_lines(protein_file)
    else:
        with open(protein_file, "r") as f:
            atom_lines = [line for line in f if line.startswith("ATOM")]

    for line in atom_lines:

        element = (line[76:78].strip() or line[12:13].strip()).upper()

        # Explicit hydrogens are read in a separate structure for H-bonds;
        # they cannot participate in contacts, rings, or pockets.
        if element == "H":
            continue

        protein_atoms.append({
            "atom_name": line[12:16].strip(),
            "residue": line[17:20].strip(),
            "resnum": int(line[22:26]),
            "chain": line[21].strip(),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54])
        })

    return protein_atoms

def _selected_protein_atom_lines(protein_file):
    """Select one conformation per atom, following the default PLIP behavior.

    Atoms without `altLoc` take priority. For alternatives, use A (or 1),
    otherwise the one with the highest occupancy. This prevents mixing A/B
    conformations of the same residue in a ring or donor group.
    """

    selected = {}
    order = []

    with open(protein_file, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16]
            element = (line[76:78].strip() or atom_name[:1]).upper()
            # Writers such as OpenBabel may name multiple H atoms of the same
            # residue simply "H". Preserve the serial in the key so altLoc
            # selection does not collapse them.
            key = (
                line[21], line[22:26], line[26], atom_name,
                line[6:11] if element == "H" else "",
            )
            altloc = line[16]
            occupancy_text = line[54:60].strip()
            occupancy = float(occupancy_text) if occupancy_text else 0.0
            priority = 3 if altloc == " " else 2 if altloc in ("A", "1") else 1
            candidate = (priority, occupancy, line)

            if key not in selected:
                selected[key] = candidate
                order.append(key)
            elif candidate[:2] > selected[key][:2]:
                selected[key] = candidate

    return [selected[key][2] for key in order]

def get_explicit_protein_polar_hydrogens(protein_file, protein_atoms):
    """Map explicit PDB hydrogens to their N/O/S donors.

    The function uses the same altLoc selection as the PLIP profile. If the
    protein is already protonated, those coordinates take precedence over any
    virtual hydrogen reconstructed by F2.
    """

    donor_atoms = [
        atom for atom in protein_atoms
        if atom["atom_name"] and atom["atom_name"][0] in ("N", "O", "S")
    ]
    hydrogens = {}

    for line in _selected_protein_atom_lines(protein_file):

        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()

        if element != "H":
            continue

        try:
            residue = line[17:20].strip()
            resnum = int(line[22:26])
            chain = line[21].strip()
            h_coord = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ], dtype=float)
        except ValueError:
            continue

        candidates = [
            atom for atom in donor_atoms
            if atom["chain"] == chain
            and atom["resnum"] == resnum
            and atom["residue"] == residue
        ]

        if not candidates:
            continue

        parent = min(
            candidates,
            key=lambda atom: np.linalg.norm(
                h_coord - np.array([atom["x"], atom["y"], atom["z"]])
            ),
        )
        parent_coord = np.array([
            parent["x"], parent["y"], parent["z"]
        ], dtype=float)

        if np.linalg.norm(h_coord - parent_coord) <= 1.3:
            key = (
                parent["chain"], parent["resnum"],
                parent["residue"], parent["atom_name"],
            )
            hydrogens.setdefault(key, []).append(h_coord)

    return hydrogens

def assign_bond_orders_deterministically(reference, target):
    """Transfer bond orders without relying on an arbitrary match.

    Symmetric ligands may have more than one valid topological correspondence.
    RDKit chooses the first and emits a warning; explicitly choose the lowest
    index correspondence to keep the result reproducible and preserve target
    coordinates.
    """

    reference_original = Chem.Mol(reference)
    reference_match = Chem.Mol(reference)
    target = Chem.Mol(target)
    matching = target.GetSubstructMatch(reference_match)

    if matching:
        return target

    for bond in target.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)

    for bond in reference_match.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)

    for atom in reference_match.GetAtoms():
        atom.SetFormalCharge(0)

    matches = target.GetSubstructMatches(reference_match, uniquify=False)

    if not matches:
        raise ValueError("No correspondence found between the SMILES and ligand.")

    matching = min(matches)

    for bond in reference_original.GetBonds():
        target_bond = target.GetBondBetweenAtoms(
            matching[bond.GetBeginAtomIdx()], matching[bond.GetEndAtomIdx()]
        )
        target_bond.SetBondType(bond.GetBondType())
        target_bond.SetIsAromatic(bond.GetIsAromatic())

    for atom in reference_original.GetAtoms():
        target_atom = target.GetAtomWithIdx(matching[atom.GetIdx()])
        target_atom.SetHybridization(atom.GetHybridization())
        target_atom.SetIsAromatic(atom.GetIsAromatic())
        target_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        target_atom.SetFormalCharge(atom.GetFormalCharge())

    Chem.SanitizeMol(target)
    return target

def template_reference_coordinates(template):
    """Return the template's original MOPAC pose in fixed atom order."""
    conformer = template.GetConformer()
    return np.asarray([
        [
            conformer.GetAtomPosition(index).x,
            conformer.GetAtomPosition(index).y,
            conformer.GetAtomPosition(index).z,
        ]
        for index in range(template.GetNumAtoms())
    ], dtype=float)
