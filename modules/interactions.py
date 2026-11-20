from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from modules.pose_loading import read_protein, get_explicit_protein_polar_hydrogens

ENABLE_PLIP_VALIDATION = False

PLIP_VALIDATION_DISTANCE_TOLERANCE = 0.15  # Å

contact_cutoff = 4.0

INTERACTION_METHOD = "K3 — explicit H per pose; virtual OH scenario scored separately"

HBOND_DIST_CUTOFF = 4.1          # Å, donor...acceptor (heavy atoms)

HBOND_HA_CUTOFF = 3.2             # Å, hydrogen...acceptor

HBOND_ANGLE_MIN = 90.0           # degrees, donor-H...acceptor angle (at H)

HYDROXYL_SELECTION_MODE = "rdkit"

H_STERIC_WEIGHT = 1.0

H_HBOND_WEIGHT = 1.0

H_ACCEPTANCE_MARGIN = 0.0

N_H_BOND_LENGTH = 1.01           # Å, idealized N-H bond length

OH_H_BOND_LENGTH = 0.96          # Å, idealized O-H bond length

OH_PARENT_H_ANGLE = 109.5        # degrees, tetrahedral C-O-H angle

OH_TORSION_SAMPLES = 36          # 10° rotations around the C-O bond

HYDROXYL_DONOR_PARENTS = {
    ("TYR", "OH"): "CZ",
    ("SER", "OG"): "CB",
    ("THR", "OG1"): "CB",
}

HYDROPHOBIC_DIST_CUTOFF = 4.0    # Å, nonpolar carbon...nonpolar carbon (aligned with PLIP HYDROPH_DIST_MAX)

PISTACK_DIST_CUTOFF = 5.5        # Å, ring centroid...ring centroid

PISTACK_PARALLEL_ANGLE_MAX = 30.0     # degrees, face-to-face stacking

PISTACK_TSHAPED_ANGLE_MIN = 60.0      # degrees, T-shaped stacking

PISTACK_TSHAPED_ANGLE_MAX = 90.0

PISTACK_OFFSET_MAX = 2.0         # Å, maximum lateral offset between ring centroids (PLIP PISTACK_OFFSET_MAX)

CATION_PI_DIST_CUTOFF = 6.0      # Å, ring centroid...cationic center

SALT_BRIDGE_DIST_CUTOFF = 4.0    # Å, anionic center...cationic center

HALOGEN_DIST_CUTOFF = 3.5        # Å, halogen...acceptor

HALOGEN_ANGLE_MIN = 165.0        # degrees, C-X...acceptor angle (near 180°)

PROTEIN_SIDECHAIN_HBOND_ROLES = {
    ("SER", "OG"): {"donor", "acceptor"},
    ("THR", "OG1"): {"donor", "acceptor"},
    ("TYR", "OH"): {"donor", "acceptor"},
    ("CYS", "SG"): {"donor", "acceptor"},
    ("ASN", "OD1"): {"acceptor"},
    ("ASN", "ND2"): {"donor"},
    ("GLN", "OE1"): {"acceptor"},
    ("GLN", "NE2"): {"donor"},
    ("ASP", "OD1"): {"acceptor"},
    ("ASP", "OD2"): {"acceptor"},
    ("GLU", "OE1"): {"acceptor"},
    ("GLU", "OE2"): {"acceptor"},
    ("LYS", "NZ"): {"donor"},
    ("ARG", "NE"): {"donor"},
    ("ARG", "NH1"): {"donor"},
    ("ARG", "NH2"): {"donor"},
    ("HIS", "ND1"): {"donor", "acceptor"},
    ("HIS", "NE2"): {"donor", "acceptor"},
    ("TRP", "NE1"): {"donor"},
    ("MET", "SD"): {"acceptor"},
}

PROTEIN_AROMATIC_RINGS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}

PROTEIN_CATIONIC_GROUPS = {
    "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2", "CZ"],
}

PROTEIN_ANIONIC_GROUPS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}

RESIDUE_FUNCTIONAL_GROUPS = {
    "PHE": "Aromatic", "TYR": "Aromatic", "TRP": "Aromatic",
    "GLY": "Aliphatic", "ALA": "Aliphatic", "VAL": "Aliphatic",
    "LEU": "Aliphatic", "ILE": "Aliphatic", "MET": "Aliphatic",
    "PRO": "Aliphatic",
    "HIS": "Pos. Charged", "LYS": "Pos. Charged", "ARG": "Pos. Charged",
    "ASP": "Neg. Charged", "GLU": "Neg. Charged",
    "SER": "Polar", "THR": "Polar", "CYS": "Polar", "ASN": "Polar",
    "GLN": "Polar",
}

PROTEIN_HYDROPHOBIC_ATOMS = {
    "ALA": ["CB"],
    "VAL": ["CB", "CG1", "CG2"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "PRO": ["CB", "CG"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CB", "CG", "CD2", "CE3", "CZ2", "CZ3", "CH2"],
    "MET": ["CB"],
    "THR": ["CG2"],
    "ASN": ["CB"],
    "GLN": ["CB", "CG"],
    "ASP": ["CB"],
    "GLU": ["CB", "CG"],
    "LYS": ["CB", "CG", "CD"],
    "ARG": ["CB", "CG"],
    "HIS": ["CB"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2"],
}

"""Independent polar-H preparation for CALS G variants.

This module does not read PLIP materials. It only transforms the user-selected
PDB into H coordinates understood by the interaction classifier.
"""

HYDROXYL_DONOR_PARENTS = {
    ("TYR", "OH"): "CZ",
    ("SER", "OG"): "CB",
    ("THR", "OG1"): "CB",
}

def make_temp_directory(prefix: str) -> tempfile.TemporaryDirectory:
    """Return a temporary directory that must remain alive until execution ends."""
    return tempfile.TemporaryDirectory(prefix=prefix)

def protonate_with_rdkit(protein_file, temp_dir: str | Path) -> Path:
    """Generate a PDB with H and 3D coordinates using RDKit only."""
    # RDKit's PDB reader on Windows does not reliably accept Unicode paths
    # (for example, "Protein.pdb"). The temporary ASCII copy prevents the
    # user-selected name from limiting G2B/G3.
    temp_dir = Path(temp_dir)
    source_copy = temp_dir / "protein_input.pdb"
    shutil.copy2(protein_file, source_copy)
    try:
        mol = Chem.MolFromPDBFile(
            str(source_copy), sanitize=False, removeHs=False,
            proximityBonding=True
        )
    except OSError as exc:
        raise RuntimeError(
            "RDKit could not open the selected protein PDB: " + str(exc)
        ) from exc
    if mol is None or mol.GetNumAtoms() == 0:
        raise RuntimeError("RDKit could not read the selected protein PDB.")
    if mol.GetNumConformers() == 0:
        raise RuntimeError("The selected protein does not contain 3D coordinates.")

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Protein structures often have partial PDB connectivity. AddHs can
        # still operate on the loaded graph without full sanitization.
        pass

    try:
        protonated = Chem.AddHs(mol, addCoords=True)
    except Exception as exc:
        raise RuntimeError(
            "RDKit could not add H to the selected protein: " + str(exc)
        ) from exc

    output = Path(temp_dir) / "protein_rdkit_H.pdb"
    # AddHs creates H without PDB metadata (UNL/no chain). Copy the bonded
    # heavy-atom metadata so CALS can associate H with N/O/S in the same
    # chain and residue, as it does for externally protonated PDBs.
    for atom in protonated.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetPDBResidueInfo() is not None:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            continue
        parent_info = neighbors[0].GetPDBResidueInfo()
        if parent_info is None:
            continue
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"H{atom.GetIdx():>3}"[-4:])
        info.SetResidueName(parent_info.GetResidueName())
        info.SetResidueNumber(parent_info.GetResidueNumber())
        info.SetChainId(parent_info.GetChainId())
        info.SetInsertionCode(parent_info.GetInsertionCode())
        info.SetAltLoc(parent_info.GetAltLoc())
        info.SetIsHeteroAtom(parent_info.GetIsHeteroAtom())
        atom.SetMonomerInfo(info)
    try:
        Chem.MolToPDBFile(protonated, str(output))
    except Exception as exc:
        raise RuntimeError(
            "RDKit could not write the temporary protonated protein: " + str(exc)
        ) from exc
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("RDKit did not generate a usable protonated PDB.")
    return output

def hydroxyl_key(atom: dict):
    return (atom["chain"], atom["resnum"], atom["residue"], atom["atom_name"])

def is_hydroxyl(atom: dict) -> bool:
    return (atom["residue"], atom["atom_name"]) in HYDROXYL_DONOR_PARENTS

def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-10:
        raise ValueError("Zero vector during OH reconstruction.")
    return vector / length

def _orthogonal(axis: np.ndarray) -> np.ndarray:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, seed))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(axis, seed))

def hydroxyl_rotamers(protein_atoms: list[dict], donor: dict,
                      bond_length: float = 0.96,
                      parent_angle: float = 109.5,
                      samples: int = 36) -> list[np.ndarray]:
    """Create equally spaced tetrahedral rotations for an OH group."""
    parent_name = HYDROXYL_DONOR_PARENTS[(donor["residue"], donor["atom_name"])]
    parent = next((atom for atom in protein_atoms if
                   atom["chain"] == donor["chain"] and
                   atom["resnum"] == donor["resnum"] and
                   atom["residue"] == donor["residue"] and
                   atom["atom_name"] == parent_name), None)
    if parent is None:
        return []

    oxygen = np.array([donor["x"], donor["y"], donor["z"]], dtype=float)
    carbon = np.array([parent["x"], parent["y"], parent["z"]], dtype=float)
    axis = _unit(carbon - oxygen)
    u = _orthogonal(axis)
    v = _unit(np.cross(axis, u))
    theta = math.radians(parent_angle)
    directions = []
    for index in range(samples):
        phi = 2.0 * math.pi * index / samples
        direction = (
            math.cos(theta) * axis
            + math.sin(theta) * (math.cos(phi) * u + math.sin(phi) * v)
        )
        directions.append(oxygen + bond_length * _unit(direction))
    return directions

def _steric_penalty(candidate: np.ndarray, heavy_coords: np.ndarray,
                    donor_coord: np.ndarray, parent_coord: np.ndarray) -> float:
    distances = np.linalg.norm(heavy_coords - candidate, axis=1)
    exclude = ((np.linalg.norm(heavy_coords - donor_coord, axis=1) < 1e-6) |
               (np.linalg.norm(heavy_coords - parent_coord, axis=1) < 1e-6))
    relevant = distances[(distances <= 4.0) & ~exclude]
    if not len(relevant):
        return 0.0
    return float(np.sum((1.7 / np.maximum(relevant, 0.7)) ** 12))

def _angle(first: np.ndarray, vertex: np.ndarray, second: np.ndarray) -> float:
    a = first - vertex
    b = second - vertex
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))

def select_pose_hydroxyl_hydrogens(protein_atoms: list[dict], base_h: dict,
                                   pose: dict, ligand_profile: dict,
                                   hbond_da_cutoff: float = 4.1,
                                   hbond_ha_cutoff: float | None = 3.2,
                                   hbond_angle_min: float = 90.0,
                                   steric_weight: float = 1.0,
                                   hbond_weight: float = 1.0,
                                   acceptance_margin: float = 0.0):
    """Select a pose-specific OH orientation without using a PLIP reference.

    Return selected coordinates and one diagnostic row per OH. The directional
    score is global and can only be calibrated in G3 development mode.
    """
    selected = {key: list(value) for key, value in base_h.items()}
    heavy = np.array([[a["x"], a["y"], a["z"]] for a in protein_atoms], dtype=float)
    ligand_atoms = pose["atoms"]
    acceptors = ligand_profile["acceptor_atoms"]
    diagnostics = []

    for donor in protein_atoms:
        if not is_hydroxyl(donor):
            continue
        key = hydroxyl_key(donor)
        rotamers = hydroxyl_rotamers(protein_atoms, donor)
        if not rotamers:
            continue
        parent_name = HYDROXYL_DONOR_PARENTS[(donor["residue"], donor["atom_name"])]
        parent = next(atom for atom in protein_atoms if
                      atom["chain"] == donor["chain"] and atom["resnum"] == donor["resnum"] and
                      atom["residue"] == donor["residue"] and atom["atom_name"] == parent_name)
        donor_coord = np.array([donor["x"], donor["y"], donor["z"]], dtype=float)
        parent_coord = np.array([parent["x"], parent["y"], parent["z"]], dtype=float)
        candidates = []
        for number, hydrogen in enumerate(rotamers):
            steric = _steric_penalty(hydrogen, heavy, donor_coord, parent_coord)
            best_quality = 0.0
            best_acceptor = None
            for atom_index in acceptors:
                acceptor = ligand_atoms[atom_index]["coord"]
                da = float(np.linalg.norm(donor_coord - acceptor))
                ha = float(np.linalg.norm(hydrogen - acceptor))
                angle = _angle(donor_coord, hydrogen, acceptor)
                if (da <= hbond_da_cutoff and angle >= hbond_angle_min and
                        (hbond_ha_cutoff is None or ha <= hbond_ha_cutoff)):
                    distance_quality = (1.0 if hbond_ha_cutoff is None else
                                        (hbond_ha_cutoff - ha) / hbond_ha_cutoff)
                    quality = ((angle - hbond_angle_min) / (180.0 - hbond_angle_min)) * distance_quality
                    if quality > best_quality:
                        best_quality = quality
                        best_acceptor = atom_index
            score = hbond_weight * best_quality - steric_weight * steric
            candidates.append((score, steric, best_quality, best_acceptor, number, hydrogen))

        no_contact_score = -steric_weight * min(item[1] for item in candidates)
        winner = max(candidates, key=lambda item: item[0])
        if winner[2] <= 0.0 or winner[0] < no_contact_score + acceptance_margin:
            winner = min(candidates, key=lambda item: item[1])
            accepted = False
        else:
            accepted = True
        selected[key] = [winner[5]]
        diagnostics.append({
            "Residuo": f"{donor['residue']}{donor['resnum']}",
            "Cadeia": donor["chain"],
            "Atomo": donor["atom_name"],
            "Rotamero": winner[4],
            "Esterico": winner[1],
            "Qualidade_HBond": winner[2],
            "Aceito": accepted,
            "Atomo_aceptor": (ligand_atoms[winner[3]]["name"] if winner[3] is not None else ""),
        })
    return selected, diagnostics

def select_variant_hydroxyl_hydrogens(
        protein_atoms: list[dict], base_h: dict, pose: dict,
        ligand_profile: dict, mode: str, *,
        hbond_da_cutoff: float = 4.1,
        hbond_ha_cutoff: float | None = 3.2,
        hbond_angle_min: float = 90.0,
        steric_weight: float = 1.0,
        hbond_weight: float = 1.0,
        acceptance_margin: float = 0.0):
    """Choose OH hydrogens for H variants without changing G2B.

    ``weighted`` uses the steric/directional score; ``steric`` ignores the
    ligand and retains the least repulsive rotamer; ``rdkit`` retains the 3D
    H atoms originally supplied in ``base_h``. Every mode returns the same
    interface so classification remains auditable.
    """
    if mode == "rdkit":
        selected = {key: list(value) for key, value in base_h.items()}
        diagnostics = []
        for donor in protein_atoms:
            if not is_hydroxyl(donor):
                continue
            key = hydroxyl_key(donor)
            diagnostics.append({
                "Residuo": f"{donor['residue']}{donor['resnum']}",
                "Cadeia": donor["chain"],
                "Atomo": donor["atom_name"],
                "Rotamero": "RDKit",
                "Esterico": None,
                "Qualidade_HBond": None,
                "Aceito": bool(selected.get(key)),
                "Atomo_aceptor": "",
            })
        return selected, diagnostics

    if mode == "steric":
        # A zero directional weight makes the maximum score exactly the lowest
        # repulsion value; the pose does not influence the selected orientation.
        return select_pose_hydroxyl_hydrogens(
            protein_atoms, base_h, pose, ligand_profile,
            hbond_da_cutoff=hbond_da_cutoff,
            hbond_ha_cutoff=hbond_ha_cutoff,
            hbond_angle_min=hbond_angle_min,
            steric_weight=1.0,
            hbond_weight=0.0,
            acceptance_margin=0.0,
        )

    if mode != "weighted":
        raise ValueError(
            "Invalid OH orientation mode: " + repr(mode)
        )

    return select_pose_hydroxyl_hydrogens(
        protein_atoms, base_h, pose, ligand_profile,
        hbond_da_cutoff=hbond_da_cutoff,
        hbond_ha_cutoff=hbond_ha_cutoff,
        hbond_angle_min=hbond_angle_min,
        steric_weight=steric_weight,
        hbond_weight=hbond_weight,
        acceptance_margin=acceptance_margin,
    )

def format_interaction_residue(residue, resnum, chain):
    """Format a displayed residue identity, including the chain when present."""

    chain = str(chain or "").strip()
    return f"{str(residue).title()}{resnum}{chain}"

def classify_hbond_strength(interaction_type, distance_da):
    """Classify H-bond strength by donor--acceptor distance."""

    if interaction_type != "H-Bond" or distance_da is None:
        return "-"

    try:
        distance_da = float(distance_da)
    except (TypeError, ValueError):
        return "-"

    if not np.isfinite(distance_da):
        return "-"
    if distance_da <= 2.5:
        return "Strong"
    if distance_da <= 3.2:
        return "Moderate"
    return "Weak"

def classify_residue_group(residue):
    """Return the standardized functional group of a protein residue."""

    return RESIDUE_FUNCTIONAL_GROUPS.get(str(residue).upper(), "-")

def count_contacts(
        ligand_atoms,
        pocket_atoms,
        cutoff=contact_cutoff):

    contacts = 0

    for atom in ligand_atoms:

        lx, ly, lz = atom["coord"]

        for pocket_atom in pocket_atoms:

            dist = math.sqrt(
                (lx - pocket_atom["x"]) ** 2 +
                (ly - pocket_atom["y"]) ** 2 +
                (lz - pocket_atom["z"]) ** 2
            )

            if dist < cutoff:
                contacts += 1

    return contacts

def get_contact_residues(
        ligand_atoms,
        protein_atoms,
        cutoff=contact_cutoff):

    residues = set()

    for ligand_atom in ligand_atoms:

        lx, ly, lz = ligand_atom["coord"]

        for protein_atom in protein_atoms:

            dist = math.sqrt(
                (lx - protein_atom["x"]) ** 2 +
                (ly - protein_atom["y"]) ** 2 +
                (lz - protein_atom["z"]) ** 2
            )

            if dist <= cutoff:

                residues.add(
                    (
                        protein_atom["residue"],
                        protein_atom["resnum"]
                    )
                )

    return residues

def calculate_conservation_score(
        ligand_atoms,
        protein_atoms,
        residue_occupancy,
        site_counts,
        pocket,
        cutoff=contact_cutoff):

    residues = get_contact_residues(
        ligand_atoms,
        protein_atoms,
        cutoff
    )

    score = 0.0

    for residue, resnum in residues:

        key = (
            pocket,
            residue,
            resnum
        )

        if key in residue_occupancy:

            frequency = (
                residue_occupancy[key] /
                site_counts[pocket]
            )

            score += frequency

    return score

def _is_planar_ring(template, ring, max_rms=0.10):
    """Use a conservative fallback when the file does not encode aromaticity.

    PLIP accepts planar rings when aromaticity could not be perceived. Do not
    use the broad "every sp2 ring" rule because it classifies quinone rings as
    π-aromatic even when the file already contains reliable aromaticity data.
    """

    if template.GetNumConformers() == 0 or len(ring) < 3:
        return False

    conformer = template.GetConformer()
    coords = np.array([
        [
            conformer.GetAtomPosition(idx).x,
            conformer.GetAtomPosition(idx).y,
            conformer.GetAtomPosition(idx).z,
        ]
        for idx in ring
    ], dtype=float)
    centered = coords - np.mean(coords, axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    return len(singular_values) >= 3 and singular_values[-1] / math.sqrt(
        len(coords)
    ) <= max_rms

def get_ligand_interaction_profile(template):

    fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    feats = factory.GetFeaturesForMol(template)

    donor_atoms = set()
    acceptor_atoms = set()

    for feat in feats:

        if feat.GetFamily() == "Donor":
            donor_atoms.update(feat.GetAtomIds())

        elif feat.GetFamily() == "Acceptor":
            acceptor_atoms.update(feat.GetAtomIds())

    # Aromatic rings perceived by RDKit. As in PLIP, planarity fallback is
    # used only when the file has no aromaticity anywhere; a merely sp2 ring
    # is not itself a π-system.

    aromatic_rings = []
    ring_list = template.GetRingInfo().AtomRings()

    for ring in ring_list:

        ring_atoms = [template.GetAtomWithIdx(idx) for idx in ring]

        if all(a.GetIsAromatic() for a in ring_atoms):
            aromatic_rings.append(list(ring))

    if not aromatic_rings and not any(
            atom.GetIsAromatic() for atom in template.GetAtoms()):

        aromatic_rings = [
            list(ring) for ring in ring_list
            if _is_planar_ring(template, ring)
        ]

    # HALOGENS (and their neighboring carbon atom for the C-X...A angle)

    halogens = []
    halogen_parent = {}

    for atom in template.GetAtoms():

        if atom.GetSymbol() in ("F", "Cl", "Br", "I"):

            halogens.append(atom.GetIdx())

            neighbors = atom.GetNeighbors()

            if neighbors:
                halogen_parent[atom.GetIdx()] = neighbors[0].GetIdx()

    # CHARGED GROUPS (common patterns + explicit formal charge)

    cationic_smarts = [
        "[NX4+]",
        "[NX3][CX3](=[NX3+])[NX3]",
        "[nX3+]",
    ]

    anionic_smarts = [
        "[CX3](=O)[O-]",
        "[PX4](=O)([O-])[O-]",
        "[SX4](=O)(=O)[O-]",
    ]

    cationic_groups = []
    anionic_groups = []

    for smarts in cationic_smarts:

        pattern = Chem.MolFromSmarts(smarts)

        if pattern is None:
            continue

        for match in template.GetSubstructMatches(pattern):
            cationic_groups.append(list(match))

    for smarts in anionic_smarts:

        pattern = Chem.MolFromSmarts(smarts)

        if pattern is None:
            continue

        for match in template.GetSubstructMatches(pattern):
            anionic_groups.append(list(match))

    for atom in template.GetAtoms():

        if atom.GetFormalCharge() > 0:
            cationic_groups.append([atom.GetIdx()])

        elif atom.GetFormalCharge() < 0:
            anionic_groups.append([atom.GetIdx()])

    # LIGAND NONPOLAR (HYDROPHOBIC) CARBONS
    # Apply the same rule as on the protein side (PROTEIN_HYDROPHOBIC_ATOMS):
    # carbon without a directly neighboring heteroatom (N/O/S/halogen).
    # This automatically excludes carbonyls and carbons bonded to N/O/S in
    # ethers, amines, and amides. For example, in a lapachol derivative's
    # pyran ring, the gem-dimethyl carbon is included, while the two cyclic
    # ether/enol carbons bonded to ring oxygen are excluded.

    hydrophobic_atoms = set()

    for atom in template.GetAtoms():

        if atom.GetSymbol() != "C":
            continue

        if any(
            nbr.GetSymbol() not in ("C", "H")
            for nbr in atom.GetNeighbors()
        ):
            continue

        hydrophobic_atoms.add(atom.GetIdx())

    return {
        "donor_atoms": donor_atoms,
        "acceptor_atoms": acceptor_atoms,
        "aromatic_rings": aromatic_rings,
        "halogens": halogens,
        "halogen_parent": halogen_parent,
        "cationic_groups": cationic_groups,
        "anionic_groups": anionic_groups,
        "hydrophobic_atoms": hydrophobic_atoms,
    }

def get_protein_ring_and_charge_groups(protein_atoms):

    by_residue = {}

    for atom in protein_atoms:

        key = (atom["chain"], atom["resnum"], atom["residue"])
        by_residue.setdefault(key, {})[atom["atom_name"]] = atom

    aromatic_rings = []
    cationic_groups = []
    anionic_groups = []

    for (chain, resnum, resname), atoms_by_name in by_residue.items():

        if resname in PROTEIN_AROMATIC_RINGS:

            ring_atom_names = PROTEIN_AROMATIC_RINGS[resname]

            ring_atoms = [
                atoms_by_name[name]
                for name in ring_atom_names
                if name in atoms_by_name
            ]

            if len(ring_atoms) >= 3:
                aromatic_rings.append({
                    "chain": chain,
                    "resnum": resnum,
                    "residue": resname,
                    "atoms": ring_atoms,
                })

        if resname in PROTEIN_CATIONIC_GROUPS:

            group_atoms = [
                atoms_by_name[name]
                for name in PROTEIN_CATIONIC_GROUPS[resname]
                if name in atoms_by_name
            ]

            if group_atoms:
                cationic_groups.append({
                    "chain": chain,
                    "resnum": resnum,
                    "residue": resname,
                    "atoms": group_atoms,
                })

        if resname in PROTEIN_ANIONIC_GROUPS:

            group_atoms = [
                atoms_by_name[name]
                for name in PROTEIN_ANIONIC_GROUPS[resname]
                if name in atoms_by_name
            ]

            if group_atoms:
                anionic_groups.append({
                    "chain": chain,
                    "resnum": resnum,
                    "residue": resname,
                    "atoms": group_atoms,
                })

    return aromatic_rings, cationic_groups, anionic_groups

def is_hydroxyl_donor(residue, atom_name):
    """Return whether the atom is a freely rotating OH donor."""

    return (residue, atom_name) in HYDROXYL_DONOR_PARENTS

def _pose_number_from_name(name):
    """Extract the last number from a file or ligand name."""

    match = re.search(r"(\d+)(?!.*\d)", str(name))
    return int(match.group(1)) if match else None

def _plip_interaction_records(report_file):
    """Read the three relevant classes from raw PLIP XML."""

    root = ET.parse(report_file).getroot()
    records = []
    definitions = (
        ("hydrophobic_interactions", "hydrophobic_interaction",
         "Hydrophobic", "dist"),
        ("hydrogen_bonds", "hydrogen_bond",
         "Hydrogen Bond", "dist_h-a"),
        ("pi_stacks", "pi_stack", "π-π stacking", "centdist"),
    )

    for group_tag, item_tag, interaction_type, distance_tag in definitions:
        for item in root.findall(f".//{group_tag}/{item_tag}"):
            try:
                records.append({
                    "interaction_type": interaction_type,
                    "residue": item.findtext("restype", default="").strip(),
                    "resnum": int(item.findtext("resnr")),
                    "chain": item.findtext("reschain", default="").strip(),
                    "distance": float(item.findtext(distance_tag)),
                    "plip_ligand_atom": item.findtext("ligatom", default="").strip(),
                    "plip_protein_atom": item.findtext("protatom", default="").strip(),
                })
            except (TypeError, ValueError):
                continue

    return records

def _interaction_type_for_plip(interaction_type):
    if interaction_type.startswith("π-stacking"):
        return "π-π stacking"
    return {
        "H-Bond": "Hydrogen Bond",
        "Hydrophobic": "Hydrophobic",
        "Halogen Bond": "Halogen Bond",
        "cation-π": "Cation-π",
        "Salt Bridge": "Salt Bridge",
    }.get(interaction_type, interaction_type)

def build_plip_validation_rows(
        reference_dir,
        all_results,
        protein_donor_h_coordinates):
    """Generate an optional diagnostic; it never influences G2B classification."""

    reference_path = Path(reference_dir)
    rows = []

    # --- OH hydrogens: compare the internal model with the reference PDB ---
    protonated_files = sorted(
        reference_path.glob("Complexo *_protonated.pdb"),
        key=lambda path: _pose_number_from_name(path.stem) or -1,
    )

    if protonated_files:
        reference_atoms = read_protein(protonated_files[0], select_altloc=True)
        reference_hydrogens = get_explicit_protein_polar_hydrogens(
            protonated_files[0], reference_atoms
        )

        for key, reference_coords in reference_hydrogens.items():
            if not is_hydroxyl_donor(key[2], key[3]):
                continue

            calculated_coords = protein_donor_h_coordinates.get(key, [])
            if not calculated_coords:
                rows.append({
                    "Category": "OH Hydrogen",
                    "Pose": "All",
                    "Reference PLIP": f"{key[2]}{key[1]}:{key[3]}",
                    "G2B Result": "Missing polar H",
                    "PLIP Distance (Å)": None,
                    "F2 Distance (Å)": None,
                    "Difference (Å)": None,
                    "Status": "Missing",
                })
                continue

            deviation = min(
                np.linalg.norm(calculated - reference)
                for calculated in calculated_coords
                for reference in reference_coords
            )
            rows.append({
                "Category": "OH Hydrogen",
                "Pose": "All",
                "Reference PLIP": f"{key[2]}{key[1]}:{key[3]}",
                "G2B Result": "H generated by RDKit",
                "PLIP Distance (Å)": None,
                "F2 Distance (Å)": None,
                "Difference (Å)": deviation,
                "Status": (
                    "Match" if deviation <= PLIP_VALIDATION_DISTANCE_TOLERANCE
                    else "Different geometry"
                ),
            })

    # --- Interactions: one-to-one correspondence with raw XML ---
    results_by_pose = {}

    for result in all_results:
        pose_number = _pose_number_from_name(result["ligand"])
        if pose_number is not None:
            results_by_pose.setdefault(pose_number, []).extend(
                result["interactions"]
            )

    for report_file in sorted(
            reference_path.glob("Complexo *_report.xml"),
            key=lambda path: _pose_number_from_name(path.stem) or -1):

        pose_number = _pose_number_from_name(report_file.stem)
        expected = _plip_interaction_records(report_file)
        observed = list(results_by_pose.get(pose_number, []))
        used_observed = set()

        for reference in expected:
            candidates = [
                (index, interaction)
                for index, interaction in enumerate(observed)
                if index not in used_observed
                and _interaction_type_for_plip(interaction["type"])
                    == reference["interaction_type"]
                and interaction["residue"] == reference["residue"]
                and interaction["resnum"] == reference["resnum"]
                and interaction["chain"] == reference["chain"]
            ]

            if candidates:
                index, interaction = min(
                    candidates,
                    key=lambda pair: abs(
                        pair[1]["distance"] - reference["distance"]
                    ),
                )
                used_observed.add(index)
                difference = abs(interaction["distance"] - reference["distance"])
                status = (
                    "Match"
                    if difference <= PLIP_VALIDATION_DISTANCE_TOLERANCE
                    else "Different measure"
                )
                result_label = (
                    f"{interaction['ligand_atom']} / "
                    f"{interaction['protein_atom']}"
                )
                calculated_distance = interaction["distance"]
            else:
                difference = None
                status = "Missing from G2B"
                result_label = ""
                calculated_distance = None

            rows.append({
                "Category": reference["interaction_type"],
                "Pose": pose_number,
                "Reference PLIP": (
                    f"{reference['residue']}{reference['resnum']} "
                    f"({reference['plip_ligand_atom']} / "
                    f"{reference['plip_protein_atom']})"
                ),
                "G2B Result": result_label,
                "PLIP Distance (Å)": reference["distance"],
                "F2 Distance (Å)": calculated_distance,
                "Difference (Å)": difference,
                "Status": status,
            })

        for index, interaction in enumerate(observed):
            if index in used_observed:
                continue

            rows.append({
                "Category": _interaction_type_for_plip(interaction["type"]),
                "Pose": pose_number,
                "Reference PLIP": "",
                "G2B Result": (
                    f"{interaction['residue']}{interaction['resnum']} "
                    f"({interaction['ligand_atom']} / "
                    f"{interaction['protein_atom']})"
                ),
                "PLIP Distance (Å)": None,
                "F2 Distance (Å)": interaction["distance"],
                "Difference (Å)": None,
                "Status": "Extra in G2B",
            })

    return rows
