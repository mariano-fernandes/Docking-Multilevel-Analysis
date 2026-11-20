"""Interaction engine for the K variant, without GUI or PLIP dependency.

The receptor is read exclusively from ``ATOM`` records. Water, metals,
cofactors, and other ``HETATM`` records never enter the K model.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures


HBOND_DA_CUTOFF = 4.1
HBOND_HA_CUTOFF = 3.2
HBOND_ANGLE_MIN = 90.0
HYDROPHOBIC_DIST_CUTOFF = 4.0
PISTACK_DIST_CUTOFF = 5.5
PISTACK_PARALLEL_ANGLE_MAX = 30.0
PISTACK_TSHAPED_ANGLE_MIN = 60.0
PISTACK_TSHAPED_ANGLE_MAX = 90.0
PISTACK_OFFSET_MAX = 2.0
CATION_PI_DIST_CUTOFF = 6.0
SALT_BRIDGE_DIST_CUTOFF = 4.0
HALOGEN_DIST_CUTOFF = 3.5
HALOGEN_ANGLE_MIN = 165.0

_ACCEPTORS = {
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
    ("ASN", "OD1"), ("GLN", "OE1"), ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"), ("MET", "SD"),
}
_DONOR_NAMES = {
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
    ("ASN", "ND2"), ("GLN", "NE2"), ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("TRP", "NE1"),
}
_RINGS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
}
_CATIONS = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2", "CZ")}
_ANIONS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}


def _coord(atom):
    return np.array([atom["x"], atom["y"], atom["z"]], dtype=float)


def _load_pdb_rdkit(path):
    """Load a PDB even when its Windows path contains Unicode characters."""
    try:
        return Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False,
                                   proximityBonding=True)
    except OSError:
        with tempfile.TemporaryDirectory(prefix="cals_k_") as temporary:
            ascii_path = Path(temporary) / "protein.pdb"
            shutil.copy2(path, ascii_path)
            return Chem.MolFromPDBFile(str(ascii_path), sanitize=False, removeHs=False,
                                       proximityBonding=True)


def get_ligand_interaction_profile(template):
    """Ligand chemical profile shared by K execution and regression."""
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    donors, acceptors = set(), set()
    for feature in factory.GetFeaturesForMol(template):
        if feature.GetFamily() == "Donor":
            donors.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(feature.GetAtomIds())
    rings = [list(ring) for ring in template.GetRingInfo().AtomRings()
             if all(template.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]
    halogens, halogen_parent, hydrophobic = [], {}, set()
    cations, anions = [], []
    for atom in template.GetAtoms():
        index = atom.GetIdx()
        if atom.GetSymbol() in ("F", "Cl", "Br", "I"):
            halogens.append(index)
            if atom.GetDegree():
                halogen_parent[index] = atom.GetNeighbors()[0].GetIdx()
        if atom.GetSymbol() == "C" and all(nbr.GetSymbol() in ("C", "H") for nbr in atom.GetNeighbors()):
            hydrophobic.add(index)
        if atom.GetFormalCharge() > 0:
            cations.append([index])
        elif atom.GetFormalCharge() < 0:
            anions.append([index])
    for smarts, target in (("[NX4+]", cations), ("[NX3][CX3](=[NX3+])[NX3]", cations),
                           ("[nX3+]", cations), ("[CX3](=O)[O-]", anions),
                           ("[PX4](=O)([O-])[O-]", anions), ("[SX4](=O)(=O)[O-]", anions)):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            target.extend(list(match) for match in template.GetSubstructMatches(pattern))
    acceptor_metadata = {}
    for index in acceptors:
        atom = template.GetAtomWithIdx(index)
        acceptor_metadata[index] = {
            "element": atom.GetSymbol(),
            "formal_charge": atom.GetFormalCharge(),
            "aromatic": atom.GetIsAromatic(),
            "hybridization": str(atom.GetHybridization()),
            "heavy_neighbor_indices": [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                                       if neighbor.GetAtomicNum() != 1],
        }
    return {"donor_atoms": donors, "acceptor_atoms": acceptors,
            "acceptor_metadata": acceptor_metadata,
            "aromatic_rings": rings, "halogens": halogens,
            "halogen_parent": halogen_parent, "hydrophobic_atoms": hydrophobic,
            "cationic_groups": _unique_groups(cations), "anionic_groups": _unique_groups(anions)}


def _unique_groups(groups):
    seen, result = set(), []
    for group in groups:
        key = tuple(sorted(group))
        if key not in seen:
            seen.add(key)
            result.append(list(group))
    return result


def _angle(first, vertex, second):
    a, b = first - vertex, second - vertex
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def _ring_normal(coords):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 3:
        return None
    normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
    length = float(np.linalg.norm(normal))
    return normal / length if length > 1e-8 else None


def _ring_offset(first, first_normal, second, second_normal):
    first_projection = second - np.dot(second - first, first_normal) * first_normal
    second_projection = first - np.dot(first - second, second_normal) * second_normal
    return min(float(np.linalg.norm(first_projection - first)),
               float(np.linalg.norm(second_projection - second)))


def _pdb_atom_lines(path):
    """Use one altloc per atom and explicitly ignore every HETATM record."""
    selected, order = {}, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16]
        element = (line[76:78].strip() or atom_name.strip()[:1]).upper()
        key = (line[21], line[22:26], line[26], atom_name,
               line[6:11] if element == "H" else "")
        altloc = line[16]
        occupancy = float(line[54:60].strip() or 0)
        value = (3 if altloc == " " else 2 if altloc in ("A", "1") else 1,
                 occupancy, line)
        if key not in selected:
            selected[key] = value
            order.append(key)
        elif value[:2] > selected[key][:2]:
            selected[key] = value
    return [selected[key][2] for key in order]


def _read_atoms(path):
    atoms = []
    for line in _pdb_atom_lines(path):
        name = line[12:16].strip()
        element = (line[76:78].strip() or name[:1]).upper()
        if element == "H":
            continue
        try:
            atoms.append({"atom_name": name, "element": element,
                          "residue": line[17:20].strip(), "resnum": int(line[22:26]),
                          "chain": line[21].strip(), "x": float(line[30:38]),
                          "y": float(line[38:46]), "z": float(line[46:54])})
        except ValueError:
            continue
    return atoms


def _rdkit_hydrogens(protein_path, atoms):
    """Obtain H parents from RDKit bonds; use proximity only as a fallback."""
    result = defaultdict(list)
    by_key = {(a["chain"], a["resnum"], a["residue"], a["atom_name"]): a for a in atoms}
    mol = _load_pdb_rdkit(protein_path)
    if mol is not None and mol.GetNumConformers():
        conformer = mol.GetConformer()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
                continue
            parent = atom.GetNeighbors()[0]
            if parent.GetAtomicNum() not in (7, 8, 16):
                continue
            info = parent.GetPDBResidueInfo()
            if info is None or info.GetIsHeteroAtom():
                continue
            key = (info.GetChainId().strip(), info.GetResidueNumber(),
                   info.GetResidueName().strip(), info.GetName().strip())
            if key not in by_key:
                continue
            point = conformer.GetAtomPosition(atom.GetIdx())
            result[key].append({"id": atom.GetIdx(), "coord": np.array([point.x, point.y, point.z]),
                                "source": "RDKit"})
    if result:
        return result

    # Controlled fallback for PDBs whose graph could not be reconstructed.
    for serial, line in enumerate(_pdb_atom_lines(protein_path)):
        name = line[12:16].strip()
        element = (line[76:78].strip() or name[:1]).upper()
        if element != "H":
            continue
        try:
            residue, resnum, chain = line[17:20].strip(), int(line[22:26]), line[21].strip()
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        candidates = [a for a in atoms if (a["chain"], a["resnum"], a["residue"]) == (chain, resnum, residue)
                      and a["element"] in ("N", "O", "S")]
        if candidates:
            parent = min(candidates, key=lambda item: np.linalg.norm(coord - _coord(item)))
            if np.linalg.norm(coord - _coord(parent)) <= 1.3:
                key = (parent["chain"], parent["resnum"], parent["residue"], parent["atom_name"])
                result[key].append({"id": f"fallback-{serial}", "coord": coord,
                                    "source": "PDB-fallback"})
    return result


def _roles(atom, hydrogens):
    key = (atom["chain"], atom["resnum"], atom["residue"], atom["atom_name"])
    residue, name = atom["residue"], atom["atom_name"]
    roles = set()
    if name == "N" or (residue, name) in _DONOR_NAMES or residue == "HIS":
        if hydrogens.get(key):
            roles.add("donor")
    if name == "O" or (residue, name) in _ACCEPTORS:
        roles.add("acceptor")
    if residue == "HIS" and atom["element"] == "N" and not hydrogens.get(key):
        roles.add("acceptor")
    return roles


def _hydrophobic(atom):
    # RDKit adjacency is preferred; use a conservative fallback for PDBs
    # whose graph was not completed by proximity bonding.
    if atom["element"] != "C":
        return False
    neighbors = atom.get("neighbor_elements")
    if neighbors is not None:
        return all(element in ("C", "H") for element in neighbors)
    return atom["atom_name"] not in ("C", "CA") and atom["residue"] not in ("GLY",)


def prepare_protein_interaction_model(protein_path):
    atoms = _read_atoms(protein_path)
    hydrogens = _rdkit_hydrogens(protein_path, atoms)
    # RDKit connectivity is annotated when PDB correspondence is reliable.
    mol = _load_pdb_rdkit(protein_path)
    if mol is not None:
        rd_neighbors = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1 or atom.GetPDBResidueInfo() is None:
                continue
            info = atom.GetPDBResidueInfo()
            if info.GetIsHeteroAtom():
                continue
            key = (info.GetChainId().strip(), info.GetResidueNumber(), info.GetResidueName().strip(), info.GetName().strip())
            rd_neighbors[key] = [nbr.GetSymbol() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() != 1]
        for atom in atoms:
            key = (atom["chain"], atom["resnum"], atom["residue"], atom["atom_name"])
            if key in rd_neighbors:
                atom["neighbor_elements"] = rd_neighbors[key]
    for atom in atoms:
        atom["roles"] = _roles(atom, hydrogens)
        atom["hydrophobic"] = _hydrophobic(atom)

    by_residue = defaultdict(dict)
    for atom in atoms:
        by_residue[(atom["chain"], atom["resnum"], atom["residue"])][atom["atom_name"]] = atom
    rings, cations, anions = [], [], []
    for (chain, resnum, residue), named in by_residue.items():
        ring_names = _RINGS.get(residue, ())
        ring_atoms = [named[name] for name in ring_names if name in named]
        if len(ring_atoms) >= 3:
            rings.append({"chain": chain, "resnum": resnum, "residue": residue, "atoms": ring_atoms})
        for names, target in ((_CATIONS.get(residue, ()), cations), (_ANIONS.get(residue, ()), anions)):
            group = [named[name] for name in names if name in named]
            if group:
                target.append({"chain": chain, "resnum": resnum, "residue": residue, "atoms": group})
    return {"atoms": atoms, "hydrogens": hydrogens, "rings": rings, "cations": cations, "anions": anions}


def _nearby_atoms(pose_atoms, protein_atoms, cutoff=8.0):
    ligand = np.array([atom["coord"] for atom in pose_atoms])
    return [atom for atom in protein_atoms if np.linalg.norm(ligand - _coord(atom), axis=1).min() <= cutoff]


def _rank_hbond(angle, distance_ha, distance_da):
    """Highest angle, then shortest H···A and D···A, defines the best candidate."""
    return (float(angle), -float(distance_ha), -float(distance_da))


def _as_hydrogen_records(mapping, source="OH_rotamer"):
    """Normalize H coordinates or records for the engine interface."""
    normalized = {}
    for key, entries in (mapping or {}).items():
        records = []
        for number, entry in enumerate(entries):
            if isinstance(entry, dict):
                record = dict(entry)
                record["coord"] = np.asarray(record["coord"], dtype=float)
                record.setdefault("source", source)
                record.setdefault("id", f"override-{number}")
            else:
                record = {"id": f"override-{number}", "coord": np.asarray(entry, dtype=float),
                          "source": source}
            records.append(record)
        normalized[key] = records
    return normalized


def _histidine_h_coordinate(named_atoms, donor_name):
    """Reconstruct an sp2 H on the outside of the imidazole ring."""
    neighbor_names = {"ND1": ("CG", "CE1"), "NE2": ("CD2", "CE1")}[donor_name]
    donor = named_atoms.get(donor_name)
    neighbors = [named_atoms.get(name) for name in neighbor_names]
    if donor is None or any(atom is None for atom in neighbors):
        return None
    donor_coord = _coord(donor)
    vectors = []
    for neighbor in neighbors:
        vector = _coord(neighbor) - donor_coord
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            return None
        vectors.append(vector / length)
    direction = -(vectors[0] + vectors[1])
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return None
    return donor_coord + 1.01 * direction / length


def _select_histidine_tautomers(protein_atoms, ligand_atoms, ligand_profile, hydrogens):
    """Use canonical ND1-H for neutral HIS; NE2 remains an acceptor.

    Selection is not pose-oriented, which avoids switching tautomers to create
    artificial contacts. If the ring is incomplete, retain the available
    explicit RDKit H as an auditable fallback.
    """
    residues = defaultdict(dict)
    for atom in protein_atoms:
        if atom["residue"] == "HIS":
            residues[(atom["chain"], atom["resnum"], atom["residue"])][atom["atom_name"]] = atom
    diagnostics = []
    for (chain, resnum, residue), named in residues.items():
        keys = {name: (chain, resnum, residue, name) for name in ("ND1", "NE2")}
        explicit_hydrogens = {name: list(hydrogens.get(key, [])) for name, key in keys.items()}
        for key in keys.values():
            hydrogens.pop(key, None)

        donor_name = "ND1"
        hcoord = _histidine_h_coordinate(named, donor_name)
        if hcoord is not None:
            hydrogens[keys[donor_name]] = [{
                "id": f"his-{chain}-{resnum}-{donor_name}", "coord": hcoord,
                "source": "HIS_ND1_canônico",
            }]
            reason = "ND1_canônico"
        else:
            donor_name = next((name for name in ("ND1", "NE2")
                               if name in named and explicit_hydrogens[name]), None)
            if donor_name is not None:
                hydrogens[keys[donor_name]] = explicit_hydrogens[donor_name]
                reason = "fallback_RDKit_anél_incompleto"

        if donor_name is None:
            diagnostics.append({"Residuo": f"{residue}{resnum}", "Cadeia": chain,
                                "Estado_HIS": "Indisponível", "Source_H": "anel_his_incompleto"})
            continue

        donor_coord = _coord(named[donor_name])
        best = None
        for hydrogen in hydrogens[keys[donor_name]]:
            for ligand_index in ligand_profile["acceptor_atoms"]:
                acceptor_coord = ligand_atoms[ligand_index]["coord"]
                distance_da = float(np.linalg.norm(donor_coord - acceptor_coord))
                if distance_da > HBOND_DA_CUTOFF:
                    continue
                distance_ha = float(np.linalg.norm(hydrogen["coord"] - acceptor_coord))
                angle = _angle(donor_coord, hydrogen["coord"], acceptor_coord)
                if angle >= HBOND_ANGLE_MIN and distance_ha <= HBOND_HA_CUTOFF:
                    candidate = (_rank_hbond(angle, distance_ha, distance_da), ligand_index,
                                 angle, distance_ha, distance_da)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
        diagnostics.append({
            "Residuo": f"{residue}{resnum}", "Cadeia": chain,
            "Estado_HIS": f"{donor_name}-H", "Source_H": reason,
            "Atomo_Doador": donor_name,
            "Atomo_Aceptor": ligand_atoms[best[1]]["name"] if best is not None else "",
            "Doador_Aceptor_A": best[4] if best is not None else None,
            "H_Aceptor_A": best[3] if best is not None else None,
            "Angulo": best[2] if best is not None else None,
        })
    return diagnostics


def _pose_protein_hydrogens(protein_model, protein_atoms, ligand_atoms, ligand_profile,
                            protein_hydrogens):
    hydrogens = _as_hydrogen_records(protein_model["hydrogens"], source="RDKit")
    hydrogens.update(_as_hydrogen_records(protein_hydrogens, source="OH_rotamer"))
    his_diagnostics = _select_histidine_tautomers(
        protein_atoms, ligand_atoms, ligand_profile, hydrogens
    )
    return hydrogens, his_diagnostics


def classify_interactions(pose, ligand_profile, protein_model, protein_hydrogens=None,
                          hbond_diagnostics=None):
    """Classify a K3 pose confirmed with explicit H and canonical HIS ND1-H."""
    interactions, ligand_atoms = [], pose["atoms"]
    protein_atoms = [dict(atom, roles=set(atom["roles"]))
                     for atom in _nearby_atoms(ligand_atoms, protein_model["atoms"])]
    active_hydrogens, his_diagnostics = _pose_protein_hydrogens(
        protein_model, protein_atoms, ligand_atoms, ligand_profile, protein_hydrogens
    )
    for protein_atom in protein_atoms:
        if protein_atom["residue"] == "HIS" and protein_atom["atom_name"] in ("ND1", "NE2"):
            key = (protein_atom["chain"], protein_atom["resnum"], protein_atom["residue"],
                   protein_atom["atom_name"])
            protein_atom["roles"] = {"donor"} if active_hydrogens.get(key) else {"acceptor"}
    h_by_parent = defaultdict(list)
    for number, hydrogen in enumerate(pose.get("polar_h", [])):
        h_by_parent[hydrogen["parent_idx"]].append({"id": hydrogen.get("id", number),
                                                     "coord": hydrogen["coord"],
                                                     "source": "Ligand PDBQT"})

    for index, ligand_atom in enumerate(ligand_atoms):
        ligand_coord = ligand_atom["coord"]
        for protein_atom in protein_atoms:
            protein_coord = _coord(protein_atom)
            distance = float(np.linalg.norm(ligand_coord - protein_coord))
            ligand_donor = index in ligand_profile["donor_atoms"] and "acceptor" in protein_atom["roles"]
            ligand_acceptor = index in ligand_profile["acceptor_atoms"] and "donor" in protein_atom["roles"]
            if (ligand_donor or ligand_acceptor) and distance <= HBOND_DA_CUTOFF:
                candidates = h_by_parent[index] if ligand_donor else active_hydrogens.get(
                    (protein_atom["chain"], protein_atom["resnum"], protein_atom["residue"], protein_atom["atom_name"]), [])
                best = None
                for hydrogen in candidates:
                    hcoord = hydrogen["coord"]
                    angle = _angle(ligand_coord if ligand_donor else protein_coord, hcoord,
                                   protein_coord if ligand_donor else ligand_coord)
                    distance_ha = float(np.linalg.norm(
                        hcoord - (protein_coord if ligand_donor else ligand_coord)
                    ))
                    rank = _rank_hbond(angle, distance_ha, distance)
                    if (angle >= HBOND_ANGLE_MIN and distance_ha <= HBOND_HA_CUTOFF and
                            (best is None or rank > best[0])):
                        best = (rank, angle, distance_ha, hydrogen)
                if best is not None:
                    _, angle, distance_ha, hydrogen = best
                    interactions.append({"residue": protein_atom["residue"], "resnum": protein_atom["resnum"], "chain": protein_atom["chain"],
                        "type": "H-Bond", "distance": distance_ha,
                        "distance_da": distance, "distance_ha": distance_ha,
                        "angle": angle, "ligand_atom": ligand_atom["name"], "protein_atom": protein_atom["atom_name"],
                        # Each physical H can point to only one acceptor.
                        # Different H atoms of the same residue remain independent.
                        "_key": ("L", index, hydrogen["id"]) if ligand_donor else
                                ("P", protein_atom["chain"], protein_atom["resnum"],
                                 protein_atom["residue"], protein_atom["atom_name"], hydrogen["id"]),
                        "_hbond_source": hydrogen.get("source", "RDKit"),
                        "_hbond_donor_side": "ligand" if ligand_donor else "proteína",
                        "_hbond_hydrogen_id": hydrogen["id"]})
            if index in ligand_profile["hydrophobic_atoms"] and protein_atom["hydrophobic"] and distance <= HYDROPHOBIC_DIST_CUTOFF:
                interactions.append({"residue": protein_atom["residue"], "resnum": protein_atom["resnum"], "chain": protein_atom["chain"], "type": "Hydrophobic", "distance": distance, "angle": None, "ligand_atom": ligand_atom["name"], "protein_atom": protein_atom["atom_name"]})
            if index in ligand_profile["halogens"] and protein_atom["element"] in ("N", "O") and "acceptor" in protein_atom["roles"] and distance <= HALOGEN_DIST_CUTOFF:
                parent = ligand_profile["halogen_parent"].get(index)
                angle = _angle(ligand_atoms[parent]["coord"], ligand_coord, protein_coord) if parent is not None else None
                if angle is not None and angle >= HALOGEN_ANGLE_MIN:
                    interactions.append({"residue": protein_atom["residue"], "resnum": protein_atom["resnum"], "chain": protein_atom["chain"], "type": "Halogen Bond", "distance": distance, "angle": angle, "ligand_atom": ligand_atom["name"], "protein_atom": protein_atom["atom_name"]})

    ligand_rings = [{"coords": [ligand_atoms[i]["coord"] for i in ring], "names": [ligand_atoms[i]["name"] for i in ring]} for ring in ligand_profile["aromatic_rings"]]
    for ring in ligand_rings:
        centroid, normal, names = np.mean(ring["coords"], axis=0), _ring_normal(ring["coords"]), ",".join(ring["names"])
        for protein_ring in protein_model["rings"]:
            coords = [_coord(atom) for atom in protein_ring["atoms"]]
            pcentroid, pnormal = np.mean(coords, axis=0), _ring_normal(coords)
            distance = float(np.linalg.norm(centroid - pcentroid))
            if distance > PISTACK_DIST_CUTOFF or normal is None or pnormal is None or _ring_offset(centroid, normal, pcentroid, pnormal) > PISTACK_OFFSET_MAX:
                continue
            angle = _angle(normal, np.zeros(3), pnormal); angle = min(angle, 180 - angle)
            if angle <= PISTACK_PARALLEL_ANGLE_MAX:
                kind = "π-stacking (Parallel)"
            elif PISTACK_TSHAPED_ANGLE_MIN <= angle <= PISTACK_TSHAPED_ANGLE_MAX:
                kind = "π-stacking (Perpendicular)"
            else:
                continue
            interactions.append({"residue": protein_ring["residue"], "resnum": protein_ring["resnum"], "chain": protein_ring["chain"], "type": kind, "distance": distance, "angle": angle, "ligand_atom": f"Aromatic Ring ({names})", "protein_atom": "Aromatic Ring (" + ",".join(a["atom_name"] for a in protein_ring["atoms"]) + ")"})

    def groups(indices):
        return [{"coords": [ligand_atoms[i]["coord"] for i in group], "names": [ligand_atoms[i]["name"] for i in group]} for group in indices]
    _add_group_interactions(interactions, ligand_rings, protein_model["cations"], "cation-π", CATION_PI_DIST_CUTOFF, "ring-group")
    _add_group_interactions(interactions, groups(ligand_profile["cationic_groups"]), protein_model["rings"], "cation-π", CATION_PI_DIST_CUTOFF, "group-ring")
    _add_salt_bridges(interactions, groups(ligand_profile["cationic_groups"]), protein_model["anions"], "Cationic Group", "Anionic Group")
    _add_salt_bridges(interactions, groups(ligand_profile["anionic_groups"]), protein_model["cations"], "Anionic Group", "Cationic Group")
    result = _consolidate(interactions)
    if hbond_diagnostics is not None:
        hbond_diagnostics.extend(his_diagnostics)
        for item in result:
            if item["type"] != "H-Bond":
                continue
            hbond_diagnostics.append({
                "Residuo": f"{item['residue']}{item['resnum']}", "Cadeia": item["chain"],
                "Estado_HIS": "", "Source_H": item.get("_hbond_source", ""),
                "Atomo_Doador": item["protein_atom"] if item.get("_hbond_donor_side") == "proteína"
                                  else item["ligand_atom"],
                "Atomo_Aceptor": item["ligand_atom"] if item.get("_hbond_donor_side") == "proteína"
                                    else item["protein_atom"],
                "Doador_Aceptor_A": item.get("distance_da"),
                "H_Aceptor_A": item.get("distance_ha"), "Angulo": item.get("angle"),
            })
    for item in result:
        item.pop("_hbond_source", None)
        item.pop("_hbond_donor_side", None)
        item.pop("_hbond_hydrogen_id", None)
    return result


def _add_group_interactions(output, first_groups, second_groups, kind, cutoff, mode):
    for first in first_groups:
        first_coords = first.get("coords") or [_coord(a) for a in first["atoms"]]
        first_names = first.get("names") or [a["atom_name"] for a in first["atoms"]]
        for second in second_groups:
            second_coords = second.get("coords") or [_coord(a) for a in second["atoms"]]
            first_center, second_center = np.mean(first_coords, axis=0), np.mean(second_coords, axis=0)
            distance = float(np.linalg.norm(first_center - second_center))
            # In addition to the centroid, cation--π requires projection near
            # the ring center; distance alone produced false positives in Lys.
            ring_coords = first_coords if mode == "ring-group" else second_coords
            cation_center = second_center if mode == "ring-group" else first_center
            normal = _ring_normal(ring_coords)
            if normal is None:
                continue
            projection = cation_center - np.dot(cation_center - np.mean(ring_coords, axis=0), normal) * normal
            lateral = float(np.linalg.norm(projection - np.mean(ring_coords, axis=0)))
            if lateral > 2.0:
                continue
            if distance <= cutoff:
                output.append({"residue": second["residue"], "resnum": second["resnum"], "chain": second["chain"], "type": kind, "distance": distance, "angle": None,
                    "ligand_atom": ("Aromatic Ring" if mode == "ring-group" else "Cationic Group") + " (" + ",".join(first_names) + ")",
                    "protein_atom": ("Cationic Group" if mode == "ring-group" else "Aromatic Ring") + " (" + ",".join(a["atom_name"] for a in second["atoms"]) + ")"})


def _add_salt_bridges(output, ligand_groups, protein_groups, ligand_label, protein_label):
    for ligand in ligand_groups:
        for protein in protein_groups:
            distance = float(np.linalg.norm(np.mean(ligand["coords"], axis=0) - np.mean([_coord(a) for a in protein["atoms"]], axis=0)))
            if distance <= SALT_BRIDGE_DIST_CUTOFF:
                output.append({"residue": protein["residue"], "resnum": protein["resnum"], "chain": protein["chain"], "type": "Salt Bridge", "distance": distance, "angle": None,
                    "ligand_atom": ligand_label + " (" + ",".join(ligand["names"]) + ")", "protein_atom": protein_label + " (" + ",".join(a["atom_name"] for a in protein["atoms"]) + ")"})


def _consolidate(interactions):
    # A ring already classified as π-π does not reappear as a collection of
    # hydrophobic contacts with the same protein ring.
    stacked = set()
    for item in interactions:
        if item["type"].startswith("π-stacking"):
            ligand = frozenset(item["ligand_atom"][len("Aromatic Ring ("):-1].split(","))
            protein = frozenset(item["protein_atom"][len("Aromatic Ring ("):-1].split(","))
            stacked.add((item["chain"], item["resnum"], item["residue"], ligand, protein))
    filtered = []
    for item in interactions:
        if item["type"] != "Hydrophobic":
            filtered.append(item)
            continue
        covered = any(item["chain"] == chain and item["resnum"] == resnum and item["residue"] == residue
                      and item["ligand_atom"] in ligand and item["protein_atom"] in protein
                      for chain, resnum, residue, ligand, protein in stacked)
        if not covered:
            filtered.append(item)
    interactions = filtered
    # PLIP retains the best C--C contact per ligand atom/residue and protein atom.
    hydro, other = [it for it in interactions if it["type"] == "Hydrophobic"], [it for it in interactions if it["type"] != "Hydrophobic"]
    by_ligand_residue = {}
    for item in hydro:
        key = (item["ligand_atom"], item["chain"], item["resnum"], item["residue"])
        if key not in by_ligand_residue or item["distance"] < by_ligand_residue[key]["distance"]:
            by_ligand_residue[key] = item
    by_protein = {}
    for item in by_ligand_residue.values():
        key = (item["chain"], item["resnum"], item["residue"], item["protein_atom"])
        if key not in by_protein or item["distance"] < by_protein[key]["distance"]:
            by_protein[key] = item
    hbond_best = {}
    for item in other:
        if item["type"] != "H-Bond":
            continue
        key = item["_key"]
        if (key not in hbond_best or
                _rank_hbond(item["angle"], item["distance_ha"], item["distance_da"]) >
                _rank_hbond(hbond_best[key]["angle"], hbond_best[key]["distance_ha"],
                            hbond_best[key]["distance_da"])):
            hbond_best[key] = item
    result = [item for item in other if item["type"] != "H-Bond"] + list(by_protein.values()) + list(hbond_best.values())
    for item in result:
        item.pop("_key", None)
    return result
