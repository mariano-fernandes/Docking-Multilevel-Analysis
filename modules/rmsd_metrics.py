import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D
from .pose_loading import pose_to_template_mol

BOLTZMANN_KT = 0.593  # kcal/mol; approximate RT at 298 K

def build_rmsd_matrix(
        results,
        template,
        method="consensus"):

    n = len(results)

    matrix = np.zeros((n, n), dtype=float)

    # Convert all poses to RDKit Mol objects.
    mols = []

    for result in results:

        try:
            mol = pose_to_template_mol(
                result["atoms"],
                template
            )

        except ValueError as error:

            raise ValueError(
                f"{error} "
                f"(ligand={result.get('ligand')}, "
                f"pose={result.get('pose')})"
            ) from error

        mols.append(mol)

    for i in range(n):

        for j in range(i + 1, n):

            if method == "consensus":

                rmsd = rdMolAlign.GetBestRMS(
                    mols[i],
                    mols[j]
                )

            elif method == "binding":

                rmsd = rdMolAlign.CalcRMS(
                    mols[i],
                    mols[j]
                )

            else:

                raise ValueError(
                    f"Unknown RMSD method: {method}"
                )

            matrix[i, j] = rmsd
            matrix[j, i] = rmsd

    return matrix

def calculate_rmsd_statistics(
        rmsd_matrix,
        poses):

    n = rmsd_matrix.shape[0]

    consensus = []

    for i in range(n):

        # Remove the pose-to-self RMSD.
        values = np.delete(rmsd_matrix[i], i)

        pose = poses[i]

        consensus.append({

            "pose_index": i,

            "ligand": pose["ligand"],

            "pose": pose["pose"],

            "energy": pose["energy"],

            "pocket": pose["pocket"],

            "site": pose.get("site", pose["pocket"]),

            "conservation_score": pose["conservation_score"],

            "mean_rmsd": float(np.mean(values)),

            "median_rmsd": float(np.median(values)),

            "std_rmsd": float(np.std(values)),

            "min_rmsd": float(np.min(values)),

            "max_rmsd": float(np.max(values))

        })

    representative = min(
        consensus,
        key=lambda x: x["mean_rmsd"]
    )

    # Ensemble statistics.
    upper_triangle = rmsd_matrix[np.triu_indices(n, k=1)]

    ensemble = {

        "n_poses": n,

        "mean_rmsd": float(np.mean(upper_triangle)),

        "median_rmsd": float(np.median(upper_triangle)),

        "std_rmsd": float(np.std(upper_triangle)),

        "min_rmsd": float(np.min(upper_triangle)),

        "max_rmsd": float(np.max(upper_triangle))

    }

    return {

        "statistics": consensus,

        "representative_pose": representative,

        "ensemble": ensemble

    }

def _pose_coordinates(pose):
    return np.asarray([atom["coord"] for atom in pose["atoms"]], dtype=float)

def _coordinates_to_template_mol(coordinates, template):
    """Create a temporary template Mol while preserving fixed atom order."""
    mol = Chem.Mol(template)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (mol.GetNumAtoms(), 3):
        raise ValueError("Consensus coordinates are incompatible with the template.")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer)
    return mol

def _rmsd_to_consensus(pose, consensus_coordinates, template):
    return float(rdMolAlign.CalcRMS(
        pose_to_template_mol(pose["atoms"], template),
        _coordinates_to_template_mol(consensus_coordinates, template),
    ))

def _representative_record(poses, pose_index):
    pose = poses[pose_index]
    return {
        "pose_index": pose_index,
        "ligand": pose["ligand"],
        "pose": pose["pose"],
        "energy": pose["energy"],
        "pocket": pose["pocket"],
        "site": pose.get("site", pose["pocket"]),
        "conservation_score": pose["conservation_score"],
    }

def select_representative_boltzmann(poses, template, kT=BOLTZMANN_KT):
    """Select the real pose closest to the energy-weighted consensus."""
    if kT <= 0:
        raise ValueError("BOLTZMANN_KT must be greater than zero.")
    energies = np.asarray([pose["energy"] for pose in poses], dtype=float)
    energy_min = float(np.min(energies))
    unnormalized_weights = np.exp(-(energies - energy_min) / kT)
    weights = unnormalized_weights / np.sum(unnormalized_weights)
    coordinate_stack = np.asarray([_pose_coordinates(pose) for pose in poses])
    consensus_coordinates = np.tensordot(weights, coordinate_stack, axes=(0, 0))
    distances = np.asarray([
        _rmsd_to_consensus(pose, consensus_coordinates, template)
        for pose in poses
    ])
    representative_index = min(
        range(len(poses)),
        key=lambda index: (distances[index], energies[index], index),
    )
    diagnostics = {
        "representative_effective_sample_size": float(1.0 / np.sum(weights ** 2)),
    }
    return _representative_record(poses, representative_index), diagnostics
