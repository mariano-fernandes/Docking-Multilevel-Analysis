import os
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET
from tkinter import filedialog

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation
from openpyxl import load_workbook, Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolAlign, ChemicalFeatures
from rdkit.Geometry import Point3D

from .pose_loading import *
from .clustering import *
from .rmsd_metrics import *
from .interactions import *
from ._interaction_engine import (
    classify_interactions as classify_k_interactions,
    prepare_protein_interaction_model,
)
from .cals_network_export import *
from .ranking import add_score_bp_values, order_ranking_rows

#======================================================================#
#                              DEFINITIONS                              #
#======================================================================#

# Keep the benchmarking calculations available while omitting their worksheet
# from generated workbooks unless it is explicitly enabled.
INCLUDE_BENCHMARKING_SHEET = False

SHEET_TAB_COLORS = {
    "General Information": "FFE699",
    "Pockets": "FFE699",
    "Pocket Summary": "FFE699",
    "Site Summary": "FFE699",
    "Centroids": "FFE699",
    "Interactions by Type": "C6E0B4",
    "Interactions by Pocket": "C6E0B4",
    "Interactions by Site": "C6E0B4",
    "Conformational Analysis": "F8CBAD",
    "Ranking BP Selection": "F8CBAD",
    "Score BP": "F8CBAD",
    "Benchmarking": "F8CBAD",
    "Structural": "B4C6E7",
    "Functional": "B4C6E7",
    "Edges Structural": "B4C6E7",
    "Nodes Structural": "B4C6E7",
    "Edges Functional": "B4C6E7",
    "Nodes Functional": "B4C6E7",
    "Structural by Site": "B4C6E7",
    "Functional by Site": "B4C6E7",
    "ML Analysis": "B4C6E7",
    "Pose Coordinates": "DBDBDB",
}

def build_excel_output_path(vina_files, ligand_file):
    """Create an available output name based only on the ligand reference."""

    output_folder = Path(vina_files[0]).parent
    base_name = f"CALS - {Path(ligand_file).stem}"
    candidate = output_folder / f"{base_name}.xlsx"
    index = 1

    while candidate.exists():
        candidate = output_folder / f"{base_name} ({index}).xlsx"
        index += 1

    return str(candidate)


def build_single_pose_rmsd_statistics(pose):
    """Return compatible statistics for a group containing one pose.

    SCM/PCM and their individual variants depend on comparisons between poses.
    For n=1, export ``None`` as a blank Excel cell rather than displaying zero
    as if a comparison had been performed.
    """

    pose_row = {
        "pose_index": 0,
        "ligand": pose["ligand"],
        "pose": pose["pose"],
        "energy": pose["energy"],
        "pocket": pose["pocket"],
        "site": pose.get("site", pose["pocket"]),
        "conservation_score": pose["conservation_score"],
        "mean_rmsd": None,
        "median_rmsd": None,
        "std_rmsd": None,
        "min_rmsd": None,
        "max_rmsd": None,
    }

    return {
        "statistics": [pose_row],
        "ensemble": {
            "n_poses": 1,
            "mean_rmsd": None,
            "median_rmsd": None,
            "std_rmsd": None,
            "min_rmsd": None,
            "max_rmsd": None,
        },
    }


def format_pose_label(pose):
    """Normalize an identifier to the ``Pose N`` format for internal use."""

    label = str(pose).strip()
    return label if label.casefold().startswith("pose ") else f"Pose {label}"


BENCHMARKING_COLUMNS = [
    "Criterion", "Ligand", "Pose", "Pocket", "Site",
    "Energy (kcal/mol)",
]


def _benchmark_energy_pose_key(result):
    """Ordena poses para os desempates de benchmarking.

    The most favorable energy wins; persistent ties go to the lowest pose.
    Missing values remain at the end to keep the export auditable.
    """

    try:
        energy = float(result.get("energy"))
    except (TypeError, ValueError):
        energy = math.inf

    pose = str(result.get("pose", ""))
    pose_parts = re.findall(r"\d+|\D+", pose)
    pose_key = tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in pose_parts
    )
    return energy, pose_key


def _benchmark_row(criterion, result, fallback_ligand=None):
    """Convert a pose, or its unavailability, into an export row."""

    if result is None:
        return {
            "Criterion": criterion,
            "Ligand": fallback_ligand,
            "Pose": "N/A",
            "Pocket": "N/A",
            "Site": "N/A",
            "Energy (kcal/mol)": None,
        }

    return {
        "Criterion": criterion,
        "Ligand": result.get("ligand"),
        "Pose": result.get("pose"),
        "Pocket": result.get("site"),
        "Site": result.get("cluster", result.get("pocket")),
        "Energy (kcal/mol)": result.get("energy"),
    }


def build_benchmarking_rows(all_results, analyzable_results, ranking_rows):
    """Build the comparison without recalculating any CALS decision.

    ``ranking_rows`` is already ordered by ``order_ranking_rows``. Its first
    row belongs to the same predominant subsite used by CALS for BestPose, so
    this routine only reuses that result for export.
    """

    fallback_ligand = all_results[0].get("ligand") if all_results else None
    lowest_energy = (
        min(all_results, key=_benchmark_energy_pose_key)
        if all_results else None
    )

    primary_ranking_row = ranking_rows[0] if ranking_rows else None
    central_pose = None
    multifactorial_pose = None

    if primary_ranking_row is not None:
        primary_scope = (
            primary_ranking_row.get("site"),
            primary_ranking_row.get("cluster"),
        )
        multifactorial_pose = next(
            (
                row for row in ranking_rows
                if (row.get("site"), row.get("cluster")) == primary_scope
                and row.get("rank") == 1
            ),
            None,
        )
        primary_members = [
            result for result in analyzable_results
            if (result.get("site"), result.get("pocket")) == primary_scope
        ]

        if primary_members:
            centroid = np.mean(
                [
                    [
                        result["centroid_x"], result["centroid_y"],
                        result["centroid_z"],
                    ]
                    for result in primary_members
                ],
                axis=0,
            )
            central_pose = min(
                primary_members,
                key=lambda result: (
                    float(np.linalg.norm(np.array([
                        result["centroid_x"], result["centroid_y"],
                        result["centroid_z"],
                    ]) - centroid)),
                    *_benchmark_energy_pose_key(result),
                ),
            )

    return [
        _benchmark_row("Lowest energy", lowest_energy, fallback_ligand),
        _benchmark_row("Spatial center", central_pose, fallback_ligand),
        _benchmark_row(
            "Multifactorial method", multifactorial_pose, fallback_ligand
        ),
    ]


def write_benchmarking_sheet(workbook, rows, energy_format="0.000"):
    """Add the compact benchmarking table to the individual workbook."""

    worksheet = workbook.create_sheet(title="Benchmarking")
    worksheet.append(BENCHMARKING_COLUMNS)
    for row in rows:
        worksheet.append([row.get(column) for column in BENCHMARKING_COLUMNS])

    energy_column = BENCHMARKING_COLUMNS.index("Energy (kcal/mol)") + 1
    for row_index in range(2, worksheet.max_row + 1):
        worksheet.cell(row=row_index, column=energy_column).number_format = energy_format
    return worksheet


REPRODUCIBILITY_SOURCE_KEYS = (
    "fidelity_structural",
    "fidelity_functional",
    "fidelity_cooccurrence_structural",
    "fidelity_cooccurrence_functional",
)


def local_network_score_values(score, as_fraction=False):
    """Convert internal network scores to export fields.

    By default, the four reproducibility measures are presented as percentages.
    Ranking BP Selection export can request the 0--1 fractional scale. The
    Reproducibility Score remains fractional.
    """

    raw = {
        key: None if score is None else score.get(key)
        for key in REPRODUCIBILITY_SOURCE_KEYS
    }
    scale = 1 if as_fraction else 100
    values = {
        "reproducibility_structural": (
            None if raw["fidelity_structural"] is None
            else raw["fidelity_structural"] * scale
        ),
        "reproducibility_functional": (
            None if raw["fidelity_functional"] is None
            else raw["fidelity_functional"] * scale
        ),
        "reproducibility_cooccurrence_structural": (
            None if raw["fidelity_cooccurrence_structural"] is None
            else raw["fidelity_cooccurrence_structural"] * scale
        ),
        "reproducibility_cooccurrence_functional": (
            None if raw["fidelity_cooccurrence_functional"] is None
            else raw["fidelity_cooccurrence_functional"] * scale
        ),
        "reproducibility_score": (
            math.sqrt(raw["fidelity_structural"] * raw["fidelity_functional"])
            if raw["fidelity_structural"] is not None
            and raw["fidelity_functional"] is not None
            else None
        ),
    }
    return values


def rich_frequency_header():
    """Create the Frequency (f) header, with only f italicized."""

    return CellRichText(
        "Frequency (",
        TextBlock(InlineFont(i=True), "f"),
        ")",
    )


def assign_primary_groups(results, pocket_centroids):
    """Freeze Pocket/Cluster before GMM refinement.

    Cluster geometry and collective evaluation remain those of original CALS.
    Final classification is protected per pose: a pose does not inherit a
    neighbor's pocket if it does not reach the contact minimum itself.
    """

    clusters = []
    for result in results:
        centroid = np.array([
            result["centroid_x"], result["centroid_y"], result["centroid_z"],
        ])
        for cluster in clusters:
            compatible = all(
                centroid_distance(
                    centroid,
                    np.array([
                        member["centroid_x"], member["centroid_y"],
                        member["centroid_z"],
                    ]),
                ) <= cluster_cutoff
                for member in cluster["members"]
            )
            if compatible:
                cluster["members"].append(result)
                cluster["centroid"] = np.mean([
                    np.array([
                        member["centroid_x"], member["centroid_y"],
                        member["centroid_z"],
                    ])
                    for member in cluster["members"]
                ], axis=0)
                break
        else:
            clusters.append({"centroid": centroid, "members": [result]})

    for index, cluster in enumerate(clusters, start=1):
        cluster_name = f"C{index}"
        pocket_scores = {pocket_name: 0 for pocket_name in pocket_centroids}
        for member in cluster["members"]:
            for pocket_name, contacts in member["ranking"].items():
                pocket_scores[pocket_name] += contacts
        cluster_best_pocket = max(pocket_scores, key=pocket_scores.get)
        cluster_best_contacts = pocket_scores[cluster_best_pocket]
        cluster_distance = centroid_distance(
            cluster["centroid"], pocket_centroids[cluster_best_pocket]
        )
        cluster_matches_cavity = (
            cluster_best_contacts >= min_contacts_for_assignment
            and cluster_distance <= max_centroid_distance
        )

        for member in cluster["members"]:
            member_best_pocket = max(
                member["ranking"], key=member["ranking"].get
            )
            member_best_contacts = member["ranking"][member_best_pocket]
            primary_group = (
                member_best_pocket
                if member_best_contacts >= min_contacts_for_assignment
                else cluster_name
            )

            # Audit trail for the original spatial decision and immutable final
            # classification used by GMM and later analyses.
            member["_spatial_cluster"] = cluster_name
            member["_cluster_best_pocket"] = cluster_best_pocket
            member["_cluster_best_contacts"] = cluster_best_contacts
            member["_cluster_centroid_distance"] = cluster_distance
            member["_cluster_matches_cavity"] = cluster_matches_cavity
            member["_primary_cavity"] = (
                primary_group if primary_group != cluster_name else None
            )
            member["_primary_group"] = primary_group
            member["site"] = primary_group
            member["pocket"] = primary_group

    return clusters


def _run_analysis_staged(
        pocket_files,
        protein_file,
        vina_files,
        ligand_file,
        smiles_file):
    """Prepare an analysis and pause it before global-batch GMM."""

    ###---------- CARREGAR POCKETS ----------###
    
    pockets = {}
    pocket_centroids = {}
    
    for pocket_file in pocket_files:
    
        pocket_name = Path(pocket_file).stem
        pocket_atoms = read_pocket(pocket_file)
        pockets[pocket_name] = pocket_atoms
    
        pocket_centroids[pocket_name] = (
            get_pocket_centroid(
                pocket_atoms
            )
        )
    
    ###---------- LOAD PROTEIN ----------###
    
    protein_atoms = read_protein(protein_file)
    # The original structure continues to feed pockets, contacts, and all
    # consolidated calculations. The copy below is exclusive to the PLIP profile.
    _g2b_temp_dir = make_temp_directory("cals_g2b_")
    protein_h_file = protonate_with_rdkit(protein_file, _g2b_temp_dir.name)
    protein_atoms_plip = read_protein(protein_h_file, select_altloc=True)
    # The K model reads only ATOM records and therefore excludes FMN, water,
    # metals, and every other receptor HETATM by construction.
    protein_interaction_model = prepare_protein_interaction_model(protein_h_file)
    
    #========== REFERENCE MOLECULE ==========#
    
    template = load_ligand_template(ligand_file)
    
    # RDKit can fail to remove a reference-file hydrogen even with
    # removeHs=True when the H atom has no perceived neighbor. This breaks
    # correspondence with Vina poses, which never include hydrogens among RMSD
    # heavy atoms. Polar H is stored separately in "polar_h", so remove any
    # remaining template hydrogens manually.
    
    remaining_h = [
        atom.GetIdx()
        for atom in template.GetAtoms()
        if atom.GetAtomicNum() == 1
    ]
    
    if remaining_h:
    
        editable_template = Chem.RWMol(template)
    
        for idx in sorted(remaining_h, reverse=True):
            editable_template.RemoveAtom(idx)
    
        template = editable_template.GetMol()
    
        try:
            Chem.SanitizeMol(template)
        except Exception:
            pass
    
        print(
            f"Warning: removed {len(remaining_h)} orphan hydrogen atom(s) "
            "manually from the reference ligand because RDKit could not "
            "remove them automatically."
        )
    
    # ASSIGN BOND ORDERS/AROMATICITY FROM A REFERENCE SMILES
    #
    # The formats above vary widely in bond-order reliability. PDB contains no
    # bond order; MOL2/MOL/SDF may be wrong depending on how a file was made.
    # Use the correct ligand SMILES to transfer bond orders to the loaded
    # template while preserving 3D coordinates and atom order required by
    # Vina/PDBQT poses. This corrects only connectivity and bond order. If no
    # SMILES is supplied, retain aromaticity perceived from the ligand file.
    
    ligand_smiles = None
    
    if smiles_file:
    
        with open(smiles_file, "r", encoding="utf-8") as f:
    
            for line in f:
    
                line = line.strip()
    
                if line:
                    ligand_smiles = line
                    break
    
    if ligand_smiles:
    
        ref_mol = Chem.MolFromSmiles(ligand_smiles)
    
        if ref_mol is None:
            print(
                "Warning: RDKit could not interpret the supplied SMILES. "
                "Continuing with aromaticity perceived directly from the "
                "ligand file without correction."
            )
        else:
            try:
                template = assign_bond_orders_deterministically(
                    ref_mol, template
                )
                print(
                    "Ligand bond orders and aromaticity corrected from the "
                    "reference SMILES."
                )
            except Exception as exc:
                print(
                    "Warning: could not transfer bond orders from the SMILES "
                    f"to the ligand read from the file ({exc}). Check whether "
                    "the ligand-file connectivity matches the supplied SMILES. "
                    "Continuing with aromaticity perceived directly from the "
                    "file without correction."
                )
    
    ###---------- INTERACTION PROFILE (ONCE) ----------###
    
    ligand_profile = get_ligand_interaction_profile(template)
    
    print(
        "Ligand interaction profile — "
        f"donors: {len(ligand_profile['donor_atoms'])}, "
        f"acceptors: {len(ligand_profile['acceptor_atoms'])}, "
        f"aromatic rings: {len(ligand_profile['aromatic_rings'])}, "
        f"hydrophobic carbons: {len(ligand_profile['hydrophobic_atoms'])}, "
        f"halogens: {len(ligand_profile['halogens'])}, "
        f"cationic groups: {len(ligand_profile['cationic_groups'])}, "
        f"anionic groups: {len(ligand_profile['anionic_groups'])}"
    )
    
    plip_validation_dir = None
    
    if ENABLE_PLIP_VALIDATION:
        plip_validation_dir = filedialog.askdirectory(
            title="Select the PLIP reference folder (validation only)"
        )
        if not plip_validation_dir:
            print("PLIP validation disabled: no reference folder selected.")
    
    (
        protein_aromatic_rings,
        protein_cationic_groups,
        protein_anionic_groups
    ) = get_protein_ring_and_charge_groups(protein_atoms_plip)
    
    # The temporary RDKit PDB is the sole H source for G2B. Missing H is not
    # idealized so the method remains isolated and auditable.
    protein_donor_h_coordinates = get_explicit_protein_polar_hydrogens(
        protein_h_file, protein_atoms_plip
    )
    if not protein_donor_h_coordinates:
        raise RuntimeError(
            "RDKit generated the protein, but no polar H could be associated "
            "with N/O/S. G2B execution was stopped."
        )
    
    ###---------- LISTAR RESULTADOS ----------##
    
    all_results = []
    
    for vina_file in vina_files:
    
        poses = read_vina_poses(vina_file)
    
        '''mol = pose_to_template_mol(
            poses[0]["atoms"],
            template
        )'''
    
        ligand_name = Path(vina_file).stem
    
        for pose in poses:
    
            canonicalize_pose_atom_order(pose, template)
    
            best_pocket, best_contacts, ranking = classify_pose(
                pose["atoms"],
                pockets
            )
    
            centroid = get_centroid(
                pose["atoms"]
            )
    
            result = {
                "ligand": ligand_name,
                "pose": pose["pose"],
                "energy": pose["energy"],
                "rmsd": pose["rmsd_lb"],
                "pocket": best_pocket,
                "contacts": best_contacts,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_z": centroid[2],
                "ranking": ranking,
                "atoms": pose["atoms"],
                "polar_h": pose.get("polar_h", [])
            }
            
            all_results.append(result)

    ###---------- INTERMOLECULAR INTERACTION TYPES (BY POSE) ----------###
    # Interactions are calculated before spatial classification. Therefore,
    # poses without interactions receive no pocket/site and do not influence
    # clusters, metrics, or statistics of analyzable poses.

    for result in all_results:

        pose_hydrogens, _ = select_variant_hydroxyl_hydrogens(
            protein_atoms_plip,
            protein_donor_h_coordinates,
            result,
            ligand_profile,
            HYDROXYL_SELECTION_MODE,
            hbond_da_cutoff=HBOND_DIST_CUTOFF,
            hbond_ha_cutoff=HBOND_HA_CUTOFF,
            hbond_angle_min=HBOND_ANGLE_MIN,
            steric_weight=H_STERIC_WEIGHT,
            hbond_weight=H_HBOND_WEIGHT,
            acceptance_margin=H_ACCEPTANCE_MARGIN,
        )
        selected_oh_hydrogens = {
            key: coordinates for key, coordinates in pose_hydrogens.items()
            if is_hydroxyl({"residue": key[2], "atom_name": key[3]})
        }
        result["interactions"] = classify_k_interactions(
            result,
            ligand_profile,
            protein_interaction_model,
            protein_hydrogens=selected_oh_hydrogens,
            hbond_diagnostics=[],
        )

    analyzable_results = []

    for result in all_results:

        if result["interactions"]:
            result["excluded_from_analysis"] = False
            analyzable_results.append(result)
            continue

        result["site"] = "N/A"
        result["pocket"] = "N/A"
        result["rmsd_docking"] = None
        result["excluded_from_analysis"] = True
    
    # Pocket/Cluster classification is decided before GMM and remains frozen
    # in the _primary_group field.
    clusters = assign_primary_groups(analyzable_results, pocket_centroids)
    
    ### REFINAMENTO POR GMM ###

    # The initial pocket classification remains intact. The next stage pauses
    # so the batch can gather interactive poses for every ligand per pocket
    # before defining global sites.
    reference_pose = template_reference_coordinates(template)
    for result in analyzable_results:
        result["_gmm_reference_pose"] = reference_pose

    prepared_state = {
        "analyzable_results": analyzable_results,
        "all_results": all_results,
        "pockets": pockets,
        "pocket_centroids": pocket_centroids,
    }
    yield prepared_state

    ###---------- CONTACT COUNT BY RESIDUE ----------###
    
    residue_occupancy = {}
    
    for result in analyzable_results:
    
        contact_residues = get_contact_residues(
                    result["atoms"],
                    protein_atoms
                )
    
        for residue in contact_residues:
    
            key = (
                result["pocket"],
                residue[0],
                residue[1],
            )
    
            residue_occupancy[key] = (
                residue_occupancy.get(key, 0) + 1
            )
    
    def build_interaction_residue_occupancy(results):
        """Count interacting residues/chains once per pose and scope."""
    
        by_cavity = {}
        by_site = {}
    
        for result in results:
            pose_residues = {
                (interaction["residue"], interaction["resnum"], interaction["chain"])
                for interaction in result.get("interactions", [])
            }
    
            for residue, resnum, chain in pose_residues:
                cavity_key = (result["site"], residue, resnum, chain)
                site_key = (
                    result["site"], result["pocket"], residue, resnum, chain
                )
                by_cavity[cavity_key] = by_cavity.get(cavity_key, 0) + 1
                by_site[site_key] = by_site.get(site_key, 0) + 1
    
        return by_cavity, by_site
    
    
    # Official interaction occupancy: each residue contributes at most once per
    # pose, even when that pose has several interactions with it.
    interaction_occupancy_by_cavity, interaction_occupancy_by_site = (
        build_interaction_residue_occupancy(analyzable_results)
    )
    
    # FREQUENCY (%) BY RESIDUE
    
    site_counts = {}
    site_totals = {}
    interaction_site_totals = {}
    
    for result in analyzable_results:
    
        site = result["pocket"]
    
        site_counts[site] = (
                site_counts.get(site, 0) + 1
        )

        interaction_scope = (result["site"], result["pocket"])
        interaction_site_totals[interaction_scope] = (
            interaction_site_totals.get(interaction_scope, 0) + 1
        )
    
        base_site = result["site"]
    
        site_totals[base_site] = (
            site_totals.get(base_site, 0) + 1
        )
    
    # CONSERVATION SCORE FOR EACH POSE
    
    for result in analyzable_results:
    
        result["conservation_score"] = (
            calculate_conservation_score(
                result["atoms"],
                protein_atoms,
                residue_occupancy,
                site_counts,
                result["pocket"]
            )
        )
    
    ###***-------------------- RMSD CONSENSUS --------------------***###
    
    groups = group_results_by_pocket(analyzable_results)
    
    consensus_results = {}
    
    for (cavity_name, pocket_name), poses in groups.items():

        if len(poses) == 1:
            pose = poses[0]
            # A single pose is, by definition, its site's representative.
            # RMSDdocking measures distance to the representative, so it is 0.
            pose["rmsd_docking"] = 0.0
            pose["is_representative"] = True
            consensus = build_single_pose_rmsd_statistics(pose)
            binding = build_single_pose_rmsd_statistics(pose)
            binding["representative_pose"] = {
                "pose_index": 0,
                "ligand": pose["ligand"],
                "pose": pose["pose"],
                "energy": pose["energy"],
                "pocket": pose["pocket"],
                "site": pose.get("site", pose["pocket"]),
                "conservation_score": pose["conservation_score"],
            }

            site_key = cavity_name, pocket_name
            consensus_results[site_key] = {
                "poses": poses,
                "consensus": {
                    "matrix": np.zeros((1, 1), dtype=float),
                    "statistics": consensus,
                },
                "binding": {
                    "matrix": np.zeros((1, 1), dtype=float),
                    "statistics": binding,
                },
            }
            continue
    
        consensus_matrix = build_rmsd_matrix(
            poses,
            template,
            method="consensus"
        )
    
        binding_matrix = build_rmsd_matrix(
            poses,
            template,
            method="binding"
        )
    
        consensus = calculate_rmsd_statistics(
            consensus_matrix,
            poses
        )
    
        binding = calculate_rmsd_statistics(
            binding_matrix,
            poses
        )
        representative_pose, representative_diagnostics = select_representative_boltzmann(poses, template)
        binding["representative_pose"] = representative_pose
        binding["representative_diagnostics"] = representative_diagnostics
        print(
            f"Representative (Boltzmann-weighted consensus): {cavity_name}/{pocket_name}, "
            f"pose {representative_pose['pose']}; diagnostics {representative_diagnostics}."
        )
    
        # RMSDdocking: PCM without alignment between each pose and the site
        # representative selected by the configured method. Calculate it once
        # here and attach it to each pose for reuse in individual pose tables.
        representative_pose = binding["representative_pose"]
        representative_idx = representative_pose["pose_index"]
    
        for idx, pose in enumerate(poses):
    
            pose["rmsd_docking"] = binding_matrix[idx][representative_idx]
            pose["is_representative"] = (idx == representative_idx)
    
        site_key = cavity_name, pocket_name

        consensus_results[site_key] = {
    
            "poses": poses,
    
            "consensus": {
    
                "matrix": consensus_matrix,
    
                "statistics": consensus
    
            },
    
            "binding": {
    
                "matrix": binding_matrix,
    
                "statistics": binding
    
            }
    
        }
    
    ###***---------------- RMSD CONSENSUS BY POCKET ----------------***###
    # Same calculation as above, but group by Pocket instead of Site; use only
    # for the "Pocket Summary" sheet.
    
    groups_by_cavity = {}
    
    for result in analyzable_results:
        groups_by_cavity.setdefault(result["site"], []).append(result)
    
    cavity_results = {}
    
    for cavity_name, poses in groups_by_cavity.items():

        if len(poses) == 1:
            cavity_results[cavity_name] = {
                "poses": poses,
                "consensus": build_single_pose_rmsd_statistics(poses[0]),
                "binding": build_single_pose_rmsd_statistics(poses[0]),
            }
            continue
    
        cavity_consensus_matrix = build_rmsd_matrix(
            poses,
            template,
            method="consensus"
        )
    
        cavity_binding_matrix = build_rmsd_matrix(
            poses,
            template,
            method="binding"
        )
    
        cavity_results[cavity_name] = {
            "poses": poses,
            "consensus": calculate_rmsd_statistics(
                cavity_consensus_matrix,
                poses
            ),
            "binding": calculate_rmsd_statistics(
                cavity_binding_matrix,
                poses
            ),
        }

    # RANKING DAS POSES POR POCKET
    
    ranking_rows = []

    cavity_pose_counts = {}
    site_pose_counts = {}

    for result in analyzable_results:
        cavity_name, pocket_name = get_site_key(result)
        cavity_pose_counts[cavity_name] = (
            cavity_pose_counts.get(cavity_name, 0) + 1
        )
        site_key = (cavity_name, pocket_name)
        site_pose_counts[site_key] = site_pose_counts.get(site_key, 0) + 1
    
    
    for (site_name, cluster_name), data in consensus_results.items():
    
        consensus = data["consensus"]["statistics"]
        binding = data["binding"]["statistics"]
    
        poses = data["poses"]
    
        consensus_statistics = consensus["statistics"]
        binding_statistics = binding["statistics"]
    
        # Sorting:
        # 1º energy mais negativa
        # 2º menor RMSDdocking
        # 3º menor PCMi
    
        ranked = sorted(
            consensus_statistics,
            key=lambda x: (
                x["energy"],
                poses[x["pose_index"]]["rmsd_docking"],
                binding_statistics[x["pose_index"]]["mean_rmsd"],
            )
        )
    
        for rank, pose in enumerate(ranked, start=1):
    
            binding_pose = binding_statistics[pose["pose_index"]]
    
            ranking_rows.append({
    
                "ligand": pose["ligand"],
    
                "site": site_name,
    
                "cluster": cluster_name,
    
                "rank": rank,
    
                "pose": pose["pose"],
    
                "energy": pose["energy"],
    
                "rmsd_docking":
                    poses[pose["pose_index"]]["rmsd_docking"],
    
                # SCMi
    
                "mean_rmsd_consensus":
                    pose["mean_rmsd"],
    
                "std_consensus":
                    pose["std_rmsd"],
    
                # PCMi
    
                "mean_rmsd_binding":
                    binding_pose["mean_rmsd"],
    
                "std_binding":
                    binding_pose["std_rmsd"]
    
            })
    
    df_ranking = pd.DataFrame(
        order_ranking_rows(
            ranking_rows,
            cavity_pose_counts,
            site_pose_counts,
        ),
        columns=[
            "ligand", "site", "cluster", "rank", "pose", "energy",
            "rmsd_docking", "mean_rmsd_consensus", "std_consensus",
            "mean_rmsd_binding", "std_binding",
        ],
    )
    
    # RESUMO POR POCKET
    
    summary_rows = []
    
    for (site_name, cluster_name), data in consensus_results.items():
    
        consensus = data["consensus"]["statistics"]
        binding = data["binding"]["statistics"]
    
        poses = data["poses"]
    
        statistics = consensus["statistics"]
    
        ensemble = consensus["ensemble"]
        ensemble_binding = binding["ensemble"]
    
        # Melhor energy de afinidade
    
        best_energy_pose = min(
            poses,
            key=lambda x: x["energy"]
        )
    
        # Localizar o RMSD consensus da pose de melhor energy
    
        best_energy_stats = next(
            row for row in statistics
            if row["pose"] ==
            best_energy_pose["pose"]
        )
    
        summary_rows.append({
    
            "ligand": poses[0]["ligand"],
    
            "site": site_name,
    
            "cluster": cluster_name,
    
            "n_poses": ensemble["n_poses"],
    
            "best_energy": best_energy_pose["energy"],
    
            "best_energy_pose": best_energy_pose["pose"],
    
            "best_energy_pose_mean_rmsd":
                best_energy_stats["mean_rmsd"],
    
            "ensemble_mean_rmsd":
                ensemble["mean_rmsd"],
    
            "ensemble_median_rmsd":
                ensemble["median_rmsd"],
    
            "ensemble_std_rmsd":
                ensemble["std_rmsd"],
    
            "ensemble_mean_rmsd_binding":
                ensemble_binding["mean_rmsd"],
    
            "ensemble_median_rmsd_binding":
                ensemble_binding["median_rmsd"],
    
            "ensemble_std_rmsd_binding":
                ensemble_binding["std_rmsd"]
    
        })
    
    df_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "ligand", "site", "cluster", "n_poses", "best_energy",
            "best_energy_pose", "best_energy_pose_mean_rmsd",
            "ensemble_mean_rmsd", "ensemble_median_rmsd",
            "ensemble_std_rmsd", "ensemble_mean_rmsd_binding",
            "ensemble_median_rmsd_binding", "ensemble_std_rmsd_binding",
        ],
    )
    
    # POCKET SUMMARY
    
    summary_cavity_rows = []
    
    for cavity_name, data in cavity_results.items():
    
        consensus = data["consensus"]
        binding = data["binding"]
    
        poses = data["poses"]
    
        statistics = consensus["statistics"]
    
        ensemble = consensus["ensemble"]
        ensemble_binding = binding["ensemble"]
    
        best_energy_pose = min(
            poses,
            key=lambda x: x["energy"]
        )
    
        best_energy_stats = next(
            row for row in statistics
            if row["pose"] ==
            best_energy_pose["pose"]
        )
    
        summary_cavity_rows.append({
    
            "ligand": poses[0]["ligand"],
    
            "site": cavity_name,
    
            "n_poses": ensemble["n_poses"],
    
            "best_energy": best_energy_pose["energy"],
    
            "best_energy_pose": best_energy_pose["pose"],
    
            "best_energy_pose_mean_rmsd":
                best_energy_stats["mean_rmsd"],
    
            "ensemble_mean_rmsd":
                ensemble["mean_rmsd"],
    
            "ensemble_median_rmsd":
                ensemble["median_rmsd"],
    
            "ensemble_std_rmsd":
                ensemble["std_rmsd"],
    
            "ensemble_mean_rmsd_binding":
                ensemble_binding["mean_rmsd"],
    
            "ensemble_median_rmsd_binding":
                ensemble_binding["median_rmsd"],
    
            "ensemble_std_rmsd_binding":
                ensemble_binding["std_rmsd"]
    
        })
    
    df_summary_cavity = pd.DataFrame(
        summary_cavity_rows,
        columns=[
            "ligand", "site", "n_poses", "best_energy", "best_energy_pose",
            "best_energy_pose_mean_rmsd", "ensemble_mean_rmsd",
            "ensemble_median_rmsd", "ensemble_std_rmsd",
            "ensemble_mean_rmsd_binding", "ensemble_median_rmsd_binding",
            "ensemble_std_rmsd_binding",
        ],
    )
    
    #
    consensus_pose_rows = []
    consolidated_conformational_rows = []
    
    for (site_name, cluster_name), data in consensus_results.items():
    
        poses = data["poses"]
    
        consensus = data["consensus"]["statistics"]
        binding = data["binding"]["statistics"]
    
        binding_statistics = binding["statistics"]
    
        # Site representative pose = method configured for this variant.
        representative = binding["representative_pose"]
        representative_idx = representative["pose_index"]
    
        for row in consensus["statistics"]:
    
            new_row = row.copy()
    
            # SCMi: retain only mean and standard deviation; median/minimum/
            # maximum were removed to keep the table concise.
            new_row.pop("median_rmsd", None)
            new_row.pop("min_rmsd", None)
            new_row.pop("max_rmsd", None)
    
            binding_row = binding_statistics[row["pose_index"]]
    
            # PCMi: likewise, only mean and standard deviation.
            new_row["mean_rmsd_binding"] = binding_row["mean_rmsd"]
            new_row["std_rmsd_binding"] = binding_row["std_rmsd"]
    
            # RMSDdocking was already calculated once and attached to the pose.
            new_row["rmsd_docking"] = poses[row["pose_index"]]["rmsd_docking"]
    
            new_row["representative"] = (
                row["pose_index"] == representative_idx
            )
            if new_row["representative"]:
                new_row.update(binding.get("representative_diagnostics", {}))
    
            consensus_pose_rows.append(new_row)
            consolidated_row = new_row.copy()
            consolidated_row["combined_pose_label"] = poses[
                row["pose_index"]
            ].get("_combined_pose_label")
            consolidated_conformational_rows.append(consolidated_row)

    df_consensus = pd.DataFrame(consensus_pose_rows)
    if df_consensus.empty:
        df_consensus = pd.DataFrame(columns=[
            "pose_index", "ligand", "pose", "energy", "pocket", "site",
            "conservation_score", "mean_rmsd", "std_rmsd",
            "mean_rmsd_binding", "std_rmsd_binding", "rmsd_docking",
            "representative",
        ])

    # This list belongs exclusively to the general batch report. The individual workbook
    # continua usando df_consensus, sem colunas adicionais.
    prepared_state["conformational_rows"] = consolidated_conformational_rows
    
    ###***-------------------- CRIAR PLANILHA --------------------***###
    
    # English column names standardized across every sheet.
    
    COLUMN_NAMES_PT = {
        "ligand": "Ligand",
        "site": "Pocket",
        "pocket": "Site",
        "cluster": "Site",
        "pose": "Pose",
        "energy": "Energy (kcal/mol)",
        "rmsd": "RMSD (Vina)",
        "contacts": "Contacts",
        "centroid_x": "Centroid X",
        "centroid_y": "Centroid Y",
        "centroid_z": "Centroid Z",
        "conservation_score": "Conservation Score",
        "n_poses": "Pose Count",
        "rank": "Ranking",
    
        # SCMi / SCM -- structural consistency metric (RMSD with alignment).
        # "i" is the individual pose value; without it, the value applies to
        # the site/pocket across all pairwise poses.
        "mean_rmsd": "SCMi (Å)",
        "std_rmsd": "SCMi - Standard Deviation (Å)",
        "mean_rmsd_consensus": "SCMi (Å)",
        "std_consensus": "SCMi - Standard Deviation (Å)",
        "ensemble_mean_rmsd": "SCM (Å)",
        "ensemble_std_rmsd": "SCM - Standard Deviation (Å)",
        "ensemble_nposes": "Pose Count (Ensemble)",
    
        # PCMi / PCM -- positional consistency metric (RMSD without alignment).
        "mean_rmsd_binding": "PCMi (Å)",
        "std_binding": "PCMi - Standard Deviation (Å)",
        "std_rmsd_binding": "PCMi - Standard Deviation (Å)",
        "ensemble_mean_rmsd_binding": "PCM (Å)",
        "ensemble_std_rmsd_binding": "PCM - Standard Deviation (Å)",
    
        # RMSDdocking -- PCM between the pose and the site representative
        # selected by the configured method.
        "rmsd_docking": "RMSDdocking (Å)",
        "representative": "Representative Pose (Boltzmann Consensus)",
        "representative_effective_sample_size": "Effective Sample Size (Kish)",
    
        "best_energy": "Best Energy",
        "best_energy_pose": "Pose (Best Energy)",
        "best_energy_pose_mean_rmsd": "SCMi (Best Energy) (Å)",
        "pose_index": "Pose Index",
        "energy_factor": "Energy Factor",
        "rmsd_factor": "RMSD Factor",
        "conservation_factor": "Conservation Factor",
        "reproducibility_structural": "Structural Reproducibility (%)",
        "reproducibility_functional": "Functional Reproducibility (%)",
        "reproducibility_cooccurrence_structural": "Co-occurrence Structural Reproducibility (%)",
        "reproducibility_cooccurrence_functional": "Co-occurrence Functional Reproducibility (%)",
        "reproducibility_score": "Reproducibility Score",
        "energy_exp": "Energy (exp)",
        "score_bp": "Score BP",
        "CALS": "CALS",
        "Ranking": "Ranking",
    }
    
    # Identifier columns always precede metric columns. Other columns retain
    # their existing order.
    # Ligand, Pose, Pocket (site), Site (pocket/cluster)
    ID_COLUMN_ORDER = ["ligand", "pose", "site", "cluster", "pocket"]
    
    # After identifier columns, retain the requested metric sequence whenever
    # those fields are present; other columns retain their existing order.
    METRIC_COLUMN_ORDER = [
        "energy",
        "rmsd_docking",
        "mean_rmsd", "mean_rmsd_consensus",
        "mean_rmsd_binding",
    ]
    
    
    def reorder_id_columns(df):
    
        front = [c for c in ID_COLUMN_ORDER if c in df.columns]
    
        metrics = [
            c for c in METRIC_COLUMN_ORDER
            if c in df.columns and c not in front
        ]
    
        rest = [
            c for c in df.columns
            if c not in front and c not in metrics
        ]
    
        return df[front + metrics + rest]
    
    
    def rename_columns_pt(df):
        df = reorder_id_columns(df)
        return df.rename(columns=COLUMN_NAMES_PT)


    def format_columns(ws, formats, header_row=1, data_last_row=None):
        """Format columns by header NAME rather than a fixed letter.

        This prevents formatting from breaking when sheet columns are added,
        removed, or reordered.
        """
    
        headers = [str(cell.value) if cell.value is not None else None for cell in ws[header_row]]
        data_last_row = data_last_row or ws.max_row
    
        for header, number_format in formats.items():
    
            if header not in headers:
                continue
    
            col_idx = headers.index(header) + 1
    
            for row in range(header_row + 1, data_last_row + 1):
                ws.cell(row=row, column=col_idx).number_format = number_format
    
    
    def write_df_sheet(wb, title, df):
        """Create a sheet from a DataFrame with final column names.

        Final sheet order is set later in DESIRED_SHEET_ORDER.
        """
    
        ws = wb.create_sheet(title=title)
    
        ws.append(list(df.columns))
    
        for row in df.itertuples(index=False):
            ws.append(list(row))
    
        return ws
    
    
    def append_df_section(ws, section_title, df):
        """Append an identified table to an existing sheet.

        Return the header and final-data rows so each table can receive its
        own numeric formatting.
        """
    
        ws.append([section_title])
        ws.append(list(df.columns))
        header_row = ws.max_row
    
        for row in df.itertuples(index=False):
            ws.append(list(row))
    
        return header_row, ws.max_row


    def write_local_network_sheet(title, node_rows, edge_rows):
        """Exporta, lado a lado, os nodes e as arestas de uma rede local."""

        ws = wb.create_sheet(title=title)
        node_df = pd.DataFrame(node_rows, columns=LOCAL_NODE_COLUMNS)
        node_df.columns = [
            "Residue" if column == "Node" and "Structural" in title
            else "Res-Int" if column == "Node" else column
            for column in node_df.columns
        ]
        edge_df = pd.DataFrame(edge_rows, columns=LOCAL_EDGE_COLUMNS)

        def write_table(start_column, section_title, frame):
            ws.cell(row=1, column=start_column, value=section_title)
            for offset, column in enumerate(frame.columns):
                cell = ws.cell(row=2, column=start_column + offset, value=column)
                if column == "Frequency (f)":
                    cell.value = rich_frequency_header()
            for row_index, row in enumerate(frame.itertuples(index=False), start=3):
                for offset, value in enumerate(row):
                    ws.cell(row=row_index, column=start_column + offset, value=value)
            return 2, max(2, len(frame) + 2)

        node_header, node_last = write_table(1, "Nodes", node_df)
        edge_start = len(node_df.columns) + 3
        edge_header, edge_last = write_table(edge_start, "Edges", edge_df)
        for header, last in ((node_header, node_last), (edge_header, edge_last)):
            format_columns(ws, {
                "Frequency (f)": FACTOR_FMT,
                "Hub Strength": FACTOR_FMT,
                "Centrality": FACTOR_FMT,
                "Raw Weight": FACTOR_FMT,
                "Normalized Weight": FACTOR_FMT,
            }, header, last)
        for column in range(edge_start, edge_start + len(edge_df.columns)):
            if str(ws.cell(row=edge_header, column=column).value) == "Frequency (f)":
                for row in range(edge_header + 1, edge_last + 1):
                    ws.cell(row=row, column=column).number_format = FACTOR_FMT
        return ws
    
    
    # Default numeric formats: two decimal places to reduce excess digits and
    # one decimal place for percentages.
    
    RMSD_FMT = "0.000"
    ENERGY_FMT = "0.000"
    COORD_FMT = "0.000"
    FACTOR_FMT = "0.000"
    PERCENT_FMT = "0.0"
    
    ###---------- SHEET: GENERAL INFORMATION ----------###
    
    GENERAL_COLUMN_ORDER = [
        "ligand", "pose", "site", "pocket", "energy", "rmsd",
        "centroid_x", "centroid_y", "centroid_z",
    ]
    
    excel_results = []
    
    for result in all_results:
    
        temp = {
            key: result.get(key)
            for key in GENERAL_COLUMN_ORDER
        }
    
        excel_results.append(temp)
    
    df_general = rename_columns_pt(pd.DataFrame(excel_results))
    
    excel_file = build_excel_output_path(vina_files, ligand_file)
    
    from openpyxl import Workbook
    
    wb = Workbook()
    ws = wb.active
    ws.title = "General Information"
    
    # Writing is deferred until reproducibility scores are available, without
    # changing previous pose calculations.

    ###---------- SHEET: POCKETS ----------###

    df_cavities = pd.DataFrame(
        build_cavity_rows(pockets),
        columns=["Pocket", "Residue", "Group"],
    )
    write_df_sheet(wb, "Pockets", df_cavities)
    
    ###---------- SHEET: INTERACTIONS AND TYPES ----------###
    # One row per (pose, residue, interaction type) -- base for networks of
    # hub residues with intermolecular interaction type and distance (Å).
    
    interaction_rows = []
    network_interaction_rows = []
    
    for result in all_results:

        if not result["interactions"]:
            # The interactions sheet must retain one row per pose, even
            # quando nenhuma interaction molecular foi detectada. Essa linha
            # informativa not entra nas redes nem no Random Forest.
            interaction_rows.append({
                "ligand": result["ligand"],
                "site": result["site"],
                "pocket": result["pocket"],
                "pose": result["pose"],
                "energy": result["energy"],
                "residue": None,
                "interaction_type": "No interactions detected",
                "strength": None,
                "distance_contact": None,
                "distance_da": None,
                "distance_ha": None,
                "angle": None,
                "ligand_atom": None,
                "protein_atom": None,
            })
            continue

        for interaction in result["interactions"]:

            interaction_row = {
                "ligand": result["ligand"],
                "site": result["site"],
                "pocket": result["pocket"],
                "pose": result["pose"],
                "energy": result["energy"],
                "residue": format_interaction_residue(
                    interaction["residue"], interaction["resnum"], interaction["chain"]
                ),
                "interaction_type": interaction["type"],
                "strength": classify_hbond_strength(
                    interaction["type"], interaction.get("distance_da")
                ),
                "distance_contact": (interaction["distance"] if interaction["type"] != "H-Bond" else None),
                "distance_da": interaction.get("distance_da"),
                "distance_ha": interaction.get("distance_ha"),
                "angle": interaction["angle"],
                "ligand_atom": interaction["ligand_atom"],
                "protein_atom": interaction["protein_atom"],
            }
            interaction_rows.append(interaction_row)
            network_interaction_rows.append(interaction_row)
    
    df_interactions = rename_columns_pt(
        pd.DataFrame(interaction_rows)
    ).rename(columns={
        "residue": "Residue",
        "interaction_type": "Interaction Type",
        "strength": "Strength",
        "distance_contact": "Interaction Distance (Å)",
        "distance_da": "Donor···Acceptor (Å)",
        "distance_ha": "H···Acceptor (Å)",
        "angle": "Angle (°)",
        "ligand_atom": "Ligand Atom",
        "protein_atom": "Protein Atom",
    })
    
    ws_interactions = write_df_sheet(
        wb, "Interactions by Type", df_interactions
    )
    
    format_columns(ws_interactions, {
        "Interaction Distance (Å)": RMSD_FMT,
        "Donor···Acceptor (Å)": RMSD_FMT,
        "H···Acceptor (Å)": RMSD_FMT,
        "Angle (°)": "0.0",
    })

    # The following six tables use only interactions already calculated above;
    # they do not introduce a new contact calculation.
    network_tables = build_network_tables(network_interaction_rows)
    local_network_tables = build_local_network_fidelity(network_interaction_rows)
    local_score_by_pose = {
        (row["ligand"], row["site"], row["pocket"], row["pose_key"]): row
        for row in local_network_tables["pose_scores"]
    }

    def individual_score_for(row, pocket_key="pocket"):
        return local_score_by_pose.get(
            (
                canonical_network_ligand(row["ligand"]),
                row["site"], row[pocket_key],
                (row["ligand"], row["pose"]),
            )
        )

    def general_score_for(row):
        return local_score_by_pose.get(
            (
                canonical_network_ligand(row["Ligand"]),
                row["Pocket"], row["Site"],
                (row["Ligand"], row["Pose"]),
            )
        )

    general_score_headers = {
        "reproducibility_structural": "Structural Reproducibility (%)",
        "reproducibility_functional": "Functional Reproducibility (%)",
        "reproducibility_cooccurrence_structural": (
            "Co-occurrence Structural Reproducibility (%)"
        ),
        "reproducibility_cooccurrence_functional": (
            "Co-occurrence Functional Reproducibility (%)"
        ),
    }
    for source_column, display_column in general_score_headers.items():
        df_general[display_column] = [
            local_network_score_values(general_score_for(row)).get(source_column)
            for row in df_general.to_dict("records")
        ]

    general_bp_rows = []
    for row in df_general.to_dict("records"):
        values = local_network_score_values(
            general_score_for(row), as_fraction=True
        )
        general_bp_rows.append({
            "site": row["Pocket"],
            "pocket": row["Site"],
            "energy": row["Energy (kcal/mol)"],
            "reproducibility_structural": values["reproducibility_structural"],
            "reproducibility_functional": values["reproducibility_functional"],
        })
    add_score_bp_values(general_bp_rows, ("site", "pocket"))
    df_general["Score BP"] = [row["score_bp"] for row in general_bp_rows]
    df_general = df_general.drop(columns=[
        "Contacts", "Score BP",
        "Co-occurrence Structural Reproducibility (%)",
        "Co-occurrence Functional Reproducibility (%)",
    ], errors="ignore")

    for frame, pocket_key, as_fraction in (
        (df_consensus, "pocket", False),
        (df_ranking, "cluster", True),
    ):
        score_columns = (
            "reproducibility_structural",
            "reproducibility_functional",
            "reproducibility_cooccurrence_structural",
            "reproducibility_cooccurrence_functional",
        )
        for column in score_columns:
            frame[column] = [
                local_network_score_values(
                    individual_score_for(row, pocket_key), as_fraction
                ).get(column)
                for row in frame.to_dict("records")
            ]

    consensus_bp_rows = []
    for row in df_consensus.to_dict("records"):
        values = local_network_score_values(
            individual_score_for(row, "pocket"), as_fraction=True
        )
        consensus_bp_rows.append({
            "site": row["site"],
            "pocket": row["pocket"],
            "energy": row["energy"],
            "reproducibility_structural": values["reproducibility_structural"],
            "reproducibility_functional": values["reproducibility_functional"],
        })
    add_score_bp_values(consensus_bp_rows, ("site", "pocket"))
    df_consensus["score_bp"] = [row["score_bp"] for row in consensus_bp_rows]

    df_ranking = pd.DataFrame(
        order_ranking_rows(
            df_ranking.to_dict("records"),
            cavity_pose_counts,
            site_pose_counts,
        )
    )

    ws.append(list(df_general.columns))
    for row in df_general.itertuples(index=False):
        ws.append(list(row))
    format_columns(ws, {
        "Energy (kcal/mol)": ENERGY_FMT,
        "RMSD (Vina)": RMSD_FMT,
        "Centroid X": COORD_FMT,
        "Centroid Y": COORD_FMT,
        "Centroid Z": COORD_FMT,
        "Score BP": FACTOR_FMT,
        "Structural Reproducibility (%)": PERCENT_FMT,
        "Functional Reproducibility (%)": PERCENT_FMT,
        "Co-occurrence Structural Reproducibility (%)": PERCENT_FMT,
        "Co-occurrence Functional Reproducibility (%)": PERCENT_FMT,
    })
    ws_structural = write_df_sheet(
        wb, "Structural",
        pd.DataFrame(network_tables["structural"], columns=STRUCTURAL_COLUMNS),
    )
    ws_functional = write_df_sheet(
        wb, "Functional",
        pd.DataFrame(network_tables["functional"], columns=FUNCTIONAL_COLUMNS),
    )
    ws_random_forest = write_df_sheet(
        wb, "ML Analysis",
        pd.DataFrame(network_tables["random_forest"], columns=RANDOM_FOREST_COLUMNS),
    )
    format_columns(ws_random_forest, {"Energy (kcal/mol)": ENERGY_FMT})
    ws_edges_structural = write_df_sheet(
        wb, "Edges Structural",
        pd.DataFrame(network_tables["edges_structural"], columns=EDGE_COLUMNS),
    )
    ws_nodes_structural = write_df_sheet(
        wb, "Nodes Structural",
        pd.DataFrame(network_tables["nodes_structural"], columns=NODE_COLUMNS),
    )
    ws_edges_functional = write_df_sheet(
        wb, "Edges Functional",
        pd.DataFrame(network_tables["edges_functional"], columns=EDGE_COLUMNS),
    )
    ws_nodes_functional = write_df_sheet(
        wb, "Nodes Functional",
        pd.DataFrame(network_tables["nodes_functional"], columns=NODE_COLUMNS),
    )
    write_local_network_sheet(
        "Structural by Site",
        local_network_tables["local_nodes_structural"],
        local_network_tables["local_edges_structural"],
    )
    write_local_network_sheet(
        "Functional by Site",
        local_network_tables["local_nodes_functional"],
        local_network_tables["local_edges_functional"],
    )
    
    # Optional validation is still calculated when enabled, but its result is
    # no longer exported as an auxiliary sheet.
    if plip_validation_dir:
        build_plip_validation_rows(
            plip_validation_dir,
            analyzable_results,
            protein_donor_h_coordinates,
        )
    
    ###---------- SHEET: POSE COORDINATES ----------###
    # One row per pose (Site, Cluster, centroid coordinates), useful for 3D
    # scatter plots and visualizing zones with more poses.
    
    pose_coord_rows = []
    
    for result in all_results:
    
        pose_coord_rows.append({
            "ligand": result["ligand"],
            "site": result["site"],
            "pocket": result["pocket"],
            "pose": result["pose"],
            "centroid_x": result["centroid_x"],
            "centroid_y": result["centroid_y"],
            "centroid_z": result["centroid_z"],
        })
    
    df_pose_coords = rename_columns_pt(pd.DataFrame(pose_coord_rows))
    
    ws_pose_coords = write_df_sheet(
        wb, "Pose Coordinates", df_pose_coords
    )
    
    format_columns(ws_pose_coords, {
        "Centroid X": COORD_FMT,
        "Centroid Y": COORD_FMT,
        "Centroid Z": COORD_FMT,
    })

    ###---------- SHEET: POCKET AND SITE CENTROIDS ----------###
    
    cluster_centroid_rows = []
    
    clusters_grouped = group_results_by_pocket(analyzable_results)
    
    for (site_name, cluster_name), members in clusters_grouped.items():
    
        xs = [m["centroid_x"] for m in members]
        ys = [m["centroid_y"] for m in members]
        zs = [m["centroid_z"] for m in members]
    
        cluster_centroid_rows.append({
            "site": site_name,
            "pocket": cluster_name,
            "n_poses": len(members),
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
            "centroid_z": float(np.mean(zs)),
        })
    
    df_cluster_centroids = rename_columns_pt(
        pd.DataFrame(
            cluster_centroid_rows,
            columns=[
                "site", "pocket", "n_poses", "centroid_x", "centroid_y",
                "centroid_z",
            ],
        ).sort_values(["site", "pocket"])
    )
    
    df_pocket_centroids = rename_columns_pt(
        pd.DataFrame([
            {
                "site": pocket_name,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_z": centroid[2],
            }
            for pocket_name, centroid in pocket_centroids.items()
        ])

    )
    
    ws_centroids = wb.create_sheet(title="Centroids")
    header_cavities, last_cavities = append_df_section(
        ws_centroids, "Pockets", df_pocket_centroids
    )
    
    format_columns(ws_centroids, {
        "Centroid X": COORD_FMT,
        "Centroid Y": COORD_FMT,
        "Centroid Z": COORD_FMT,
    }, header_cavities, last_cavities)
    
    # Leave a blank row between the two tables to preserve separate reading.
    ws_centroids.append([])
    header_sites, last_sites = append_df_section(
        ws_centroids, "Sites", df_cluster_centroids
    )
    
    format_columns(ws_centroids, {
        "Centroid X": COORD_FMT,
        "Centroid Y": COORD_FMT,
        "Centroid Z": COORD_FMT,
    }, header_sites, last_sites)
    
    ###---------- SHEETS: INTERACTIONS BY RESIDUE ----------###
    # Duas abas separadas: uma agregada por Pocket (independente do site/
    # subcluster) and another by Site (fine subdivision within the pocket).
    
    interaction_cavity_rows = []
    
    for key, count in sorted(
            interaction_occupancy_by_cavity.items(),
            key=lambda x: (str(x[0][0]), -x[1])
            ):
    
        site, residue, resnum, chain = key
    
        frequency = (count / site_totals[site]) * 100
    
        interaction_cavity_rows.append({
            "cavity": site,
            "residue": format_interaction_residue(residue, resnum, chain),
            "interactions": count,
            "frequency": frequency,
        })
    
    df_interaction_cavity = pd.DataFrame(
        interaction_cavity_rows,
        columns=["cavity", "residue", "interactions", "frequency"],
    ).rename(columns={
        "cavity": "Pocket",
        "residue": "Residue",
        "interactions": "Interactions",
        "frequency": "Frequency (%)",
    })
    
    ws_interaction_cavity = write_df_sheet(
        wb, "Interactions by Pocket", df_interaction_cavity
    )
    
    format_columns(ws_interaction_cavity, {"Frequency (%)": PERCENT_FMT})
    
    interaction_site_rows = []
    
    for key, count in sorted(
            interaction_occupancy_by_site.items(),
            key=lambda x: (str(x[0][0]), str(x[0][1]), -x[1])
            ):
    
        cavity_name, cluster_name, residue, resnum, chain = key
    
        frequency = (count / interaction_site_totals[(cavity_name, cluster_name)]) * 100
    
        interaction_site_rows.append({
            "cavity": cavity_name,
            "site": cluster_name,
            "residue": format_interaction_residue(residue, resnum, chain),
            "interactions": count,
            "frequency": frequency,
        })
    
    df_interaction_site = pd.DataFrame(
        interaction_site_rows,
        columns=["cavity", "site", "residue", "interactions", "frequency"],
    ).rename(columns={
        "cavity": "Pocket",
        "site": "Site",
        "residue": "Residue",
        "interactions": "Interactions",
        "frequency": "Frequency (%)",
    })
    
    ws_interaction_site = write_df_sheet(
        wb, "Interactions by Site", df_interaction_site
    )
    
    format_columns(ws_interaction_site, {"Frequency (%)": PERCENT_FMT})
    
    ###---------- SHEET: CONFORMATIONAL ANALYSIS ----------###
    
    df_consensus_pt = rename_columns_pt(
        df_consensus.drop(columns=[
            "conservation_score", "pose_index", "rmsd_docking",
            "mean_rmsd", "std_rmsd", "mean_rmsd_consensus", "std_consensus",
            "ensemble_mean_rmsd", "ensemble_std_rmsd",
            "mean_rmsd_binding", "std_binding", "std_rmsd_binding",
            "ensemble_mean_rmsd_binding", "ensemble_std_rmsd_binding",
            "best_energy_pose_mean_rmsd", "representative",
            "representative_effective_sample_size",
            "reproducibility_cooccurrence_structural",
            "reproducibility_cooccurrence_functional",
        ], errors="ignore")
    )
    
    ws_consensus = write_df_sheet(
        wb, "Conformational Analysis", df_consensus_pt
    )
    
    format_columns(ws_consensus, {
        "Score BP": FACTOR_FMT,
        "Structural Reproducibility (%)": PERCENT_FMT,
        "Functional Reproducibility (%)": PERCENT_FMT,
    })
    
    ###---------- SHEET: POCKET SUMMARY ----------###
    
    df_summary_cavity_pt = rename_columns_pt(df_summary_cavity.drop(columns=[
        "mean_rmsd", "std_rmsd", "mean_rmsd_consensus", "std_consensus",
        "ensemble_mean_rmsd", "ensemble_std_rmsd", "ensemble_nposes",
        "mean_rmsd_binding", "std_binding", "std_rmsd_binding",
        "ensemble_mean_rmsd_binding", "ensemble_std_rmsd_binding",
        "best_energy_pose_mean_rmsd",
    ], errors="ignore"))
    
    ws_summary_cavity = write_df_sheet(
        wb, "Pocket Summary", df_summary_cavity_pt
    )
    
    format_columns(ws_summary_cavity, {
        "Best Energy": ENERGY_FMT,
    })
    
    ###---------- SHEET: SITE SUMMARY ----------###
    
    df_summary_pt = rename_columns_pt(df_summary.drop(columns=[
        "mean_rmsd", "std_rmsd", "mean_rmsd_consensus", "std_consensus",
        "ensemble_mean_rmsd", "ensemble_std_rmsd", "ensemble_nposes",
        "mean_rmsd_binding", "std_binding", "std_rmsd_binding",
        "ensemble_mean_rmsd_binding", "ensemble_std_rmsd_binding",
        "best_energy_pose_mean_rmsd",
    ], errors="ignore"))
    
    ws_summary = write_df_sheet(wb, "Site Summary", df_summary_pt)
    
    format_columns(ws_summary, {
        "Best Energy": ENERGY_FMT,
    })
    
    ###---------- ABA: RANKING CONSENSUS ----------###
    
    ranking_columns = [
        "ligand", "pose", "site", "cluster", "energy",
        "reproducibility_structural", "reproducibility_functional",
        "score_bp", "rank",
    ]
    df_ranking_pt = df_ranking[ranking_columns].rename(columns=COLUMN_NAMES_PT)
    for header in (
        "Structural Reproducibility (%)",
        "Functional Reproducibility (%)",
    ):
        df_ranking_pt[header] = df_ranking_pt[header] * 100
    
    ws_ranking = write_df_sheet(wb, "Ranking BP Selection", df_ranking_pt)
    
    format_columns(ws_ranking, {
        "Energy (kcal/mol)": ENERGY_FMT,
        "Structural Reproducibility (%)": PERCENT_FMT,
        "Functional Reproducibility (%)": PERCENT_FMT,
        "Score BP": FACTOR_FMT,
    })

    score_bp_columns = [
        "ligand", "pose", "site", "cluster", "energy", "energy_exp",
        "reproducibility_structural", "reproducibility_functional", "score_bp",
    ]
    df_score_bp = df_ranking[score_bp_columns].rename(columns=COLUMN_NAMES_PT)
    for header in (
        "Structural Reproducibility (%)",
        "Functional Reproducibility (%)",
    ):
        df_score_bp[header] = df_score_bp[header] * 100
    ws_score_bp = write_df_sheet(wb, "Score BP", df_score_bp)
    format_columns(ws_score_bp, {
        "Energy (kcal/mol)": ENERGY_FMT,
        "Energy (exp)": FACTOR_FMT,
        "Structural Reproducibility (%)": PERCENT_FMT,
        "Functional Reproducibility (%)": PERCENT_FMT,
        "Score BP": FACTOR_FMT,
    })

    if INCLUDE_BENCHMARKING_SHEET:
        write_benchmarking_sheet(
            wb,
            build_benchmarking_rows(
                all_results,
                analyzable_results,
                df_ranking.to_dict("records"),
            ),
            ENERGY_FMT,
        )
    
    # FINAL SHEET ORDER
    # Defined here to avoid depending on the order in which sheets were created.
    
    DESIRED_SHEET_ORDER = [
        "General Information",
        "Pockets",
        "Pocket Summary",
        "Site Summary",
        "Centroids",
        "Interactions by Type",
        "Interactions by Pocket",
        "Interactions by Site",
        "Conformational Analysis",
        "Ranking BP Selection",
        "Score BP",
        *(["Benchmarking"] if INCLUDE_BENCHMARKING_SHEET else []),
        "Structural",
        "Functional",
        "Edges Structural",
        "Nodes Structural",
        "Edges Functional",
        "Nodes Functional",
        "Structural by Site",
        "Functional by Site",
        "ML Analysis",
        "Pose Coordinates",
    ]
    
    wb._sheets = [
        wb[name]
        for name in DESIRED_SHEET_ORDER
        if name in wb.sheetnames
    ]
    for name, color in SHEET_TAB_COLORS.items():
        if name in wb.sheetnames:
            wb[name].sheet_properties.tabColor = color
    
    wb.save(excel_file)
    

    return excel_file


class BatchAnalysisError(RuntimeError):
    """Include the ligand whose batch preparation or export failed."""

    def __init__(self, ligand_name, cause):
        self.ligand_name = ligand_name
        self.cause = cause
        super().__init__(f"{ligand_name}: {cause}")


def _global_site_label(index):
    """Return the sequential letter for a site, without a Z limit."""

    # Convention equivalent to Excel columns: A..Z, AA..AZ, BA... This does
    # not limit the number of principal spatial sites.
    number = index + 1
    letters = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return f"Site {''.join(reversed(letters))}"


def _global_primary_sites(members):
    """Group poses in one pocket according to the original CALS spatial rule."""

    clusters = []
    for member in members:
        centroid = np.array([
            member["centroid_x"],
            member["centroid_y"],
            member["centroid_z"],
        ])

        for cluster in clusters:
            # Retain the original criterion: a pose enters a site only when it
            # is within cluster_cutoff of every existing member.
            compatible = all(
                centroid_distance(
                    centroid,
                    np.array([
                        existing_member["centroid_x"],
                        existing_member["centroid_y"],
                        existing_member["centroid_z"],
                    ]),
                ) <= cluster_cutoff
                for existing_member in cluster["members"]
            )
            if compatible:
                cluster["members"].append(member)
                cluster["centroid"] = np.mean(
                    [
                        np.array([
                            existing_member["centroid_x"],
                            existing_member["centroid_y"],
                            existing_member["centroid_z"],
                        ])
                        for existing_member in cluster["members"]
                    ],
                    axis=0,
                )
                break
        else:
            clusters.append({"centroid": centroid, "members": [member]})

    return clusters


def _apply_global_gmm_sites(prepared_states):
    """Refine global Sites without changing the primary Pocket/Cluster."""

    poses_by_cavity = {}
    for state in prepared_states:
        for result in state["analyzable_results"]:
            if result.get("site") == "N/A" or result.get("pocket") == "N/A":
                continue
            primary_group = result.get("_primary_group", result["site"])
            poses_by_cavity.setdefault(primary_group, []).append(result)

    for cavity_name in sorted(poses_by_cavity, key=natural_sort_key):
        members = poses_by_cavity[cavity_name]
        primary_sites = _global_primary_sites(members)

        for site_index, primary_site in enumerate(primary_sites):
            primary_label = _global_site_label(site_index)
            subclusters = gmm_subclustering(
                primary_site["members"],
                template=None,
                reference_pose=None,
            )

            for subsite_index, subcluster in enumerate(subclusters, start=1):
                # An unsplit site remains "Site A". Only a GMM split receives
                # subsite numbering: A1, A2...
                site_label = (
                    primary_label
                    if len(subclusters) == 1
                    else f"{primary_label}{subsite_index}"
                )
                for member in subcluster:
                    # GMM determines only Site. The classification completed
                    # before it is explicitly restored for every pose.
                    member["site"] = member.get("_primary_group", cavity_name)
                    member["pocket"] = site_label

        print(
            f"Global sites: pocket {cavity_name}, "
            f"{len(members)} interactive poses, {len(primary_sites)} site(s)."
        )


def _is_analyzable_pose(result):
    """Indicate whether a pose participates in analyses without hiding raw data."""

    return (
        not result.get("excluded_from_analysis", False)
        and result.get("site") != "N/A"
        and result.get("pocket") != "N/A"
    )


def _assign_global_pose_labels(prepared_states):
    """Assign one Pose N sequence to the general batch report."""

    pose_number = 1
    for state in prepared_states:
        for result in state["all_results"]:
            result["_combined_pose_label"] = f"Pose {pose_number}"
            result["_combined_pose_index"] = pose_number
            pose_number += 1


def _plain_ligand_name(value):
    """Remove only the descriptive suffix in parentheses, if present."""

    return str(value).split(" (", 1)[0]


def _write_general_sheet(wb, title, columns, rows, formats=None):
    """Write a continuous general-report table with a single header."""

    ws = wb.create_sheet(title=title)
    ws.append(columns)
    for row in rows:
        ws.append([row.get(column) for column in columns])

    for header, number_format in (formats or {}).items():
        if header not in columns:
            continue
        column_index = columns.index(header) + 1
        for row_index in range(2, ws.max_row + 1):
            ws.cell(row=row_index, column=column_index).number_format = number_format
    return ws


def _write_local_network_sheet(wb, title, node_rows, edge_rows, formats):
    """Write local nodes and edges side by side in the general report."""

    ws = wb.create_sheet(title=title)

    def write_table(start_column, section_title, columns, rows):
        ws.cell(row=1, column=start_column, value=section_title)
        for offset, column in enumerate(columns):
            cell = ws.cell(row=2, column=start_column + offset, value=column)
            if column == "Frequency (f)":
                cell.value = rich_frequency_header()
        for row_index, row in enumerate(rows, start=3):
            for offset, column in enumerate(columns):
                source_column = "Node" if column in {"Residue", "Res-Int"} else column
                ws.cell(
                    row=row_index,
                    column=start_column + offset,
                    value=row.get(source_column),
                )
        return 2, max(2, len(rows) + 2)

    node_columns = [
        "Residue" if column == "Node" and "Structural" in title
        else "Res-Int" if column == "Node" else column
        for column in LOCAL_NODE_COLUMNS
    ]
    node_header, node_last = write_table(1, "Nodes", node_columns, node_rows)
    edge_start = len(LOCAL_NODE_COLUMNS) + 3
    edge_header, edge_last = write_table(
        edge_start, "Edges", LOCAL_EDGE_COLUMNS, edge_rows
    )
    for header_row, last_row in ((node_header, node_last), (edge_header, edge_last)):
        headers = [str(cell.value) if cell.value is not None else None for cell in ws[header_row]]
        for header, number_format in formats.items():
            if header not in headers:
                continue
            column_index = headers.index(header) + 1
            for row_index in range(header_row + 1, last_row + 1):
                ws.cell(row=row_index, column=column_index).number_format = number_format
    for column in range(edge_start, edge_start + len(LOCAL_EDGE_COLUMNS)):
        if str(ws.cell(row=edge_header, column=column).value) == "Frequency (f)":
            for row_index in range(edge_header + 1, edge_last + 1):
                ws.cell(row=row_index, column=column).number_format = formats["Frequency (f)"]
    return ws


def _next_general_excel_path(output_folder):
    """Create an available CALS - General path without overwriting results."""

    output_folder = Path(output_folder)
    candidate = output_folder / "CALS - General.xlsx"
    index = 1
    while candidate.exists():
        candidate = output_folder / f"CALS - General ({index}).xlsx"
        index += 1
    return candidate


def _build_general_excel(prepared_states, output_folder):
    """Generate consolidated Excel from the batch's already calculated results."""

    all_results = [
        result
        for state in prepared_states
        for result in state["all_results"]
    ]
    analyzable_results = [
        result for result in all_results if _is_analyzable_pose(result)
    ]
    first_state = prepared_states[0]
    pockets = first_state["pockets"]
    pocket_centroids = first_state["pocket_centroids"]

    number_formats = {
        "Energy (kcal/mol)": "0.000",
        "RMSD (Vina)": "0.000",
        "RMSDdocking (Å)": "0.000",
        "SCMi (Å)": "0.000",
        "SCMi - Standard Deviation (Å)": "0.000",
        "PCMi (Å)": "0.000",
        "PCMi - Standard Deviation (Å)": "0.000",
        "Centroid X": "0.000",
        "Centroid Y": "0.000",
        "Centroid Z": "0.000",
        "x": "0.000",
        "y": "0.000",
        "z": "0.000",
        "Energy": "0.000",
        "RMSD": "0.000",
        "Frequency (%)": "0.0",
        "Frequency (f)": "0.000",
        "Score BP": "0.000",
        "Structural Reproducibility (%)": "0.0",
        "Functional Reproducibility (%)": "0.0",
        "Co-occurrence Structural Reproducibility (%)": "0.0",
        "Co-occurrence Functional Reproducibility (%)": "0.0",
        "Interaction Distance (Å)": "0.000",
        "Donor···Acceptor (Å)": "0.000",
        "H···Acceptor (Å)": "0.000",
        "Angle (°)": "0.0",
    }

    wb = Workbook()
    wb.remove(wb.active)

    general_columns = [
        "Ligand", "Pose", "Pocket", "Site", "Energy (kcal/mol)",
        "RMSD (Vina)", "Centroid X", "Centroid Y",
        "Centroid Z", "Structural Reproducibility (%)",
        "Functional Reproducibility (%)",
    ]
    general_rows = [
        {
            "Ligand": _plain_ligand_name(result["ligand"]),
            "Pose": result["_combined_pose_label"],
            "Pocket": result.get("site"),
            "Site": result.get("pocket"),
            "Energy (kcal/mol)": result.get("energy"),
            "RMSD (Vina)": result.get("rmsd"),
            "Centroid X": result.get("centroid_x"),
            "Centroid Y": result.get("centroid_y"),
            "Centroid Z": result.get("centroid_z"),
        }
        for result in all_results
    ]
    _write_general_sheet(
        wb, "Pockets", ["Pocket", "Residue", "Group"],
        build_cavity_rows(pockets), number_formats,
    )

    summary_cavity_columns = [
        "Pocket", "Pose Count", "Best Energy", "Pose (Best Energy)",
    ]
    summary_cavity_rows = []
    by_cavity = {}
    for result in analyzable_results:
        by_cavity.setdefault(result["site"], []).append(result)
    for cavity, members in sorted(by_cavity.items(), key=lambda item: natural_sort_key(item[0])):
        best = min(members, key=lambda result: result["energy"])
        summary_cavity_rows.append({
            "Pocket": cavity,
            "Pose Count": len(members),
            "Best Energy": best["energy"],
            "Pose (Best Energy)": best["_combined_pose_label"],
        })
    _write_general_sheet(
        wb, "Pocket Summary", summary_cavity_columns,
        summary_cavity_rows, number_formats,
    )

    summary_site_columns = [
        "Pocket", "Site", "Pose Count", "Best Energy",
        "Pose (Best Energy)",
    ]
    summary_site_rows = []
    by_site = {}
    for result in analyzable_results:
        by_site.setdefault((result["site"], result["pocket"]), []).append(result)
    for (cavity, site), members in sorted(
            by_site.items(),
            key=lambda item: (natural_sort_key(item[0][0]), natural_sort_key(item[0][1])),
    ):
        best = min(members, key=lambda result: result["energy"])
        summary_site_rows.append({
            "Pocket": cavity,
            "Site": site,
            "Pose Count": len(members),
            "Best Energy": best["energy"],
            "Pose (Best Energy)": best["_combined_pose_label"],
        })
    _write_general_sheet(
        wb, "Site Summary", summary_site_columns,
        summary_site_rows, number_formats,
    )

    interaction_columns = [
        "Ligand", "Pose", "Pocket", "Site", "Energy (kcal/mol)",
        "Residue", "Interaction Type", "Strength", "Interaction Distance (Å)",
        "Donor···Acceptor (Å)", "H···Acceptor (Å)", "Angle (°)",
        "Ligand Atom", "Protein Atom",
    ]
    interaction_rows = []
    network_interaction_rows = []
    for result in all_results:
        base_row = {
            "Ligand": _plain_ligand_name(result["ligand"]),
            "Pose": result["_combined_pose_label"],
            "Pocket": result.get("site"),
            "Site": result.get("pocket"),
            "Energy (kcal/mol)": result.get("energy"),
        }
        if not result.get("interactions"):
            interaction_rows.append({
                **base_row,
                "Residue": None,
                "Interaction Type": "No interactions detected",
                "Strength": None,
                "Interaction Distance (Å)": None,
                "Donor···Acceptor (Å)": None,
                "H···Acceptor (Å)": None,
                "Angle (°)": None,
                "Ligand Atom": None,
                "Protein Atom": None,
            })
            continue

        for interaction in result["interactions"]:
            interaction_rows.append({
                **base_row,
                "Residue": format_interaction_residue(
                    interaction["residue"], interaction["resnum"], interaction["chain"]
                ),
                "Interaction Type": interaction["type"],
                "Strength": classify_hbond_strength(
                    interaction["type"], interaction.get("distance_da")
                ),
                "Interaction Distance (Å)": (
                    interaction["distance"] if interaction["type"] != "H-Bond" else None
                ),
                "Donor···Acceptor (Å)": interaction.get("distance_da"),
                "H···Acceptor (Å)": interaction.get("distance_ha"),
                "Angle (°)": interaction.get("angle"),
                "Ligand Atom": interaction["ligand_atom"],
                "Protein Atom": interaction["protein_atom"],
            })
            network_interaction_rows.append({
                "ligand": _plain_ligand_name(result["ligand"]),
                "pose": result["_combined_pose_label"],
                "site": result.get("site"),
                "pocket": result.get("pocket"),
                "energy": result["energy"],
                "residue": format_interaction_residue(
                    interaction["residue"], interaction["resnum"], interaction["chain"]
                ),
                "chain": "",
                "interaction_type": interaction["type"],
                "network_pose_key": result["_combined_pose_label"],
                "global_pose_label": result["_combined_pose_label"],
            })
    _write_general_sheet(
        wb, "Interactions by Type", interaction_columns, interaction_rows, number_formats
    )

    network_tables = build_network_tables(network_interaction_rows)
    local_network_tables = build_local_network_fidelity(network_interaction_rows)
    local_score_by_pose = {
        (row["ligand"], row["site"], row["pocket"], row["pose_key"]): row
        for row in local_network_tables["pose_scores"]
    }
    for row, result in zip(general_rows, all_results):
        score = local_score_by_pose.get(
            (
                canonical_network_ligand(_plain_ligand_name(result["ligand"])),
                result.get("site"), result.get("pocket"),
                result.get("_combined_pose_label"),
            )
        )
        values = local_network_score_values(score)
        row.update({
            "Structural Reproducibility (%)": values["reproducibility_structural"],
            "Functional Reproducibility (%)": values["reproducibility_functional"],
        })
    general_bp_rows = []
    for result in all_results:
        score = local_score_by_pose.get(
            (
                canonical_network_ligand(_plain_ligand_name(result["ligand"])),
                result.get("site"), result.get("pocket"),
                result.get("_combined_pose_label"),
            )
        )
        values = local_network_score_values(score, as_fraction=True)
        general_bp_rows.append({
            "site": result.get("site"),
            "pocket": result.get("pocket"),
            "energy": result.get("energy"),
            "reproducibility_structural": values["reproducibility_structural"],
            "reproducibility_functional": values["reproducibility_functional"],
        })
    add_score_bp_values(general_bp_rows, ("site", "pocket"))
    _write_general_sheet(
        wb, "General Information", general_columns, general_rows, number_formats
    )
    for title, key, columns in (
        ("Structural", "structural", STRUCTURAL_COLUMNS),
        ("Functional", "functional", FUNCTIONAL_COLUMNS),
        ("Edges Structural", "edges_structural", EDGE_COLUMNS),
        ("Nodes Structural", "nodes_structural", NODE_COLUMNS),
        ("Edges Functional", "edges_functional", EDGE_COLUMNS),
        ("Nodes Functional", "nodes_functional", NODE_COLUMNS),
        ("ML Analysis", "random_forest", RANDOM_FOREST_COLUMNS),
    ):
        _write_general_sheet(
            wb, title, list(columns), network_tables[key], number_formats
        )
    local_network_formats = {
        "Frequency (f)": "0.000",
        "Hub Strength": "0.000",
        "Centrality": "0.000",
        "Raw Weight": "0.000",
        "Normalized Weight": "0.000",
    }
    _write_local_network_sheet(
        wb, "Structural by Site",
        local_network_tables["local_nodes_structural"],
        local_network_tables["local_edges_structural"],
        local_network_formats,
    )
    _write_local_network_sheet(
        wb, "Functional by Site",
        local_network_tables["local_nodes_functional"],
        local_network_tables["local_edges_functional"],
        local_network_formats,
    )

    coordinate_columns = [
        "Ligand", "Pose", "Pocket", "Site", "Centroid X",
        "Centroid Y", "Centroid Z",
    ]
    coordinate_rows = [
        {
            "Ligand": _plain_ligand_name(result["ligand"]),
            "Pose": result["_combined_pose_label"],
            "Pocket": result.get("site"),
            "Site": result.get("pocket"),
            "Centroid X": result.get("centroid_x"),
            "Centroid Y": result.get("centroid_y"),
            "Centroid Z": result.get("centroid_z"),
        }
        for result in all_results
    ]
    _write_general_sheet(
        wb, "Pose Coordinates", coordinate_columns, coordinate_rows, number_formats
    )

    ws_centroids = wb.create_sheet(title="Centroids")
    ws_centroids.append(["Pockets"])
    cavity_centroid_columns = ["Pocket", "Centroid X", "Centroid Y", "Centroid Z"]
    ws_centroids.append(cavity_centroid_columns)
    for cavity, centroid in sorted(pocket_centroids.items(), key=lambda item: natural_sort_key(item[0])):
        ws_centroids.append([cavity, centroid[0], centroid[1], centroid[2]])
    ws_centroids.append([])
    ws_centroids.append(["Sites"])
    site_centroid_columns = [
        "Pocket", "Site", "Pose Count", "Centroid X", "Centroid Y", "Centroid Z",
    ]
    ws_centroids.append(site_centroid_columns)
    for (cavity, site), members in sorted(
            by_site.items(),
            key=lambda item: (natural_sort_key(item[0][0]), natural_sort_key(item[0][1])),
    ):
        ws_centroids.append([
            cavity, site, len(members),
            float(np.mean([member["centroid_x"] for member in members])),
            float(np.mean([member["centroid_y"] for member in members])),
            float(np.mean([member["centroid_z"] for member in members])),
        ])
    for header_row in (2, ws_centroids.max_row - len(by_site)):
        headers = [cell.value for cell in ws_centroids[header_row]]
        for header in ("Centroid X", "Centroid Y", "Centroid Z"):
            if header in headers:
                column_index = headers.index(header) + 1
                for row_index in range(header_row + 1, ws_centroids.max_row + 1):
                    ws_centroids.cell(row=row_index, column=column_index).number_format = "0.000"

    interaction_cavity_counts = {}
    interaction_site_counts = {}
    cavity_totals = {}
    site_totals = {}
    for result in analyzable_results:
        cavity_totals[result["site"]] = cavity_totals.get(result["site"], 0) + 1
        site_key = (result["site"], result["pocket"])
        site_totals[site_key] = site_totals.get(site_key, 0) + 1
        residues = {
            (interaction["residue"], interaction["resnum"], interaction["chain"])
            for interaction in result.get("interactions", [])
        }
        for residue, resnum, chain in residues:
            cavity_key = (result["site"], residue, resnum, chain)
            interaction_cavity_counts[cavity_key] = interaction_cavity_counts.get(cavity_key, 0) + 1
            detailed_site_key = (result["site"], result["pocket"], residue, resnum, chain)
            interaction_site_counts[detailed_site_key] = interaction_site_counts.get(detailed_site_key, 0) + 1

    cavity_interaction_columns = ["Pocket", "Residue", "Interactions", "Frequency (%)"]
    cavity_interaction_rows = [
        {
            "Pocket": cavity,
            "Residue": format_interaction_residue(residue, resnum, chain),
            "Interactions": count,
            "Frequency (%)": (count / cavity_totals[cavity]) * 100,
        }
        for (cavity, residue, resnum, chain), count in sorted(
            interaction_cavity_counts.items(), key=lambda item: (natural_sort_key(item[0][0]), -item[1])
        )
    ]
    _write_general_sheet(
        wb, "Interactions by Pocket", cavity_interaction_columns,
        cavity_interaction_rows, number_formats,
    )

    site_interaction_columns = ["Pocket", "Site", "Residue", "Interactions", "Frequency (%)"]
    site_interaction_rows = [
        {
            "Pocket": cavity,
            "Site": site,
            "Residue": format_interaction_residue(residue, resnum, chain),
            "Interactions": count,
            "Frequency (%)": (count / site_totals[(cavity, site)]) * 100,
        }
        for (cavity, site, residue, resnum, chain), count in sorted(
            interaction_site_counts.items(),
            key=lambda item: (natural_sort_key(item[0][0]), natural_sort_key(item[0][1]), -item[1]),
        )
    ]
    _write_general_sheet(
        wb, "Interactions by Site", site_interaction_columns,
        site_interaction_rows, number_formats,
    )

    translation = {
        "ligand": "Ligand", "pose": "Pose", "site": "Pocket", "pocket": "Site",
        "energy": "Energy (kcal/mol)",
        "reproducibility_structural": "Structural Reproducibility (%)",
        "reproducibility_functional": "Functional Reproducibility (%)",
        "score_bp": "Score BP",
    }
    conformational_rows = []
    for state in prepared_states:
        for row in state.get("conformational_rows", []):
            final_row = row.copy()
            final_row["ligand"] = _plain_ligand_name(final_row["ligand"])
            final_row["pose"] = final_row.pop("combined_pose_label")
            score = local_score_by_pose.get(
                (
                    canonical_network_ligand(final_row["ligand"]),
                    final_row.get("site"), final_row.get("pocket"), final_row["pose"],
                )
            )
            final_row.update(local_network_score_values(score))
            final_row.pop("reproducibility_score", None)
            final_row.pop("pose_index", None)
            final_row.pop("conservation_score", None)
            for key in (
                "rmsd_docking", "mean_rmsd", "std_rmsd",
                "mean_rmsd_consensus", "std_consensus",
                "ensemble_mean_rmsd", "ensemble_std_rmsd", "ensemble_nposes",
                "mean_rmsd_binding", "std_binding", "std_rmsd_binding",
                "ensemble_mean_rmsd_binding", "ensemble_std_rmsd_binding",
                "best_energy_pose_mean_rmsd", "representative",
                "representative_effective_sample_size",
                "reproducibility_cooccurrence_structural",
                "reproducibility_cooccurrence_functional",
            ):
                final_row.pop(key, None)
            conformational_rows.append(final_row)
    conformational_bp_rows = []
    for row in conformational_rows:
        score = local_score_by_pose.get(
            (
                canonical_network_ligand(row["ligand"]),
                row.get("site"), row.get("pocket"), row["pose"],
            )
        )
        values = local_network_score_values(score, as_fraction=True)
        conformational_bp_rows.append({
            "site": row.get("site"),
            "pocket": row.get("pocket"),
            "energy": row.get("energy"),
            "reproducibility_structural": values["reproducibility_structural"],
            "reproducibility_functional": values["reproducibility_functional"],
        })
    add_score_bp_values(conformational_bp_rows, ("site", "pocket"))
    for row, bp_row in zip(conformational_rows, conformational_bp_rows):
        row["score_bp"] = bp_row["score_bp"]
    preferred_conformation = [
        "ligand", "pose", "site", "pocket", "energy",
        "reproducibility_structural", "reproducibility_functional",
        "score_bp",
    ]
    all_conformation_keys = {
        key for row in conformational_rows for key in row if key != "combined_pose_label"
    }
    conformational_keys = [key for key in preferred_conformation if key in all_conformation_keys]
    conformational_keys.extend(
        key for key in all_conformation_keys if key not in conformational_keys
    )
    conformational_columns = [translation.get(key, key) for key in conformational_keys]
    rendered_conformational_rows = [
        {translation.get(key, key): row.get(key) for key in conformational_keys}
        for row in conformational_rows
    ]
    _write_general_sheet(
        wb, "Conformational Analysis", conformational_columns,
        rendered_conformational_rows, number_formats,
    )

    desired_sheet_order = [
        "General Information", "Pockets", "Pocket Summary", "Site Summary",
        "Centroids",
        "Interactions by Type", "Interactions by Pocket", "Interactions by Site",
        "Conformational Analysis", "Structural", "Functional", "Edges Structural",
        "Nodes Structural", "Edges Functional", "Nodes Functional",
        "Structural by Site", "Functional by Site", "ML Analysis",
        "Pose Coordinates",
    ]
    wb._sheets = [wb[name] for name in desired_sheet_order]
    for name, color in SHEET_TAB_COLORS.items():
        if name in wb.sheetnames:
            wb[name].sheet_properties.tabColor = color
    excel_file = _next_general_excel_path(output_folder)
    wb.save(excel_file)
    return str(excel_file)


def _resume_staged_analysis(staged_analysis):
    """Resume an analysis after GMM and return its Excel output when complete."""

    try:
        next(staged_analysis)
    except StopIteration as completion:
        return completion.value
    raise RuntimeError("Analysis paused unexpectedly after the global GMM.")


def run_analysis(pocket_files, protein_file, vina_files, ligand_file, smiles_file):
    """Run individual analysis while retaining canonical GMM sites."""

    staged_analysis = _run_analysis_staged(
        pocket_files, protein_file, vina_files, ligand_file, smiles_file
    )
    prepared_state = next(staged_analysis)
    _apply_global_gmm_sites([prepared_state])
    return _resume_staged_analysis(staged_analysis)


def run_batch_analysis(
        pocket_files,
        protein_file,
        batches,
        on_excel_created=None,
        on_general_excel_created=None):
    """Run a batch with one GMM per pocket for every ligand."""

    prepared_analyses = []
    for batch in batches:
        ligand_name = batch["ligand_name"]
        staged_analysis = _run_analysis_staged(
            pocket_files,
            protein_file,
            batch["vina_files"],
            batch["reference_file"],
            batch["smiles_file"],
        )
        try:
            prepared_state = next(staged_analysis)
        except Exception as exc:
            raise BatchAnalysisError(ligand_name, exc) from exc
        prepared_analyses.append((batch, staged_analysis, prepared_state))

    _apply_global_gmm_sites(
        [prepared_state for _, _, prepared_state in prepared_analyses]
    )
    _assign_global_pose_labels(
        [prepared_state for _, _, prepared_state in prepared_analyses]
    )

    excel_files = []
    for batch, staged_analysis, _ in prepared_analyses:
        ligand_name = batch["ligand_name"]
        try:
            excel_file = _resume_staged_analysis(staged_analysis)
            if on_excel_created is not None:
                excel_file = on_excel_created(batch, excel_file)
        except Exception as exc:
            raise BatchAnalysisError(ligand_name, exc) from exc
        excel_files.append(excel_file)

    if on_general_excel_created is not None:
        try:
            general_source_folder = Path(batches[0]["vina_files"][0]).parent
            general_excel = _build_general_excel(
                [prepared_state for _, _, prepared_state in prepared_analyses],
                general_source_folder,
            )
            general_excel = on_general_excel_created(general_excel)
        except Exception as exc:
            raise BatchAnalysisError("CALS - General", exc) from exc
        excel_files.append(general_excel)

    return excel_files
