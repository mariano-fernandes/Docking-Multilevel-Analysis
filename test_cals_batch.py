"""Headless tests for the CALS batch orchestrator."""

import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).parent
LAUNCHER = PROJECT_ROOT / "CALS.py"


def load_launcher():
    """Load the launcher without requiring RDKit or the real engine."""

    package = types.ModuleType("modules")
    package.__path__ = []
    engine = types.ModuleType("modules.engine")
    engine.BatchAnalysisError = RuntimeError
    engine.run_batch_analysis = lambda *args, **kwargs: []
    previous_modules = {
        name: sys.modules.get(name) for name in ("modules", "modules.engine")
    }
    sys.modules["modules"] = package
    sys.modules["modules.engine"] = engine
    try:
        spec = importlib.util.spec_from_file_location("cals", LAUNCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous


class BatchValidationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_launcher()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.names = ("LAP1", "LAP1a", "LAP2")

        self.folders = []
        self.references = []
        self.smiles = []
        for name, suffix in zip(self.names, (".pdb", ".mol", ".sdf")):
            folder = self.root / name
            folder.mkdir()
            (folder / "results.pdbqt").write_text("MODEL 1", encoding="utf-8")
            self.folders.append(folder)

            reference = self.root / f"{name}{suffix}"
            reference.write_text("reference", encoding="utf-8")
            self.references.append(reference)

            smile = self.root / f"{name}.txt"
            smile.write_text("C", encoding="utf-8")
            self.smiles.append(smile)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pairs_all_ligands_and_ignores_nested_pdbqt(self):
        nested = self.folders[0] / "nested"
        nested.mkdir()
        (nested / "ignored.pdbqt").write_text("MODEL 1", encoding="utf-8")

        batches = self.module.validate_batch(
            self.folders, self.references, self.smiles
        )

        self.assertEqual([batch["ligand_name"] for batch in batches], list(self.names))
        self.assertEqual(len(batches[0]["vina_files"]), 1)

    def test_rejects_unpaired_reference(self):
        extra_reference = self.root / "EXTRA.pdb"
        extra_reference.write_text("reference", encoding="utf-8")

        with self.assertRaisesRegex(
            self.module.BatchValidationError,
            "Reference without a matching Vina folder",
        ):
            self.module.validate_batch(
                self.folders, self.references + [extra_reference], self.smiles
            )

    def test_rejects_duplicate_reference_name(self):
        alternative_folder = self.root / "alternative"
        alternative_folder.mkdir()
        duplicate = alternative_folder / "LAP1.mol"
        duplicate.write_text("reference", encoding="utf-8")

        with self.assertRaisesRegex(
            self.module.BatchValidationError, "Duplicate name in references"
        ):
            self.module.validate_batch(
                self.folders, self.references + [duplicate], self.smiles
            )

    def test_rejects_folder_without_direct_pdbqt(self):
        empty_folder = self.root / "EMPTY"
        empty_folder.mkdir()
        (empty_folder / "nested").mkdir()
        (empty_folder / "nested" / "only_nested.pdbqt").write_text(
            "MODEL 1", encoding="utf-8"
        )
        reference = self.root / "EMPTY.pdb"
        reference.write_text("reference", encoding="utf-8")
        smile = self.root / "EMPTY.txt"
        smile.write_text("C", encoding="utf-8")

        with self.assertRaisesRegex(
            self.module.BatchValidationError, "contains no .pdbqt files at its top level"
        ):
            self.module.validate_batch(
                self.folders + [empty_folder],
                self.references + [reference],
                self.smiles + [smile],
            )

    def test_requires_smiles_for_each_vina_folder(self):
        with self.assertRaisesRegex(
            self.module.BatchValidationError, "Missing SMILES for ligand 'LAP2'"
        ):
            self.module.validate_batch(self.folders, self.references, self.smiles[:-1])

    def test_output_name_uses_numeric_suffix_without_overwriting(self):
        output = self.root / "output"
        output.mkdir()
        (output / "CALS - LAP1.xlsx").write_text("existing", encoding="utf-8")

        self.assertEqual(
            self.module.build_destination_path(output, "LAP1").name,
            "CALS - LAP1 (1).xlsx",
        )

    def test_global_sites_keep_primary_clusters_and_gmm_only_splits_subsites(self):
        from modules import engine as real_engine

        def pose(ligand, cavity, x):
            return {
                "ligand": ligand,
                "site": cavity,
                "pocket": "old",
                "centroid_x": x,
                "centroid_y": 0.0,
                "centroid_z": 0.0,
            }

        lap1_a = pose("LAP1", "Pocket 1", 0.0)
        lap1a_a = pose("LAP1a", "Pocket 1", 1.0)
        lap2_b = pose("LAP2", "Pocket 1", 20.0)
        lap3_c = pose("LAP3", "Pocket 1", 40.0)
        lap2_cavity_2 = pose("LAP2", "Pocket 2", 0.0)
        not_interactive = {"ligand": "NoInteraction", "site": "N/A", "pocket": "N/A"}
        states = [
            {"analyzable_results": [lap1_a, lap1a_a, not_interactive]},
            {"analyzable_results": [lap2_b, lap3_c, lap2_cavity_2]},
        ]
        calls = []

        def fake_gmm(members, template, reference_pose):
            calls.append([member["ligand"] for member in members])
            if members[0] is lap1_a:
                return [members[:1], members[1:]]
            return [members]

        with patch.object(real_engine, "gmm_subclustering", side_effect=fake_gmm):
            real_engine._apply_global_gmm_sites(states)

        self.assertEqual(
            calls,
            [["LAP1", "LAP1a"], ["LAP2"], ["LAP3"], ["LAP2"]],
        )
        self.assertEqual([lap1_a["pocket"], lap1a_a["pocket"]], ["Site A1", "Site A2"])
        self.assertEqual(lap2_b["pocket"], "Site B")
        self.assertEqual(lap3_c["pocket"], "Site C")
        self.assertEqual(lap2_cavity_2["pocket"], "Site A")
        self.assertNotIn(not_interactive["ligand"], [ligand for call in calls for ligand in call])

    def test_primary_cavity_or_cluster_is_frozen_before_gmm(self):
        from modules import engine as real_engine

        def pose(name, ranking, x):
            return {
                "ligand": name,
                "ranking": ranking,
                "centroid_x": x,
                "centroid_y": 0.0,
                "centroid_z": 0.0,
            }

        in_cavity = pose("DentroA", {"Pocket A": 30, "Pocket B": 0}, 3.0)
        zero_contacts = pose("SemContato", {"Pocket A": 0, "Pocket B": 0}, 0.0)
        few_contacts = pose("Poucos", {"Pocket A": 11, "Pocket B": 0}, 1.0)
        in_other_cavity = pose("DentroB", {"Pocket A": 14, "Pocket B": 16}, 2.0)
        far_but_strong = pose("ForteDistante", {"Pocket A": 40, "Pocket B": 0}, 20.0)
        results = [in_cavity, zero_contacts, few_contacts, in_other_cavity, far_but_strong]

        real_engine.assign_primary_groups(
            results,
            {"Pocket A": np.zeros(3), "Pocket B": np.array([2.0, 0.0, 0.0])},
        )

        self.assertEqual("Pocket A", in_cavity["site"])
        self.assertTrue(zero_contacts["site"].startswith("C"))
        self.assertEqual(zero_contacts["site"], few_contacts["site"])
        self.assertEqual("Pocket B", in_other_cavity["site"])
        self.assertEqual("Pocket A", far_but_strong["site"])
        self.assertEqual("Pocket B", in_other_cavity["_primary_group"])

        excluded = {"ligand": "N/A", "site": "N/A", "pocket": "N/A"}
        gmm_groups = []

        def fake_gmm(members, template, reference_pose):
            gmm_groups.append([member["ligand"] for member in members])
            return [members]

        with patch.object(real_engine, "gmm_subclustering", side_effect=fake_gmm):
            real_engine._apply_global_gmm_sites([{
                "analyzable_results": results,
                "all_results": results + [excluded],
            }])

        self.assertEqual("Pocket A", in_cavity["site"])
        self.assertEqual(zero_contacts["_primary_group"], zero_contacts["site"])
        self.assertEqual("Pocket B", in_other_cavity["site"])
        self.assertEqual("Pocket A", far_but_strong["site"])
        self.assertEqual("N/A", excluded["site"])
        self.assertEqual("N/A", excluded["pocket"])
        self.assertEqual(
            [["SemContato", "Poucos"], ["DentroA"], ["ForteDistante"], ["DentroB"]],
            gmm_groups,
        )

    def test_global_site_labels_continue_after_z(self):
        from modules import engine as real_engine

        self.assertEqual(real_engine._global_site_label(0), "Site A")
        self.assertEqual(real_engine._global_site_label(25), "Site Z")
        self.assertEqual(real_engine._global_site_label(26), "Site AA")

    def test_local_network_fidelity_is_scoped_and_weighted_by_hubs(self):
        from modules.cals_network_export import build_local_network_fidelity

        def interaction(pose, residue, interaction_type, site="Site A"):
            return {
                "ligand": "LAP1", "site": "Pocket 1", "pocket": site,
                "pose": pose, "energy": -7.0, "residue": residue,
                "chain": "", "interaction_type": interaction_type,
            }

        rows = []
        # Four poses repeat the ALA1/GLY2 core; one contains three rare nodes.
        for pose in range(1, 5):
            rows.extend([
                interaction(pose, "ALA1", "H-Bond"),
                interaction(pose, "GLY2", "Hydrophobic"),
            ])
        rows.extend([
            interaction(5, "SER3", "H-Bond"),
            interaction(5, "THR4", "Hydrophobic"),
            interaction(5, "TYR5", "π-stacking (Parallel)"),
        ])
        # Another site from the same ligand cannot contaminate reference A.
        rows.append(interaction(1, "ASP99", "Salt Bridge", site="Site B"))

        tables = build_local_network_fidelity(
            rows,
            scope_poses=[{
                "ligand": "LAP1", "site": "Pocket 1", "pocket": "Site C",
                "pose": 1,
            }],
        )
        scores = {
            (row["site"], row["pocket"], row["pose_key"]): row
            for row in tables["pose_scores"]
        }

        core = scores[("Pocket 1", "Site A", ("LAP1", 1))]
        peripheral = scores[("Pocket 1", "Site A", ("LAP1", 5))]
        isolated = scores[("Pocket 1", "Site B", ("LAP1", 1))]
        empty = scores[("Pocket 1", "Site C", ("LAP1", 1))]

        # A few high-weight nodes outperform more rare peripheral nodes.
        self.assertGreater(core["fidelity_structural"], peripheral["fidelity_structural"])
        self.assertGreater(core["fidelity_functional"], peripheral["fidelity_functional"])
        self.assertGreater(
            core["fidelity_cooccurrence_structural"],
            peripheral["fidelity_cooccurrence_structural"],
        )
        self.assertAlmostEqual(core["fidelity_structural"], 2.304 / 2.436)
        self.assertAlmostEqual(peripheral["fidelity_structural"], 0.132 / 2.436)
        self.assertGreater(core["fidelity_structural"], 0.9)
        self.assertLess(peripheral["fidelity_structural"], 0.1)
        self.assertAlmostEqual(isolated["fidelity_structural"], 1.0)
        self.assertIsNone(isolated["fidelity_cooccurrence_structural"])
        self.assertEqual(empty["fidelity_structural"], 0.0)
        self.assertIsNone(empty["fidelity_cooccurrence_structural"])

        local_a_nodes = [
            row for row in tables["local_nodes_structural"]
            if row["Site"] == "Site A"
        ]
        self.assertNotIn("Asp99", {row["Node"] for row in local_a_nodes})

    def test_numbered_ligand_replicas_share_one_local_reference(self):
        from modules.cals_network_export import build_local_network_fidelity

        def interaction(ligand, pose, residue, interaction_type):
            return {
                "ligand": ligand, "site": "Pocket 1", "pocket": "Site A",
                "pose": pose, "residue": residue, "chain": "",
                "interaction_type": interaction_type,
            }

        rows = []
        for ligand in ("LAP1(7)", "LAP1(42)"):
            rows.extend([
                interaction(ligand, 1, "TYR217A", "Hydrophobic"),
                interaction(ligand, 1, "TYR345A", "H-Bond"),
            ])
        rows.append(interaction("LAP1(99)", 1, "ASP189A", "Salt Bridge"))

        tables = build_local_network_fidelity(rows)
        scores = {
            row["pose_key"]: row
            for row in tables["pose_scores"]
            if row["ligand"] == "LAP1"
        }
        first = scores[("LAP1(7)", 1)]
        second = scores[("LAP1(42)", 1)]
        for metric in (
            "fidelity_structural",
            "fidelity_functional",
            "fidelity_cooccurrence_structural",
            "fidelity_cooccurrence_functional",
        ):
            self.assertEqual(first[metric], second[metric])

        local_nodes = tables["local_nodes_structural"]
        self.assertEqual({row["Ligand"] for row in local_nodes}, {"LAP1"})
        self.assertEqual({row["Pose Count"] for row in local_nodes}, {3})

    def test_ranking_uses_energy_window_and_score_bp(self):
        from modules.ranking import order_ranking_rows

        rows = [
            {
                "ligand": "LAP1", "site": "Pocket 1", "cluster": "Site A",
                "rank": 1, "pose": 1, "energy": -9.7, "rmsd_docking": 0.2,
                "mean_rmsd_binding": 0.2,
                "reproducibility_structural": 0.70,
                "reproducibility_functional": 0.70,
            },
            {
                "ligand": "LAP1", "site": "Pocket 1", "cluster": "Site A",
                "rank": 2, "pose": 2, "energy": -9.4, "rmsd_docking": 3.0,
                "mean_rmsd_binding": 3.0,
                "reproducibility_structural": 0.90,
                "reproducibility_functional": 0.90,
            },
            {
                "ligand": "LAP1", "site": "Pocket 1", "cluster": "Site A",
                "rank": 3, "pose": 3, "energy": -8.6, "rmsd_docking": 0.1,
                "mean_rmsd_binding": 0.1,
                "reproducibility_structural": 0.99,
                "reproducibility_functional": 0.99,
            },
        ]
        counts = {"Pocket 1": 3}
        site_counts = {("Pocket 1", "Site A"): 3}
        ranked = order_ranking_rows(rows, counts, site_counts)

        self.assertEqual([row["pose"] for row in ranked], [2, 1, 3])
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])
        self.assertAlmostEqual(ranked[1]["energy_exp"], 1.0)
        self.assertAlmostEqual(
            ranked[0]["energy_exp"], math.exp(-0.3 / 2.0)
        )
        self.assertAlmostEqual(
            ranked[0]["score_bp"],
            (math.exp(-0.3 / 2.0) * 0.90 * 0.90) ** (1 / 3),
        )
        # Pose 3 has Score BP but cannot exceed the +1 energy window.
        self.assertIsNotNone(ranked[2]["score_bp"])
        self.assertEqual(ranked[2]["pose"], 3)

    def test_ranking_breaks_visible_score_bp_ties_by_energy_then_rmsd(self):
        from modules.ranking import order_ranking_rows

        equal_repro = math.sqrt(
            (0.8 ** 3) / math.exp(-0.1 / 2.0)
        )
        best_energy_repro = math.sqrt(0.8 ** 3)

        rows = [
            {
                "ligand": "LAP1", "site": "Pocket 1", "cluster": "Site A",
                "rank": 1, "pose": 1, "energy": -9.5, "rmsd_docking": 0.1,
                "mean_rmsd_binding": 0.1,
                "reproducibility_structural": equal_repro,
                "reproducibility_functional": equal_repro,
                "reproducibility_cooccurrence_structural": 99.0,
                "reproducibility_cooccurrence_functional": 99.0,
            },
            {
                "ligand": "LAP1", "site": "Pocket 1", "cluster": "Site A",
                "rank": 2, "pose": 2, "energy": -9.6, "rmsd_docking": 4.0,
                "mean_rmsd_binding": 4.0,
                "reproducibility_structural": best_energy_repro,
                "reproducibility_functional": best_energy_repro,
                "reproducibility_cooccurrence_structural": 1.0,
                "reproducibility_cooccurrence_functional": 1.0,
            },
            {
                "ligand": "LAP1", "site": "Pocket 1", "cluster": "Site A",
                "rank": 3, "pose": 3, "energy": -9.6, "rmsd_docking": 0.8,
                "mean_rmsd_binding": 0.8,
                "reproducibility_structural": best_energy_repro,
                "reproducibility_functional": best_energy_repro,
                "reproducibility_cooccurrence_structural": 2.0,
                "reproducibility_cooccurrence_functional": 2.0,
            },
        ]
        counts = {"Pocket 1": 3}
        site_counts = {("Pocket 1", "Site A"): 3}

        ranked = order_ranking_rows(rows, counts, site_counts)

        # Todos exibem Score BP 0,800: energy desempata 2/3 de 1 e RMSD
        # It ranks 3 ahead of 2. Co-occurrences do not participate.
        self.assertEqual([row["pose"] for row in ranked], [3, 2, 1])
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])
        self.assertEqual([round(row["score_bp"], 3) for row in ranked], [0.8] * 3)

    def test_reproducibility_scores_support_fractional_ranking_export(self):
        from modules.engine import local_network_score_values

        raw_scores = {
            "fidelity_structural": 0.81,
            "fidelity_functional": 0.64,
            "fidelity_cooccurrence_structural": 0.49,
            "fidelity_cooccurrence_functional": 0.36,
        }
        values = local_network_score_values(raw_scores)
        fractional_values = local_network_score_values(raw_scores, as_fraction=True)

        self.assertAlmostEqual(values["reproducibility_structural"], 81.0)
        self.assertAlmostEqual(values["reproducibility_functional"], 64.0)
        self.assertAlmostEqual(values["reproducibility_score"], 0.72)
        self.assertAlmostEqual(
            fractional_values["reproducibility_structural"], 0.81
        )
        self.assertAlmostEqual(
            fractional_values["reproducibility_functional"], 0.64
        )
        self.assertAlmostEqual(
            fractional_values["reproducibility_cooccurrence_structural"], 0.49
        )
        self.assertAlmostEqual(
            fractional_values["reproducibility_cooccurrence_functional"], 0.36
        )
        self.assertAlmostEqual(fractional_values["reproducibility_score"], 0.72)
        self.assertNotIn("test_score", fractional_values)

    def test_benchmarking_reuses_cals_best_pose_and_primary_subsite(self):
        from modules import engine as real_engine

        def pose(number, energy, x, site="Pocket 1", pocket="Site A"):
            return {
                "ligand": "LAP1", "pose": number, "energy": energy,
                "site": site, "pocket": pocket,
                "centroid_x": x, "centroid_y": 0.0, "centroid_z": 0.0,
            }

        primary_best = pose(1, -8.0, 0.0)
        primary_center = pose(2, -9.0, 2.0)
        primary_far = pose(3, -8.0, 10.0)
        outside = pose(9, -12.0, 30.0, "N/A", "N/A")
        other_site = pose(4, -10.0, 40.0, "Pocket 1", "Site B")
        ranking_rows = [
            {
                "ligand": "LAP1", "pose": 1, "energy": -8.0,
                "site": "Pocket 1", "cluster": "Site A", "rank": 1,
            },
            {
                "ligand": "LAP1", "pose": 2, "energy": -9.0,
                "site": "Pocket 1", "cluster": "Site A", "rank": 2,
            },
            {
                "ligand": "LAP1", "pose": 4, "energy": -10.0,
                "site": "Pocket 1", "cluster": "Site B", "rank": 1,
            },
        ]

        rows = real_engine.build_benchmarking_rows(
            [primary_best, primary_center, primary_far, outside, other_site],
            [primary_best, primary_center, primary_far, other_site],
            ranking_rows,
        )

        self.assertEqual([row["Criterion"] for row in rows], [
            "Lowest energy", "Spatial center", "Multifactorial method",
        ])
        self.assertEqual(rows[0]["Pose"], 9)
        self.assertEqual(rows[1]["Pose"], 2)
        self.assertEqual(rows[2]["Pose"], 1)
        self.assertEqual(rows[2]["Site"], "Site A")

    def test_benchmarking_breaks_center_ties_by_energy_then_pose(self):
        from modules import engine as real_engine

        def pose(number, energy, x):
            return {
                "ligand": "LAP1", "pose": number, "energy": energy,
                "site": "Pocket 1", "pocket": "Site A",
                "centroid_x": x, "centroid_y": 0.0, "centroid_z": 0.0,
            }

        lower_energy = pose(3, -9.0, -1.0)
        higher_energy = pose(2, -8.0, 1.0)
        ranking_rows = [{
            "ligand": "LAP1", "pose": 2, "energy": -8.0,
            "site": "Pocket 1", "cluster": "Site A", "rank": 1,
        }]
        rows = real_engine.build_benchmarking_rows(
            [lower_energy, higher_energy],
            [lower_energy, higher_energy],
            ranking_rows,
        )
        self.assertEqual(rows[1]["Pose"], 3)

        same_energy_lower_pose = pose(2, -9.0, 1.0)
        rows = real_engine.build_benchmarking_rows(
            [lower_energy, same_energy_lower_pose],
            [lower_energy, same_energy_lower_pose],
            ranking_rows,
        )
        self.assertEqual(rows[1]["Pose"], 2)

    def test_benchmarking_sheet_has_expected_columns_and_energy_format(self):
        from modules import engine as real_engine
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.remove(workbook.active)
        worksheet = real_engine.write_benchmarking_sheet(workbook, [{
            "Criterion": "Lowest energy", "Ligand": "LAP1", "Pose": 1,
            "Pocket": "N/A", "Site": "N/A", "Energy (kcal/mol)": -9.5,
        }])

        self.assertEqual(worksheet.title, "Benchmarking")
        self.assertEqual(
            [cell.value for cell in worksheet[1]],
            real_engine.BENCHMARKING_COLUMNS,
        )
        self.assertEqual(worksheet.cell(2, 6).number_format, "0.000")

    def test_benchmarking_keeps_lowest_energy_when_cals_ranking_is_unavailable(self):
        from modules import engine as real_engine

        unclassified_pose = {
            "ligand": "LAP1", "pose": 7, "energy": -10.0,
            "site": "N/A", "pocket": "N/A",
            "centroid_x": 0.0, "centroid_y": 0.0, "centroid_z": 0.0,
        }
        rows = real_engine.build_benchmarking_rows(
            [unclassified_pose], [], []
        )

        self.assertEqual(rows[0]["Pose"], 7)
        self.assertEqual(rows[1]["Pose"], "N/A")
        self.assertEqual(rows[2]["Pose"], "N/A")

    def test_general_excel_uses_global_poses_and_excludes_na_from_analyses(self):
        from modules import engine as real_engine

        def interactive_pose(ligand, energy, x):
            return {
                "ligand": ligand,
                "pose": 1,
                "site": "Pocket 1",
                "pocket": "Site A",
                "energy": energy,
                "rmsd": 0.5,
                "contacts": 1,
                "centroid_x": x,
                "centroid_y": 0.0,
                "centroid_z": 0.0,
                "conservation_score": 0.4,
                "rmsd_docking": 0.2,
                "excluded_from_analysis": False,
                "interactions": [{
                    "residue": "ALA", "resnum": 1, "chain": "A",
                    "type": "Hydrophobic", "distance": 3.0,
                    "angle": None, "ligand_atom": "C1", "protein_atom": "CB",
                }],
            }

        pose_1 = interactive_pose("LAP1", -8.0, 0.0)
        pose_na = {
            "ligand": "LAP1", "pose": 2, "site": "N/A", "pocket": "N/A",
            "energy": -7.0, "rmsd": 1.0, "contacts": 0,
            "centroid_x": 1.0, "centroid_y": 0.0, "centroid_z": 0.0,
            "conservation_score": None, "rmsd_docking": None,
            "excluded_from_analysis": True, "interactions": [],
        }
        pose_3 = interactive_pose("LAP2", -9.0, 2.0)
        state = {
            "all_results": [pose_1, pose_na, pose_3],
            "analyzable_results": [pose_1, pose_3],
            "pockets": {"Pocket 1": []},
            "pocket_centroids": {"Pocket 1": np.array([0.0, 0.0, 0.0])},
            "conformational_rows": [
                {
                    "pose_index": 0, "ligand": "LAP1", "pose": 1,
                    "site": "Pocket 1", "pocket": "Site A", "energy": -8.0,
                    "combined_pose_label": "Pose 1",
                },
                {
                    "pose_index": 0, "ligand": "LAP2", "pose": 1,
                    "site": "Pocket 1", "pocket": "Site A", "energy": -9.0,
                    "combined_pose_label": "Pose 3",
                },
            ],
        }
        real_engine._assign_global_pose_labels([state])
        output = real_engine._build_general_excel([state], self.root)
        workbook = load_workbook(output, data_only=True)

        self.assertEqual(
            workbook.sheetnames,
            [
                "General Information", "Pockets", "Pocket Summary", "Site Summary",
                "Centroids", "Interactions by Type", "Interactions by Pocket",
                "Interactions by Site", "Conformational Analysis", "Structural",
                "Functional", "Edges Structural", "Nodes Structural",
                "Edges Functional", "Nodes Functional", "Structural by Site",
                "Functional by Site", "ML Analysis", "Pose Coordinates",
            ],
        )
        self.assertNotIn("Ranking BP Selection", workbook.sheetnames)
        self.assertNotIn("Benchmarking", workbook.sheetnames)
        self.assertNotIn("ScatterPlot", workbook.sheetnames)
        self.assertIn("ML Analysis", workbook.sheetnames)
        self.assertTrue(
            workbook["General Information"].sheet_properties.tabColor.rgb.endswith("FFE699")
        )
        self.assertTrue(
            workbook["Interactions by Type"].sheet_properties.tabColor.rgb.endswith("C6E0B4")
        )
        self.assertTrue(
            workbook["Conformational Analysis"].sheet_properties.tabColor.rgb.endswith("F8CBAD")
        )
        self.assertTrue(
            workbook["ML Analysis"].sheet_properties.tabColor.rgb.endswith("B4C6E7")
        )
        self.assertTrue(
            workbook["Pose Coordinates"].sheet_properties.tabColor.rgb.endswith("DBDBDB")
        )
        self.assertEqual(
            [cell.value for cell in workbook["Pose Coordinates"][1]],
            [
                "Ligand", "Pose", "Pocket", "Site",
                "Centroid X", "Centroid Y", "Centroid Z",
            ],
        )
        self.assertEqual(workbook["Pose Coordinates"]["E2"].number_format, "0.000")
        self.assertEqual(workbook["Pose Coordinates"]["C2"].value, "Pocket 1")
        self.assertEqual(workbook["Pose Coordinates"]["D2"].value, "Site A")
        self.assertEqual(
            workbook["General Information"]["B2"].value, "Pose 1"
        )
        self.assertEqual(
            workbook["General Information"]["B3"].value, "Pose 2"
        )
        self.assertEqual(
            workbook["General Information"]["B4"].value, "Pose 3"
        )
        self.assertEqual(workbook["Pocket Summary"]["B2"].value, 2)
        self.assertEqual(workbook["Pocket Summary"]["D2"].value, "Pose 3")
        self.assertEqual(workbook["Structural"]["A2"].value, "Pose 1")
        self.assertEqual(workbook["Structural"]["A3"].value, "Pose 3")
        self.assertEqual(workbook["Conformational Analysis"].max_row, 3)
        self.assertIn("Structural by Site", workbook.sheetnames)
        self.assertIn("Functional by Site", workbook.sheetnames)
        local_structural = workbook["Structural by Site"]
        self.assertEqual(local_structural["A1"].value, "Nodes")
        self.assertEqual(local_structural["D2"].value, "Residue")
        self.assertEqual(local_structural["G2"].value, "Frequency (f)")
        self.assertEqual(local_structural["A3"].value, "LAP1")
        self.assertEqual(local_structural["G3"].value, 1.0)
        self.assertEqual(local_structural["G3"].number_format, "0.000")

        general_headers = [cell.value for cell in workbook["General Information"][1]]
        self.assertIn("Centroid X", general_headers)
        self.assertIn("Centroid Y", general_headers)
        self.assertIn("Centroid Z", general_headers)
        self.assertNotIn("x", general_headers)
        self.assertNotIn("y", general_headers)
        self.assertNotIn("z", general_headers)
        self.assertEqual(workbook["General Information"]["G2"].number_format, "0.000")
        centroid_sheet = workbook["Centroids"]
        self.assertEqual(
            [cell.value for cell in centroid_sheet[2][:4]],
            ["Pocket", "Centroid X", "Centroid Y", "Centroid Z"],
        )
        self.assertEqual(
            [cell.value for cell in centroid_sheet[6]],
            [
                "Pocket", "Site", "Pose Count",
                "Centroid X", "Centroid Y", "Centroid Z",
            ],
        )
        self.assertEqual(centroid_sheet["B3"].number_format, "0.000")
        self.assertEqual(centroid_sheet["D7"].number_format, "0.000")
        self.assertNotIn("Contacts", general_headers)
        self.assertNotIn("Score BP", general_headers)
        self.assertNotIn(
            "Co-occurrence Structural Reproducibility (%)", general_headers
        )
        self.assertNotIn(
            "Co-occurrence Functional Reproducibility (%)", general_headers
        )
        self.assertNotIn("Score de Reproducibility", general_headers)
        self.assertNotIn("Conservation Score", general_headers)
        conformational_headers = [cell.value for cell in workbook["Conformational Analysis"][1]]
        self.assertIn("Score BP", conformational_headers)
        self.assertNotIn("Score de Reproducibility", conformational_headers)
        self.assertNotIn("Conservation Score", conformational_headers)
        self.assertNotIn("Pose Index", conformational_headers)
        for header in (
            "RMSDdocking (Å)", "SCMi (Å)", "SCMi - Standard Deviation (Å)",
            "SCM (Å)", "SCM - Standard Deviation (Å)", "PCMi (Å)",
            "PCMi - Standard Deviation (Å)", "PCM (Å)",
            "PCM - Standard Deviation (Å)",
            "Representative Pose (Boltzmann Consensus)",
            "Effective Sample Size (Kish)",
            "Co-occurrence Structural Reproducibility (%)",
            "Co-occurrence Functional Reproducibility (%)",
        ):
            self.assertNotIn(header, conformational_headers)

    def test_general_excel_uses_numeric_suffix_without_overwriting(self):
        from modules import engine as real_engine

        (self.root / "CALS - General.xlsx").write_text("existing", encoding="utf-8")
        self.assertEqual(
            real_engine._next_general_excel_path(self.root).name,
            "CALS - General (1).xlsx",
        )

    def test_batch_prepares_all_ligands_before_global_gmm_and_exports(self):
        from modules import engine as real_engine

        events = []
        batches = [
            {
                "ligand_name": "LAP1",
                "vina_files": ("LAP1.pdbqt",),
                "reference_file": "LAP1.pdb",
                "smiles_file": "LAP1.txt",
            },
            {
                "ligand_name": "LAP2",
                "vina_files": ("LAP2.pdbqt",),
                "reference_file": "LAP2.pdb",
                "smiles_file": "LAP2.txt",
            },
        ]

        def fake_staged(_pockets, _protein, vina_files, *_files):
            ligand_name = Path(vina_files[0]).stem
            events.append(f"prepare:{ligand_name}")
            result = {
                "ligand": ligand_name,
                "site": "Pocket 1",
                "pocket": "old",
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "centroid_z": 0.0,
            }
            yield {
                "analyzable_results": [result],
                "all_results": [result],
            }
            events.append(f"export:{ligand_name}")
            return f"{ligand_name}.xlsx"

        def fake_gmm(members, template=None, reference_pose=None):
            events.append("gmm")
            return [members]

        def on_excel_created(batch, excel_file):
            events.append(f"move:{batch['ligand_name']}")
            return f"destino/{excel_file}"

        def fake_general_excel(_states, _source_folder):
            events.append("general")
            return "CALS - General.xlsx"

        def on_general_excel_created(excel_file):
            events.append("move:General")
            return f"destino/{excel_file}"

        with (
            patch.object(real_engine, "_run_analysis_staged", side_effect=fake_staged),
            patch.object(real_engine, "gmm_subclustering", side_effect=fake_gmm),
            patch.object(real_engine, "_build_general_excel", side_effect=fake_general_excel),
        ):
            output = real_engine.run_batch_analysis(
                [], "protein.pdb", batches,
                on_excel_created=on_excel_created,
                on_general_excel_created=on_general_excel_created,
            )

        self.assertEqual(
            events,
            [
                "prepare:LAP1",
                "prepare:LAP2",
                "gmm",
                "export:LAP1",
                "move:LAP1",
                "export:LAP2",
                "move:LAP2",
                "general",
                "move:General",
            ],
        )
        self.assertEqual(
            output,
            [
                "destino/LAP1.xlsx", "destino/LAP2.xlsx",
                "destino/CALS - General.xlsx",
            ],
        )


if __name__ == "__main__":
    unittest.main()
