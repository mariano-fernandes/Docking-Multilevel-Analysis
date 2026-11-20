"""Interaction incidence tables and networks exported by CALS."""

from collections import Counter
from itertools import combinations
import re


FUNCTIONAL_CODES = {
    "H-Bond": "HB",
    "Hydrophobic": "Hyd",
    "π-stacking (Parallel)": "πstcP",
    "π-stacking (Perpendicular)": "πstcT",
    "cation-π": "catπ",
    "Salt Bridge": "SB",
    "Halogen Bond": "Hal-B",
}

STRUCTURAL_COLUMNS = ("Pose", "Residue")
FUNCTIONAL_COLUMNS = ("Pose", "Res-Int")
RANDOM_FOREST_COLUMNS = ("Pose", "Res-Int", "Energy (kcal/mol)")
EDGE_COLUMNS = ("Source", "Target", "Type", "Weight")
NODE_COLUMNS = ("Label", "Id")

LOCAL_NODE_COLUMNS = (
    "Ligand", "Pocket", "Site", "Node", "Present Poses",
    "Pose Count", "Frequency (f)", "Hub Strength",
    "Centrality", "Raw Weight", "Normalized Weight",
)
LOCAL_EDGE_COLUMNS = (
    "Ligand", "Pocket", "Site", "Source", "Target",
    "Co-occurring Poses", "Pose Count", "Frequency (f)",
    "Normalized Weight",
)


def _display_residue(residue, chain=""):
    """Convert ``TYR217``/``A`` to ``Tyr217A`` without changing number/insertion."""

    match = re.match(r"([A-Za-z]+)(.*)", str(residue))
    label = f"{match.group(1).title()}{match.group(2)}" if match else str(residue)
    chain = str(chain or "").strip()
    return f"{label}{chain}" if chain else label


def _pose_key(row):
    """Keep poses from different Vina files as distinct instances."""

    return row.get("network_pose_key", (row["ligand"], row["pose"]))


def _add_ordered(container, value):
    """Insert once while preserving the Interactions by Type sheet order."""

    container.setdefault(value, None)


def _edge_rows(per_pose_nodes):
    weights = Counter()
    for nodes in per_pose_nodes:
        for first, second in combinations(nodes, 2):
            source, target = sorted((first, second))
            weights[(source, target)] += 1
    return [
        {"Source": source, "Target": target, "Type": "Undirected", "Weight": weight}
        for (source, target), weight in sorted(weights.items())
    ]


def _node_rows(nodes):
    return [{"Label": node, "Id": node} for node in nodes]


def build_network_tables(interaction_rows):
    """Build network tables and the ML Analysis data from computed interactions.

    The first occurrence of each pose defines its global number. Presence is
    binary within a pose, so an edge weight is the number of poses containing
    both of its nodes.
    """

    structural_by_pose = {}
    functional_by_pose = {}
    pose_labels = {}
    pose_energies = {}
    structural_nodes = {}
    functional_nodes = {}

    for row in interaction_rows:
        key = _pose_key(row)
        if key not in pose_labels:
            pose_labels[key] = row.get(
                "global_pose_label", f"Pose {len(pose_labels) + 1}"
            )
            structural_by_pose[key] = {}
            functional_by_pose[key] = {}
            pose_energies[key] = row.get("energy")

        residue = _display_residue(row["residue"], row.get("chain", ""))
        try:
            code = FUNCTIONAL_CODES[row["interaction_type"]]
        except KeyError as exc:
            raise ValueError(
                f"Interaction type without a functional code: {row['interaction_type']}"
            ) from exc

        functional = f"{residue}-{code}"
        _add_ordered(structural_by_pose[key], residue)
        _add_ordered(functional_by_pose[key], functional)
        _add_ordered(structural_nodes, residue)
        _add_ordered(functional_nodes, functional)

    structural_rows = [
        {"Pose": pose_labels[key], "Residue": residue}
        for key, residues in structural_by_pose.items()
        for residue in residues
    ]
    functional_rows = [
        {"Pose": pose_labels[key], "Res-Int": functional}
        for key, functional_nodes_for_pose in functional_by_pose.items()
        for functional in functional_nodes_for_pose
    ]
    random_forest_rows = [
        {
            "Pose": pose_labels[key],
            "Res-Int": functional,
            "Energy (kcal/mol)": pose_energies[key],
        }
        for key, functional_nodes_for_pose in functional_by_pose.items()
        for functional in functional_nodes_for_pose
    ]

    return {
        "structural": structural_rows,
        "functional": functional_rows,
        "random_forest": random_forest_rows,
        "edges_structural": _edge_rows(structural_by_pose.values()),
        "nodes_structural": _node_rows(structural_nodes),
        "edges_functional": _edge_rows(functional_by_pose.values()),
        "nodes_functional": _node_rows(functional_nodes),
    }


def _local_scope_key(row):
    """Identify a ligand reference network in a specific site."""

    return (
        canonical_network_ligand(row.get("ligand", "")),
        str(row.get("site", "")),
        str(row.get("pocket", "")),
    )


def canonical_network_ligand(ligand):
    """Group Vina replicates ``LAP1(7)`` under reference ``LAP1``.

    Normalization is used only in the reference network. Original file/pose
    labels remain available in the other tables for auditing.
    """

    return re.sub(r"\(\d+\)\s*$", "", str(ligand)).rstrip()


def _normalised_weights(weights):
    """Normalize a dictionary of weights while preserving the empty case."""

    total = sum(weights.values())
    if total == 0:
        return {key: 0.0 for key in weights}
    return {key: value / total for key, value in weights.items()}


def _build_local_network(scope, pose_nodes, network_kind):
    """Calculate weights and fidelity for a Structural or Functional network."""

    pose_keys = list(pose_nodes)
    total_poses = len(pose_keys)
    node_counts = Counter(
        node
        for nodes in pose_nodes.values()
        for node in nodes
    )
    node_frequency = {
        node: count / total_poses
        for node, count in node_counts.items()
    }

    edge_counts = Counter()
    for nodes in pose_nodes.values():
        for first, second in combinations(sorted(nodes), 2):
            edge_counts[(first, second)] += 1
    edge_frequency = {
        edge: count / total_poses
        for edge, count in edge_counts.items()
    }

    node_strength = {node: 0.0 for node in node_frequency}
    for (first, second), frequency in edge_frequency.items():
        node_strength[first] += frequency
        node_strength[second] += frequency
    maximum_strength = max(node_strength.values(), default=0.0)
    centrality = {
        node: (strength / maximum_strength if maximum_strength else 0.0)
        for node, strength in node_strength.items()
    }
    raw_node_weight = {
        node: node_frequency[node] ** 2 * (
            1.0 + node_frequency[node] * centrality[node]
        )
        for node in node_frequency
    }
    node_weight = _normalised_weights(raw_node_weight)
    edge_weight = _normalised_weights({
        edge: frequency ** 2
        for edge, frequency in edge_frequency.items()
    })

    ligand, cavity, site = scope
    node_rows = [
        {
            "Ligand": ligand,
            "Pocket": cavity,
            "Site": site,
            "Node": node,
            "Present Poses": node_counts[node],
            "Pose Count": total_poses,
            "Frequency (f)": node_frequency[node],
            "Hub Strength": node_strength[node],
            "Centrality": centrality[node],
            "Raw Weight": raw_node_weight[node],
            "Normalized Weight": node_weight[node],
        }
        for node in sorted(node_frequency)
    ]
    edge_rows = [
        {
            "Ligand": ligand,
            "Pocket": cavity,
            "Site": site,
            "Source": first,
            "Target": second,
            "Co-occurring Poses": edge_counts[(first, second)],
            "Pose Count": total_poses,
            "Frequency (f)": edge_frequency[(first, second)],
            "Normalized Weight": edge_weight[(first, second)],
        }
        for first, second in sorted(edge_frequency)
    ]
    scores = {
        pose_key: {
            f"fidelity_{network_kind}": sum(
                node_weight[node] for node in nodes
            ),
            f"fidelity_cooccurrence_{network_kind}": (
                sum(
                    edge_weight[edge]
                    for edge in combinations(sorted(nodes), 2)
                    if edge in edge_weight
                )
                if edge_weight else None
            ),
        }
        for pose_key, nodes in pose_nodes.items()
    }
    return node_rows, edge_rows, scores


def build_local_network_fidelity(interaction_rows, scope_poses=()):
    """Score each pose by fidelity to the already exported local networks.

    Each reference network is restricted to ``canonical ligand × pocket ×
    site``; for example, ``LAP1(7)`` and ``LAP1(42)`` use ``LAP1``.
    Structural nodes are residues and Functional nodes are Res-Int. No
    interaction type receives an external chemical weight: importance emerges
    from frequency and centrality in its own local network. Frequency is
    squared so the score favors the recurrent core without requiring an
    arbitrary "key interaction" cutoff.
    """

    by_scope = {}
    for row in scope_poses:
        scope = _local_scope_key(row)
        pose_key = row.get("network_pose_key", (row.get("ligand"), row.get("pose")))
        by_scope.setdefault(scope, {}).setdefault(
            pose_key, {"structural": {}, "functional": {}}
        )
    for row in interaction_rows:
        scope = _local_scope_key(row)
        pose_key = _pose_key(row)
        group = by_scope.setdefault(scope, {})
        pose = group.setdefault(pose_key, {"structural": {}, "functional": {}})
        residue = _display_residue(row["residue"], row.get("chain", ""))
        functional = f"{residue}-{FUNCTIONAL_CODES[row['interaction_type']]}"
        pose["structural"][residue] = None
        pose["functional"][functional] = None

    tables = {
        "local_nodes_structural": [],
        "local_edges_structural": [],
        "local_nodes_functional": [],
        "local_edges_functional": [],
        "pose_scores": [],
    }
    for scope, poses in by_scope.items():
        combined_scores = {pose_key: {} for pose_key in poses}
        for network_kind in ("structural", "functional"):
            pose_nodes = {
                pose_key: set(values[network_kind])
                for pose_key, values in poses.items()
            }
            node_rows, edge_rows, scores = _build_local_network(
                scope, pose_nodes, network_kind
            )
            tables[f"local_nodes_{network_kind}"].extend(node_rows)
            tables[f"local_edges_{network_kind}"].extend(edge_rows)
            for pose_key, values in scores.items():
                combined_scores[pose_key].update(values)

        ligand, cavity, site = scope
        for pose_key, scores in combined_scores.items():
            tables["pose_scores"].append({
                "ligand": ligand,
                "site": cavity,
                "pocket": site,
                "pose_key": pose_key,
                **scores,
            })
    return tables
