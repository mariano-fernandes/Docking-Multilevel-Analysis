import math
import re
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation
from .pose_loading import pose_to_template_mol, template_reference_coordinates
from .interactions import count_contacts, format_interaction_residue, classify_residue_group
from .rmsd_metrics import _pose_coordinates

centroid_cutoff = 5.0

binding_rmsd_cutoff = 10.0  # Do not change.

GMM_ORIENTATION_WEIGHT = 1.0

GMM_MAX_COMPONENTS = 2

GMM_MIN_SILHOUETTE = 0.7

GMM_REG_COVAR = 1e-3

min_contacts_for_assignment = 4

max_centroid_distance = 10.0

cluster_cutoff = 7.0

def get_centroid(atoms):

    coords = np.array(
        [
            atom["coord"]
            for atom in atoms
        ]
    )

    return coords.mean(axis=0)

def centroid_distance(c1, c2):

    return math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2 +
        (c1[2] - c2[2]) ** 2
    )

def get_pocket_centroid(pocket_atoms):

    coords = np.array([

        [atom["x"], atom["y"], atom["z"]]
        for atom in pocket_atoms
    ])

    return coords.mean(axis = 0)

def classify_pose(pose_atoms, pockets):

    pocket_contacts = {}

    for pocket_name, pocket_atoms in pockets.items():

        contacts = count_contacts(
            pose_atoms,
            pocket_atoms
        )

        pocket_contacts[pocket_name] = contacts

    best_pocket = max(
        pocket_contacts,
        key = pocket_contacts.get
    )

    best_contacts = pocket_contacts[best_pocket]

    if best_contacts == 0:

        return ("Unclassified", 0, pocket_contacts)

    return (
        best_pocket,
        best_contacts,
        pocket_contacts
    )

def build_cavity_rows(pockets):
    """Build one row for each residue present in each pocket."""

    rows = []
    for cavity, atoms in sorted(pockets.items(), key=lambda item: natural_sort_key(item[0])):
        residues = {
            (
                str(atom.get("residue", "")).upper(),
                atom.get("resnum"),
                str(atom.get("chain", "")).strip(),
            )
            for atom in atoms
            if atom.get("resnum") is not None
        }
        for residue, resnum, chain in sorted(
                residues,
                key=lambda item: (int(item[1]), item[2], item[0])):
            rows.append({
                "Pocket": cavity,
                "Residue": format_interaction_residue(residue, resnum, chain),
                "Group": classify_residue_group(residue),
            })
    return rows

def gmm_subclustering(
        cluster_members,
        template,
        reference_pose,
        max_components=GMM_MAX_COMPONENTS,
        min_silhouette=GMM_MIN_SILHOUETTE,
        reg_covar=GMM_REG_COVAR):
    """Subcluster a site with GMM using pose position and orientation.

    Silhouette selection with a minimum threshold is deliberately conservative:
    BIC tends to fragment sites and must not be reintroduced here.
    """
    del template  # The signature retains explicit compatibility with site assignment.
    if len(cluster_members) <= 2:
        return [cluster_members]

    features = []
    for member in cluster_members:
        centroid = [
            member["centroid_x"],
            member["centroid_y"],
            member["centroid_z"],
        ]
        member_reference_pose = member.get(
            "_gmm_reference_pose", reference_pose
        )
        if member_reference_pose is None:
            raise ValueError("Pose without a reference for GMM orientation.")
        features.append(
            centroid + pose_rotation_vector(
                member, member_reference_pose
            ).tolist()
        )
    scaled_features = StandardScaler().fit_transform(np.asarray(features, dtype=float))
    scaled_features[:, 3:] *= GMM_ORIENTATION_WEIGHT

    maximum = min(max_components, len(cluster_members) - 1)
    best_components = 1
    best_silhouette = None
    for component_count in range(1, maximum + 1):
        model, _ = _fit_gmm(scaled_features, component_count, reg_covar)
        if model is None or component_count == 1:
            continue

        labels = model.predict(scaled_features)
        label_count = len(np.unique(labels))
        if not 1 < label_count < len(cluster_members):
            print(
                "GMM warning: invalid silhouette labels for "
                f"n={component_count}; candidate ignored."
            )
            continue
        try:
            score = float(silhouette_score(scaled_features, labels))
        except Exception as silhouette_error:
            print(
                "GMM warning: silhouette calculation failed for "
                f"n={component_count} ({silhouette_error}); candidate ignored."
            )
            continue
        if best_silhouette is None or score > best_silhouette:
            best_components = component_count
            best_silhouette = score

    # Without a split above the threshold, n=1 preserves the full site.
    if best_silhouette is None or best_silhouette <= min_silhouette:
        best_components = 1

    final_model, covariance_type = _fit_gmm(
        scaled_features, best_components, reg_covar
    )
    if final_model is None:
        print("GMM warning: final fit failed; keeping one subcluster.")
        return [cluster_members]

    labels = final_model.predict(scaled_features)
    grouped_indices = {}
    for index, label in enumerate(labels):
        grouped_indices.setdefault(int(label), []).append(index)
    print(
        f"GMM: {len(cluster_members)} poses, n={best_components}, "
        f"silhouette={best_silhouette if best_silhouette is not None else 'n/a'}, "
        f"covariance={covariance_type}."
    )
    return [
        [cluster_members[index] for index in indices]
        for _, indices in sorted(grouped_indices.items(), key=lambda item: item[1][0])
    ]

def _fit_gmm(features, component_count, reg_covar):
    """Fit a tied GMM; use spherical only if tied fails numerically."""
    try:
        return GaussianMixture(
            n_components=component_count,
            covariance_type="tied",
            reg_covar=reg_covar,
            random_state=42,
            n_init=3,
        ).fit(features), "tied"
    except Exception as tied_error:
        print(
            "GMM warning: tied covariance failed for "
            f"n={component_count} ({tied_error}); using spherical."
        )
        try:
            return GaussianMixture(
                n_components=component_count,
                covariance_type="spherical",
                reg_covar=reg_covar,
                random_state=42,
                n_init=3,
            ).fit(features), "spherical"
        except Exception as spherical_error:
            print(
                "GMM warning: spherical fit also failed for "
                f"n={component_count} ({spherical_error})."
            )
            return None, None

def pose_rotation_vector(pose, reference_pose):
    """Calculate pose-to-reference orientation by Kabsch in axis-angle form."""
    pose_coordinates = _pose_coordinates(pose)
    reference_coordinates = np.asarray(reference_pose, dtype=float)
    if pose_coordinates.shape != reference_coordinates.shape:
        raise ValueError("Pose and template have different atom counts for GMM.")

    pose_centered = pose_coordinates - np.mean(pose_coordinates, axis=0)
    reference_centered = reference_coordinates - np.mean(reference_coordinates, axis=0)
    covariance = pose_centered.T @ reference_centered
    left_vectors, _, right_vectors = np.linalg.svd(covariance)
    rotation_matrix = right_vectors.T @ left_vectors.T
    if np.linalg.det(rotation_matrix) < 0:
        right_vectors[-1, :] *= -1
        rotation_matrix = right_vectors.T @ left_vectors.T

    rotation_vector = Rotation.from_matrix(rotation_matrix).as_rotvec()
    angle = np.linalg.norm(rotation_vector)
    # (theta, axis) and (theta - 2π, axis) encode the same orientation.
    # Keep the representation within π to avoid separating identical poses.
    if angle > np.pi:
        rotation_vector = rotation_vector * (1.0 - (2.0 * np.pi / angle))
    return rotation_vector

def group_results_by_pocket(results):
    """Group poses by (Pocket, Site) while retaining external labels."""

    groups = {}

    for result in results:

        groups.setdefault(get_site_key(result), []).append(result)

    return groups

def get_site_key(result):
    """Chave interna que impede misturar sites de pockets distintas."""

    pocket = result["pocket"]
    return result.get("site", pocket), pocket

def natural_sort_key(value):
    """Natural sort key for pocket labels."""

    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    ]
