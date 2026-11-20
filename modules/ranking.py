import math


ENERGY_COMPETITIVE_WINDOW = 1.0
ENERGY_DECAY_SCALE = 2.0


def _numeric_sort_value(value):
    """Sort numbers before unavailable values without comparing None."""

    if value is None:
        return (1, 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (1, 0.0)
    return (0, number) if math.isfinite(number) else (1, 0.0)


def _rounded_score_sort_value(value):
    """Sort a displayed score rounded to three decimal places, highest first."""

    missing, number = _numeric_sort_value(value)
    return (missing, -round(number, 3))


def score_bp_values(energy, best_energy, reproducibility_structural,
                    reproducibility_functional):
    """Calculate Energy (exp) and Score BP for a pose in the local scope."""

    energy_missing, energy_value = _numeric_sort_value(energy)
    best_missing, best_value = _numeric_sort_value(best_energy)
    if energy_missing or best_missing:
        return None, None

    energy_exp = math.exp(-(energy_value - best_value) / ENERGY_DECAY_SCALE)
    structural_missing, structural = _numeric_sort_value(
        reproducibility_structural
    )
    functional_missing, functional = _numeric_sort_value(
        reproducibility_functional
    )
    if structural_missing or functional_missing or structural < 0 or functional < 0:
        return energy_exp, None

    score_bp = (energy_exp * structural * functional) ** (1 / 3)
    return energy_exp, score_bp


def add_score_bp_values(rows, scope_fields, energy_field="energy",
                        structural_field="reproducibility_structural",
                        functional_field="reproducibility_functional"):
    """Attach Score BP to records using the minimum ΔG in each scope."""

    grouped_rows = {}
    for row in rows:
        scope = tuple(row.get(field) for field in scope_fields)
        grouped_rows.setdefault(scope, []).append(row)

    for scope_rows in grouped_rows.values():
        energies = [
            number
            for row in scope_rows
            for missing, number in [_numeric_sort_value(row.get(energy_field))]
            if not missing
        ]
        best_energy = min(energies) if energies else None
        for row in scope_rows:
            energy_exp, score_bp = score_bp_values(
                row.get(energy_field),
                best_energy,
                row.get(structural_field),
                row.get(functional_field),
            )
            row["energy_exp"] = energy_exp
            row["score_bp"] = score_bp
    return rows


def order_ranking_rows(ranking_rows, cavity_pose_counts, site_pose_counts):
    """Order Ranking BP Selection according to best-pose classification.

    Global priority remains pocket and site population. In each local scope,
    poses within 1 kcal/mol of the best energy are compared by Score BP
    rounded to three decimal places, then by energy and RMSDdocking. Energy
    (exp) and Score BP are calculated and retained for every pose, including
    those outside the competitive window.
    """

    grouped_rows = {}
    for row in ranking_rows:
        scope = (row["site"], row["cluster"])
        grouped_rows.setdefault(scope, []).append(dict(row))

    ordered_groups = []
    for scope, rows in grouped_rows.items():
        energies = [
            float(row["energy"])
            for row in rows
            if _numeric_sort_value(row.get("energy"))[0] == 0
        ]
        best_energy = min(energies) if energies else None

        add_score_bp_values(rows, ("site", "cluster"))

        def within_energy_window(row):
            if best_energy is None:
                return False
            missing, energy = _numeric_sort_value(row.get("energy"))
            return not missing and energy <= best_energy + ENERGY_COMPETITIVE_WINDOW

        ranked_rows = sorted(
            rows,
            key=lambda row: (
                0 if within_energy_window(row) else 1,
                _rounded_score_sort_value(row.get("score_bp"))
                if within_energy_window(row) else _numeric_sort_value(row.get("energy")),
                _numeric_sort_value(row.get("energy")),
                _numeric_sort_value(row.get("rmsd_docking")),
                str(row.get("ligand", "")),
                str(row.get("pose", "")),
            ),
        )
        for rank, row in enumerate(ranked_rows, start=1):
            row["rank"] = rank
        ordered_groups.append((scope, ranked_rows))

    ordered_groups.sort(
        key=lambda item: (
            -cavity_pose_counts[item[0][0]],
            -site_pose_counts[item[0]],
            str(item[0][0]),
            str(item[0][1]),
        )
    )
    return [row for _, rows in ordered_groups for row in rows]
