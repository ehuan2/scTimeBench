"""
Helper function for miscellaneous tasks.
"""


def parse_equivalence(file_path):
    """
    Parse a cell equivalence file and create a dictionary mapping equivalent names.

    Parameters:
    -----------
    file_path : str
        Path to the equivalence file (split by ,)

    Returns:
    --------
    dict
        Dictionary mapping normalized cell type names to one or more equivalent names
    """
    if file_path is None:
        return {}

    with open(file_path, "r") as f:
        content = f.read().strip()

    equivalence = {}

    # for each row in the file:
    for row in content.splitlines():
        # Split by , and clean up first
        cells = [cell.strip() for cell in row.split(",") if cell.strip()]

        if not cells:
            continue

        # then let's build the equivalence mapping
        key = cells[0]
        values = cells[1:]

        if not values:
            continue

        if key not in equivalence:
            equivalence[key] = []

        for value in values:
            if value not in equivalence[key]:
                equivalence[key].append(value)

    return equivalence


def parse_cell_lineage(file_path, equivalence_file_path=None):
    """
    Parse a cell lineage file and create a dictionary mapping source to root.

    Parameters:
    -----------
    file_path : str
        Path to the lineage file (split by =>)

    Returns:
    --------
    dict
        Dictionary mapping normalized cell types to their root
    """
    equivalence_dict = parse_equivalence(equivalence_file_path)

    with open(file_path, "r") as f:
        content = f.read().strip()

    lineage = {}

    # for each row in the file:
    for row in content.splitlines():
        # Split by => and clean up first
        cells = [cell.strip() for cell in row.split("=>") if cell.strip()]

        # replace names using equivalence mapping, then normalize to lists
        normalized_cells = []
        for cell in cells:
            if cell in equivalence_dict:
                mapped = equivalence_dict[cell]
                normalized_cells.append(
                    mapped if isinstance(mapped, list) else [mapped]
                )
            else:
                normalized_cells.append([cell])

        # then let's build the lineage mapping, expanding one-to-many mappings
        for i in range(len(normalized_cells) - 1):
            sources = normalized_cells[i]
            targets = normalized_cells[i + 1]
            for source in sources:
                if source not in lineage:
                    lineage[source] = []
                for target in targets:
                    if target not in lineage[source]:
                        lineage[source].append(target)

    return lineage
