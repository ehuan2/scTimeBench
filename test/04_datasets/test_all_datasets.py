from scTimeBench.shared.constants import RequiredOutputFiles
import pytest

from test.test_suite_helpers import (
    build_method_config,
    dataset_tag_to_name,
    GEX_DATASET_NAMES,
    get_output_paths_for_method,
    sql_fetchone,
    write_config,
)


def _metric_for_dataset(dataset_name: str) -> str:
    if dataset_name in GEX_DATASET_NAMES:
        return "WassersteinOTLoss"
    return "JaccardSimilarity"


DATASET_TAG_NAME_ITEMS = sorted(dataset_tag_to_name().items())


@pytest.mark.parametrize("dataset_tag,dataset_name", DATASET_TAG_NAME_ITEMS)
def test_dataset_smoke_with_simplest_scnode(
    dataset_tag,
    dataset_name,
    workspace,
    run_bench,
):
    """
    Validate each shared dataset tag with simple scNODE config.

    Uses:
    - JaccardSimilarity for ontology-style datasets,
    - WassersteinOTLoss for gex-prediction datasets.
    """
    metric_name = _metric_for_dataset(dataset_name)

    config = build_method_config(
        "scNODE",
        [{"tag": dataset_tag}],
        [{"name": metric_name}],
    )

    config_path = write_config(config, workspace, f"dataset_{dataset_tag}.yaml")

    result = run_bench(
        config_path,
        "auto_train_test",
        workspace,
        f"dataset_{dataset_tag}.log",
        extra_args=["--force_rerun"],
    )
    assert result.returncode == 0

    db_path = workspace / "scTimeBench.db"

    eval_count = sql_fetchone(
        db_path,
        """
        SELECT COUNT(*)
        FROM evals e
        JOIN metrics m ON m.id = e.metric_id
        JOIN method_outputs mo ON mo.id = e.method_output_id
        JOIN datasets d ON d.id = mo.dataset_id
        WHERE mo.name = 'scNODE' AND m.name = ? AND d.name = ?
        """,
        (metric_name, dataset_name),
    )[0]
    assert (
        eval_count > 0
    ), f"No eval row found for dataset tag {dataset_tag} ({dataset_name}) with metric {metric_name}."

    # Basic output check for these metrics with scNODE: next timepoint GEX should exist no matter what.
    output_paths = get_output_paths_for_method(db_path, "scNODE")
    assert output_paths, "No output paths recorded for scNODE."

    expected_file = RequiredOutputFiles.NEXT_TIMEPOINT_GENE_EXPRESSION.value
    assert any(
        (output_path / expected_file).exists() for output_path in output_paths
    ), f"Expected required output {expected_file} not found for dataset tag {dataset_tag}."
