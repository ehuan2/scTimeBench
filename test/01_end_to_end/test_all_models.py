import pytest
from pathlib import Path
from crispy_fishstick.metrics.base import METRIC_REGISTRY

# 1. Define the path to your configs
CONFIG_DIR = Path(__file__).parent / "configs"
# 2. Automatically find all .yaml files
CONFIG_FILES = list(CONFIG_DIR.glob("*.yaml"))

TEST_DIR = Path(__file__).parent.parent


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_config_execution(config_path, workspace, run_bench):
    """
    Test that a given config file can be executed end-to-end
    and produces the expected outputs.
    """
    log_file = Path(config_path).name + ".log"
    result = run_bench(config_path, "auto_train_test", workspace, log_file)

    assert result.returncode == 0

    output_files = list((workspace / "outputs").iterdir())
    assert len(output_files) > 0, f"Config {config_path.name} produced no output."

    # let's verify that we have:
    # 1. database object
    # 2. metric outputs
    # 3. dataset pickle
    # 4. model config

    # then for each metric collect a set of all required output files
    with open(config_path, "r") as f:
        import yaml

        config_data = yaml.safe_load(f)

    metrics = config_data.get("metrics", [])
    output_required_files = set()
    for metric in metrics:
        metric_name = metric["name"]
        output_required_files.add(
            METRIC_REGISTRY[metric_name](None, None, {}).output_path_name.value
        )

    output_required_files.update(
        [
            "dataset.pkl",
            "model_config.yaml",
        ]
    )

    # here we parse for the proper output path:
    # by looking under the logs for the line: Output path for model: <path>
    with open(TEST_DIR / "logs" / log_file, "r") as log_file_handle:
        for line in log_file_handle:
            if "Output path for" in line:
                parts = line.strip().split("Output path for model: ")
                if len(parts) == 2:
                    output_path = parts[1]
                    output_model_dir = Path(output_path).name
                    break

    found_output_files = set(
        f.name for f in (workspace / "outputs" / output_model_dir).iterdir()
    )

    missing_files = output_required_files - found_output_files
    assert (
        len(missing_files) == 0
    ), f"Running config {config_path.name} missing output files: {missing_files}"

    # finally, verify that the database holds the expected evaluations
    # this is good to test hierarchical metric evaluations as well

    # create a minimal config object for database access
    from crispy_fishstick.database import DatabaseManager

    class dummy_class:
        def __init__(self, database_path):
            self.database_path = database_path

    test_config = dummy_class(str(workspace / "crispy_fishstick.db"))
    db_manager = DatabaseManager(test_config)

    db_contents = db_manager.return_all()

    # now we parse line by line of the db_contents to make sure that every single
    # metric gets hit, as long as they are non-hierarchical because we will be testing that
    # config is going to include BaseMetric (i.e. all of them)
    print(db_contents)

    for metric in metrics:
        metric_name = metric["name"]
        print("Metric name:", metric_name)

        # now we should iterate over all the submetrics if it is hierarchical
        if len(METRIC_REGISTRY[metric_name].submetrics) == 0:
            # skip hierarchical metrics
            evals = db_manager.get_evals_per_metric(
                metric_name,
                METRIC_REGISTRY[metric_name](
                    None, None, {}
                )._get_param_encoding(),  # ** Note: have to use default parametrization **
            )
            print(
                f"Metric {metric_name} produced {len(evals)} evaluations in the database."
            )
            assert (
                len(evals) > 0
            ), f"Metric {metric_name} produced no evaluations in the database."
        else:
            for submetric in METRIC_REGISTRY[metric_name].submetrics:
                submetric_name = submetric.__name__
                evals = db_manager.get_evals_per_metric(
                    submetric_name,
                    submetric(
                        None, None, {}
                    )._get_param_encoding(),  # ** Note: have to use default parametrization **
                )
                print(
                    f"Submetric {submetric_name} of {metric_name} produced {len(evals)} evaluations in the database."
                )
                assert (
                    len(evals) > 0
                ), f"Submetric {submetric_name} of {metric_name} produced no evaluations in the database."

    db_manager.close()
