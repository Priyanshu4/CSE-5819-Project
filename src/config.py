from pathlib import Path

CONFIGS_PATH = Path(__file__).parent.parent / "configs"
DATASETS_CONFIG_PATH = CONFIGS_PATH / "datasets.json"
ALL_RESULTS_PATH = Path(__file__).parent.parent / "results"


def get_results_path(name: str):
    """ Returns the path to the results of the experiment with the given name.
    """
    if not ALL_RESULTS_PATH.exists():
        ALL_RESULTS_PATH.mkdir()

    results_path = ALL_RESULTS_PATH / name

    if not results_path.exists():
        results_path.mkdir()

    return results_path.resolve()