from pathlib import Path

CONFIGS_PATH = Path(__file__).parent.parent / "configs"
DATASETS_CONFIG_PATH = CONFIGS_PATH / "datasets.json"
RESULTS_PATH = Path(__file__).parent.parent / "results"
EMBEDDINGS_PATH = RESULTS_PATH / "embeddings"
LOGS_PATH = RESULTS_PATH / "logs"

if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir()

if not LOGS_PATH.exists():
    LOGS_PATH.mkdir()

if not EMBEDDINGS_PATH.exists():
    EMBEDDINGS_PATH.mkdir()
