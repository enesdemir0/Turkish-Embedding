# Turkish-Embedding: generic Colab experiment launcher.
#
# This ONE script runs any experiment in configs/experiments/ — to run a different
# experiment, change CONFIG_PATH below, don't write a new Colab script per model.
#
# Usage in Google Colab (A100 GPU runtime):
#   !git clone https://github.com/enesdemir0/Turkish-Embedding.git
#   %cd Turkish-Embedding
#   Add a Colab secret named DAGSHUB_USER_TOKEN (recommended), set CONFIG_PATH below,
#   then run all cells.

# %%
CONFIG_PATH = "configs/experiments/exp001_mean_composition_cosine.yaml"  # <- only thing to change per run

# %%
# --- Install dependencies ---
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)

# %%
# --- Confirm GPU ---
import torch

assert torch.cuda.is_available(), "No GPU detected — Runtime > Change runtime type > A100"
print("GPU:", torch.cuda.get_device_name(0))

# %%
# --- DagsHub token: prefer Colab secrets over pasting it into this file ---
import os

try:
    from google.colab import userdata  # type: ignore

    os.environ["DAGSHUB_USER_TOKEN"] = userdata.get("DAGSHUB_USER_TOKEN")
except Exception:
    if "DAGSHUB_USER_TOKEN" not in os.environ:
        raise RuntimeError(
            "Set a Colab secret named DAGSHUB_USER_TOKEN (left sidebar -> key icon), "
            "or set os.environ['DAGSHUB_USER_TOKEN'] manually before continuing."
        )

# %%
# --- Run the experiment ---
sys.path.insert(0, ".")

from src.config import ExperimentConfig
from src.pipeline import Pipeline
from src.tracking.experiment_tracker import init_tracking

init_tracking()
config = ExperimentConfig.load(CONFIG_PATH)
results = Pipeline(config).run()
print(results)
