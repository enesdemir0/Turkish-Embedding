# Turkish-Embedding: generic Colab experiment launcher.
#
# This ONE script runs any experiment in configs/experiments/ — to run a different
# experiment, change CONFIG_PATH below, don't write a new Colab script per model.
#
# Usage in Google Colab (A100 GPU runtime):
#   !git clone https://github.com/enesdemir0/Turkish-Embedding.git
#   %cd Turkish-Embedding
#   Add Colab secrets DAGSHUB_USER_TOKEN and HF_TOKEN (left sidebar -> key icon),
#   set CONFIG_PATH below, then run all cells.
#
# HF_TOKEN needs read access to google/embeddinggemma-300m specifically — that repo
# is gated, so visit https://huggingface.co/google/embeddinggemma-300m while logged
# in and accept the license BEFORE running this, or the clone step will fail with a
# GatedRepoError regardless of the token being valid.

# %%
CONFIG_PATH = "configs/experiments/exp009c_compositional_root_suffix_30pct_mse.yaml"  # <- only thing to change per run

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
# --- Tokens: prefer Colab secrets over pasting them into this file ---
import os


def _load_secret(name: str) -> None:
    try:
        from google.colab import userdata  # type: ignore

        os.environ[name] = userdata.get(name)
    except Exception:
        if name not in os.environ:
            raise RuntimeError(
                f"Set a Colab secret named {name} (left sidebar -> key icon), "
                f"or set os.environ['{name}'] manually before continuing."
            )


_load_secret("DAGSHUB_USER_TOKEN")
_load_secret("HF_TOKEN")  # needs read access to google/embeddinggemma-300m (gated — accept its license first)

from huggingface_hub import login as hf_login

hf_login(token=os.environ["HF_TOKEN"])

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
