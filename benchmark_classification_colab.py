# Turkish Classification-only benchmark for trmteb/turkish-embedding-model
# using the mteb_tr fork (https://github.com/selmanbaysan/mteb_tr).
#
# Usage in Google Colab (T4 GPU runtime):
#   !git clone <this-repo-url>
#   %cd <this-repo-name>
#   !python benchmark_classification_colab.py

# %%
# --- Install dependencies (mteb_tr fork + sentence-transformers) ---
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/selmanbaysan/mteb_tr.git",
        "sentence-transformers",
    ],
    check=True,
)

# %%
# --- Confirm we actually have the T4 GPU ---
import torch

assert torch.cuda.is_available(), "No GPU detected — set Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

# %%
# --- Load the model ---
from sentence_transformers import SentenceTransformer

MODEL_NAME = "trmteb/turkish-embedding-model"
model = SentenceTransformer(MODEL_NAME, device="cuda")

# %%
# --- Select only the Turkish Classification tasks (not the full benchmark) ---
import mteb

tasks = mteb.get_tasks(task_types=["Classification"], languages=["tur"])
print(f"Selected {len(tasks)} tasks:")
for task in tasks:
    print(" -", task.metadata.name)

# %%
# --- Run the evaluation ---
OUTPUT_FOLDER = f"results/{MODEL_NAME.replace('/', '__')}"

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(
    model,
    output_folder=OUTPUT_FOLDER,
    encode_kwargs={"batch_size": 64},
)

# %%
# --- Summarize results into a simple table ---
import glob
import json

import pandas as pd

rows = []
for path in glob.glob(f"{OUTPUT_FOLDER}/**/*.json", recursive=True):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    task_name = data.get("task_name")
    if task_name is None:
        continue

    for split_scores in data.get("scores", {}).values():
        for entry in split_scores:
            rows.append(
                {
                    "task": task_name,
                    "split": entry.get("hf_subset", "default"),
                    "accuracy": entry.get("accuracy") or entry.get("main_score"),
                }
            )

df = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
print(df.to_string(index=False))
print(f"\nMean accuracy: {df['accuracy'].mean():.4f}")
