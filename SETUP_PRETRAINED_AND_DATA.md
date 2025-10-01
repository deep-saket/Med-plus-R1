# Med‑R1: Pretrained Checkpoints and Dataset Setup (macOS/MPS)

This guide shows how to: (1) place the authors’ pretrained checkpoints under `checkpoints/Med-R1`, (2) download/extract the OmniMedVQA dataset under `data/`, and (3) run a quick smoke test on a Mac with MPS.

What you’ll get
- Pretrained weights organized under `checkpoints/Med-R1/Qwen_2.5_3B_nothink/...`
- OmniMedVQA dataset under `data/OmniMedVQA/OmniMedVQA/` with `Images/` and `QA_information/`
- Verified paths you can pass to `inference.py` and evaluation scripts

Prereqs
- macOS with Apple Silicon recommended (MPS)
- Python 3.11 or 3.12
- A clean virtual environment and the repo requirements

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r src/requirements.txt
```

Confirm MPS

```zsh
python - <<'PY'
import torch
print('MPS available:', torch.backends.mps.is_available(), 'torch', torch.__version__)
PY
```

1) Authenticate with Hugging Face (once)

```zsh
pip install -U huggingface_hub
huggingface-cli login  # paste your token
```

2) Download authors’ pretrained checkpoints (selective, resumable)

The repo README links to the authors’ Hugging Face model: `yuxianglai117/Med-R1`.
We’ll download only the subfolders we need into `checkpoints/Med-R1/`.

Tips for large files and slow networks
- Enable the Rust/aria2 transfer helper for better throughput.
- Increase the Hub timeout and make downloads resumable.

```zsh
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1

# Create the target folder
mkdir -p checkpoints/Med-R1

# Example 1: download a single task-specific VQA checkpoint (e.g., CT)
huggingface-cli download yuxianglai117/Med-R1 \
  --include "Qwen_2.5_3B_nothink/VQA_CT/*" \
  --local-dir checkpoints/Med-R1 \
  --local-dir-use-symlinks False \
  --resume-download

# Example 2: download a task-specific checkpoint (e.g., Anatomy Identification)
huggingface-cli download yuxianglai117/Med-R1 \
  --include "Qwen_2.5_3B_nothink/VQA_AI/*" \
  --local-dir checkpoints/Med-R1 \
  --local-dir-use-symlinks False \
  --resume-download
```

If you prefer a Python snippet (fine‑grained filtering):

```zsh
python - <<'PY'
from huggingface_hub import snapshot_download
# Choose exactly what you need to reduce disk + time
allow = [
    'Qwen_2.5_3B_nothink/VQA_CT/*',
    # 'Qwen_2.5_3B_nothink/VQA_AI/*',
    # 'Qwen_2.5_3B_nothink/VQA_LG/*',
    # 'Qwen_2.5_3B_nothink/VQA_MR/*',
    # ... add others as needed
]
snapshot_download(
    repo_id='yuxianglai117/Med-R1',
    local_dir='checkpoints/Med-R1',
    allow_patterns=allow,
    resume_download=True,
    max_workers=4,
)
print('Done. Check checkpoints/Med-R1 for downloaded folders.')
PY
```

What to expect in each checkpoint folder
- Configuration: `config.json`, `generation_config.json`
- Tokenizer: `tokenizer.json`, `tokenizer_config.json`, `merges.txt`, `vocab.json`, `special_tokens_map.json`
- Weights: multi‑part `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, plus `model.safetensors.index.json`

Note: These files together can be many GBs per checkpoint. Ensure you have sufficient disk space.

3) Prepare the OmniMedVQA dataset

The repository already includes `data/OmniMedVQA/OmniMedVQA.zip` in some setups. If it exists, extract it. Otherwise, download from the authors’ dataset on the Hub: `foreverbeliever/OmniMedVQA`.

Option A — extract the provided zip (fastest)

```zsh
if [ -f data/OmniMedVQA/OmniMedVQA.zip ]; then
  echo 'Found local zip. Extracting...'
  mkdir -p data/OmniMedVQA
  unzip -n data/OmniMedVQA/OmniMedVQA.zip -d data/OmniMedVQA
else
  echo 'No local zip found. Use Option B to download.'
fi
```

Option B — download from Hugging Face (dataset repo)

```zsh
# Download only the main archive to save time
huggingface-cli download foreverbeliever/OmniMedVQA \
  --repo-type dataset \
  --include "OmniMedVQA.zip" \
  --local-dir data/OmniMedVQA \
  --local-dir-use-symlinks False \
  --resume-download

unzip -n data/OmniMedVQA/OmniMedVQA.zip -d data/OmniMedVQA
```

Expected dataset structure after extraction

```text
data/OmniMedVQA/OmniMedVQA/
├── Images/
└── QA_information/
```

- `Images/` contains images for open‑access subsets.
- `QA_information/Open-access/*.json` and `QA_information/Restricted-access/*.json` contain per‑dataset QA entries. See `data/OmniMedVQA/README.md` for full details.

4) Quick smoke tests

List the checkpoints you downloaded:

```zsh
find checkpoints/Med-R1 -maxdepth 3 -type f -name 'config.json' | sed 's#/config.json##'
```

Try single‑image inference (replace the image path with one from your dataset):

```zsh
python inference.py \
  --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \
  --image_path "data/OmniMedVQA/OmniMedVQA/Images/<some_folder>/<some_image>.png" \
  --question "What abnormalities do you observe in this image?" \
  --device mps
```

Batch inference with a questions file

`inference.py` accepts a JSON list of items with `{image, question}` fields:

```json
[
  {"image": "/absolute/path/to/image1.png", "question": "..."},
  {"image": "/absolute/path/to/image2.png", "question": "..."}
]
```

Run batch mode (writes a results JSON):

```zsh
python inference.py \
  --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \
  --batch_file /absolute/path/to/questions.json \
  --output_file ./eval_results/results.json \
  --batch_size 1 \
  --device mps
```

5) Troubleshooting downloads

- Read timeout from Hugging Face
  - Increase timeouts: `export HF_HUB_TIMEOUT=300`
  - Resume: add `--resume-download` to CLI or `resume_download=True` in Python
  - Reduce concurrency: `max_workers=2`
- Disk space errors
  - Download only the subfolders you need using `--include` or `allow_patterns`
- Tokenizer/weights mismatch
  - Ensure the tokenizer and `generation_config.json` are taken from the same checkpoint directory as the weights

6) Where things should end up

- Checkpoints you downloaded (examples):
  - `checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT/`
  - `checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI/`
- Dataset after extraction:
  - `data/OmniMedVQA/OmniMedVQA/Images/`
  - `data/OmniMedVQA/OmniMedVQA/QA_information/`

You can now run `inference.py` and any evaluation scripts against these paths on macOS using MPS.

