# Med‑R1 Evaluation Guide

This guide explains how the repository performs evaluation, how to run inference and evaluation on a Mac (using MPS), where the relevant code and checkpoints live, the GRPO implementation and reward math (as implemented in `src/r1-v/src/open_r1/grpo.py`), and suggested improvements and additional evaluation ideas.

Checklist (what this document contains)
- Quick map of code & entry points used for training, inference and evaluation
- How `torchrun --nproc_per_node` works and which script is the entry point
- How to run inference locally on macOS (MPS) with `inference.py` (examples)
- Exact evaluation metrics used by the code / authors and their mathematical definitions
- GRPO implementation: where the math / reward functions are implemented and a concise math summary of the algorithm used (what is computed and optimized)
- Practical steps to reproduce inference/evaluation when you have (a) the authors' checkpoints or (b) only the repository code and dataset
- Suggestions for further evaluation metrics and improvements

---

1) Code map and entry points

- Training (GRPO): `src/r1-v/src/open_r1/grpo.py`
  - This file builds dataset -> conversation prompts and instantiates a GRPO trainer class (via `open_r1.trainer.Qwen2VLGRPOTrainer` or `Qwen2VLGRPOVLLMTrainer`).
  - The script uses a CLI parser (`TrlParser`) to combine `GRPOScriptArguments`, `GRPOConfig`, and `ModelConfig` (from `trl`).
  - Example training launcher: `src/r1-v/run_grpo.sh` (invokes `torchrun` and passes `src/open_r1/grpo.py` as the script to run).

- Inference: `inference.py` (top-level)
  - Single-image and batch inference script supporting MPS (`--device mps`) and CPU.
  - Loads the model & processor from a checkpoint directory and runs generate().
  - CLI flags (key ones): `--model_path`, `--device`, `--image_path`, `--question`, `--batch_file`, `--output_file`, `--batch_size`, `--max_new_tokens`, `--do_sample`, `--temperature`.

- Checkpoints & data locations in the repo
  - Checkpoints are expected under `checkpoints/Med-R1/` (the repo also contains code that downloads HuggingFace snapshots when running). The large `Qwen_2.5_3B_nothink/...` subfolders in `checkpoints/Med-R1` are the authors' model checkpoints (if they have been downloaded).
  - Dataset samples or zipped datasets live under `data/OmniMedVQA/` or `data/tiny_vqa_ds/`. The repo contains script helpers to prepare datasets (see `prepare_dataset.py`, `src/distill_r1/create_hf_dataset.py`).
  - `Images/` in repo: contains figures and result images provided by the authors (plots, heatmaps), not per-example medical images.

---

2) How `torchrun --nproc_per_node` works and the entry point

- `torchrun --nproc_per_node=N <script> [args]` starts N processes on the current node. Each process is typically bound to a single GPU device (LOCAL_RANK environment variable). When you run distributed training with GPUs, set `--nproc_per_node` equal to the number of GPU devices you want to use.

- Important notes for this repo:
  - The `torchrun` commands in `src/r1-v/run_grpo.sh` and other launch scripts pass the Python script path after the `torchrun` options — e.g.:

    torchrun --nproc_per_node="8" ... src/open_r1/grpo.py --output_dir <...> --model_name_or_path <...>

    Here, `src/open_r1/grpo.py` is the Python entry point run under `torchrun`.

  - On macOS (MPS) you normally do NOT run distributed training with `torchrun` (no GPU processes to spawn). Use the single-process inference flows (no torchrun) or run `torchrun --nproc_per_node=1` if you want to keep the same invocation style.

---

3) Running inference on macOS (MPS) — practical steps

Pre-reqs (short):
- Python 3.11+ recommended (this repo used 3.13 in logs, but 3.11/3.12 are more common/stable).
- Install dependencies (create a venv first):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r src/requirements.txt
```

Notes about PyTorch/MPS:
- Ensure you have a mac that supports MPS (Apple Silicon / macOS with Metal + PyTorch built with MPS). The `torch` in `src/requirements.txt` is pinned to 2.5.1 which contains MPS support; if you encounter compatibility issues, install a PyTorch wheel built for your macOS and MPS.
- To confirm MPS availability in Python:

```python
import torch
print(torch.backends.mps.is_available(), torch.__version__)
```

Inference examples using `inference.py` (single image):

```bash
# Single image on MPS
python inference.py \
  --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
  --image_path /path/to/image.jpg \
  --question "What abnormalities do you observe in this image?" \
  --device mps

# Batch inference (questions.json is a list or dict with 'questions' key)
python inference.py \
  --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \
  --batch_file /path/to/questions.json \
  --output_file ./eval_results/results.json \
  --batch_size 1 \
  --device mps
```

File formats expected by `--batch_file`:
- A JSON list of objects like {"image": "/abs/path/to/image.png", "question": "...", "question_type": "open"}
- Or a dict with key `questions` whose value is a list of items above.

If you don't have the authors' checkpoint locally, `inference.py` will show available model directories under `checkpoints/Med-R1/Qwen_2.5_3B_nothink` before exiting.

---

4) Authors' evaluation metrics — explicit math

This repository (and `grpo.py`) uses reward functions for training that are binary (0/1) and also the usual VQA / classification metrics at evaluation time. Below is the mathematical description of the metrics commonly used here.

4.a Exact Match (EM)
- Definition: EM = 1 if predicted answer exactly equals the ground-truth answer string after normalization; otherwise 0.
- For a dataset of N examples, Exact Match accuracy is:

  Accuracy_EM = (1/N) * sum_{i=1..N} indicator(pred_i == gt_i)

4.b Counting accuracy (numeric answers)
- If ground truth and prediction are numeric, extract integer values and compare equality.

  CountingAccuracy = (#examples where extracted_number(pred_i) == extracted_number(gt_i)) / N

4.c F1 score (token-level, common for VQA)
- Let pred_tokens and gt_tokens be token sets. Precision = |pred ∩ gt| / |pred|, Recall = |pred ∩ gt| / |gt|, F1 = 2 * (P * R) / (P + R).

4.d BLEU / BLEU-4 (for free-text generation)
- Standard BLEU score with n-gram precisions and brevity penalty. See: BLEU = BP * exp(sum_{n=1..4} w_n log p_n), with w_n = 1/4.

4.e The repo's training reward math (binary rewards):
- The `accuracy_reward` in `grpo.py` returns r_i ∈ {0.0, 1.0} per example, using two checks in order:
  1) Symbolic verification: it attempts to parse the model's solution and the ground truth using `math_verify.parse` and `math_verify.verify`. If verification yields a positive correctness signal, reward r_i = 1.0.
  2) Otherwise, it falls back to string extraction between `<answer>...</answer>` tags and compares the extracted token(s); if equal, r_i = 1.0.
  3) Otherwise r_i = 0.0.

- `format_reward` is also available and enforces that the model outputs both `<think>...</think>` and `<answer>...</answer>` tags. It is also binary (1 if the pattern matches, otherwise 0).

Practically these binary rewards are used by the GRPO trainer to compute sequence-level rewards R(s,a) used by the policy gradient updates (details below).

---

5) GRPO implementation: where the math is and what is being optimized

Where the reward math is implemented:
- `src/r1-v/src/open_r1/grpo.py` contains the reward functions `accuracy_reward` and `format_reward` (see the code). These provide numeric reward signals to the trainer.
- The actual policy optimization algorithm is implemented by the TRL `GRPOTrainer` (imported via `from trl import GRPOConfig, GRPOTrainer, ...`). The repository uses a custom wrapper `open_r1.trainer.Qwen2VLGRPOTrainer` which configures model-specific behavior and hooks into the TRL trainer.

Concise math summary of the (typical) approach used by GRPO-style trainers (policy-gradient on sequence generation):

- Objective: maximize expected reward J(\theta) = E_{x ~ D, y ~ p_\theta(.|x)}[R(y, x)]
  - x is the input (prompt + possibly image), y is the model-generated sequence/answer, R(y,x) is the scalar reward computed by the reward function(s).

- Policy gradient (REINFORCE-style estimator):
  - ∇_\theta J(\theta) ≈ (1/M) * sum_{i=1..M} R(y_i, x_i) * ∇_\theta log p_\theta(y_i | x_i)
  - y_i are samples from p_\theta (or PG with importance sampling / weighting) and a baseline b may be subtracted to reduce variance: (R - b) * ∇ log p.

- Sequence log-prob decomposition:
  - log p_\theta(y|x) = sum_{t=1..T} log p_\theta(y_t | x, y_{<t})

- Implementation details commonly used by TRL trainers (and relevant to GRPO variants):
  - Reward is typically computed at sequence level and broadcast to per-token gradient via multiply-by-log-prob.
  - Baselines / reward normalization: rewards are often normalized (mean/variance) across the minibatch to stabilize training.
  - KL penalties or value network baselines: many RL-from-LM frameworks add a KL penalty against the reference LM to prevent degeneration and control distributional shift. If present, that would be configured through `GRPOConfig` or in the trainer wrapper.

Concrete tie-back to this repo:
- The repo's `grpo.py` wires the binary reward functions into a `Qwen2VLGRPOTrainer` instance. The trainer class is responsible for applying the TRL optimization loop with the configured reward, baselines, and optimization hyperparameters (learning rate, batch size, gradient accumulation, etc.).
- For exact implementation details of the gradient estimator, advantage calculation, and clipping/penalties, inspect the TRL `GRPOTrainer` implementation in the installed `trl` package (the repo imports it). The repo itself does not re-implement the low-level RL math inside `grpo.py`—it defines the reward and dataset formatting.

---

6) Exact command examples for evaluation (when you have authors' checkpoints and dataset)

A. Download / prepare authors' checkpoint(s)
- If authors published their models on HuggingFace, you can use `huggingface-cli` or Python `huggingface_hub.snapshot_download` to download the repo snapshot into `checkpoints/Med-R1/`.
- The repo's earlier logs show downloads into `checkpoints/Med-R1/Qwen_2.5_3B_nothink/...` — if those folders already exist, inference will pick them up.

B. Prepare dataset
- If the authors shared a HF dataset or a zip, extract it under `data/` and point scripts to it. Example expected dataset layout for `inference.py` batch mode: a JSON list where each item includes absolute `image` paths.

C. Run batch inference on Mac (MPS):

```bash
# example: produce results.json
python inference.py \
  --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_CT \
  --batch_file /absolute/path/to/questions.json \
  --output_file ./eval_results/results.json \
  --batch_size 1 \
  --device mps
```

D. Simple evaluation script (recommended):
- Use the output of `inference.py` (which includes `extracted_answer` per item) and compute exact match / F1 / counting accuracy.
- You can reuse the evaluation logic from the earlier `evaluate_mac.py` snippet (the repository does not have a dedicated `evaluate.py` at the top-level). If you want, I can add a robust `evaluate.py` that reproduces the authors' metrics and runs on macOS.

---

7) Where the images and model weights come from (summary)

- Images in `Images/` folder: these are plotted figures and result visualizations provided by the authors; they are not the dataset image instances used for training/eval.
- Datasets: `data/OmniMedVQA/` (zips and HF dataset files) — look inside to find raw images and QA files if present. If the authors distributed their dataset publicly, use their instructions (often included in `data/OmniMedVQA/README.md`) to download the dataset.
- Pretrained model weights: The repo shows downloads of large `safetensors` files into `checkpoints/Med-R1/Qwen_2.5_3B_nothink/...` (these were obtained via `huggingface_hub` in the user's earlier logs). If you want to reuse the authors' official checkpoints, download the same model repo id they reference (check the paper/readme for the HF model id) or use the `checkpoints` folder if the repo provides pre-downloaded weights.

---

8) Additional evaluation approaches (suggested improvements)

Short list of recommended additions to better quantify model quality:

- Soft / graded rewards instead of binary: if you can compute a graded score (semantic similarity, partial credit), use a continuous reward in [0,1] to provide richer signal. Example: token-level F1 or normalized edit distance.

- BLEU / ROUGE / BERTScore / BLEURT: for free-text answers include textual similarity metrics (BERTScore/BLEURT are more robust to paraphrase than BLEU).

- Calibration & confidence: evaluate the model's probability/confidence calibration for its predictions (ECE, reliability diagrams). This is important in clinical settings.

- Hallucination detection: compare model outputs to evidence in the image+metadata and compute hallucination rates (semantic checks using medical ontologies or fact-checking modules).

- Human-in-the-loop evaluation: sample failure cases and have medical experts rate correctness and clinical safety.

- Robustness tests: evaluate on perturbed images (blur, noise, different contrast) to measure degradation under realistic imaging variability.

- Cross-dataset generalization: measure performance of a single checkpoint across multiple medical VQA datasets (radiology, pathology, dermoscopy) to evaluate transfer.

- Uncertainty-aware evaluation: compute prediction intervals if you ensemble multiple generations or use MC-dropout; evaluate coverage vs. nominal.

- Chain-of-thought verification (for GRPO): use the `<think>` output to run an automatic verifier (e.g., symbolic verifier for counting/math tasks) and compute the fraction of times the chain-of-thought contains a valid reasoning trace.

---

9) Practical next steps I can take for you (I can implement any of these)

- Add a small `evaluate.py` that consumes `inference.py`'s JSON output and computes EM, F1, counting accuracy, and produces a CSV/JSON summary and confusion breakdown.
- Add a shell example and a helper that downloads the authors' HuggingFace checkpoint(s) into `checkpoints/Med-R1/` (requires the HF repo id or a working Internet connection).
- Add a `requirements-mac.txt` that pins minimal packages for running inference on mac MPS (smaller than the full `src/requirements.txt`).
- Implement a continuous reward wrapper that uses fuzzy matching / normalized edit distance instead of binary exact-match.

If you'd like, tell me which of the next steps I should do now and I'll implement them (I can create `evaluate.py`, add a `requirements-mac.txt`, and add example commands to this guide).

---

Appendix: Useful quick commands

1) Check MPS and torch version:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available(), 'torch', torch.__version__)"
```

2) Run single-image inference (MPS):

```bash
python inference.py \
  --model_path checkpoints/Med-R1/Qwen_2.5_3B_nothink/VQA_AI \
  --image_path /absolute/path/to/image.jpg \
  --question "Describe abnormality" \
  --device mps
```

3) Run training locally (not recommended on Mac for large models) but to follow the same invocation style, use `--nproc_per_node=1` and the grpo script:

```bash
# Example: single-process training run (small debugging mode)
torchrun --nproc_per_node=1 src/open_r1/grpo.py --model_name_or_path <path> --dataset_name <path-or-hf-id> --output_dir ./debug_out
```




