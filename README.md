# Do Not Assume Users Are Privacy-Aware: Evaluating Proactive Privacy-Aware Assistance in Large Language Models

This repository contains the code and scripts for evaluating proactive privacy-aware assistance in large language models. The benchmark focuses on ordinary user interactions where users may unintentionally expose sensitive information, and evaluates whether models can proactively warn about privacy risks and avoid reproducing sensitive values.

## Repository Structure

```text
.
├── model/
│   ├── __init__.py
│   ├── base.py
│   ├── llama31.py
│   ├── mistral.py
│   └── qwen25.py
├── model_outputs/
│   ├── Standard/
│   │   ├── token.zip
│   │   ├── password.zip
│   │   ├── personal.zip
│   │   ├── financial.zip
│   │   └── open-source.zip
│   ├── Multi-Exposure.zip
│   └── Position and Length.zip
├── run/
│   ├── run_all.sh
│   ├── run_all_ass.sh
│   ├── run_all_fake.sh
│   ├── run_eval.sh
│   ├── run_eval_multi.sh
│   └── run_position_and_length.sh
├── eval.py
├── evaluation.zip
├── get_answer.py
├── get_answer_a.py
├── get_answer_fake.py
├── get_answer_position_and_length.py
├── open_source_model_run.py
├── privacy_eval_multi.py
├── README.md
└── README_RESULT.md
```

The benchmark data are not stored directly in this GitHub repository. They are released on Hugging Face:

```text
https://huggingface.co/datasets/LeoWang0318/PrivAwareBench
```

After downloading the dataset, place it in a local directory such as `DATA/` or modify the script arguments to point to your dataset location. The expected dataset organization contains the standard, multi-exposure, and position/length benchmark inputs.

## Released Results

Generated model outputs and evaluation results are provided as compressed archives. This is because the benchmark intentionally contains simulated privacy-sensitive artifacts, including synthetic API-token-like strings, in order to test whether models warn about and suppress sensitive information. Directly uploading the raw output directories may trigger GitHub secret-scanning push protection, even though the values are synthetic benchmark artifacts. Therefore, released outputs and evaluation results are distributed in compressed form.

The output archives are organized as follows:

```text
model_outputs/
├── Standard/
│   ├── token.zip
│   ├── password.zip
│   ├── personal.zip
│   ├── financial.zip
│   └── open-source.zip
├── Multi-Exposure.zip
└── Position and Length.zip
```

For the `Standard` setting, each sensitive category is compressed separately. Each category archive contains the standard, assisted, and induced no-warning outputs for that category. The outputs of the three local open-source models are compressed together in `open-source.zip`.

The full evaluation directory is compressed as:

```text
evaluation.zip
```

## Extracting All Released Results

From the repository root, run the following commands to extract all released outputs and evaluation results:

```bash
unzip -o evaluation.zip -d .

unzip -o "model_outputs/Standard/token.zip" -d model_outputs/Standard/
unzip -o "model_outputs/Standard/password.zip" -d model_outputs/Standard/
unzip -o "model_outputs/Standard/personal.zip" -d model_outputs/Standard/
unzip -o "model_outputs/Standard/financial.zip" -d model_outputs/Standard/
unzip -o "model_outputs/Standard/open-source.zip" -d model_outputs/Standard/

unzip -o "model_outputs/Multi-Exposure.zip" -d model_outputs/
unzip -o "model_outputs/Position and Length.zip" -d model_outputs/
```

After extraction, the result directories are restored to the same layout as before compression:

```text
model_outputs/
├── Standard/
├── Multi-Exposure/
└── Position and Length/

evaluation/
├── Standard/
├── Multi-Exposure/
└── Position and Length/
```

The extracted directory layout is described in detail in `README_RESULT.md`.

## Downloading Benchmark Data

The benchmark inputs are hosted at:

```text
https://huggingface.co/datasets/LeoWang0318/PrivAwareBench
```

One convenient option is to download or clone the dataset into a local `DATA/` directory before running the scripts:

```bash
git lfs install
git clone https://huggingface.co/datasets/LeoWang0318/PrivAwareBench DATA
```

If you store the dataset somewhere else, replace the `DATA/...` paths in the commands below with your actual local dataset path.

## API Configuration

All closed-source model scripts use an OpenAI-compatible API interface.

```bash
export BASE_URL="<OPENAI_COMPATIBLE_BASE_URL>"
export OPENAI_API_KEY="<YOUR_OPENAI_COMPATIBLE_API_KEY>"
```

Model targets follow this format:

```bash
--target "<MODEL_NAME>|<MODEL_GROUP>|${BASE_URL}|OPENAI_API_KEY"
```

where `<MODEL_GROUP>` is usually one of:

```text
thinking
instant
```

Example model configuration:

```bash
TARGET_ARGS+=(--target "gpt-5.1-thinking|thinking|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-4o|instant|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-5.4|instant|${BASE_URL}|OPENAI_API_KEY")
TARGET_ARGS+=(--target "gpt-5.1|instant|${BASE_URL}|OPENAI_API_KEY")
```

## Standard Setting

The standard setting evaluates proactive privacy-aware behavior without any additional privacy-assistance prompt.

```bash
TYPE=token bash run/run_all.sh
TYPE=password bash run/run_all.sh
TYPE=personal_id bash run/run_all.sh
TYPE=financial bash run/run_all.sh
```

## Assisted Setting

The assisted setting evaluates the same underlying inputs with an additional privacy-aware assistance prompt.

```bash
TYPE=token_ass bash run/run_all_ass.sh
TYPE=password_ass bash run/run_all_ass.sh
TYPE=personal_ass bash run/run_all_ass.sh
TYPE=financial_ass bash run/run_all_ass.sh
```

## Induced No-Warning Setting

The induced no-warning setting evaluates model behavior under an instruction that discourages privacy warnings.

```bash
TYPE=token_fake bash run/run_all_fake.sh
TYPE=password_fake bash run/run_all_fake.sh
TYPE=personal_fake bash run/run_all_fake.sh
TYPE=financial_fake bash run/run_all_fake.sh
```

## Multi-Exposure Setting

The multi-exposure setting evaluates whether models can identify and warn about multiple sensitive categories in the same input.

```bash
TYPE=Multi-Exposure DATA_JSONL="DATA/Multi-Exposure/multi-exposure.jsonl" bash run/run_all.sh
```

Evaluate multi-exposure outputs:

```bash
bash run/run_eval_multi.sh
```

Generated outputs are saved under:

```text
model_outputs/Multi-Exposure/
```

Evaluation results are saved under:

```text
evaluation/Multi-Exposure/
```

If you use the released results, first extract the corresponding archives as described above.

## Position and Length Setting

The position and length setting evaluates whether privacy-aware behavior is affected by sensitive-value position and context length.

```bash
TYPE=256 DATA_JSONL="DATA/Position and Length/256" bash run/run_position_and_length.sh
TYPE=512 DATA_JSONL="DATA/Position and Length/512" bash run/run_position_and_length.sh
TYPE=1k DATA_JSONL="DATA/Position and Length/1k" bash run/run_position_and_length.sh
TYPE=1.5k DATA_JSONL="DATA/Position and Length/1.5k" bash run/run_position_and_length.sh
TYPE=2k DATA_JSONL="DATA/Position and Length/2k" bash run/run_position_and_length.sh
```

Outputs are saved under:

```text
model_outputs/Position and Length/<length>/<position>/
```

Evaluation results are saved under:

```text
evaluation/Position and Length/<length>/<position>/
```

where `<position>` is one of:

```text
front
middle
end
```

## Running Open-Source Models Locally

For local open-source models, use:

```bash
python open_source_model_run.py
```

The script loads local models through the wrappers under `model/` and generates outputs in the same JSONL format as the closed-source API-based scripts.

Example:

```bash
python open_source_model_run.py \
  --data_dir DATA/Standard \
  --out_root model_outputs/Standard \
  --models llama31_8b \
  --model_path llama31_8b=<PATH_TO_LLAMA31_8B_MODEL> \
  --setting proactive \
  --with_assistant 1 \
  --limit 300 \
  --resume 1
```

To run multiple open-source models:

```bash
python open_source_model_run.py \
  --data_dir DATA/Standard \
  --out_root model_outputs/Standard \
  --models llama31_8b,mistral_7b_v03,qwen25_7b \
  --model_path llama31_8b=<PATH_TO_LLAMA31_8B_MODEL> \
  --model_path mistral_7b_v03=<PATH_TO_MISTRAL_7B_MODEL> \
  --model_path qwen25_7b=<PATH_TO_QWEN25_7B_MODEL> \
  --setting proactive \
  --with_assistant 1 \
  --limit 300 \
  --resume 1
```

The supported model aliases should correspond to the model wrappers under:

```text
model/
├── base.py
├── llama31.py
├── mistral.py
└── qwen25.py
```

The generated outputs will be saved under:

```text
model_outputs/Standard/<model_alias>/
model_outputs/Standard/<model_alias>_ass/
```

## Running Output Evaluation

Evaluate generated outputs for the standard, assisted, or induced no-warning settings:

```bash
TYPE=token DIF=standard bash run/run_eval.sh
TYPE=token DIF=ass bash run/run_eval.sh
TYPE=token DIF=fake bash run/run_eval.sh
```

Supported `TYPE` values:

```text
token
password
personal_id
financial
```

Supported `DIF` values:

```text
standard
ass
fake
```

## Output Summary

Each evaluation directory may contain:

```text
<model_name>.privacy_eval.jsonl
summary.csv
summary.md
```

The main metrics are:

```text
warn_privacy_risk
redacted
```

For multi-exposure evaluation, additional metrics include:

```text
warn_generic
warn_token
warn_password
warn_personal_id
warn_financial
warn_partial
warn_all
has_mitigation
redacted
```

## Notes

This benchmark is designed for non-adversarial sensitive-information exposure. The goal is not to test whether models comply with malicious users, but whether they can assist benign users who may not realize that their inputs contain sensitive information.

The benchmark evaluates two independent behaviors:

1. whether the model explicitly warns about privacy or security risk;
2. whether the model avoids reproducing the sensitive value.

A model may warn while still exposing the value, or redact the value without giving a warning. These two dimensions should therefore be analyzed separately.
