# Released Model Outputs and Evaluation Results

This file describes the released model outputs and evaluation results for PrivAwareBench.

The benchmark input data are hosted separately on Hugging Face:

```text
https://huggingface.co/datasets/LeoWang0318/PrivAwareBench
```

This GitHub repository contains the code, scripts, compressed model outputs, and compressed evaluation results.

## Why Results Are Compressed

The released results are compressed because PrivAwareBench intentionally includes simulated sensitive artifacts, including synthetic token-like strings, to evaluate whether models can proactively warn about privacy risks and avoid reproducing sensitive values.

Although these values are synthetic benchmark artifacts, some of them resemble real credentials. GitHub secret-scanning push protection may therefore reject direct uploads of the raw output files. To avoid this upload failure while preserving the benchmark results, model outputs and evaluation results are released as compressed archives.

## Released Archives

The released output files are organized as:

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

For the `Standard` setting, each sensitive category is compressed separately:

```text
token.zip
password.zip
personal.zip
financial.zip
```

Each category archive contains the corresponding standard, assisted, and induced no-warning outputs. For example, after extracting `token.zip`, the restored standard-output folders include:

```text
model_outputs/Standard/token/
model_outputs/Standard/token_ass/
model_outputs/Standard/token_fake/
```

The outputs of the three local open-source models are compressed together in:

```text
model_outputs/Standard/open-source.zip
```

After extracting `open-source.zip`, the restored folders include:

```text
model_outputs/Standard/llama31_8b/
model_outputs/Standard/llama31_8b_ass/
model_outputs/Standard/mistral_7b_v03/
model_outputs/Standard/mistral_7b_v03_ass/
model_outputs/Standard/qwen25_7b/
model_outputs/Standard/qwen25_7b_ass/
```

The released evaluation results are compressed as:

```text
evaluation.zip
```

## Extracting All Results

From the repository root, run:

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

## Model Outputs

The `model_outputs/` directory stores raw model responses generated for different benchmark settings. Each output file is in JSONL format and contains one model response per benchmark instance.

### Output Format

Each line is a JSON object:

```json
{
  "pid": 1,
  "answer": "model response"
}
```

### Restored Directory Structure

After extracting all output archives, the restored `model_outputs/` directory has the following structure:

```text
model_outputs/
├── Standard/
│   ├── token/
│   ├── token_ass/
│   ├── token_fake/
│   ├── password/
│   ├── password_ass/
│   ├── password_fake/
│   ├── personal_id/
│   ├── personal_ass/
│   ├── personal_fake/
│   ├── financial/
│   ├── financial_ass/
│   ├── financial_fake/
│   ├── llama31_8b/
│   ├── llama31_8b_ass/
│   ├── mistral_7b_v03/
│   ├── mistral_7b_v03_ass/
│   ├── qwen25_7b/
│   └── qwen25_7b_ass/
│
├── Multi-Exposure/
│   ├── <model_name>.jsonl
│   └── ...
│
└── Position and Length/
    ├── 256/
    │   ├── front/
    │   ├── middle/
    │   └── end/
    ├── 512/
    │   ├── front/
    │   ├── middle/
    │   └── end/
    ├── 1k/
    │   ├── front/
    │   ├── middle/
    │   └── end/
    ├── 1.5k/
    │   ├── front/
    │   ├── middle/
    │   └── end/
    └── 2k/
        ├── front/
        ├── middle/
        └── end/
```

### Standard Setting Outputs

The standard setting evaluates proactive privacy-aware behavior without any additional privacy-assistance prompt.

Example paths:

```text
model_outputs/Standard/token/
model_outputs/Standard/password/
model_outputs/Standard/personal_id/
model_outputs/Standard/financial/
```

Each folder contains one JSONL file per evaluated model:

```text
model_outputs/Standard/token/<model_name>.jsonl
model_outputs/Standard/password/<model_name>.jsonl
```

### Assisted Setting Outputs

The assisted setting evaluates the same inputs with an additional privacy-aware assistance prompt.

Example paths:

```text
model_outputs/Standard/token_ass/
model_outputs/Standard/password_ass/
model_outputs/Standard/personal_ass/
model_outputs/Standard/financial_ass/
```

Each folder contains one JSONL file per evaluated model.

### Induced No-Warning Outputs

The induced no-warning setting evaluates model behavior under an instruction that discourages privacy warnings.

Example paths:

```text
model_outputs/Standard/token_fake/
model_outputs/Standard/password_fake/
model_outputs/Standard/personal_fake/
model_outputs/Standard/financial_fake/
```

Each folder contains one JSONL file per evaluated model.

### Open-Source Local Model Outputs

Open-source local model outputs are organized by model name rather than by sensitive category.

Example paths:

```text
model_outputs/Standard/llama31_8b/
model_outputs/Standard/llama31_8b_ass/
model_outputs/Standard/mistral_7b_v03/
model_outputs/Standard/mistral_7b_v03_ass/
model_outputs/Standard/qwen25_7b/
model_outputs/Standard/qwen25_7b_ass/
```

Each open-source model folder contains one JSONL file per sensitive category:

```text
model_outputs/Standard/llama31_8b/token.jsonl
model_outputs/Standard/llama31_8b/password.jsonl
model_outputs/Standard/llama31_8b/personal_id.jsonl
model_outputs/Standard/llama31_8b/financial.jsonl
```

The corresponding assisted-output folder uses the `_ass` suffix:

```text
model_outputs/Standard/llama31_8b_ass/token.jsonl
model_outputs/Standard/llama31_8b_ass/password.jsonl
model_outputs/Standard/llama31_8b_ass/personal_id.jsonl
model_outputs/Standard/llama31_8b_ass/financial.jsonl
```

Open-source model outputs are not currently provided for:

```text
token_fake
password_fake
personal_fake
financial_fake
Multi-Exposure
Position and Length
```

### Multi-Exposure Outputs

The multi-exposure setting evaluates model behavior when multiple sensitive categories appear in the same input.

Example path:

```text
model_outputs/Multi-Exposure/
```

Each file corresponds to one model:

```text
model_outputs/Multi-Exposure/<model_name>.jsonl
```

### Position and Length Outputs

The position and length setting evaluates robustness under different context lengths and sensitive-value positions.

Example paths:

```text
model_outputs/Position and Length/256/front/
model_outputs/Position and Length/1k/middle/
model_outputs/Position and Length/2k/end/
```

Each folder contains one JSONL file per evaluated model.

## Evaluation Results

The `evaluation/` directory stores automatic privacy-awareness judgments and summary files for generated model outputs.

### Restored Directory Structure

After extracting `evaluation.zip`, the restored `evaluation/` directory has the following structure:

```text
evaluation/
├── Standard/
│   ├── token/
│   ├── token_ass/
│   ├── token_fake/
│   ├── password/
│   ├── password_ass/
│   ├── password_fake/
│   ├── personal_id/
│   ├── personal_ass/
│   ├── personal_fake/
│   ├── financial/
│   ├── financial_ass/
│   └── financial_fake/
│
├── Multi-Exposure/
│   ├── <model_name>.privacy_eval.jsonl
│   ├── summary.csv
│   └── summary.md
│
└── Position and Length/
    ├── 256/
    ├── 512/
    ├── 1k/
    ├── 1.5k/
    └── 2k/
```

### Standard Evaluation Format

For standard, assisted, and induced no-warning settings, each evaluated output file is stored as:

```text
<model_name>.privacy_eval.jsonl
```

Each line contains the model output and its privacy-awareness labels. The core fields are:

```text
warn_privacy_risk
redacted
warn_evidence
redact_evidence
```

The two primary metrics are:

```text
warn_privacy_risk
redacted
```

`warn_privacy_risk` measures whether the model explicitly warns about privacy or security risk.

`redacted` measures whether the model avoids reproducing the sensitive value.

### Multi-Exposure Evaluation Format

For multi-exposure evaluation, each model has a corresponding file:

```text
evaluation/Multi-Exposure/<model_name>.privacy_eval.jsonl
```

The main fields include:

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

The summary files are:

```text
evaluation/Multi-Exposure/summary.csv
evaluation/Multi-Exposure/summary.md
```

### Position and Length Evaluation Format

For position and length experiments, results are organized by context length and sensitive-value position:

```text
evaluation/Position and Length/<length>/<position>/
```

where `<length>` is one of:

```text
256
512
1k
1.5k
2k
```

and `<position>` is one of:

```text
front
middle
end
```

Each directory contains evaluation files and summaries for the corresponding condition.

## Notes

The released output archives contain raw model generations only.

The released `evaluation.zip` contains automatic privacy-awareness labels and summary files.

The benchmark input data should be downloaded separately from Hugging Face.

A model may warn about privacy risk while still reproducing the sensitive value, or it may redact the value without explicitly warning the user. Therefore, warning and redaction are reported as separate dimensions.
