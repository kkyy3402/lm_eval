# vLLM LLM Evaluation (lm-eval + Guardrail)

`run_eval.py` runs two evaluations against a model served on port `8000` (or any URL), and reads all runtime options from YAML:

1. `lm-eval-harness` task metrics (default: `mmlu`)
2. Guardrail metrics from public benchmark sets (default: HarmBench + ToxicChat)

Then it merges everything into one JSON under `results/`.

## Install

```bash
pip install -r requirements.txt
```

## Example

```bash
python run_eval.py --config config.yaml
```

## Start vLLM (Qwen3-4B)

```bash
./start_vllm_qwen.sh
```

This script:
- creates `.venv`
- installs `uv` first if missing
- installs `vllm` and `requirements.txt` via `uv pip`
- serves `Qwen/Qwen3-4B-Instruct-2507` on port `8000` in background
- waits until `/v1/models` is ready

After it is ready:

```bash
source .venv/bin/activate
python run_eval.py --config config.yaml
```

## Output

Each run creates:

- `results/<run_name>/combined_results.json`
- `results/<run_name>/guardrail_traces.jsonl`
- `results/<run_name>/guardrail_traces.csv`
- `results/<run_name>/lm_eval_raw/*` (raw lm-eval output)

## Notes

- The script assumes vLLM OpenAI-compatible API (`/v1/chat/completions`) is available.
- `lm-eval` backend defaults to `local-completions`.
- Configure `model`, `server_base_url`, `tasks`, and benchmark options in `config.yaml`.
- YAML accepts both `server_base_url` and `server-base-url` style keys.
- Guardrail prompts are loaded from datasets in `guardrail_benchmarks`; no hardcoded eval prompt list is used.
- Supported guardrail presets:
  - `harmbench` (default: `swiss-ai/harmbench`, `DirectRequest`)
  - `toxicchat` (default: `lmsys/toxic-chat`, `toxicchat0124`)
  - `hf_generic` (custom Hugging Face dataset mapping)
- If your lm-eval backend args differ by version, set `lm_eval_model` / `lm_eval_model_args` in YAML.
