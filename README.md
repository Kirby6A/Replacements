# Replacements — pre-edited InternAgent files

Drop-in files for the cloud box. Copy each to its destination inside the cloned
InternAgent repo, overwriting the original.

## Config files

| File here | Goes to (inside InternAgent/) | Edits applied |
|---|---|---|
| `openrouter_config.yaml` | `config/openrouter_config.yaml` | 1) committee model → `anthropic/claude-sonnet-5` (near-Opus quality; intro pricing through 2026-08-31) · 2) builder `experiment.model` → `claude-sonnet-5` (billed to Claude Max OAuth) · 3) embedding model → `BAAI/bge-base-en-v1.5` (turns semantic memory ON; matches InternAgent's own docs) · 4) `loop_rounds: 10 → 2` (calibration run) · 5) `gpu_per_experiment: 1.0 → 0.5` (single-GPU pod, 2 experiments share it) · 6) all three `temperature: 0.7 → 1.0` (global + evolution + exp_analyze — Sonnet 5 400s on non-default sampling params) |
| `config_simple.yaml` | `internagent/mas/agents/dr_agents/config_simple.yaml` | 1) DR default `o4-mini` → `gpt-5-mini` (workers/skimmers) · 2) planner + synthesizer + coordinator → `gpt-5.5` (structure-setting and briefing-writing roles; coordinator dormant in simple mode) · 3) all `volc_search` → `tavily_search` (makes TAVILY_API_KEY actually get used) |

```bash
cp ../Repo_Maker/Replacements/openrouter_config.yaml config/openrouter_config.yaml
cp ../Repo_Maker/Replacements/config_simple.yaml internagent/mas/agents/dr_agents/config_simple.yaml
```

## TrustFilter/ — THE REAL TASK (ready to deploy)

Built from Humain's "Trust the Model: Compact VLMs as In-Context Judges for
Image-Text Data Quality" (IEEE MLSP 2025). Improve an image-text data-filtration
method: detect synthetically corrupted COCO caption pairs. Baseline = the
CLIP-similarity filter the paper argues against; primary metric = `detection_f1`
(ungameable — scored against known corruption labels, keep-ratio fixed).

Deploy:

```bash
cp -r ../Repo_Maker/Replacements/TrustFilter tasks/TrustFilter
pip install torch transformers pillow numpy
cd tasks/TrustFilter && python code/experiment.py && mv final_info.json run_0/ \
  && rm run_0/PLACEHOLDER.md && cd ../..

python launch_discovery.py --task TrustFilter --config config/openrouter_config.yaml \
  --mode report --exp_backend claudecode        # ideas-only first
python launch_discovery.py --task TrustFilter --config config/openrouter_config.yaml \
  --exp_backend claudecode                      # full 2-round run
```

First baseline run downloads COCO val2017 (~1GB, cached in `tasks/TrustFilter/data/`).
Only the SCORING section of `experiment.py` is fair game for the agents — the
corruption generator and evaluation are frozen (they're the ground truth).

## PaperTask/ — blank task stub (template for future papers)

Template for "improve on a company's paper". Three pieces:

| File | What to do with it |
|---|---|
| `PaperTask/prompt.json` | Fill every `<REPLACE: ...>` field. `task_description` matters most — it's the Brainstormer's seed directions. Metric keys must match what `experiment.py` writes. |
| `PaperTask/code/experiment.py` | Implement the paper's method in the three marked sections (data / method / evaluate). Keep the scaffold: tunables as argparse `default=`s, results written to `final_info.json` in the script's parent dir. |
| `PaperTask/run_0/` | Don't hand-write anything — see `run_0/PLACEHOLDER.md`: run the baseline once, move `final_info.json` in, delete the placeholder. |

**Where it goes:** the whole folder → `InternAgent/tasks/<TaskName>/` (rename freely;
folder name = task name):

```bash
cp -r ../Repo_Maker/Replacements/PaperTask tasks/PaperTask
cd tasks/PaperTask && python code/experiment.py && mv final_info.json run_0/ && cd ../..
```

Notes:
- `code_summary.json` is auto-generated on first run — don't create one.
- A folder with `prompt.json` (and no `task_info.json`) auto-detects as an `auto`
  (improve-the-baseline) task. That's the right mode.
- Keep the baseline FAST (seconds to ~a minute per run) — the builder re-runs it
  constantly, and slow baselines multiply wall-clock and Max-window burn.

## Env vars to export on the box

```bash
export OPENROUTER_API_KEY=...       # committee
export CLAUDE_CODE_OAUTH_TOKEN=...  # builder (from `claude setup-token` on laptop)
export OPENAI_API_KEY=...           # deep research (gpt-5-mini workers + gpt-5.5 heads)
export SERPER_API_KEY=...           # librarian web search
export TAVILY_API_KEY=...           # deep research web search
unset ANTHROPIC_API_KEY             # CRITICAL: would override the Max token and bill API credits
```

## Run order

```bash
# 1. Smoke test on the shipped toy task (proves keys + config, ~free):
python launch_discovery.py --task AutoDebug --config config/openrouter_config.yaml \
  --mode report --exp_backend claudecode

# 2. Ideas-only on YOUR task (sanity-check idea quality before spending builder time):
python launch_discovery.py --task PaperTask --config config/openrouter_config.yaml \
  --mode report --exp_backend claudecode

# 3. The real 2-round end-to-end:
python launch_discovery.py --task PaperTask --config config/openrouter_config.yaml \
  --exp_backend claudecode
```

After calibrating: bump `loop_rounds` back up, revisit `gpu_per_experiment` if the
pod has >1 GPU, and check the OpenRouter/OpenAI dashboards for real per-round costs.
