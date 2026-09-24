# LLM Analysis: BUG_FIX vs. NON_BUG_FIX classification of Terraform commits

This module uses a Large Language Model to classify Terraform commits as `BUG_FIX` or `NON_BUG_FIX`.
It can use the commit message, the Terraform diff, or both.

## Research objective

The miner (`src/miner/`) selects *candidate* bug-fix commits with a keyword heuristic: the message
contains `fix`, `bug`, `issue`, `resolve`, `error`, `failure` or `incorrect`. That heuristic has false
positives such as "fix typo in README", "resolve merge conflicts" and "add error_message variable".
This module answers two questions:

1. **Which candidates are actually bug fixes?** Answering this lets the study estimate the
   heuristic's precision. It also produces a cleaner defect dataset for deriving Terraform mutation
   operators.
2. **Does the code change add information beyond the commit message?** The same commits are
   classified from the message alone, from the diff alone, and from both, and the results are compared.

## Relationship with the rest of the repository

```
src/miner/  (unchanged)  ->  ../data/dataset/<owner>_<repo>/data.jsonl   one record per (commit, .tf file)
                                  |
src/llm_analysis/01_prepare_commits.py   -> commits_prepared.jsonl      one record per commit
src/llm_analysis/02_classify_commits.py  -> classifications_<mode>.jsonl
src/llm_analysis/03_summarize_results.py -> reports/<experiment_id>/*
```

The module only reads the miner's output. It never modifies the mined JSONL files, and it does not
touch `src/nlp_analysis/` or its results. The NLP pipeline clusters commit *messages*; this pipeline
classifies *commits* (message and/or diff). The two can be run independently.

> **Dataset requirement.** Before 2026-09-23 the miner stored inverted diffs (`commit.diff(parent)`).
> Classify only data mined after that fix; see the root `CLAUDE.md` for how to re-mine.

## Two experiment modes (`experiment.mode`)

| Mode | Input | Question | What the numbers mean |
|---|---|---|---|
| `candidate_validation` | the keyword-filtered miner dataset | Among commits selected by keywords, which are bug fixes? | The `BUG_FIX` share estimates the heuristic's **precision** (false-positive rate = 1 − precision). |
| `full_classification` | an **unfiltered** collection of Terraform commits | Among arbitrary Terraform commits, which are bug fixes? | Rates and metrics describe Terraform commits in general, provided the corpus really was unfiltered. |

### ⚠ Methodological warning: keyword-prefiltered data

The current miner dataset contains **only** commits whose message matches a bug-fix keyword. Therefore:

- The miner's selection is **not** ground truth. Nothing in this module treats it as ground truth.
- In `candidate_validation`, supervised metrics (precision, recall, F1, MCC, accuracy, specificity)
  computed against gold labels are **conditional on keyword-selected commits**. They measure how well
  the LLM separates real fixes from keyword false positives. They do **not** estimate how the LLM would
  perform on arbitrary Terraform commits. In particular, bug fixes whose messages contain none of the
  keywords are absent from the data, so recall over *all* bug fixes cannot be measured.
- Every `summary.json` carries a `scope_note` that states this, and so does every gold evaluation.
- Stage 01 reports `share_of_messages_matching_miner_keywords`. If `mode` is `full_classification` and
  that share is ≥ `keyword_prefilter_warning_threshold`, it logs a warning because the corpus looks
  prefiltered.

`full_classification` needs a commit collection mined **without** the keyword filter. The current
miner does not produce one. This module accepts one if it has the same record format.

## Classification definition

The definition lives in `prompts.py` (`SYSTEM_PROMPT_V1`) and is identical for all input modes.

- **BUG_FIX**: the primary purpose is to correct existing erroneous, defective, unintended, unsafe or
  invalid behavior or configuration. Examples: wrong attribute values, missing required configuration,
  broken references, invalid lifecycle, provider/resource interaction errors, regressions, security
  misconfigurations, wrong `count`/`for_each`/conditional logic, and plan/apply/validation failures.
- **NON_BUG_FIX**: features, new resources, refactoring, formatting, docs, examples, version bumps
  without a demonstrated defect, test-only changes, cleanup, renames, migrations, and merges or
  conflict resolution by themselves.

The prompt tells the model explicitly that bug-related words are not evidence. It must separate
"the message contains *fix*" from "the change repairs previously defective behavior". It must cite
only evidence present in the input, and put missing information in `uncertainty` rather than invent
a rationale. Commit text is treated as data, so instructions embedded in it are ignored.

Response schema (JSON only):

```json
{"label": "BUG_FIX|NON_BUG_FIX", "confidence": 0.0-1.0, "message_evidence": "...",
 "diff_evidence": "...", "reason": "...", "uncertainty": "..."}
```

`confidence` is self-reported by the model. It is **not** a calibrated probability.

## Three input modes (`experiment.input_modes`)

| Mode | The model sees |
|---|---|
| `message_only` | commit message |
| `diff_only` | Terraform diffs (file paths and patches) |
| `message_and_diff` | both |

All three use the same system prompt and the same commits. The user prompt states which evidence was
withheld, and the model must put `"NOT_PROVIDED"` in the corresponding evidence field. Stage 02 runs
all modes of a commit before moving to the next commit, so even an interrupted run covers the same
commits in every mode. Stage 03 compares the modes on the commits labeled in all of them.

Repository name, author and date are **not** shown in any mode. The model judges only the message
and the change.

## Input format and aggregation (stage 01)

Input is any set of miner JSONL files (`input.dataset_glob`) with records like:

```json
{"repo": "...", "commit_hash": "...", "author": "...", "date": "...", "message": "...",
 "file": "main.tf", "patch": "@@ ...", "head_hash": "..."}
```

The unit of classification is the **commit**. Records are grouped by `(repo, commit_hash)`, and files
of the same commit are never classified separately. For each commit, stage 01:

- keeps `repo, commit_hash, author, date, message, head_hash` (from the first record; conflicts are counted);
- lists every changed `.tf` file in `files`, including files with an empty patch (`files_without_patch`);
- drops empty patches and repeated `(file, patch)` records;
- stores identical patch text applied to several files once, with the other paths in `also_in`
  (common for copied modules);
- sorts `diffs` by path and computes `content_sha256` over everything that can reach a prompt.

Output: `commits_prepared.jsonl` (lossless, not truncated) and `prepare_stats.json` (counts, the
SHA-256 of every input file, the sample settings, and the keyword-match share).
`preparation.sample_size` / `--sample-size` draw a seeded random sample of commits.

## Diff representation and large diffs

Each file appears as `FILE: <path>` followed by a fenced `diff` block. `@@` headers and the `+`, `-`
and space prefixes are preserved exactly. Large commits are cut deterministically (`diff_context.py`):

1. At most `max_files_per_commit` patches are shown, in path order. The names of the other files are
   still listed.
2. `max_diff_chars` is split across the shown files by water-filling. Small patches get their full
   size, the remainder is shared equally, and each file is capped at `max_chars_per_file`. The first
   file therefore cannot consume the whole budget.
3. A patch is cut at hunk boundaries, then at line boundaries (mid-line only when a single line
   exceeds the allotment).
4. Every cut is announced in the prompt (`NOTE: patch truncated ...`). Each classification records
   `diff_truncated`, `original_diff_chars`, `submitted_diff_chars`, `files_submitted` and `files_omitted`.

With the default limits (24,000 / 8,000 / 20), 443 of 8,355 commits in the replayed dataset of 2026-09-24
(5.3%) have truncated diffs. Use `--dry-run` to measure this for other limits.

## Configuration (`config.yaml`)

Relative paths are resolved against the **repository root**. Paths passed on the command line are
resolved against the current directory. Main sections:

- `input`: `dataset_glob` and `miner_config` (the latter only supplies keywords for the prefilter check).
- `output`: `directory`, `prepared_commits_file`, `store_prompts` (store the full user prompt in every record).
- `experiment`: `mode`, `input_modes`, `keyword_prefilter_warning_threshold`.
- `preparation`: `sample_size`, `seed`.
- `llm`: `provider`, `model`, `temperature` (keep `0` for the main experiments), `max_output_tokens`,
  `timeout_seconds`, `retries`, `retry_backoff_seconds`, `workers`, `api_key_env`, `base_url`, `seed`,
  `json_mode`, `max_tokens_param` and `extra_params`. Unknown keys are rejected.
- `context`: `max_diff_chars`, `max_chars_per_file`, `max_files_per_commit`, `max_listed_file_names`.
- `classification`: `prompt_version`.
- `evaluation`: `gold_labels`, `gold_commit_column`, `gold_label_column`.

**Credentials** come only from environment variables: `ANTHROPIC_API_KEY` (default provider),
`OPENAI_API_KEY`, or the variable named in `llm.api_key_env`. Never put keys in `config.yaml`.

## Running

From the repository root (the scripts also work from `src/llm_analysis`):

```bash
uv pip install -r requirements.txt          # adds anthropic, openai, PyYAML
export ANTHROPIC_API_KEY=...                 # fish: set -x ANTHROPIC_API_KEY ...

python src/llm_analysis/01_prepare_commits.py                      # all commits
python src/llm_analysis/01_prepare_commits.py --sample-size 500    # or a seeded sample

python src/llm_analysis/02_classify_commits.py --dry-run           # prompts + truncation stats, no API calls
python src/llm_analysis/02_classify_commits.py --provider mock --limit 20   # offline plumbing test (FAKE labels)
python src/llm_analysis/02_classify_commits.py                     # the real run
python src/llm_analysis/02_classify_commits.py --commits-file commits_analysis/gold.csv   # only gold commits

python src/llm_analysis/03_summarize_results.py [--gold-labels gold.csv]
```

Other stage 02 options: `--input-modes`, `--model`, `--workers`, `--limit`, `--force` and
`--retry-parse-failures`. All stages accept `--config` and `--output-dir`.

Provider `mock` returns labels derived from a hash of the prompt. It exists only to test the
pipeline, and stage 03 warns when it summarizes mock results.

## Outputs (`analysis_results_llm/`, gitignored)

| File | Content |
|---|---|
| `commits_prepared.jsonl`, `prepare_stats.json` | stage 01 (see above) |
| `classifications_<input_mode>.jsonl` | one line per request, append-only (fields below) |
| `experiments/<experiment_id>.json` | full experiment settings, system prompt text and template |
| `runs.jsonl` | one line per stage 02 invocation (arguments, counts, timestamps) |
| `pipeline.log` | log of all stages |
| `reports/<experiment_id>/summary.json` | per-mode counts, label distribution, parse/API failures, truncated commits, confidence distribution, cross-mode disagreement counts, the LLM-estimated heuristic precision (candidate_validation), gold metrics (if any) and the scope note |
| `reports/<experiment_id>/summary.csv` | one row per input mode |
| `reports/<experiment_id>/classification_counts.csv` | `input_mode, label, count, fraction` (labels include `PARSE_FAILURE`, `API_FAILURE`) |
| `reports/<experiment_id>/errors.jsonl` | every failed request with its error / raw response |
| `reports/<experiment_id>/disagreements.csv` | commits where `message_only != message_and_diff` or `diff_only != message_and_diff`, with the labels, confidences and reasons of every mode (for qualitative analysis) |

Each classification record has these fields: `record_id, experiment_id, experiment_mode, repo,
commit_hash, input_mode, provider, model, response_model, temperature, max_output_tokens,
prompt_version, prompt_sha256, timestamp, label, confidence, reason, message_evidence, diff_evidence,
uncertainty, diff_truncated, original_diff_chars, submitted_diff_chars, files_submitted,
files_omitted, prepared_commit_sha256, user_prompt_sha256, user_prompt, raw_response, parse_success,
parse_error, parse_warnings, error, error_type, attempts, input_tokens, output_tokens, stop_reason,
latency_seconds`.

**Audit trail:** each record links the repository and SHA to the commit's content hash, the exact
user prompt (or its hash), the system prompt (via `prompt_version`, `prompt_sha256` and
`experiments/<id>.json`), the model and the raw response.

## Reproducibility and resuming

- `experiment_id` is a hash of experiment mode, provider, model, temperature, max output tokens, seed,
  JSON mode, extra parameters, prompt version and template hash, context limits and diff-format
  version. `record_id` is a hash of `(experiment_id, repo, commit_hash, input_mode)`.
- Every result is appended and flushed as soon as its request finishes. After a crash or Ctrl-C,
  rerunning the same command skips completed `record_id`s. API failures are retried on the next run.
  Parse failures are kept unless `--retry-parse-failures` is given, because the model did answer and
  its raw output is stored. `--force` reclassifies everything. Old lines are kept, and the last line
  of a `record_id` wins.
- Changing any identity setting (for example the model) starts a new experiment in the same files.
  Stage 03 summarizes one experiment at a time (the configured one by default, or `--experiment-id`).
- Stage 02 refuses to resume when stored results were built from a different `commits_prepared.jsonl`
  (content hash mismatch, for example after re-mining). It also refuses when the text of a prompt
  version was edited after use. **Never edit a used prompt: add a new version to `prompts.PROMPTS`.**
- Retries are handled by this module (SDK retries are disabled): timeouts, connection errors, 408,
  409, 429 and 5xx responses, with exponential backoff.

## Gold labels (optional)

Supervised metrics are computed only when a gold file is supplied (`evaluation.gold_labels` or
`--gold-labels`). The file is `.csv` or `.jsonl` with:

- a commit column: `commit_hash`, `commit_sha`, `sha` or `commit_url`. GitHub commit URLs and
  spreadsheet `=HYPERLINK("...")` cells work, and so do abbreviated SHAs that are unique;
- a label column: `gold_label`, `label` or `is_bugfix`. `BUG_FIX`, `true`, `1` and `yes` mean bug fix;
  `NON_BUG_FIX`, `false`, `0` and `no` mean not a bug fix. Other values (for example `MAYBE`) are
  ignored and counted.

Rows are matched to classifications by commit SHA, because the miner's `repo` field is a clone
directory name. Commits with contradictory labels in the file are excluded and reported. Stage 03
reports the confusion matrix, precision, recall, F1 and MCC for `BUG_FIX`, per input mode:

- `paired`: commits with a gold label **and** a parsed label in every mode. This is the fair
  comparison of the three modes;
- `unpaired`: every gold commit each mode could label, with coverage counts.

The per-annotator files in `commits_analysis/sample_<name>.csv` are **not** a gold standard as they
are. Annotators disagree on some commits, and some use `MAYBE` or leave cells blank. Adjudicate them
into one file first, and report inter-annotator agreement separately. Use
`02_classify_commits.py --commits-file <gold file>` to classify just those commits.

## Adding an LLM provider

1. Subclass `LLMClient` in `llm_client.py` and implement `_complete(system, user) -> LLMResponse`,
   making a single request with no retries. Read credentials from an environment variable via
   `self._api_key(...)`. Override `is_retryable` if the SDK's exceptions do not expose `status_code`.
2. Register it: add it to `PROVIDERS`, or call `register_provider("name", MyClient)`.
3. Set `llm.provider: name` in `config.yaml`. Add the SDK to `requirements.txt` with a comment saying why.

For local models, an OpenAI-compatible server usually needs no new code: set `provider: openai` and
`base_url: http://localhost:11434/v1` (Ollama), plus `seed`, and optionally `json_mode: true`.

## Tests

```bash
python -m pytest tests/test_llm_*.py
```

The tests use a fake client and the `mock` provider and never call a real API. They cover grouping,
duplicate and empty patches, deterministic formatting, truncation, response parsing and validation,
retries, resume/force/stale detection, metrics, gold-label loading, and an end-to-end run of the three
stages.

## Known limitations

- **Merge commits**: the miner diffs a commit against its first parent, so a merge commit shows the
  whole merged branch. The prompt tells the model that a merge by itself is not a bug fix, but such
  diffs can be large and mix many changes.
- **Only `.tf` files** are in the miner's output. A fix whose essential part is in a `.tfvars`, script
  or provider code shows only its `.tf` side.
- **Model drift**: hosted models can change behind the same name. `response_model` records the name
  the provider reports; prefer dated model snapshots when a provider offers them.
- **Self-reported confidence** is not calibrated. Use it for ranking and triage, not as a probability.
