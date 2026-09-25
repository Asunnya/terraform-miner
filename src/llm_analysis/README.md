# LLM Analysis: BUG_FIX vs. NON_BUG_FIX classification of Terraform commits

This module uses a Large Language Model to classify Terraform commits as `BUG_FIX` or `NON_BUG_FIX`.
It can use the commit message, the Terraform diff, or both.

## Research objective

The miner (`src/miner/`) selects *candidate* bug-fix commits with a keyword heuristic: the message
contains `fix`, `bug`, `issue`, `resolve`, `error`, `failure` or `incorrect`. That heuristic has false
positives such as "fix typo in README", "resolve merge conflicts" and "add error_message variable".
This module answers two questions:

1. **Which candidates are actually bug fixes?** Among keyword-selected commits that change
   infrastructure code (after the eligibility filter), the share the LLM classifies as `BUG_FIX` is reported
   as `eligible_candidate_bugfix_rate`. It also produces a cleaner defect dataset for deriving Terraform
   mutation operators.

The three steps have separate roles: **candidate retrieval** (miner: article-derived keyword heuristic +
changed `.tf` files), **eligibility** (stage 01: scope-based selection of infrastructure changes) and
**classification** (stage 02: LLM `BUG_FIX` / `NON_BUG_FIX`). The keyword heuristic identifies candidates;
it is not ground truth.
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
| `candidate_validation` | the keyword-filtered miner dataset, after eligibility filtering | Among keyword-selected commits with eligible infrastructure changes, which are bug fixes? | The `BUG_FIX` share in that population (not the heuristic's precision over all raw keyword-selected commits). |
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

The definition lives in `prompt_templates/v1/system.txt` and is identical for all input modes.

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

### Eligibility filtering (scope)

Stage 01 runs **aggregation → eligibility filtering → sampling**, so a sample is always drawn from the
eligible population. The target population is **Terraform commits that modify infrastructure-defining
`.tf` code**: the next research step catalogs defects in infrastructure code, so commits whose changes are
only tests, examples, documentation, vendored or generated content, and generic merge commits, are left
out before any LLM call. This is a **scope** decision only: it never looks at bug-related words and never
decides whether a commit fixes a bug.

- **File scope** comes from whole directory segments (`eligibility.excluded_scope_segments`, case-insensitive;
  the outermost excluded segment decides): `tests`/`testdata`/`fixtures` → TEST, `examples`/`example` →
  EXAMPLE, `docs`/`doc` → DOCUMENTATION, `vendor` → VENDOR, `.terraform` → GENERATED, `.github` →
  REPOSITORY_METADATA, anything else → INFRASTRUCTURE. A directory named `test` is excluded **only as the
  first directory** (`eligibility.root_only_scope_segments`): `test/` at the repository root is a test suite
  (Terratest/Kitchen convention), while `environments/test/main.tf` can be real, deployed infrastructure and
  is kept. The miner applies no scope filter
  (it exports every `.tf` file of a candidate commit), so this stage sees the complete evidence.
- **Mixed commits are kept, minus the out-of-scope files**: implementation + tests → keep the commit, send
  only the implementation diff. `files`, `diffs` (including shared patches in `also_in`) and
  `files_without_patch` keep only in-scope paths; the removed ones are listed in `excluded_files`, which
  never reaches the prompt. `content_sha256` is recomputed from the filtered commit.
- **Generic merges** are detected by message patterns (`eligibility.merge_patterns`, e.g.
  `^Merge pull request #\d+ from .*`), a heuristic, since the miner records carry no parent count.
  A message that merely contains the word "merge" is not a merge.
- **States**: `IN_SCOPE` (`eligible`), `OUT_OF_SCOPE` (`generic_merge`, `only_test_files`,
  `only_example_files`, `only_documentation_files`, `only_vendor_files`, `only_generated_files`,
  `only_excluded_files`, `no_in_scope_terraform_files`) and `INSUFFICIENT_DATA` (`no_in_scope_patch`:
  in-scope files exist but none has a patch). An empty commit message is not a reason.

Excluded and insufficient commits go to `eligibility_exclusions.jsonl` (state, reason, original,
in-scope and excluded files with their scope); only `IN_SCOPE` commits reach `commits_prepared.jsonl`.

Output: `commits_prepared.jsonl` (lossless, not truncated), `eligibility_exclusions.jsonl` and
`prepare_stats.json` (counts, eligibility counts by state and reason, the SHA-256 of every input file,
the sample settings, and the keyword-match share).
`preparation.sample_size` / `--sample-size` draw a seeded random sample of the eligible commits.

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

## Prompt files

The prompt text is kept out of the Python code, one folder per version:

```
prompt_templates/v1/system.txt   system prompt: the classification definition and output schema
prompt_templates/v1/user.yaml    message/diff headers, empty-message text, one note per input mode
```

`prompts.py` only loads and assembles them. To change the wording, copy `v1/` to `v2/`, edit the copy,
and set `classification.prompt_version: "v2"`. Each record stores the version and the SHA-256 of the
loaded files, so a version can never be edited silently after producing results.

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
- `classification`: `prompt_version` (a folder in `prompt_templates/`).
- `batch`: `max_requests_per_batch`, `max_file_mb`, `poll_interval_seconds`.
- `evaluation`: `gold_labels`, `gold_commit_column`, `gold_label_column`.

**Credentials** come only from environment variables: each vendor's standard one, read by LiteLLM
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...), or the one named in `llm.api_key_env`.
Never put keys in `config.yaml`.
The easiest way is a `.env` file at the repository root, which is gitignored and loaded by stage 02:

```bash
cp .env.example .env     # then fill in OPENAI_API_KEY=...
```

Variables already exported in the shell take precedence over `.env`.

## Running

From the repository root (the scripts also work from `src/llm_analysis`):

```bash
uv pip install -r requirements.txt          # adds litellm, PyYAML, python-dotenv
cp .env.example .env                         # put OPENAI_API_KEY=... in .env (or export it)

python src/llm_analysis/01_prepare_commits.py                      # all commits
python src/llm_analysis/01_prepare_commits.py --sample-size 500    # or a seeded sample

python src/llm_analysis/02_classify_commits.py --dry-run           # prompts + truncation stats, no API calls
python src/llm_analysis/02_classify_commits.py --provider mock --limit 20   # offline plumbing test (FAKE labels)
python src/llm_analysis/02_classify_commits.py                     # the real run
python src/llm_analysis/02_classify_commits.py --commits-file commits_analysis/gold.csv   # only gold commits

python src/llm_analysis/03_summarize_results.py [--gold-labels gold.csv]
```

### Cheaper runs: batch

`--batch` sends the same requests through LiteLLM's batch API (half the price on OpenAI, results within
24 hours). Supported vendors: `openai`, `azure`, `vertex_ai`, `bedrock`, `hosted_vllm`; not `anthropic`
(LiteLLM cannot create Anthropic batches yet) or `gemini` (use `vertex_ai`). Before submitting, the
configured parameters are checked against what the vendor accepts.

```bash
python src/llm_analysis/02_classify_commits.py --batch submit          # upload the pending requests
python src/llm_analysis/02_classify_commits.py --batch status          # progress
python src/llm_analysis/02_classify_commits.py --batch collect         # write finished results
python src/llm_analysis/02_classify_commits.py --batch collect --wait  # or poll until everything is done
python src/llm_analysis/02_classify_commits.py --batch cancel
```

- `submit` accepts the same selection flags as a normal run (`--limit`, `--commits-file`,
  `--input-modes`, `--force`, `--retry-parse-failures`). Submissions over 50,000 requests or 200 MB
  are split into several batches automatically.
- Batch results are ordinary records with the same `experiment_id`, plus `batch_id`. Stage 03 treats
  them exactly like synchronous results, and a batch run can be finished synchronously or the other
  way round.
- A request in a batch that has not been collected is never sent again, by either path. Requests
  that failed, or that got no answer (an expired or cancelled batch), go back into the plan: run
  `--batch submit` again to resend them.
- Everything is kept in `batches/`: the state file of each batch, the exact uploaded JSONL, and the
  raw output and error files.
- If OpenAI rejects a batch with `token_limit_exceeded` (the tokens your account tier may queue per
  model), submit in parts: `--batch submit --limit 2000`, `--batch collect --wait`, and repeat.
- For the full dataset, 25,065 requests, 177 MB and about 41M input tokens fit in one batch.

Other stage 02 options: `--input-modes`, `--model`, `--workers`, `--limit`, `--force` and
`--retry-parse-failures`. All stages accept `--config` and `--output-dir`.

Provider `mock` returns labels derived from a hash of the prompt. It exists only to test the
pipeline, and stage 03 warns when it summarizes mock results.

## Outputs (`analysis_results_llm/`, gitignored)

| File | Content |
|---|---|
| `commits_prepared.jsonl`, `eligibility_exclusions.jsonl`, `prepare_stats.json` | stage 01 (see above) |
| `classifications_<input_mode>.jsonl` | one line per request, append-only (fields below) |
| `experiments/<experiment_id>.json` | full experiment settings, system prompt text and template |
| `runs.jsonl` | one line per stage 02 invocation (arguments, counts, timestamps) |
| `batches/` | Batch API state, uploaded input and raw output of each batch |
| `pipeline.log` | log of all stages |
| `reports/<experiment_id>/summary.json` | per-mode counts, label distribution, parse/API failures, truncated commits, confidence distribution, cross-mode disagreement counts and Cohen's kappa, `eligible_candidate_bugfix_rate` (eligible commits classified BUG_FIX / eligible commits successfully classified, with its 95% Wilson interval; not precision), gold metrics (if any: precision/recall with Wilson intervals, F1, MCC, exact McNemar tests between modes with Holm-adjusted p-values) and the scope note |
| `reports/<experiment_id>/summary.csv` | one row per input mode |
| `reports/<experiment_id>/classification_counts.csv` | `input_mode, label, count, fraction` (labels include `PARSE_FAILURE`, `API_FAILURE`) |
| `reports/<experiment_id>/errors.jsonl` | every failed request with its error / raw response |
| `reports/<experiment_id>/disagreements.csv` | commits where `message_only != message_and_diff` or `diff_only != message_and_diff`, with the labels, confidences and reasons of every mode (for qualitative analysis) |

Each classification record has these fields: `record_id, experiment_id, experiment_mode, repo,
commit_hash, input_mode, provider, model, response_model, temperature, max_output_tokens,
prompt_version, prompt_sha256, timestamp, label, confidence, reason, message_evidence, diff_evidence,
uncertainty, diff_truncated, original_diff_chars, submitted_diff_chars, files_submitted,
files_omitted, prepared_commit_sha256, user_prompt_sha256, user_prompt, raw_response, parse_success,
parse_error, parse_warnings, error, error_type, attempts, batch_id, input_tokens, output_tokens,
stop_reason, latency_seconds`.

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
  version was edited after use. **Never edit a used prompt: add a new version folder (see below).**
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

## Models and providers

Every real model is reached through [LiteLLM](https://docs.litellm.ai/docs/providers) (`provider: litellm`),
named `"<vendor>/<model>"`: `openai/gpt-4.1-nano`, `anthropic/claude-sonnet-5`, `vertex_ai/gemini-2.5-pro`,
`hosted_vllm/<model>` or `ollama/<model>` with `base_url`. The only other provider is `mock` (fake labels,
tests only). A parameter the vendor does not accept raises an error instead of being silently dropped
(e.g. set `seed: null` for vendors without seeds).

To add a different access path, subclass `LLMClient`, implement `_complete(system, user)` (one request, no
retries) and register it with `register_provider("name", MyClient)`.

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

### Report schema change (2026-09-25)

`summary.json` no longer has `llm_estimated_keyword_heuristic_precision`, the per-mode
`bug_fix_share_of_parsed(_ci95)` or `keyword_heuristic_precision_gold(_ci95)`. They are replaced by
`eligible_candidate_bugfix_rate` (top level, with its definition, and per mode) and
`gold_eligible_candidate_bugfix_rate`; `summary.csv` has `eligible_candidate_bugfix_rate(_ci95_low|_ci95_high)`.
The old names called an LLM-label rate "precision", which it is not without gold labels.
