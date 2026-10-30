# PEFT for extractive BioASQ QA — reproducibility package

This package contains the code, configurations, validation trajectories and
aggregate results for a comparison of full fine-tuning (FFT), LoRA and DoRA on
`BioLinkBERT-base` for extractive BioASQ factoid question answering. Nine runs
were performed — three methods × three training seeds (42, 123, 1024) — on an
Apple M1 Pro (16 GiB) using the PyTorch MPS backend. All nine runs share one
fixed grouped data split and one evaluator; each run trains for two epochs
(1,122 optimizer steps) with AdamW, batch size 4, float32, no learning-rate
scheduler and no gradient clipping, and selects its checkpoint by maximum
validation pair F1 (earliest checkpoint retained on ties).

## Dataset

`BioASQ-train-factoid-6b-full-annotated.json`, from the Kaggle BioASQ-Dataset
collection: https://www.kaggle.com/datasets/maverickss26/bioasq-dataset

- Size: 9,433,581 bytes
- SHA-256: `b4d62be43a448b2f600664e219b23aa047f2dd80248acde3d8838b7959b41bc5`

**The dataset file is not redistributed in this package.** To reproduce anything
below, obtain the file from Kaggle and verify it before use:

```
shasum -a 256 BioASQ-train-factoid-6b-full-annotated.json
```

The digest must equal the value above. The file's internal `BioASQ6b` label is
not a Kaggle release number, and the Kaggle release/version of the local copy
used here was not recorded.

## Base model

`michiyasunaga/BioLinkBERT-base` at commit
`b71f5d70f063d1c8f1124070ce86f1ee463ca1fe`.

The runs loaded this from a local snapshot directory with `--local-files-only`.
Per-file SHA-256 digests of the six snapshot files are recorded in
`configs/protocol.json` under `model_file_hashes`.

## Regenerating the grouped split (seed 42)

The split logic is fully contained in `code/qa_core.py`. The split is produced
by this call, where `SOURCE` is the verified dataset file:

```python
from qa_core import load_rows, merge_examples, split_examples

splits = split_examples(merge_examples(load_rows(SOURCE)), seed=42)
```

`split_examples` already defaults to `seed=42`. It merges exact
question/context duplicates, forms connected components linked by original
question ID, normalized question text or normalized context, and allocates
components 70/10/20. This yields 2,536 examples in 423 components, split into
1,787/247/502 examples, 304/44/91 original question IDs and 295/43/85
components for train/validation/test.

To write the `split_manifest.json` file that `run_experiment.py` and
`run_nine.py` consume, run `audit.py` from inside `code/`:

```
cd code
python3 audit.py \
  --source <path to verified BioASQ-train-factoid-6b-full-annotated.json> \
  --source-url https://www.kaggle.com/datasets/maverickss26/bioasq-dataset
```

This writes `code/artifacts/split_manifest.json` and `code/artifacts/audit.json`.
`--out` defaults to `artifacts/` beside the script, which is exactly where
`run_nine.py` looks for the manifest, so no further argument is needed. The
manifest has keys `schema_version`, `split_seed` (42), `source_sha256`,
`policy`, `splits` and `manifest_sha256`, the last being `qa_core.digest()` of
the object with that key removed; `run_experiment.py` re-checks that digest on
startup and refuses to train if it does not match.

The manifest is built from the source file alone and reproduces exactly: the
regenerated `manifest_sha256` for the checksummed source file at split seed 42
is `ad1ebb4838ae9dc5d8edfcd8945c6563640b6d9ae390b1d442e397a6a32a30ab`.

`audit.py` additionally reports on the superseded earlier split and on archived
training logs by scanning `FFT_seed_results/`, `lora_seed_results/`,
`dora_seed_results/` and `lightning_logs/` under `--project`. Those directories
are deliberately not part of this package, so those sections of `audit.json`
come out empty. This does not affect `split_manifest.json`, which is unaffected
by their absence.

## Rerunning the nine runs

The driver is `code/run_nine.py`:

```
python3 run_nine.py --model <snapshot dir for BioLinkBERT-base> --out <new output dir>
```

Both arguments are required and `--out` must not already exist. The driver
requires MPS to be available (it refuses to substitute CPU), at least 6 GiB of
free disk space, and expects the split manifest at `artifacts/split_manifest.json`
relative to its own directory — generate it with `audit.py` first, as described
above. It fixes the learning rates internally — FFT 4e-5, LoRA 3e-4, DoRA 2e-4 —
and runs the nine jobs in a rotated method order.

`run_nine.py` snapshots `run_experiment.py`, `qa_core.py`, `summarize_runs.py`
and `plot_runs.py` into `<out>/source/`, runs the nine jobs, then invokes
`summarize_runs.py` and `plot_runs.py` to write `<out>/summary/` and
`<out>/figures/`. All four scripts are included here, so the driver runs
end to end once the manifest exists and the base model snapshot is available.

Individual runs can also be launched directly; this is the exact command
`run_nine.py` constructs for each job:

```
python3 run_experiment.py --manifest <manifest> --model <snapshot dir> --local-files-only \
  --method fft --seed 42 --learning-rate 4e-5 --device mps --epochs 2 --batch-size 4 \
  --max-length 384 --stride 128 --max-answer-tokens 30 --validations-per-epoch 4 \
  --out <new run dir>
```

Aggregation across the nine completed run directories:

```
python3 summarize_runs.py <run dir> [<run dir> ...] --out <new summary dir>
```

### Primary and Secondary Evaluation Metrics

The manuscript reports pair-weighted Exact Match (EM) and F1 scores as the primary evaluation metrics. These scores are averaged over merged question/context pairs.

The experimental artifacts also contain secondary original-question macro-averaged EM and F1 scores. These metrics assign equal weight to original question IDs rather than individual question/context pairs and are retained for transparency and supplementary analysis.

The secondary metrics produce different numerical model rankings, highlighting the sensitivity of the results to the aggregation method. Neither evaluation corresponds to the official BioASQ factoid benchmark protocol. The study does not establish superiority of one adaptation method over another.

## What is included

- `code/` — preparation and scoring (`qa_core.py`), the single-run trainer
  (`run_experiment.py`), the nine-run driver (`run_nine.py`), aggregation and
  bootstrap (`summarize_runs.py`), the data audit and split-manifest writer
  (`audit.py`), the figure generator (`plot_runs.py`), the
  tokenization/coverage audit (`audit_tokenization.py`) and unit tests
  (`test_qa_core.py`, `test_summarize.py`)
- `configs/` — the frozen batch protocol and the nine per-run configurations
- `runs/` — the nine validation trajectories (loss and metrics at steps 141,
  282, 423, 561, 702, 843, 984 and 1,122)
- `results/` — per-seed and aggregate metrics, the efficiency table, and the
  consolidated resource measurement records

## What is not included

- **The dataset file and the split manifest.** Both contain third-party source
  question and context text and are not redistributed. See the Dataset section
  for how to obtain and verify the source file.
- **Model checkpoints** (`best_state.pt`, 431–442 MB each, 3.9 GB in total) —
  omitted for size.
- **Per-example test predictions and the raw per-run result records** —
  available from the corresponding author on request.
- **Earlier results.** Results produced before this protocol used a different,
  non-grouped split with train/test leakage and a different evaluator. They are
  superseded, are not comparable to the numbers here, and are excluded.

## Determinism

`run_experiment.py` seeds Python, NumPy and PyTorch and requests deterministic
algorithms with `torch.use_deterministic_algorithms(True, warn_only=True)`.
Unsupported operations therefore fall back to nondeterministic kernels on MPS,
and **bitwise reproducibility is not guaranteed**. Reported resource figures are
sampled maxima at a 20 ms interval, not guaranteed peaks; MPS and process RSS
values overlap in unified memory and must not be summed.

## License

Released under the MIT License; see [LICENSE](LICENSE).
