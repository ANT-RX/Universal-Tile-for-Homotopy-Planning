# Universal Tile-Based Obstacle Map Synthesis via Diffusion for Homotopy Planning

Reference implementation and evaluation code for the paper *Universal Tile-Based
Obstacle Map Synthesis via Diffusion for Homotopy Planning* (ACDSA 2027).

The method learns a permutation-equivariant set diffusion model over small
**universal** obstacle tiles -- tiles the HPPM path-planning engine can traverse
along all 12 port-to-centre routes, three equispaced ports per edge -- and
assembles them into larger mosaics, validated up to 8x8, whose corner-to-corner
route is then verified with the same engine. This repository contains the code
that produced the tables and figures of the paper.

> **Assets are not in this repository.** It is code only. The dataset (171 MB)
> and the six checkpoints (1.4 GB) are hosted separately and fetched by
> `scripts/download_assets.py`. See [§3](#3-download-the-data).

---

## 1. Requirements

Python 3.10+ and a C compiler.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the campaign was run against. Two
dependencies are easy to miss and break the import chain when absent:

* **scipy** -- used by `src/evaluation/paper_metrics.py`
  (`scipy.stats.norm`, `scipy.spatial.cKDTree`).
* **matplotlib** -- imported by the four scripts that render review figures
  (`src/pipeline/assembler.py`, `src/pipeline/mosaic_no_hppm.py`,
  `src/evaluation/visual_review.py`, `src/evaluation/final_evaluation.py`).

`matplotlib` is deliberately left **unpinned**: it is only used with the `Agg`
backend to draw figures, so its version cannot affect any reported number, and
pinning it would needlessly narrow the environments this code installs in. Pin
it yourself if you want byte-identical figures.

`requirements.txt` also pulls `torch` from the CUDA wheel index
(`--extra-index-url .../cu124`). Edit or drop that line if your machine has a
different CUDA toolkit or no GPU.

Before running anything at volume on a new machine:

```bash
python3 -m src.pipeline.server_diagnostics --expected-workers <N> --min-disk-gb 20
```

It reports how many cores Python actually sees (`os.cpu_count()` is what the
`--workers` defaults trust -- on a shared or containerised box that is often the
host's core count, not your quota), whether `torch.cuda` sees a GPU with bf16
support, free disk, and whether `gcc` is on `PATH`.

## 2. Compile the HPPM engine

Do this **before** anything else:

```bash
gcc -O2 -o engine/hppm engine/HPPMDinamico3.c -lm
chmod +x engine/hppm
```

> ### Recompile on **every** machine.
>
> `engine/hppm` is a compiled binary and is deliberately **not** versioned (see
> `.gitignore`). An inherited binary -- one built on a different machine and
> copied over -- still runs and still exits successfully, but silently produces
> **zero successes across millions of candidates**. The campaign lost an entire
> phase to exactly this failure mode, which is why the engine is never shipped
> and why the sanity check below is not optional.
>
> ```bash
> python3 -m src.dataset.check_engine --dir <universal_generator output dir> \
>   --hppm-exe engine/hppm -n 200
> ```
>
> If that reports failures on tiles already known to be universal, the engine's
> behaviour changed. Stop before training or evaluating anything with it.

Two things about the engine that are easy to get wrong:

* **`--hppm-exe` path.** The campaign scripts default this flag to
  `artifacts/hppm_v2`, the path the campaign actually used. Either pass
  `--hppm-exe engine/hppm` explicitly (the examples below do) or compile to
  `artifacts/hppm_v2` instead. Both paths are gitignored.
* **`m1`/`m2` are dead configuration.** `leer_archivos_configuracion` in
  `engine/HPPMDinamico3.c` reads the `m1`/`m2` fields of `pendientes.txt` and
  then assigns `1.0` and `4.0` hardcoded, so they do not affect the planned
  path at all. The `--m1`/`--m2` CLI flags were removed for that reason; the
  fixed values are documented in `src/common/domain_scaling.py`.

`engine/` contains only `HPPMDinamico3.c`. The engine's geometry constants (for
example `MAX_OBSTACULOS`, mirrored by `ENGINE_MAX_OBSTACLES` in
`src/common/domain_scaling.py`) are part of the compiled artefact -- change them
in the `.c` and recompile, and update the Python mirror by hand.

## 3. Download the data

```bash
python3 scripts/download_assets.py
```

The script downloads into the exact paths the code expects, is **resumable**
(HTTP range requests) and **idempotent** (a file whose sha256 already matches is
never fetched again), and **verifies every file against `checksums.txt`** before
accepting it. The dataset is checked against

```
95a20e222a46057d1bfc3cb1d55c359fa1bca9d9ac064b8302c4bd2ceabdfcfe
```

which is the value reported in Table I of the paper; the script aborts if it
does not match. Use `--check` to verify what is already on disk and download
nothing:

```bash
python3 scripts/download_assets.py --check
```

The assets live in two Hugging Face repositories:

* Dataset, plus both held-out splits and their sidecars:
  <https://huggingface.co/datasets/jantrom/tile-obstacle-maps>
* The six checkpoints, under `artifacts/ckpt_tiles_seed{101..106}/`:
  <https://huggingface.co/jantrom/tile-obstacle-diffusion>

The full evaluation battery was run with **seed 101**; the other five seeds were
used to quantify run-to-run variance and only need the diversity tests.

## 4. Reproducing the results

### 4.1 The held-out test split

The `.npz` holdouts are not versioned here -- they are derived from the dataset.
What is shipped in this repository are their `.meta.json` sidecars, which record
exactly how each split was built (`test_indices`, `data_rows`, `data_sha256`,
and for the deduplicated file `removed_row_positions`). The `.npz` files
themselves are available in the Hugging Face dataset repository if you would
rather download them than rebuild them.

`artifacts/test_holdout_final.npz` is carved out of the dataset once, with a
fixed seed that is independent of the training `--seed`
(`TEST_SPLIT_SEED = 20261009` in `src/model/set_diffusion.py`), so every seed
trains and is evaluated against the identical held-out tiles. It is written by
the first training run (see below) and reused unchanged afterwards.

`artifacts/test_holdout_dedup.npz` is that same holdout with near-duplicate rows
removed. Build it once, after the holdout exists:

```bash
python3 -m src.evaluation.check_split_leakage \
  --data tile-obstacle-maps.npz \
  --test artifacts/test_holdout_final.npz \
  --dedup-out artifacts/test_holdout_dedup.npz \
  --output artifacts/leakage_report_dedup.json
```

Use the **deduplicated** file as `--reference-data` when evaluating the six
seeds. A near-duplicate tile surviving in the reference would make two runs'
diversity and memorization numbers disagree for a reason that has nothing to do
with the model. The shipped `artifacts/leakage_report_dedup.json` is the audit
that was run over this split: all five n-bins fall below the 5% alert threshold
with 0 exact descriptor duplicates.

If `tile-obstacle-maps.npz` ever changes, both holdouts must be regenerated and
the leakage re-checked before any new metric is trusted.

### 4.2 Training -- seeds 101 to 106

One run per seed, same data and same held-out split; only `--seed` and `--dir`
differ:

```bash
for SEED in 101 102 103 104 105 106; do
  python3 -m src.model.set_diffusion train \
    --data tile-obstacle-maps.npz \
    --dir artifacts/ckpt_tiles_seed${SEED} \
    --test-out artifacts/test_holdout_final.npz \
    --seed ${SEED} 2>&1 | tee train_seed${SEED}.log
done
```

The training hyperparameters are the campaign's values, frozen as module
constants rather than flags: steps 100000, batch 512, lr 3e-4, warmup 2000,
ema 0.9995, cfg-dropout 0.15, T 1000, d-model 256, layers 8, heads 8,
test-frac 0.05, val-frac 0.05 and success-only (`True`, matching the six
checkpoints, which all record `success_only: true`), with AdamW
`weight_decay=0.01`, a cosine schedule and bf16 autocast.

The published checkpoints come from exactly these settings, with one exception:
seed 105 was checkpointed at 96,000 steps rather than 100,000. It is included in
every reported aggregate.

Every run must print `Test holdout REUSED` at startup. If any prints
`(written)`, its `--data` is not the file the holdout was generated from and
that run's "unseen" tiles are not comparable to the other five -- stop and fix
it before training.

Keep the training log: `paper_metrics.py --training-log` parses the wall-clock
cost from its last `step ... | N min` line.

### 4.3 Evaluation -- seed 42

The evaluation RNG seed is 42 for every checkpoint. Pass `--workers` explicitly
rather than trusting the `os.cpu_count()-2` default (see §1).

The campaign's battery was split across two invocations:

```bash
# seed 101: cost accounting, diversity and the mosaic battery
python3 -m src.evaluation.paper_metrics \
  --model-dir artifacts/ckpt_tiles_seed101 \
  --hppm-exe engine/hppm \
  --training-data tile-obstacle-maps.npz \
  --reference-data artifacts/test_holdout_dedup.npz \
  --output-dir artifacts/metrics_seed101 \
  --training-log train_seed101.log \
  --seed 42 --workers 30 \
  --tests cost diversity mosaics

# seeds 102-106: diversity only
for SEED in 102 103 104 105 106; do
  python3 -m src.evaluation.paper_metrics \
    --model-dir artifacts/ckpt_tiles_seed${SEED} \
    --hppm-exe engine/hppm \
    --training-data tile-obstacle-maps.npz \
    --reference-data artifacts/test_holdout_dedup.npz \
    --output-dir artifacts/metrics_seed${SEED} \
    --training-log train_seed${SEED}.log \
    --seed 42 --workers 30 \
    --tests diversity
done
```

Compare the **C2ST confidence intervals** across the six seeds, not just the
point estimates, before claiming the model is stable across seeds.

The DDIM sweep and the per-`n` rate table were run with `--tests ddim_ablation`
and `--tests rate_by_n` respectively. The sample sizes per test
(`--n-per-bin 6000`, `--n-cost-rejection 200000`, `--n-cost-diffusion 5000`,
`--n-ablation 3000`, `--n-diversity 8000`, `--n-reference 1000`,
`--chamfer-pairs 2000`, `--memorization-samples 20000`) are the campaign's
values.

Two flags are deliberately **exposed** because the paper sweeps them:

* `--ddim-steps-to-test` (default `10 25 50 100 250`) -- the Table IV sweep.
* `--mosaic-sizes` (default `3 5 8`) -- the Table VI sweep.

`--n` / `--n-per-bin` and the other sample sizes are also flags, so a run can be
repeated with more samples.

Sampling itself is not swept: DDIM with 100 steps, `eta = 0` and guidance scale
`1.0`, which is the configuration behind every number in the paper.

### 4.4 Configuration dump

```bash
python3 -m src.evaluation.dump_config \
  --model-dir artifacts/ckpt_tiles_seed101 --output-dir artifacts/config_dump
```

Writes `config_dump.json` / `config_dump.md` with the checkpoint configuration,
loss normalization, schedule, EMA and guidance, the `domain_scaling.py`
constants, the `#define`s of `HPPMDinamico3.c`, and the route-resolved
verification criterion read from `verify_maps.py` at dump time.

## 5. Which script produces what

| Command | Output directory | What it is |
|---|---|---|
| `scripts/download_assets.py` | `tile-obstacle-maps.npz` | dataset: 600,000 tiles, sha256 `95a20e22…dcfe` (**Table I**) |
| `set_diffusion train` | `artifacts/ckpt_tiles_seed*/` | the six trained checkpoints; also writes the held-out split |
| `check_split_leakage` | `artifacts/leakage_report*.json` | per-`n`-bin near-duplicate audit of the split |
| `paper_metrics --tests rate_by_n` | `1_rate_by_n/` | success rate vs number of obstacles, with Wilson CI (**Table II**) |
| `paper_metrics --tests cost` | `2_cost/` | sampling cost: pure rejection vs diffusion + filter (**Table III**) |
| `paper_metrics --tests ddim_ablation` | `3_ddim_ablation/` | DDIM steps 10/25/50/100/250 (**Table IV**) |
| `paper_metrics --tests diversity` | `4_diversity/` | C2ST AUC (+ CI and p-value), Chamfer diversity, memorization (**Table V**) |
| `paper_metrics --tests mosaics` | `5_mosaics/` | mosaic assembly, rate before/after repair, Wilson CI (**Table VI**) |
| `dump_config` | `config_dump.{json,md}` | effective configuration, read from code at dump time |

Each test writes a `summary.json` (aggregate) and per-candidate
`results_detail.json`, which together are the traceability of every number
in the paper. The `artifacts/metrics_seed*/` directories shipped in this
repository are the campaign's actual outputs.

Two caveats when reading them:

* The **mosaic verification scope is corner-to-corner only**, a single HPPM
  route, not the per-tile 4·P port-to-centre criterion that each individual tile
  passed before placement. See `MOSAIC_VERIFICATION_SCOPE` in
  `src/common/domain_scaling.py`.
* `rate_by_n` and `cost`/`ddim_ablation` use **different sampling regimes** --
  a uniform number of candidates per `n` bin versus `n` drawn from the
  checkpoint's own distribution. `summary.json` labels each block with
  `sampling_regime`. Quote one primary aggregate and state which regime it is.

## 6. License

Released under the MIT License -- see [LICENSE](LICENSE).

## 7. Citation

```bibtex
@inproceedings{ruizvera2027universaltiles,
  title     = {Universal Tile-Based Obstacle Map Synthesis via Diffusion for
               Homotopy Planning},
  author    = {Ruiz-Vera, Daniel Yush{\'e}d and Romero-Melo, Jes{\'u}s Antonio
               and Diaz-Arango, Gerardo and Vazquez-Leal, Hector},
  booktitle = {ACDSA},
  year      = {2027},
  note      = {To appear}
}
```

<!-- TODO(owner): add the DOI and page numbers once the paper is published. -->

Faculty of Electronic Instrumentation, Universidad Veracruzana, Veracruz,
Mexico.
