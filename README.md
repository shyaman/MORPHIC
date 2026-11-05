# MORPHIC

MORPHIC scores **directed, nearby cell pairs** for 1,821 ligand–receptor mechanisms using an H&E image. It uses frozen [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) nucleus embeddings and the selected gated head in `models/morphic_head.pt`. RNA is used to construct training labels; inference needs only H&E and nuclei. Scores are raw sigmoid outputs, **not calibrated probabilities or proof of physical signaling**.

![MORPHIC overview: spatial transcriptomics supplies training labels, and H&E images are used to predict cell-pair ligand–receptor scores](overview.png)

## Set up

Python 3.12 and a CUDA GPU are recommended. Install [uv](https://docs.astral.sh/uv/) and run from the repository root:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/huggingface"
uv sync --frozen
```

UNI2-h is gated. Visit the [UNI2-h model page](https://huggingface.co/MahmoodLab/UNI2-h), sign in with an institutional email, request access, and accept its terms. After approval, authenticate locally:

```bash
uv run --frozen hf auth login
```

Each user must obtain their own access. UNI2-h weights are not redistributed here. The download script checks their SHA256 against the weights used for this head. UNI2-h and models trained from its outputs are subject to its [noncommercial research terms](https://huggingface.co/MahmoodLab/UNI2-h).

## Run on a new H&E image

`--mpp` is the **microns represented by one full-resolution image pixel**. Use the scanner's value; an incorrect value changes crop size and pair distances. Input images must be RGB TIFF or OME-TIFF.

With automatic H&E nucleus segmentation:

```bash
uv run --frozen python scripts/download_data.py --skip-hest
uv run --frozen python scripts/infer.py --image /path/to/slide.ome.tif --mpp 0.5 --out outputs/slide
```

With an aligned integer nucleus label TIFF of the same width and height:

```bash
uv run --frozen python scripts/infer.py --image /path/to/slide.ome.tif --mpp 0.5 \
  --nuclei /path/to/nuclei.tif --out outputs/slide
```

With cell annotations, add `--annotations /path/to/cells.csv`. The CSV requires unique `cell_id,x_px,y_px` columns in full-resolution H&E pixel coordinates. Additional columns such as `cell_type` are carried into `cells.parquet`; **cell type is not an input to the model**. MORPHIC matches each annotation to a detected or supplied nucleus within 10 µm. Unmatched annotations are reported and omitted; duplicate matches cause an error. Cell boundary annotations can be exported as centroids in this CSV. If you also have nucleus outlines, convert them to the aligned label TIFF and pass `--nuclei`.

Inference writes `cells.parquet`, `edges.parquet`, `scores.parquet`, and `run.json`. `scores.parquet` contains sender ID, receiver ID, mechanism name, and uncalibrated score. A slide can yield many cell-pair–mechanism rows; output is written in batches. Nuclei near the image boundary that cannot supply a complete 112 µm crop are omitted. Candidate pairs use the 40 µm radius and 12-neighbor rule. Nuclei are proxies for cells; proximity does not establish membrane contact.

To score a reproducible percentage of cells, add `--sample-percent 10 --seed 17` to any inference command. The percentage selects anchor cells uniformly from the crop-ready detected or annotated cells; each anchor keeps its nearby neighbors and both pair directions. Training uses all eligible neighboring cell pairs. Consequently, `cells.parquet` can contain more than 10% of eligible cells. The percentage, seed, eligible count, and anchor count are saved in `run.json`. Omit the option to use every eligible cell.

The small synthetic [inference example](examples/inference/README.md) checks the supplied-nuclei and cell-annotation inputs. Its scores have no biological meaning.

## Reproduce training

HEST files come from the pinned [HEST dataset](https://huggingface.co/datasets/MahmoodLab/hest) revision. Accept its access conditions if requested. `data/cohort.csv` defines **37 training, six validation, and 35 held-out test slides**. `data/lr_catalog.csv` contains the fixed 1,821 mechanisms and thresholds fitted on the ten threshold-fitting training sections. Test RNA is used only to evaluate predictions.

```bash
uv run --frozen python scripts/download_data.py --split train validation
uv run --frozen python scripts/preprocess.py --split train validation --stage targets
uv run --frozen python scripts/preprocess.py --split train validation --stage embeddings
uv run --frozen python scripts/train.py
```

For a small preprocessing check, add `--samples TENX132 --split train` to the download and preprocessing commands. Existing outputs are protected against overwriting. Use a new `--out` directory to rebuild targets; `--stage embeddings` can then add embeddings to those targets. Caches from the older sampled-cell preprocessing are incompatible and must be rebuilt.

Preprocessing uses all unique, crop-ready nuclei with at least 30 accepted transcripts (QV ≥20), selects up to 12 neighbors within 40 µm, and includes both directions. Isolated nuclei are omitted. It stores sender-ordered cell-pair/target shards of approximately 100,000 directed pairs, keeping complete sender neighborhoods within a shard. Crops are 224 × 224 at 0.5 µm/pixel; UNI2-h is frozen and nucleus-mask pooled. Training uses all eligible neighboring cell pairs each epoch, with a directed-pair batch size of **4,096** and seed 17.

The objective includes every positive mechanism and up to **16 hard-mixture negatives per edge**: up to eight from the 32 highest nonzero subthreshold targets, then zero-valued negatives, filling any shortfall from remaining negatives. Per-edge mean BCE is combined with 0.02 paired direction loss. AdamW uses learning rate 0.001 and weight decay 0.01, cosine decay, and gradient clipping at 1. Each epoch validates on a fixed uniform sample of 4,096 directed pairs per validation slide, averaging binary Brier equally over all measured slide–mechanism combinations. The lowest Brier checkpoint is saved. Early stopping uses patience 2, a minimum of three epochs, and improvement threshold 0.00001, with at most ten epochs.

Outputs are `outputs/trained/best.pt`, `config.json`, and `history.json`. Defaults reproduce the paper protocol; changing `--batch` or `--epochs` changes the experiment. The included head is the **paper checkpoint, epoch 3 (one-based)**, SHA256 `4871703f1fa58bf29cf6e4508cb03a1ec19d73f3d8dc35fac2d70bf236f96ecc`. Its exact original run configuration is in `models/training_config.json`; paths there describe the research run. The portable release uses `configs/model.yaml`. GPU/kernel differences can affect retraining numerics and the selected epoch.

## Evaluate the model

Prepare the held-out slides and evaluate the included paper checkpoint:

```bash
uv run --frozen python scripts/download_data.py --split test --skip-encoder --skip-segmentation
uv run --frozen python scripts/preprocess.py --split test --stage targets
uv run --frozen python scripts/preprocess.py --split test --stage embeddings
uv run --frozen python scripts/evaluate.py --split test
```

To evaluate a newly trained head, add `--checkpoint outputs/trained/best.pt --out outputs/trained_test`. For development validation, use `--split validation --out outputs/validation`. Validation was used for model selection and is not an independent test. `--samples NCBI876` restricts an evaluation to one held-out slide.

Evaluation uniformly samples up to **20,000 senders per slide**, using the original sample-specific seed, and evaluates each sampled sender with all its eligible neighboring receivers. `--senders 0` evaluates all senders; `--seed` defaults to 17. This benchmark sampling differs from `infer.py --sample-percent`, which constructs neighborhoods around sampled anchors.

Outputs:

- `metrics.csv`: per-slide/mechanism binary Brier, log loss, 20-bin ECE, AUROC, AP, AP excess (AP minus prevalence), macro sender AP excess against the exact random-ranking expectation, and direction accuracy.
- `summary.csv`: mechanism averages within each slide. Ranking metrics require at least five positive and five negative directed pairs; benchmark averages also require an informative sender. Brier, log loss, and ECE include all measured mechanisms.
- `groups.csv` and `overall.csv`: average slides within donor/source groups, then weight groups equally.
- `run.json`: checkpoint hash, sample list, sampling settings, and metric scope.

For direction accuracy, each unordered pair incident to a sampled sender is counted once; its reverse is additionally scored if necessary. Those added reverse rows do not enter ranking or probability metrics. Tied directions receive half credit. Scores remain uncalibrated; no calibrator is applied.

**Grouping correction:** NCBI885–888 are conservatively grouped as one GSE315411 TMA source because their donor/core relationships are unresolved. The full test cohort therefore has **26 source groups**, rather than the manuscript workbook's original 29. The workbook in `paper/` is retained as the original reference and has not been revised. New group summaries should not be presented as its original 29-group results. This repository currently implements the final model and its benchmark; baselines, ablations, lung-specific analyses, and survival analyses will be added separately.

See [verification notes](VERIFICATION.md) for checks performed and remaining full-scale reproduction work.

For offline compute nodes, download and sync on a networked node, then use `uv run --frozen --offline`. Training and UNI2-h feature extraction require substantial GPU time and storage; preprocessing streams the large HEST transcript files and TIFFs.

## Provenance and terms

`models/manifest.json` records checkpoint, encoder, catalog (including thresholds), cohort, original configuration, and research source hashes. The released checkpoint contains only the trained gated head; the UNI2-h encoder is fetched separately after individual access approval. The HEST dataset, UNI2-h model, and InstanSeg model retain their respective source terms. See their linked source pages before redistributing derived assets or using MORPHIC outside noncommercial research.
