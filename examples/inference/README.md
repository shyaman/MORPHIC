# Small inference example

`image.tif`, `nuclei.tif`, and `annotations.csv` are synthetic fixtures. After `uv sync --frozen` and the authorized UNI2-h download, run from the repository root:

```bash
uv run --frozen python scripts/download_data.py --skip-hest --skip-segmentation
uv run --frozen python scripts/infer.py \
  --image examples/inference/image.tif --mpp 0.5 \
  --nuclei examples/inference/nuclei.tif \
  --annotations examples/inference/annotations.csv \
  --out outputs/example
```

Inspect `outputs/example/cells.parquet`, `edges.parquet`, and `scores.parquet`. The synthetic image tests the file and command contract; its scores have no biological interpretation. Omit `--nuclei` to run automatic segmentation on a real H&E image.
