# Data

This repository includes a **sample** of the Reactiva Perú loan certificate dataset for reproducibility.

## Included in the repo

- `sample/reactiva_peru_sample.csv` — 5,000 firm-level records extracted from the full public dataset.

## Not included in the repo

- The full dataset (~500k rows, ~53 MB) is stored locally under `data/raw/reactiva_peru.csv` and is ignored by git.

## Source

The data comes from the Reactiva Perú government-backed loan program (certificates emitted through October 2020). The original study used this dataset for credit-risk modelling with Lasso and Ridge regression.

If you need the full dataset, obtain it from the same public source used in the original paper and place it at:

```text
data/raw/reactiva_peru.csv
```

Then run the scripts from the repository root with the virtual environment activated.

## Header rows

The raw CSV contains two title rows before the column names. Scripts handle this via:

- `SAMPLE_HEADER_ROW = 0` for the sample file
- `CSV_HEADER_ROW = 2` for the full raw file
