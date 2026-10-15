# LendingClub Loan Default & Profitability Prediction — Baseline

Replication of Li & Han (Stanford): default classifiers, annualised-return (NAR) regressors, and a threshold-based loan selection strategy.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset (not included in the repo)
Download from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
Use the **accepted** file `accepted_2007_to_2018Q4.csv.gz` (no need to unzip it). The script keeps only 2012–2015 loans with a final status.

## Run
Clean baseline (post-origination columns removed — realistic, no leakage):
```bash
python baseline.py --data loans.csv.gz --out results_clean --sample 200000 --nn-iter 200
```
Paper-style run (keeps post-origination columns, i.e. leaky — for comparison only):
```bash
python baseline.py --data loans.csv.gz --out results_leaky --sample 200000 --nn-iter 200 --keep-leaky
```
Omit `--sample` for the full dataset (needs lots of RAM). Each run writes `metrics.json`, `roc_rf.png` and `investment_return_trend.png`.

## Results (200k-loan sample, test set)
| | Paper | Clean | Leaky |
|---|---|---|---|
| LogReg weighted F1 | ~0.88 | 0.69 | ~1.00 |
| RF Default F1 | 0.70 | 0.09 | 0.97 |
| Ridge test R² | 0.238 | 0.027 | 0.669 |
| RF-10 test R² | 0.315 | 0.029 | 0.721 |

Takeaway: the paper's scores are far above what application-time features give, and match models that use post-origination columns (`recoveries`, `total_rec_prncp`, `last_fico_range_*`, ...). The clean run is the honest baseline to improve.

## Notes / deviations from the paper
- NAR uses the standard annualisation `(payments/loan)^(365/D) - 1`; the paper prints the exponent as `1/(365/D)`.
- Imputation statistics are computed on the training split only.
- NN classifier uses Adam; RF regressor uses 100 trees (paper does not specify).

## For the optimiser
Compare against `results_clean/metrics.json` using the same `--sample 200000` and keep the clean setup.
