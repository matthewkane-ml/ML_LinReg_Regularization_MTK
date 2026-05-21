# Lasso Regression — Predicting County-Level Heart Disease

> Regression pipeline on US county-level sociodemographic and health data: heavy feature pruning from a wide dataset, SelectKBest pre-filtering, and a Lasso (L1-regularized) model that zeroes out weak predictors — demonstrating why regularization matters when features outnumber signal.

---

## Problem

Sociodemographic and health resource data has been collected at the county level across the United States (2018–2019). The goal is to determine whether there is a meaningful relationship between sociodemographic factors — poverty, education, age distribution, race, employment — and heart disease burden. The target is a raw count: total people per county diagnosed with heart disease.

## Dataset

- **Source:** US county-level demographic health dataset (2018–2019)
- **Target:** `Heart disease_number` — total heart disease cases per county (continuous)
- **Raw shape:** Wide dataset with ~50+ columns covering population demographics, income, education, employment, race distribution, and disease prevalence/counts

## EDA & Preprocessing Pipeline

**Step 1 — Aggressive column pruning** before any analysis:

| Category dropped | Reason |
|---|---|
| `Heart disease_prevalence`, CI bounds | Data leakage — derived from the target |
| Other disease raw counts & prevalence | Outcome variables, not predictors |
| All CI bound columns | Redundant with point estimates |
| Raw age/race population counts | % equivalents retained instead |
| Raw employment/poverty counts | Rate/% equivalents retained instead |

| Step | Action |
|---|---|
| Duplicates | None found |
| Null handling | Rows with null target dropped; remaining nulls filled with column median |
| Outlier capping | IQR method on TOT_POP, MEDHHINC_2018, GQ_ESTIMATES_2018 |
| String columns | Any remaining object columns (e.g. county name) dropped |
| Scaling | MinMaxScaler on all feature columns |
| Feature selection | SelectKBest (f_regression, k=15) — generous k because Lasso performs further pruning during training |
| Split | 80/20 train/test |

**Top predictors identified (by correlation with target):**

| Feature | Direction |
|---|---|
| TOT_POP (county population) | Strong positive — larger counties have more cases by definition |
| Obesity_prevalence | Positive |
| % Black-alone | Positive |
| PCTPOVALL_2018 (poverty rate) | Positive |
| 80+ y/o % of total pop | Positive |
| MEDHHINC_2018 (median household income) | Negative |
| Bachelor's degree % | Negative |
| Active Physicians per 100k | Negative |

## Model

**Two models compared:**

| Model | Description |
|---|---|
| `LinearRegression` | Baseline — no regularization |
| `Lasso(alpha=1.0)` | L1 regularization — shrinks weak coefficients to exactly zero |

Lasso is the right choice for this dataset because: (1) the feature set is wide relative to signal, (2) many sociodemographic indicators are correlated with each other, and (3) L1 regularization produces a sparse model that automatically identifies which predictors are genuinely useful.

The trained Lasso model is saved to `models/lasso_alpha-1.0.sav`.

## Key Takeaways

- **Population is not a feature — it's a confounder:** TOT_POP must be included because a county of 1 million will always have more heart disease cases than a county of 5,000 regardless of sociodemographic factors. Without it, the model learns county size, not health risk factors.
- **Leakage in wide datasets is everywhere:** Heart disease prevalence, CI bounds, and derived count columns all needed to be dropped before any analysis — they encode the answer rather than predicting it.
- **L1 vs L2 regularization:** Lasso (L1) drives weak coefficients to zero, producing a sparse, interpretable model. Ridge (L2) shrinks all coefficients but keeps them all. When the goal is to identify which sociodemographic factors matter, Lasso's sparsity is more useful.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_LinReg_Regularization_MTK.git
cd ML_LinReg_Regularization_MTK
pip install -r requirements.txt
python src/app.py
```

## What I'd Do Next

- Tune `alpha` with cross-validation (`LassoCV`) instead of using a fixed value of 1.0 — the optimal regularization strength depends on the data
- Engineer a **per-capita** target (`heart_disease_number / TOT_POP`) to study health risk factors independently of population size
- Compare Lasso against **ElasticNet** (L1 + L2 hybrid) to see whether combining both penalties improves out-of-sample R²

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
