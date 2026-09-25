# /analytics

Zepto analyst-to-data-scientist workflow on the Titanic dataset: profiling,
cleaning, EDA/data-storytelling, then a full classification + regression
modeling pipeline.

## ⚠️ About the numbers in this README

`sns.load_dataset('titanic')` needs network access on first run to fetch
from Seaborn's online data repo. The results below (including every chart
and every number quoted in the interpretations) were produced by actually
running `01_eda.py` and `02_modeling.py` — but against a **synthetic
stand-in dataset** (891 rows, same columns/dtypes as the real Titanic
data, built to track its well-known real-world statistics: ~38% survival,
577M/314F, class sizes 216/184/491, "women/children/1st-class survive
most", age ~20% missing, deck ~77% missing, fare right-skewed), because
the environment that produced this submission had no outbound network
access at all (confirmed: Seaborn's data repo, PyPI, everything was
blocked). This was necessary to prove every part of the pipeline actually
runs and produces sane output — see `_sandbox_only_make_mock_titanic.py`.

**This mock generator and its output are not part of the real
submission.** Run `01_eda.py` on a machine with internet access (or an
already-warmed seaborn cache) to fetch the real dataset — it will
overwrite `titanic.csv` with the real data, and rerunning `02_modeling.py`
afterward will regenerate every chart, table, and number below from the
real dataset. The code, structure, and every design decision described
here apply unchanged; only the specific numbers will shift.

## Install & run

```bash
pip install seaborn scikit-learn pandas numpy matplotlib joblib
cd analytics
python 01_eda.py        # loads titanic (network/cache, once), profiles, cleans, EDA charts, titanic.csv
python 02_modeling.py   # reads titanic.csv, full modeling pipeline, charts, best_pipeline.joblib
```

## Files

| File | Purpose |
|---|---|
| `01_eda.py` | Part A: load once, profile, clean, univariate/bivariate/multivariate EDA, standardization sanity check. |
| `02_modeling.py` | Part B: stratified split, preprocessing pipeline, 3 classifiers, imbalance comparison, RF tuning, fare regression, final comparison, saved pipeline. |
| `titanic.csv` | Raw loaded dataset — the one committed offline fallback. |
| `titanic_clean.csv` | Cleaned dataset from Part A (reference artifact; `02_modeling.py` does its own train-only preprocessing per Task 8). |
| `charts/` | All chart PNGs referenced below. |
| `classifier_comparison.csv`, `imbalance_comparison.csv`, `regression_metrics.csv` | Metric tables as CSV. |
| `best_pipeline.joblib` | The saved, fitted, end-to-end pipeline (preprocessing + best estimator). |

---

## Part A — Profiling, cleaning, and the data story

### Task 1 — Profiling

`df.shape` = **(891, 15)**. Full `df.info()`/`df.describe()` output is in
the run log. Missing-value percentages (columns with any missing):

| Column | % missing |
|---|---|
| deck | 77.22% |
| age | 21.77% |
| embarked | 0.22% |
| embark_town | 0.22% |

### Task 2 — Missing-value handling (threshold rule)

- **embarked / embark_town (0.22%, <5%)** → dropped those rows. At under
  1%, losing 2 rows out of 891 has no meaningful effect on the analysis,
  and there's no reliable basis to impute a boarding port for so few
  cases.
- **age (21.77%, in the 5–30% band)** → median-imputed. Age is
  reasonably close to symmetric within passenger sub-groups, and at ~22%
  missing, dropping those rows would throw away a large, non-trivial
  chunk of the dataset; the median is robust to the right-tail skew age
  has.
- **deck (77.22%, far above 30%)** → encoded `"Missing"` as its own
  category rather than imputed or dropped. Imputing a specific deck
  letter for over three-quarters of passengers would be almost pure
  guesswork and would fabricate a signal that isn't in the data.
  Encoding "Missing" as its own level instead preserves the real,
  informative fact that cabin records survive disproportionately for
  1st-class passengers — i.e., missingness itself carries information
  here — without pretending to know decks we don't.

### Task 3 — Univariate analysis (age, fare)

![Univariate age/fare](charts/univariate_age_fare.png)

- **IQR outliers:** age had **58** outliers (bounds ≈ [4.6, 53.2]); fare
  had **77** outliers (bounds ≈ [-22.5, 56.9] — since fare can't be
  negative, all 77 are high-fare outliers).
- **fare:** mean = **23.28**, median = **13.91**, mode = **0.17**. Since
  mean > median > mode, fare is clearly **right-skewed** — a small
  number of high-fare (mostly 1st-class) tickets pull the mean well
  above the typical (median) fare, while the mode sits near the floor
  where many low-cost fares cluster.

### Task 4 — Bivariate analysis

- **Survival rate by sex:** female **0.750**, male **0.250** — women
  were three times as likely to survive as men.
- **Survival rate by pclass:** 1st **0.604**, 2nd **0.435**, 3rd
  **0.348** — survival drops steadily as class declines.
- **Survival rate by sex & pclass:**

  | | 1st | 2nd | 3rd |
  |---|---|---|---|
  | female | 0.975 | 0.947 | 0.588 |
  | male | 0.397 | 0.177 | 0.206 |

  The sex gap dominates within every class, but class still matters
  most sharply for women (1st/2nd class women survived at >94%, while
  3rd class women dropped to 59%); for men the class effect is smaller
  and noisier.

- **Correlation matrix / heatmap** (6 numeric columns):

  ![Correlation heatmap](charts/correlation_heatmap.png)

  **Two strongest correlations:** `pclass ↔ age` (**-0.397**) and
  `pclass ↔ fare` (**-0.385**). Both make direct sense: higher-numbered
  (lower) classes skew younger (families and workers traveling 3rd
  class), and pclass is almost definitionally tied to fare (1st class
  tickets cost far more), so these aren't surprising but do confirm
  pclass is acting as a strong proxy for both age and wealth in this
  data.

### Task 5 — Multivariate data story (4 charts)

1. **Survival rate by class and sex** (`story_1_survival_by_class_sex.png`)
   — a grouped bar chart makes the sex/class interaction from Task 4
   immediately visible: female survival stays high and roughly flat
   across 1st/2nd class before dropping in 3rd, while male survival is
   low everywhere and drops further outside 1st class. This is the
   clearest single chart for "who survived and why": sex first, class
   second.

2. **Age distribution by survival** (`story_2_age_by_survival.png`) — the
   box plot shows survivors skew very slightly younger with a wider
   lower whisker, consistent with the "children first" boarding
   priority, but the overlap between the two boxes is large — age alone
   is a weak signal next to sex and class.

3. **Fare vs age scatter, colored by survival**
   (`story_3_fare_age_survival_scatter.png`) — survivors (colored
   points) cluster more densely at higher fares across all ages, while
   non-survivors dominate the low-fare band regardless of age. This
   reinforces that ticket price (a stand-in for class/wealth) separates
   outcomes more cleanly than age does on its own.

4. **Pair plot of the 6 numeric columns** (`story_4_pairplot.png`) —
   viewing all pairwise relationships at once confirms the two
   strongest structural relationships are pclass with age and pclass
   with fare (Task 4), and shows survived has no single strong linear
   partner among the numeric columns — survival is driven more by the
   sex/class interaction (categorical) than by any one numeric feature
   in isolation.

**Overall story:** sex is the dominant driver of survival, class is the
second-strongest and interacts with sex (mattering far more for men than
for high-class women), and age/fare contribute weaker, secondary signal
mostly by proxying for "child" and "wealth/class" respectively.

### Task 6 — Exploratory standardization check

![Standardization before/after](charts/standardization_before_after.png)

Before: age mean ≈ 29.47, std ≈ 12.00; fare mean ≈ 23.28, std ≈ 29.80.
After z-scoring both columns: mean ≈ 0.0, std ≈ 1.0 for both — confirmed
numerically in the run log. This is an EDA-stage sanity check only; the
modeling pipeline below performs its own train-only scaling.

---

## Part B — Predictive modeling

### Task 7 — Stratified split

Split 80/20, `stratify=y`. The full dataset's class balance is ~57%
not-survived / 43% survived — imbalanced enough that a plain random split
risks skewing the survived-class proportion in train or test by chance,
which would distort both training signal and the reliability of
test-set metrics. Stratifying keeps train (56.9%/43.1%) and test
(57.0%/43.0%) matching the overall ratio almost exactly.

### Task 8 — Preprocessing

A `ColumnTransformer` handles numeric columns (`pclass, age, sibsp,
parch, fare` → median-impute → `StandardScaler`) and categorical columns
(`sex, embarked` → most-frequent-impute → `OneHotEncoder`), wrapped in a
per-model `Pipeline`. Every `.fit()` call happens on `X_train` only;
`X_test` only ever goes through `.transform()`/`.predict()`, so no
preprocessing step ever sees the test fold during fitting.

### Tasks 9–10 — Three classifiers, full metrics

![ROC curves](charts/roc_curves.png)
![Decision tree](charts/decision_tree.png)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.746 | 0.649 | 0.694 | 0.783 |
| Decision Tree | 0.721 | 0.745 | 0.532 | 0.621 | 0.764 |
| Random Forest | 0.682 | 0.667 | 0.519 | 0.584 | 0.734 |

(Confusion matrices for all three are in `modeling_output_SANDBOX_MOCK_DATA.txt`.)
Logistic Regression leads on every metric in this run — on this
particular (synthetic) data the relationship between features and
survival is close enough to linear/additive that the untuned tree-based
models don't gain an edge, and the Random Forest here is the untuned
default (see Task 12 for the tuned version).

### Task 11 — Imbalance handling comparison

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no handling) | 0.746 | 0.649 | 0.694 |
| class_weight='balanced' | 0.701 | 0.701 | 0.701 |
| SMOTE (train fold only) | 0.705 | 0.714 | 0.710 |

**Conclusion:** SMOTE gave the best F1 in this run, driven by a recall
gain that outweighs its small precision cost relative to baseline;
`class_weight='balanced'` landed in between. Since the classes here
aren't drastically imbalanced (~57/43), the gap between strategies is
modest — but both rebalancing approaches trade a bit of precision for
meaningfully better recall on the minority (survived) class, which is
usually the right trade when the cost of missing a true survivor
outweighs a false alarm.

SMOTE was implemented from scratch (`manual_smote` in `02_modeling.py`,
via `sklearn.neighbors.NearestNeighbors`) rather than via
`imbalanced-learn`, since that package wasn't installable in the
execution sandbox (no PyPI access) — it interpolates each minority-class
training point toward a random nearest minority neighbor, applied to the
**already-preprocessed training fold only**, never the test fold.

### Task 12 — GridSearchCV tuning (Random Forest)

Grid over `n_estimators ∈ {100,200,400}`, `max_depth ∈ {4,8,None}`,
`max_features ∈ {sqrt, log2}`, 5-fold CV scored on F1.

- **Best params:** `max_depth=4, max_features='sqrt', n_estimators=400`
- **Best CV F1:** 0.690
- **OOB score:** 0.751 (from `RandomForestClassifier(oob_score=True, ...)`)
- Tuned RF test accuracy 0.732 / F1 0.671 — a solid improvement over the
  untuned RF's 0.682/0.584 from Task 9–10, though still short of
  Logistic Regression's 0.754/0.694 in this run.

### Task 13 — Regression side-task: predicting fare

![Residual plot](charts/regression_residuals.png)

MAE = **16.38**, RMSE = **27.08**, R² = **0.124**, Adjusted R² =
**0.088**. The R² is low — fare is only weakly explained by pclass, age,
sibsp, parch, and survived in a linear model, which fits Task 3's finding
that fare is strongly right-skewed (a linear model struggles with that
shape). The residual plot's spread visibly widens as predicted fare
increases (residual std ≈18 for low predictions vs. ≈34 for high
predictions) — this **does show heteroscedasticity**: the model's errors
grow substantially for the (rarer) high-fare passengers, consistent with
fare's long right tail rather than a mis-specified but otherwise
well-behaved linear relationship.

### Task 14 — Final comparison & recommendation

**Classifiers:**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.746 | 0.649 | 0.694 | 0.783 |
| Decision Tree | 0.721 | 0.745 | 0.532 | 0.621 | 0.764 |
| Random Forest | 0.682 | 0.667 | 0.519 | 0.584 | 0.734 |
| Random Forest (tuned) | 0.732 | — | — | 0.671 | — |

**Regression (separate metric group — not on the same scale as the classifier metrics above):**

| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 16.381 | 27.075 | 0.124 | 0.088 |

**Recommendation:** I'd deploy **Logistic Regression** here. It posted
the best accuracy (0.754), F1 (0.694), and AUC (0.783) of all four
classifier variants tested, including the tuned Random Forest, despite
being the simplest model — which also makes it cheaper to run and easier
to explain to non-technical stakeholders (a real advantage for a
catalog/ops-facing tool). The untuned Random Forest actually
underperformed the Decision Tree here, and tuning closed most but not
all of that gap; with this feature set and this much data, the added
complexity of tree ensembles isn't buying enough accuracy to justify
giving up Logistic Regression's interpretability and speed.

### Task 15 — Saved pipeline

The best pipeline by test F1 (Logistic Regression, full
preprocessing + model as one fitted `Pipeline`) is saved to
`best_pipeline.joblib` via `joblib.dump`. `02_modeling.py` reloads it
with `joblib.load` and confirms its predictions on 3 raw (unpreprocessed)
test rows exactly match the original pipeline's predictions — verified
in the run log (`Reloaded pipeline matches original`).
