"""
01_eda.py
---------
Part A of the module: load the Titanic dataset ONCE, profile it,
clean it, and build the full EDA story. Saves:
  - titanic.csv           (raw load, committed offline fallback)
  - titanic_clean.csv     (cleaned DataFrame, consumed by 02_modeling.py)
  - charts/*.png          (every chart referenced in the README)

Run this first; 02_modeling.py picks up from titanic.csv (the raw
fallback) and does its own train-only preprocessing per Task 8.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHARTS = "charts"
import os
os.makedirs(CHARTS, exist_ok=True)


# ---------------------------------------------------------------------------
# Load (exactly once). Try the network/cache loader first; fall back to the
# committed titanic.csv if the network is unavailable at run time -- this is
# the offline-fallback behavior the assignment explicitly asks for.
# ---------------------------------------------------------------------------
try:
    df = sns.load_dataset("titanic")
    df.to_csv("titanic.csv", index=False)
    print("Loaded fresh via sns.load_dataset('titanic') and saved titanic.csv")
except Exception as e:
    print(f"sns.load_dataset failed ({e}); falling back to committed titanic.csv")
    df = pd.read_csv("titanic.csv")

print("\n" + "=" * 70)
print("TASK 1 -- Profiling")
print("=" * 70)
print("\n--- df.info() ---")
df.info()
print("\n--- df.describe() ---")
print(df.describe(include="all").T)
print(f"\n--- df.shape --- \n{df.shape}")

missing_pct = (df.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\n--- Missing value % (columns with any missing) ---")
print(missing_pct)


# ---------------------------------------------------------------------------
# TASK 2 -- Missing value handling, per the threshold rule:
#   < 5%   missing -> drop those rows
#   5-30%  missing -> impute
#   very high (unreliable to impute) -> drop column OR encode "missing" as
#                                        its own category (justify)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 2 -- Missing value handling")
print("=" * 70)

clean = df.copy()

for col, pct in missing_pct.items():
    if col == "deck":
        # deck's missing rate is very high (~77-78%) -- imputing a cabin deck
        # for 3 out of 4 passengers would be almost entirely guesswork and
        # would manufacture a signal that isn't in the data. Encoding
        # "missing" as its own category preserves the (real) information
        # that deck is simply *unrecorded* for most 2nd/3rd class
        # passengers, which is itself informative (cabin records survive
        # mostly for 1st class), without pretending we know the actual deck.
        clean[col] = clean[col].astype(object).fillna("Missing")
        print(f"{col}: {pct}% missing (>>30%) -> encoded 'Missing' as its own category (see justification above)")
    elif pct < 5:
        before = len(clean)
        clean = clean.dropna(subset=[col])
        print(f"{col}: {pct}% missing (<5%) -> dropped {before - len(clean)} rows")
    elif 5 <= pct <= 30:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = clean[col].median()
            clean[col] = clean[col].fillna(fill_val)
            print(f"{col}: {pct}% missing (5-30%) -> median-imputed with {fill_val:.2f}")
        else:
            fill_val = clean[col].mode(dropna=True)[0]
            clean[col] = clean[col].fillna(fill_val)
            print(f"{col}: {pct}% missing (5-30%) -> mode-imputed with '{fill_val}'")
    else:
        clean[col] = clean[col].astype(object).fillna("Missing")
        print(f"{col}: {pct}% missing (>30%) -> encoded 'Missing' as its own category")

# embark_town mirrors embarked 1:1 -- if embarked got row-dropped above,
# embark_town's matching rows are already gone; nothing further to do.
print(f"\nShape after cleaning: {clean.shape}")


# ---------------------------------------------------------------------------
# TASK 3 -- Univariate analysis: age & fare
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 3 -- Univariate analysis (age, fare)")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for i, col in enumerate(["age", "fare"]):
    axes[0, i].hist(clean[col], bins=30, color="#4C72B0", edgecolor="white")
    axes[0, i].set_title(f"{col} -- histogram")
    axes[1, i].boxplot(clean[col], vert=False)
    axes[1, i].set_title(f"{col} -- box plot")
plt.tight_layout()
plt.savefig(f"{CHARTS}/univariate_age_fare.png", dpi=110)
plt.close()

def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lo) | (series > hi)).sum(), lo, hi

for col in ["age", "fare"]:
    n_out, lo, hi = iqr_outliers(clean[col])
    print(f"{col}: {n_out} IQR outliers (bounds: [{lo:.2f}, {hi:.2f}])")

fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode()[0]
print(f"\nfare -- mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
skew_note = "right-skewed" if fare_mean > fare_median > fare_mode else (
    "left-skewed" if fare_mean < fare_median < fare_mode else "roughly symmetric"
)
print(f"fare distribution: {skew_note} (mean {'>' if fare_mean > fare_median else '<='} median "
      f"{'>' if fare_median > fare_mode else '<='} mode)")


# ---------------------------------------------------------------------------
# TASK 4 -- Bivariate analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 4 -- Bivariate analysis")
print("=" * 70)

survival_by_sex = {}
for s in clean["sex"].unique():
    mask = clean["sex"] == s
    survival_by_sex[s] = clean.loc[mask, "survived"].mean()
print("Survival rate by sex:", {k: round(v, 3) for k, v in survival_by_sex.items()})

survival_by_class = {}
for c in sorted(clean["pclass"].unique()):
    mask = clean["pclass"] == c
    survival_by_class[c] = clean.loc[mask, "survived"].mean()
print("Survival rate by pclass:", {k: round(v, 3) for k, v in survival_by_class.items()})

survival_by_sex_class = {}
for s in clean["sex"].unique():
    for c in sorted(clean["pclass"].unique()):
        mask = (clean["sex"] == s) & (clean["pclass"] == c)
        survival_by_sex_class[(s, c)] = clean.loc[mask, "survived"].mean()
print("Survival rate by sex & pclass:")
for k, v in sorted(survival_by_sex_class.items()):
    print(f"  {k}: {v:.3f}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()
print("\nCorrelation matrix (6 numeric columns):")
print(corr.round(3))

# top-2 strongest off-diagonal correlations
pairs = []
for i, a in enumerate(corr_cols):
    for j, b in enumerate(corr_cols):
        if i < j:
            pairs.append((a, b, corr.loc[a, b]))
pairs.sort(key=lambda t: abs(t[2]), reverse=True)
print("\nTop 2 strongest correlations:")
for a, b, v in pairs[:2]:
    print(f"  {a} <-> {b}: {v:.3f}")

plt.figure(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/correlation_heatmap.png", dpi=110)
plt.close()


# ---------------------------------------------------------------------------
# TASK 5 -- Multivariate "data story" (>= 4 charts, each with interpretation
# printed here; full prose interpretation lives in README.md)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 5 -- Multivariate data story charts")
print("=" * 70)

# Chart 1: survival rate by sex & pclass (grouped bar)
plt.figure(figsize=(6, 4.5))
sns.barplot(data=clean, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by class and sex")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(f"{CHARTS}/story_1_survival_by_class_sex.png", dpi=110)
plt.close()

# Chart 2: age distribution by survival (box)
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=clean, x="survived", y="age")
plt.title("Age distribution by survival outcome")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/story_2_age_by_survival.png", dpi=110)
plt.close()

# Chart 3: fare vs age scatter, colored by survival
plt.figure(figsize=(6, 4.5))
sns.scatterplot(data=clean, x="age", y="fare", hue="survived", alpha=0.6)
plt.title("Fare vs age, colored by survival")
plt.tight_layout()
plt.savefig(f"{CHARTS}/story_3_fare_age_survival_scatter.png", dpi=110)
plt.close()

# Chart 4: pair plot of the numeric columns colored by survival
pp = sns.pairplot(clean[corr_cols], hue="survived", diag_kind="hist", corner=True)
pp.savefig(f"{CHARTS}/story_4_pairplot.png", dpi=110)
plt.close("all")

print("Saved 4 multivariate story charts to charts/ (story_1..story_4). "
      "Written interpretations for each are in README.md.")


# ---------------------------------------------------------------------------
# TASK 6 -- Exploratory z-score standardization check (age, fare) --
# EDA-stage sanity check only; NOT used by the modeling pipeline.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 6 -- Exploratory standardization check (age, fare)")
print("=" * 70)

before_stats = clean[["age", "fare"]].agg(["mean", "std"])
print("Before standardization:\n", before_stats)

z = clean[["age", "fare"]].apply(lambda s: (s - s.mean()) / s.std())
after_stats = z.agg(["mean", "std"])
print("\nAfter z-score standardization:\n", after_stats.round(6))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(clean["age"], bins=30, alpha=0.6, label="age (raw)")
axes[0].hist(clean["fare"], bins=30, alpha=0.6, label="fare (raw)")
axes[0].set_title("Before standardization")
axes[0].legend()
axes[1].hist(z["age"], bins=30, alpha=0.6, label="age (z)")
axes[1].hist(z["fare"], bins=30, alpha=0.6, label="fare (z)")
axes[1].set_title("After standardization (~mean 0, std 1)")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{CHARTS}/standardization_before_after.png", dpi=110)
plt.close()


# ---------------------------------------------------------------------------
# Save cleaned data for 02_modeling.py's own reference / consistency checks.
# (02_modeling.py re-reads the raw titanic.csv per the module's design and
# does its own train-only preprocessing -- this file is not required by it,
# but is kept as a convenient artifact of Part A.)
# ---------------------------------------------------------------------------
clean.to_csv("titanic_clean.csv", index=False)
print(f"\nSaved cleaned data to titanic_clean.csv ({clean.shape[0]} rows)")
print("\n01_eda.py complete.")
