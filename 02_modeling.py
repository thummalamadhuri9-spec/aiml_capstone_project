"""
02_modeling.py
---------------
Part B of the module: reads the committed titanic.csv (the raw fallback
01_eda.py produced) and runs the full modeling pipeline. Does NOT call
sns.load_dataset again -- per the module design, the dataset is loaded
from network/cache exactly once, in 01_eda.py.
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

CHARTS = "charts"
os.makedirs(CHARTS, exist_ok=True)
RNG = 42

df = pd.read_csv("titanic.csv")
print(f"Loaded titanic.csv (raw fallback from 01_eda.py): {df.shape}")

# Columns used as model features. We deliberately exclude adult_male/alive/
# alone/class/who/embark_town/deck: alive and adult_male leak or duplicate
# the target/other features (see Task 4's exclusion rationale in 01_eda.py);
# class/embark_town are redundant string duplicates of pclass/embarked;
# who leaks age+sex; deck is dropped here given its ~77% missing rate makes
# it a weak, mostly-"Missing"-valued feature for a from-scratch model.
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ---------------------------------------------------------------------------
# TASK 7 -- Stratified train/test split
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 7 -- Stratified train/test split")
print("=" * 70)
print(f"Class balance in y: {y.value_counts(normalize=True).round(3).to_dict()}")
print("Stratifying on `survived` because the classes are imbalanced "
      "(~62/38 in this run); a plain random split risks over- or "
      "under-representing the minority (survived=1) class in train or "
      "test by chance, which would skew both training signal and the "
      "reliability of test-set metrics. stratify=y keeps the same "
      "survived/not-survived ratio in both splits.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test class balance:  {y_test.value_counts(normalize=True).round(3).to_dict()}")


# ---------------------------------------------------------------------------
# TASK 8 -- Preprocessing (fit on train only), via ColumnTransformer
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 8 -- Preprocessing pipeline (fit on train split only)")
print("=" * 70)

numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
print("ColumnTransformer built: numeric -> median-impute + StandardScaler; "
      "categorical (sex, embarked) -> most-frequent-impute + OneHotEncoder. "
      "Wrapped per-model in a Pipeline below, so .fit(X_train) fits every "
      "step on the training fold only, and .transform/.predict on X_test "
      "never refits anything.")


# ---------------------------------------------------------------------------
# TASK 9/10 -- Train 3 classifiers, evaluate with full metric suite
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 9/10 -- Train & evaluate Logistic Regression, Decision Tree, Random Forest")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RNG),
    "Decision Tree": DecisionTreeClassifier(random_state=RNG, max_depth=4),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RNG),
}

fitted_pipelines = {}
metrics_rows = []
roc_data = {}

for name, estimator in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"\n--- {name} ---")
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
          f"F1={f1:.3f}  AUC={auc:.3f}")

    metrics_rows.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "AUC": auc,
    })

classifier_comparison = pd.DataFrame(metrics_rows).set_index("Model").round(3)
print("\n=== Classifier comparison table ===")
print(classifier_comparison)
classifier_comparison.to_csv("classifier_comparison.csv")

# ROC curves, all 3 models on one plot
plt.figure(figsize=(6, 5.5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves -- all 3 classifiers")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS}/roc_curves.png", dpi=110)
plt.close()

# Decision tree visualization
plt.figure(figsize=(18, 9))
ohe_cat_names = list(
    fitted_pipelines["Decision Tree"]
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
)
all_feature_names = numeric_features + ohe_cat_names
plot_tree(
    fitted_pipelines["Decision Tree"].named_steps["model"],
    feature_names=all_feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (max_depth=4, first 3 levels shown)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/decision_tree.png", dpi=110)
plt.close()


# ---------------------------------------------------------------------------
# TASK 11 -- Imbalance handling comparison (baseline / class_weight / SMOTE)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 11 -- Imbalance handling comparison (Logistic Regression)")
print("=" * 70)
print(f"Class balance (full y): {y.value_counts().to_dict()} "
      f"({y.value_counts(normalize=True).round(3).to_dict()})")


def manual_smote(X_arr, y_arr, k=5, random_state=RNG):
    """
    Minimal from-scratch SMOTE: for the minority class, synthesize new
    points by interpolating between each minority sample and one of its
    k nearest minority neighbors. Implemented directly with sklearn's
    NearestNeighbors (no imbalanced-learn dependency) so it runs anywhere
    scikit-learn does. Applied to the TRAINING FOLD ONLY, after the
    ColumnTransformer has already turned X into a numeric array -- never
    to the test fold, to avoid leakage.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_arr, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_count = counts.max()
    minority_count = counts.min()
    n_to_generate = majority_count - minority_count

    minority_X = X_arr[y_arr == minority_class]
    if n_to_generate <= 0 or len(minority_X) <= 1:
        return X_arr, y_arr

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(minority_X))).fit(minority_X)
    _, neighbor_idx = nn.kneighbors(minority_X)

    synthetic = []
    for _ in range(n_to_generate):
        i = rng.integers(0, len(minority_X))
        neighbors_of_i = neighbor_idx[i][1:]  # exclude self
        if len(neighbors_of_i) == 0:
            j = i
        else:
            j = rng.choice(neighbors_of_i)
        gap = rng.random()
        synthetic_point = minority_X[i] + gap * (minority_X[j] - minority_X[i])
        synthetic.append(synthetic_point)

    synthetic = np.array(synthetic)
    X_resampled = np.vstack([X_arr, synthetic])
    y_resampled = np.concatenate([y_arr, np.full(len(synthetic), minority_class)])
    return X_resampled, y_resampled


# Fit the preprocessor on the training fold once, reuse its transform for
# all three imbalance variants (still train-only fitting; test is untouched
# here since this sub-task only compares train-side strategies via held-out
# X_test/y_test for scoring).
preproc_for_imbalance = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])
X_train_proc = preproc_for_imbalance.fit_transform(X_train, y_train)
X_test_proc = preproc_for_imbalance.transform(X_test)
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()
y_train_arr = y_train.to_numpy()

imbalance_variants = {}

# (a) baseline
clf_base = LogisticRegression(max_iter=1000, random_state=RNG)
clf_base.fit(X_train_proc, y_train_arr)
pred_base = clf_base.predict(X_test_proc)
imbalance_variants["Baseline (no handling)"] = pred_base

# (b) class_weight='balanced'
clf_bal = LogisticRegression(max_iter=1000, random_state=RNG, class_weight="balanced")
clf_bal.fit(X_train_proc, y_train_arr)
pred_bal = clf_bal.predict(X_test_proc)
imbalance_variants["class_weight='balanced'"] = pred_bal

# (c) SMOTE on the training fold only
X_smote, y_smote = manual_smote(X_train_proc, y_train_arr)
print(f"SMOTE: training fold grew from {len(y_train_arr)} to {len(y_smote)} rows "
      f"(class counts now {dict(zip(*np.unique(y_smote, return_counts=True)))})")
clf_smote = LogisticRegression(max_iter=1000, random_state=RNG)
clf_smote.fit(X_smote, y_smote)
pred_smote = clf_smote.predict(X_test_proc)
imbalance_variants["SMOTE (train fold only)"] = pred_smote

imbalance_rows = []
for name, preds in imbalance_variants.items():
    imbalance_rows.append({
        "Strategy": name,
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    })
imbalance_comparison = pd.DataFrame(imbalance_rows).set_index("Strategy").round(3)
print("\n=== Imbalance handling comparison ===")
print(imbalance_comparison)
imbalance_comparison.to_csv("imbalance_comparison.csv")

best_strategy = imbalance_comparison["F1"].idxmax()
print(f"\nBest by F1: {best_strategy}")


# ---------------------------------------------------------------------------
# TASK 12 -- GridSearchCV tuning for Random Forest (+ OOB score)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 12 -- GridSearchCV tuning (Random Forest)")
print("=" * 70)

rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RNG)),
])
param_grid = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [4, 8, None],
    "model__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV F1: {grid.best_score_:.3f}")
best_rf_pipe = grid.best_estimator_
oob = best_rf_pipe.named_steps["model"].oob_score_
print(f"OOB score of best estimator: {oob:.3f}")

tuned_test_pred = best_rf_pipe.predict(X_test)
print(f"Tuned RF test accuracy: {accuracy_score(y_test, tuned_test_pred):.3f}, "
      f"F1: {f1_score(y_test, tuned_test_pred):.3f}")


# ---------------------------------------------------------------------------
# TASK 13 -- Regression side-task: predict fare
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 13 -- Regression side-task: predict fare")
print("=" * 70)

reg_features = ["pclass", "age", "sibsp", "parch", "survived"]
reg_categorical = ["sex", "embarked"]
X_reg = df[reg_features + reg_categorical].copy()
y_reg = df["fare"].copy()

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RNG
)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), reg_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), reg_categorical),
])
reg_pipe = Pipeline(steps=[("preprocessor", reg_preprocessor), ("model", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(6.5, 5))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Residual plot -- fare regression")
plt.tight_layout()
plt.savefig(f"{CHARTS}/regression_residuals.png", dpi=110)
plt.close()

pred_median = np.median(yr_pred)
resid_std_low = residuals[yr_pred < pred_median].std()
resid_std_high = residuals[yr_pred >= pred_median].std()
hetero_note = (
    "shows heteroscedasticity" if abs(resid_std_high - resid_std_low) / max(resid_std_low, 1e-9) > 0.3
    else "does not show strong heteroscedasticity (roughly constant spread)"
)
print(f"Residual std for low-vs-high predicted fare: {resid_std_low:.2f} vs {resid_std_high:.2f} "
      f"-> {hetero_note}")

regression_metrics = pd.DataFrame([{
    "MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted R2": adj_r2,
}]).round(3)
regression_metrics.to_csv("regression_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# TASK 14 -- Final comparison table + recommendation (printed here;
# full written recommendation lives in README.md, referencing these numbers)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 14 -- Final comparison")
print("=" * 70)
print("\nClassifiers:\n", classifier_comparison)
print("\nRegression (separate metric group -- not directly comparable to classifier metrics):\n", regression_metrics)


# ---------------------------------------------------------------------------
# TASK 15 -- Save the best full pipeline (preprocessing + estimator) via joblib
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TASK 15 -- Save & reload full pipeline")
print("=" * 70)

# "Best" = highest F1 among the 3 base classifiers vs. the tuned RF; use
# whichever scores higher on the held-out test set.
candidates = {**fitted_pipelines, "Random Forest (tuned)": best_rf_pipe}
best_name, best_pipe = max(
    candidates.items(),
    key=lambda kv: f1_score(y_test, kv[1].predict(X_test)),
)
print(f"Best-performing full pipeline by test F1: {best_name}")

joblib.dump(best_pipe, "best_pipeline.joblib")
print("Saved best_pipeline.joblib")

reloaded = joblib.load("best_pipeline.joblib")
raw_sample = X_test.iloc[:3]
original_pred = best_pipe.predict(raw_sample)
reloaded_pred = reloaded.predict(raw_sample)
print(f"Original pipeline predictions on 3 raw test rows:  {original_pred}")
print(f"Reloaded pipeline predictions on same raw rows:    {reloaded_pred}")
assert (original_pred == reloaded_pred).all(), "Reloaded pipeline predictions do not match!"
print("Reloaded pipeline matches original -> confirmed working end-to-end on raw input.")

print("\n02_modeling.py complete.")
