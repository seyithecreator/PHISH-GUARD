import os
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not found — run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not found — run: pip install lightgbm")

from features import extract_features, FEATURE_NAMES
from patch_dataset import patch_dataset          # ← auto-patch before training


# ─────────────────────────────────────────────
# 1. PATCH DATASET (runs automatically)
#    Adds clean legitimate URLs to fix false
#    positives on major platforms like Google.
#    Safe to run multiple times — skips URLs
#    that are already present.
# ─────────────────────────────────────────────
print("🩹 Checking dataset for missing clean entries...")
patch_dataset("dataset.csv")
print()


# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("dataset.csv")
print(f"   Samples : {len(df)}")
print(f"   Classes :\n{df['label'].value_counts(normalize=True).round(3).to_string()}\n")


# ─────────────────────────────────────────────
# 3. EXTRACT FEATURES
# ─────────────────────────────────────────────
print("🔄 Extracting features...")
X = np.array([extract_features(url) for url in df['url']])
y = df['label'].values
print(f"   Feature matrix : {X.shape}\n")


# ─────────────────────────────────────────────
# 4. SPLIT → SCALE  (no data leakage)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 5. DEFINE MODELS
# ─────────────────────────────────────────────
models = {}

models["RandomForest"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

if HAS_XGB:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1,
        random_state=42
    )

if HAS_LGB:
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )


# ─────────────────────────────────────────────
# 6. CROSS-VALIDATED MODEL COMPARISON
# ─────────────────────────────────────────────
print("📊 Cross-validating models (5-fold)...\n")
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=cv, scoring='f1_weighted', n_jobs=-1
    )
    cv_results[name] = scores
    print(f"   {name:<20} F1: {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n🏆 Best single model: {best_model_name}\n")


# ─────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print(f"⚙️  Tuning {best_model_name} hyperparameters...")

param_grids = {
    "RandomForest": {
        'n_estimators':      [300, 500, 700],
        'max_depth':         [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt', 'log2'],
    },
    "XGBoost": {
        'n_estimators':     [300, 500, 700],
        'max_depth':        [4, 5, 6, 7],
        'learning_rate':    [0.01, 0.05, 0.1],
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha':        [0, 0.1, 0.5],
    },
    "LightGBM": {
        'n_estimators':  [300, 500, 700],
        'max_depth':     [4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves':    [31, 63, 127],
        'subsample':     [0.7, 0.8, 0.9],
        'reg_alpha':     [0, 0.1, 0.5],
    }
}

tuner = RandomizedSearchCV(
    models[best_model_name],
    param_grids[best_model_name],
    n_iter=30, cv=cv,
    scoring='f1_weighted',
    n_jobs=-1, random_state=42, verbose=0
)
tuner.fit(X_train_scaled, y_train)
tuned_model = tuner.best_estimator_
tuned_score = tuner.best_score_

print(f"   Best tuned F1 : {tuned_score:.4f}")
print(f"   Best params   : {tuner.best_params_}\n")

models[best_model_name]    = tuned_model
cv_results[best_model_name] = np.array([tuned_score] * 5)


# ─────────────────────────────────────────────
# 8. STACKING ENSEMBLE
# ─────────────────────────────────────────────
print("🔗 Building Stacking Ensemble...")
stacking_clf = StackingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    final_estimator=LogisticRegression(
        C=1.0, class_weight='balanced',
        max_iter=1000, random_state=42
    ),
    cv=5, n_jobs=-1, passthrough=True
)

stacking_scores = cross_val_score(
    stacking_clf, X_train_scaled, y_train,
    cv=cv, scoring='f1_weighted', n_jobs=-1
)
print(f"   Stacking Ensemble    F1: {stacking_scores.mean():.4f} ± {stacking_scores.std():.4f}")

if stacking_scores.mean() > cv_results[best_model_name].mean():
    print("✅ Using Stacking Ensemble as final model\n")
    final_model      = stacking_clf
    final_model_name = "StackingEnsemble"
else:
    print(f"✅ Using tuned {best_model_name} as final model\n")
    final_model      = models[best_model_name]
    final_model_name = best_model_name


# ─────────────────────────────────────────────
# 9. TRAIN ON FULL TRAINING SET
# ─────────────────────────────────────────────
print(f"🧠 Training {final_model_name} on full training set...")
final_model.fit(X_train_scaled, y_train)


# ─────────────────────────────────────────────
# 10. EVALUATE
# ─────────────────────────────────────────────
y_pred      = final_model.predict(X_test_scaled)
y_pred_prob = final_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred, average='weighted')
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print("\n" + "=" * 50)
print("        FINAL TEST SET RESULTS")
print("=" * 50)
print(f"  Accuracy :  {accuracy * 100:.2f}%")
print(f"  F1 Score :  {f1 * 100:.2f}%")
print(f"  ROC-AUC  :  {roc_auc * 100:.2f}%")
print("=" * 50)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ─────────────────────────────────────────────
# 11. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
if final_model_name != "StackingEnsemble":
    importances = final_model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]
    print(f"\n🔍 Top 15 Most Important Features:")
    for rank, idx in enumerate(top_idx, 1):
        fname = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        print(f"   {rank:>2}. {fname:<30} {importances[idx]:.4f}")


# ─────────────────────────────────────────────
# 12. SAVE
# ─────────────────────────────────────────────
joblib.dump(final_model, 'phish_model.pkl')
joblib.dump(scaler,      'scaler.pkl')
print(f"\n💾 Saved: phish_model.pkl + scaler.pkl")
print(f"🚀 Done! Final accuracy: {accuracy * 100:.2f}%")