"""
model_train.py — CreditLens Model Training Script
==================================================
Run this script to retrain all models with your current sklearn/xgboost versions
and regenerate the models/ directory (PKL files + results_summary.json + comparison_table.csv).

Usage:
    python model_train.py

Expected runtime: ~5–10 minutes depending on hardware.
"""

import numpy as np
import pandas as pd
import warnings, os, json, time
warnings.filterwarnings('ignore')

import sklearn
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder,
    PowerTransformer, FunctionTransformer
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

print("=" * 60)
print("CreditLens — Model Training Script")
print(f"scikit-learn : {sklearn.__version__}")
print("=" * 60)

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)

# ── 1. Load & preprocess data ─────────────────────────────────────────────────
print("\n[1/5] Loading data ...")
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'Credit_Card_Default.csv')

df = pd.read_csv(csv_path)

# Rename columns to match app.py expectations
df = df.rename(columns={
    'default.payment.next.month': 'DEFAULT',
    'SEX': 'GENDER'
})
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("[2/5] Engineering features ...")

df['UTIL_RATE'] = (df['BILL_AMT1'] / df['LIMIT_BAL'].replace(0, np.nan)).clip(-1, 2).fillna(0)
df['AVG_PAY_STATUS'] = df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].mean(axis=1)
df['TOTAL_BILL'] = df[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].sum(axis=1)
df['TOTAL_PAY']  = df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1)
df['PAY_RATIO']  = (df['TOTAL_PAY'] / df['TOTAL_BILL'].replace(0, np.nan)).clip(-2, 5).fillna(0)
df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
df['PAY_TREND']  = df['PAY_AMT1']  - df['PAY_AMT6']
df['MAX_PAY_DELAY'] = df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].max(axis=1)
df['CONSEC_LATE']   = (df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']] > 0).sum(axis=1)
# CREDIT_USAGE_RATIO is an alias of UTIL_RATE in app.py — not used in training features
# (app.py computes it on the fly; the 33-col input includes it but models were trained on 31)

FEATURES = [
    'LIMIT_BAL','GENDER','EDUCATION','MARRIAGE','AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
    'UTIL_RATE','PAY_RATIO','AVG_PAY_STATUS','TOTAL_BILL','TOTAL_PAY',
    'BILL_TREND','PAY_TREND','MAX_PAY_DELAY','CONSEC_LATE',
]

X = df[FEATURES].fillna(0)
y = df['DEFAULT'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

print(f"  Train : {len(X_train):,} rows  |  Default rate: {y_train.mean():.3%}")
print(f"  Test  : {len(X_test):,} rows   |  Default rate: {y_test.mean():.3%}")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# ── 3. Preprocessors ──────────────────────────────────────────────────────────
bill_cols       = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_amt_cols    = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
pay_status_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
num_base_cols   = ['LIMIT_BAL','AGE','UTIL_RATE','PAY_RATIO','TOTAL_BILL','TOTAL_PAY',
                   'AVG_PAY_STATUS','BILL_TREND','PAY_TREND','MAX_PAY_DELAY','CONSEC_LATE']
cat_cols        = ['GENDER','EDUCATION','MARRIAGE']

linear_preprocessor = ColumnTransformer([
    ('bill_amt',   PowerTransformer(method='yeo-johnson'),   bill_cols),
    ('pay_amt',    FunctionTransformer(np.log1p),            pay_amt_cols),
    ('num',        StandardScaler(),                         num_base_cols),
    ('pay_status', 'passthrough',                            pay_status_cols),
    ('cat',        OneHotEncoder(handle_unknown='ignore'),   cat_cols),
])

tree_preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

# ── 4. Train helpers ──────────────────────────────────────────────────────────
results_json = {}
all_rows     = []

def get_metrics(name, model, X_tr, X_te, y_tr, y_te, threshold=0.5):
    model.fit(X_tr, y_tr)
    tr_prob = model.predict_proba(X_tr)[:, 1]
    tr_pred = (tr_prob >= threshold).astype(int)
    te_prob = model.predict_proba(X_te)[:, 1]
    te_pred = (te_prob >= threshold).astype(int)
    return {
        'Model':          name,
        'Train Accuracy': round(accuracy_score(y_tr, tr_pred), 4),
        'Test Accuracy':  round(accuracy_score(y_te, te_pred), 4),
        'AUC':            round(roc_auc_score(y_te, te_prob),  4),
        'F1 Score':       round(f1_score(y_te, te_pred,        zero_division=0), 4),
        'Precision':      round(precision_score(y_te, te_pred, zero_division=0), 4),
        'Recall':         round(recall_score(y_te, te_pred,    zero_division=0), 4),
    }

def extract_feature_importances(model, feature_names):
    """
    Walk the pipeline to find feature_importances_ or coef_, then map them back
    to the original feature names (before ColumnTransformer expansion).

    Returns a dict {feature_name: importance} or {} if not available.
    """
    # Unwrap Pipeline / ImbPipeline to the final estimator
    estimator = model
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]   # last step of sklearn Pipeline
    elif hasattr(model, 'named_steps'):
        estimator = list(model.named_steps.values())[-1]

    # Get the preprocessor (first step) for feature-name reconstruction
    preprocessor = None
    if hasattr(model, 'steps') and len(model.steps) > 1:
        preprocessor = model.steps[0][1]
    elif hasattr(model, 'named_steps') and len(model.named_steps) > 1:
        preprocessor = list(model.named_steps.values())[0]

    # --- tree-based: feature_importances_ ---
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_

        # Try to recover transformed feature names from the ColumnTransformer
        if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
            try:
                transformed_names = list(preprocessor.get_feature_names_out())
                if len(transformed_names) == len(importances):
                    # Collapse one-hot columns back to their original feature name
                    # by summing importances that share the same base name
                    agg = {}
                    for fname, imp in zip(transformed_names, importances):
                        # sklearn names OHE cols as "cat__GENDER_1", "cat__GENDER_2", etc.
                        # strip the transformer prefix ("cat__", "remainder__", etc.)
                        parts = fname.split('__', 1)
                        base = parts[-1]                    # e.g. "GENDER_1"
                        root = base.split('_')[0]           # e.g. "GENDER"
                        # only collapse if root is a known original feature
                        key = root if root in feature_names else base
                        agg[key] = agg.get(key, 0.0) + float(imp)
                    return {k: round(v, 6) for k, v in
                            sorted(agg.items(), key=lambda x: -x[1])}
            except Exception:
                pass

        # Fallback: zip raw importances with original feature list (lengths may differ
        # if OHE expanded columns, so we only map what lines up)
        if len(importances) == len(feature_names):
            return {k: round(float(v), 6) for k, v in
                    sorted(zip(feature_names, importances), key=lambda x: -x[1])}
        # If lengths differ, return positional keys so at least something shows
        return {f'feature_{i}': round(float(v), 6)
                for i, v in enumerate(importances)}

    # --- linear: coef_ (logistic regression) ---
    if hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_[0])
        if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
            try:
                transformed_names = list(preprocessor.get_feature_names_out())
                if len(transformed_names) == len(coef):
                    agg = {}
                    for fname, imp in zip(transformed_names, coef):
                        parts = fname.split('__', 1)
                        base = parts[-1]
                        root = base.split('_')[0]
                        key = root if root in feature_names else base
                        agg[key] = agg.get(key, 0.0) + float(imp)
                    return {k: round(v, 6) for k, v in
                            sorted(agg.items(), key=lambda x: -x[1])}
            except Exception:
                pass
        if len(coef) == len(feature_names):
            return {k: round(float(v), 6) for k, v in
                    sorted(zip(feature_names, coef), key=lambda x: -x[1])}

    return {}


def save_bundle(model, display_name, fname, threshold):
    """Save PKL bundle, compute metrics and feature importances, return results dict."""
    te_prob = model.predict_proba(X_test)[:, 1]
    te_pred = (te_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, te_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, te_prob)
    bundle = {
        'model':     model,
        'features':  FEATURES,
        'threshold': threshold,
    }
    path = f'models/{fname}.pkl'
    joblib.dump(bundle, path)
    size_kb = os.path.getsize(path) / 1024
    auc_v = roc_auc_score(y_test, te_prob)
    f1_v  = f1_score(y_test, te_pred, zero_division=0)

    fi = extract_feature_importances(model, FEATURES)
    fi_status = f"  FI={len(fi)} features" if fi else "  FI=none"
    print(f"  ✅ {path:40s}  AUC={auc_v:.4f}  F1={f1_v:.4f}  ({size_kb:.0f} KB){fi_status}")

    result = {
        'auc':   round(float(auc_v), 6),
        'f1':    round(float(f1_v),  6),
        'fpr':   fpr.tolist(),
        'tpr':   tpr.tolist(),
        'prec':  prec_c.tolist(),
        'rec':   rec_c.tolist(),
        'cm':    confusion_matrix(y_test, te_pred).tolist(),
    }
    if fi:
        result['feature_importances'] = fi
    return result

print("\n[3/5] Training baseline models ...")

baseline_models = {
    'Baseline — Logistic Regression': Pipeline([
        ('prep',  linear_preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Baseline — Decision Tree': Pipeline([
        ('prep',  tree_preprocessor),
        ('model', DecisionTreeClassifier(max_depth=6, random_state=42))
    ]),
    'Baseline — Random Forest': Pipeline([
        ('prep',  tree_preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    'Baseline — Gradient Boosting': Pipeline([
        ('prep',  tree_preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ]),
    'Baseline — XGBoost': Pipeline([
        ('prep',  tree_preprocessor),
        ('model', XGBClassifier(n_estimators=300, learning_rate=0.05,
                                max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                eval_metric='logloss', random_state=42, verbosity=0))
    ]),
}

baseline_rows = []
for name, model in baseline_models.items():
    print(f"  Training: {name} ...")
    row = get_metrics(name, model, X_train, X_test, y_train, y_test, threshold=0.5)
    baseline_rows.append({**row, 'Group': '1. Baseline'})
    key = name.replace('Baseline — ', '').replace(' ', '_')
    results_json[name] = save_bundle(model, name, f'baseline_{key}', threshold=0.5)

print("\n[4/5] Training SMOTE & class-weight models ...")

smote_models = {
    'SMOTE — Logistic Regression': ImbPipeline([
        ('prep',  linear_preprocessor),
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'SMOTE — Random Forest': ImbPipeline([
        ('prep',  tree_preprocessor),
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    'SMOTE — XGBoost': ImbPipeline([
        ('prep',  tree_preprocessor),
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('model', XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0))
    ]),
}
weight_models = {
    'Weight — Logistic Regression': Pipeline([
        ('prep',  linear_preprocessor),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Weight — Random Forest': Pipeline([
        ('prep',  tree_preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                         random_state=42, n_jobs=-1))
    ]),
    'Weight — XGBoost': Pipeline([
        ('prep',  tree_preprocessor),
        ('model', XGBClassifier(scale_pos_weight=scale_pos_weight,
                                eval_metric='logloss', random_state=42, verbosity=0))
    ]),
}

imb_rows = []
for name, model in {**smote_models, **weight_models}.items():
    print(f"  Training: {name} ...")
    row = get_metrics(name, model, X_train, X_test, y_train, y_test, threshold=0.3)
    imb_rows.append({**row, 'Group': '2. Imbalance Handled'})
    key = name.replace(' — ', '_').replace(' ', '_')
    results_json[name] = save_bundle(model, name, key, threshold=0.3)

# ── Hyperparameter tuning ────────────────────────────────────────────────────
print("\n  Tuning SMOTE + XGBoost (RandomizedSearchCV, 20 iters) ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

smote_xgb_pipe = ImbPipeline([
    ('prep',  tree_preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0))
])
param_dist_smote = {
    'smote__sampling_strategy': [0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
    'model__n_estimators':      [50, 100, 200, 300, 500, 800],
    'model__max_depth':         [2, 3, 4, 6],
    'model__learning_rate':     [0.01, 0.05, 0.02, 0.03, 0.1],
    'model__subsample':         [0.5, 0.6, 0.7, 0.8],
    'model__colsample_bytree':  [0.7, 0.8, 0.5, 0.6, 0.9, 1.0],
}
search_smote = RandomizedSearchCV(
    smote_xgb_pipe, param_dist_smote,
    n_iter=20, scoring='f1', cv=cv, n_jobs=-1, random_state=42, verbose=0
)
search_smote.fit(X_train, y_train)
best_smote_xgb = search_smote.best_estimator_
print(f"  ✅ Best SMOTE XGB CV F1: {search_smote.best_score_:.4f}")

print("  Tuning Class Weight + XGBoost ...")
weight_xgb_pipe = Pipeline([
    ('prep',  tree_preprocessor),
    ('model', XGBClassifier(scale_pos_weight=scale_pos_weight,
                            eval_metric='logloss', random_state=42, verbosity=0))
])
param_dist_weight = {
    'model__n_estimators':     [100, 200, 300],
    'model__max_depth':        [3, 5, 7],
    'model__learning_rate':    [0.01, 0.05, 0.1],
    'model__subsample':        [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0],
    'model__gamma':            [0, 1, 5],
}
search_weight = RandomizedSearchCV(
    weight_xgb_pipe, param_dist_weight,
    n_iter=20, scoring='f1', cv=cv, n_jobs=-1, random_state=42, verbose=0
)
search_weight.fit(X_train, y_train)
best_weight_xgb = search_weight.best_estimator_
print(f"  ✅ Best Weight XGB CV F1: {search_weight.best_score_:.4f}")

tuned_rows = []
for name, model in [('Tuned SMOTE XGBoost', best_smote_xgb),
                     ('Tuned Weight XGBoost', best_weight_xgb)]:
    te_prob = model.predict_proba(X_test)[:, 1]
    te_pred = (te_prob >= 0.3).astype(int)
    tr_prob = model.predict_proba(X_train)[:, 1]
    tr_pred = (tr_prob >= 0.3).astype(int)
    tuned_rows.append({
        'Model': name, 'Group': '3. Tuned',
        'Train Accuracy': round(accuracy_score(y_train, tr_pred), 4),
        'Test Accuracy':  round(accuracy_score(y_test,  te_pred), 4),
        'AUC':            round(roc_auc_score(y_test,   te_prob), 4),
        'F1 Score':       round(f1_score(y_test, te_pred,        zero_division=0), 4),
        'Precision':      round(precision_score(y_test, te_pred, zero_division=0), 4),
        'Recall':         round(recall_score(y_test, te_pred,    zero_division=0), 4),
    })

results_json['Tuned SMOTE XGB']  = save_bundle(best_smote_xgb,  'Tuned SMOTE XGB',  'tuned_smote_xgb',  threshold=0.3)
results_json['Tuned Weight XGB'] = save_bundle(best_weight_xgb, 'Tuned Weight XGB', 'tuned_weight_xgb', threshold=0.3)

# ── 5. Save summary files ────────────────────────────────────────────────────
print("\n[5/5] Saving summary files ...")

all_results = pd.concat([
    pd.DataFrame(baseline_rows),
    pd.DataFrame(imb_rows),
    pd.DataFrame(tuned_rows),
], ignore_index=True)

all_results.to_csv('models/comparison_table.csv', index=False)
print("  ✅ models/comparison_table.csv saved")

with open('models/results_summary.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("  ✅ models/results_summary.json saved")

print("\n=== Saved Files ===")
for fname in sorted(os.listdir('models')):
    sz = os.path.getsize(f'models/{fname}')
    print(f"  models/{fname:45s} {sz/1024:8.1f} KB")

print(f"\n✅ All done! Models trained with scikit-learn {sklearn.__version__}")
print("   Restart your Streamlit app: streamlit run app.py")
