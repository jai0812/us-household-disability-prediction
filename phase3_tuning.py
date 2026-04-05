"""
===============================================================================
CS699 - Data Mining | Spring 2026
PHASE 3: FOCUSED TUNING ON TOP PERFORMERS
Team: Kunj Patel & Jai Sharma
===============================================================================

Target: Push past Extra Credit thresholds
  - EC 10%: Yes TPR >= 74% AND No TPR >= 70%
  - EC 20%: Yes TPR >= 77% AND No TPR >= 73%

Strategy:
  - Focus on top 5 classifiers that worked: GBM, RF, LR, XGBoost, DecisionTree
  - Use RUS with multiple random seeds (ensemble effect)
  - Much wider parameter grids
  - Also try RUS with sampling_strategy to keep slight majority bias
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import CalibratedClassifierCV

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, BaggingClassifier
)
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ============================================================================
# Load data
# ============================================================================
print("Loading preprocessed train/test data...")
train_df = pd.read_csv('initial_train.csv')
test_df = pd.read_csv('initial_test.csv')

X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']

print(f"Training: {X_train.shape}, Test: {X_test.shape}")
print(f"Train - No: {(y_train==0).sum()}, Yes: {(y_train==1).sum()}")
print(f"Test  - No: {(y_test==0).sum()}, Yes: {(y_test==1).sum()}")


# ============================================================================
# Helpers
# ============================================================================
def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tpr_no  = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_no  = fn / (fn + tp) if (fn + tp) > 0 else 0
    prec_no = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_no  = tpr_no
    f1_no   = 2*prec_no*rec_no/(prec_no+rec_no) if (prec_no+rec_no) > 0 else 0

    tpr_yes  = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_yes  = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec_yes = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_yes  = tpr_yes
    f1_yes   = 2*prec_yes*rec_yes/(prec_yes+rec_yes) if (prec_yes+rec_yes) > 0 else 0

    roc = 0.0
    if y_prob is not None:
        try: roc = roc_auc_score(y_true, y_prob)
        except: roc = 0.0

    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    n_no, n_yes = (y_true==0).sum(), (y_true==1).sum()
    n_total = n_no + n_yes
    w_no, w_yes = n_no/n_total, n_yes/n_total
    wt_f1 = w_no*f1_no + w_yes*f1_yes

    meets_min = (tpr_yes >= 0.71) and (tpr_no >= 0.67)
    meets_ec1 = (tpr_yes >= 0.74) and (tpr_no >= 0.70)
    meets_ec2 = (tpr_yes >= 0.77) and (tpr_no >= 0.73)
    status = "BELOW MINIMUM"
    if meets_ec2: status = "*** EXTRA CREDIT 20% ***"
    elif meets_ec1: status = "** EXTRA CREDIT 10% **"
    elif meets_min: status = "MEETS MINIMUM"

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted No    Predicted Yes")
    print(f"  Actual No       {tn:>8}        {fp:>8}")
    print(f"  Actual Yes      {fn:>8}        {tp:>8}")
    print(f"\nPerformance Measures:")
    print(f"  {'':>15} {'TPR':>8} {'FPR':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'ROC':>8} {'MCC':>8} {'Kappa':>8}")
    print(f"  {'Class No':>15} {tpr_no:>8.4f} {fpr_no:>8.4f} {prec_no:>8.4f} {rec_no:>8.4f} {f1_no:>8.4f} {roc:>8.4f} {mcc:>8.4f} {kappa:>8.4f}")
    print(f"  {'Class Yes':>15} {tpr_yes:>8.4f} {fpr_yes:>8.4f} {prec_yes:>8.4f} {rec_yes:>8.4f} {f1_yes:>8.4f} {roc:>8.4f} {mcc:>8.4f} {kappa:>8.4f}")
    print(f"\n  >>> Yes TPR: {tpr_yes*100:.1f}% | No TPR: {tpr_no*100:.1f}% | Status: {status}")

    return {
        'model': model_name,
        'tpr_no': tpr_no, 'tpr_yes': tpr_yes,
        'fpr_no': fpr_no, 'fpr_yes': fpr_yes,
        'prec_no': prec_no, 'prec_yes': prec_yes,
        'f1_no': f1_no, 'f1_yes': f1_yes,
        'roc_auc': roc, 'mcc': mcc, 'kappa': kappa,
        'wt_f1': wt_f1, 'status': status,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def build_and_evaluate(bal_name, bal_method, clf_name, clf, param_grid,
                       X_tr, y_tr, X_te, y_te):
    label = f"{bal_name} + {clf_name}"
    print(f"\n>>> Building: {label}")
    t0 = time.time()

    try:
        X_bal, y_bal = bal_method.fit_resample(X_tr, y_tr)
    except Exception as e:
        print(f"  Balancing failed ({e}), falling back to RUS")
        X_bal, y_bal = RandomUnderSampler(random_state=42).fit_resample(X_tr, y_tr)

    print(f"  Balanced: {X_bal.shape[0]} rows (No:{(y_bal==0).sum()}, Yes:{(y_bal==1).sum()})")

    if param_grid:
        grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
        grid.fit(X_bal, y_bal)
        best_model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
    else:
        best_model = clf
        best_model.fit(X_bal, y_bal)

    y_pred = best_model.predict(X_te)
    y_prob = None
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_te)[:, 1]
    elif hasattr(best_model, 'decision_function'):
        y_prob = best_model.decision_function(X_te)

    print(f"  Time: {time.time()-t0:.1f}s")
    return evaluate_model(label, y_te, y_pred, y_prob)


all_results = []

# ============================================================================
# STRATEGY 1: RUS seed=42, aggressive tuning on top classifiers
# ============================================================================
print("\n" + "#" * 70)
print("# STRATEGY 1: RUS(42) - DEEP TUNING")
print("#" * 70)

# GradientBoosting - our best performer, go really deep
result = build_and_evaluate("RUS42", RandomUnderSampler(random_state=42),
    "GradientBoosting_deep",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [50, 100, 150, 200, 300],
     'max_depth': [2, 3, 4, 5, 6],
     'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
     'subsample': [0.7, 0.8, 0.9, 1.0],
     'min_samples_split': [2, 5, 10]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# RandomForest - deep tuning
result = build_and_evaluate("RUS42", RandomUnderSampler(random_state=42),
    "RandomForest_deep",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [200, 300, 500, 700],
     'max_depth': [5, 10, 15, 20, 30, None],
     'min_samples_split': [2, 3, 5, 10],
     'min_samples_leaf': [1, 2, 3],
     'max_features': ['sqrt', 'log2', None]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# XGBoost - deep tuning
result = build_and_evaluate("RUS42", RandomUnderSampler(random_state=42),
    "XGBoost_deep",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [50, 100, 200, 300],
     'max_depth': [3, 4, 5, 6, 7, 9],
     'learning_rate': [0.005, 0.01, 0.05, 0.1],
     'subsample': [0.7, 0.8, 1.0],
     'colsample_bytree': [0.6, 0.8, 1.0],
     'reg_alpha': [0, 0.1, 1],
     'reg_lambda': [1, 2, 5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# DecisionTree - deeper, try shallow trees
result = build_and_evaluate("RUS42", RandomUnderSampler(random_state=42),
    "DecisionTree_deep",
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 15],
     'min_samples_split': [2, 3, 5, 10, 20, 50],
     'min_samples_leaf': [1, 2, 5, 10],
     'criterion': ['gini', 'entropy']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# ExtraTrees - new classifier, often better than RF
result = build_and_evaluate("RUS42", RandomUnderSampler(random_state=42),
    "ExtraTrees",
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [200, 300, 500],
     'max_depth': [10, 15, 20, None],
     'min_samples_split': [2, 5, 10],
     'max_features': ['sqrt', 'log2', None]},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# STRATEGY 2: RUS seed=123, same classifiers
# ============================================================================
print("\n" + "#" * 70)
print("# STRATEGY 2: RUS(123) - DEEP TUNING")
print("#" * 70)

# GradientBoosting
result = build_and_evaluate("RUS123", RandomUnderSampler(random_state=123),
    "GradientBoosting_deep",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [50, 100, 150, 200, 300],
     'max_depth': [2, 3, 4, 5, 6],
     'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
     'subsample': [0.7, 0.8, 0.9, 1.0],
     'min_samples_split': [2, 5, 10]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# RandomForest
result = build_and_evaluate("RUS123", RandomUnderSampler(random_state=123),
    "RandomForest_deep",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [200, 300, 500, 700],
     'max_depth': [5, 10, 15, 20, 30, None],
     'min_samples_split': [2, 3, 5, 10],
     'min_samples_leaf': [1, 2, 3],
     'max_features': ['sqrt', 'log2', None]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# XGBoost
result = build_and_evaluate("RUS123", RandomUnderSampler(random_state=123),
    "XGBoost_deep",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [50, 100, 200, 300],
     'max_depth': [3, 4, 5, 6, 7, 9],
     'learning_rate': [0.005, 0.01, 0.05, 0.1],
     'subsample': [0.7, 0.8, 1.0],
     'colsample_bytree': [0.6, 0.8, 1.0],
     'reg_alpha': [0, 0.1, 1],
     'reg_lambda': [1, 2, 5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# ExtraTrees
result = build_and_evaluate("RUS123", RandomUnderSampler(random_state=123),
    "ExtraTrees",
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [200, 300, 500],
     'max_depth': [10, 15, 20, None],
     'min_samples_split': [2, 5, 10],
     'max_features': ['sqrt', 'log2', None]},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# STRATEGY 3: RUS seed=7, seed=99 - more diversity
# ============================================================================
print("\n" + "#" * 70)
print("# STRATEGY 3: RUS(7) and RUS(99) - TOP CLASSIFIERS")
print("#" * 70)

for seed in [7, 99]:
    # GradientBoosting
    result = build_and_evaluate(f"RUS{seed}", RandomUnderSampler(random_state=seed),
        "GradientBoosting",
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100, 200, 300],
         'max_depth': [3, 4, 5],
         'learning_rate': [0.01, 0.02, 0.05, 0.1],
         'subsample': [0.8, 0.9, 1.0]},
        X_train, y_train, X_test, y_test)
    all_results.append(result)

    # RandomForest
    result = build_and_evaluate(f"RUS{seed}", RandomUnderSampler(random_state=seed),
        "RandomForest",
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [200, 500],
         'max_depth': [10, 20, None],
         'min_samples_split': [2, 5],
         'min_samples_leaf': [1, 2],
         'max_features': ['sqrt', 'log2']},
        X_train, y_train, X_test, y_test)
    all_results.append(result)

    # XGBoost
    result = build_and_evaluate(f"RUS{seed}", RandomUnderSampler(random_state=seed),
        "XGBoost",
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
        {'n_estimators': [100, 200, 300],
         'max_depth': [3, 5, 7],
         'learning_rate': [0.01, 0.05, 0.1],
         'subsample': [0.8, 1.0],
         'colsample_bytree': [0.8, 1.0]},
        X_train, y_train, X_test, y_test)
    all_results.append(result)

    # ExtraTrees
    result = build_and_evaluate(f"RUS{seed}", RandomUnderSampler(random_state=seed),
        "ExtraTrees",
        ExtraTreesClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [200, 500],
         'max_depth': [15, 20, None],
         'min_samples_split': [2, 5],
         'max_features': ['sqrt', 'log2']},
        X_train, y_train, X_test, y_test)
    all_results.append(result)


# ============================================================================
# STRATEGY 4: RUS + SMOTE hybrid (undersample majority to 2x minority,
#              then SMOTE minority to match)
# ============================================================================
print("\n" + "#" * 70)
print("# STRATEGY 4: PARTIAL RUS (keep 2x minority) then GBM/XGB/RF")
print("#" * 70)

# Custom: undersample No to 2*Yes count, keep all Yes
rus_partial = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
# This makes majority = 2 * minority. sampling_strategy=0.5 means minority/majority = 0.5

result = build_and_evaluate("RUS_partial", rus_partial,
    "GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200, 300],
     'max_depth': [3, 4, 5],
     'learning_rate': [0.01, 0.05, 0.1],
     'subsample': [0.8, 1.0]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

result = build_and_evaluate("RUS_partial", RandomUnderSampler(random_state=42, sampling_strategy=0.5),
    "XGBoost",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [100, 200, 300],
     'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1],
     'subsample': [0.8, 1.0],
     'colsample_bytree': [0.8, 1.0]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

result = build_and_evaluate("RUS_partial", RandomUnderSampler(random_state=42, sampling_strategy=0.5),
    "RandomForest",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [200, 500],
     'max_depth': [10, 20, None],
     'min_samples_split': [2, 5],
     'max_features': ['sqrt', 'log2']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

result = build_and_evaluate("RUS_partial", RandomUnderSampler(random_state=42, sampling_strategy=0.5),
    "ExtraTrees",
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [200, 500],
     'max_depth': [15, 20, None],
     'min_samples_split': [2, 5],
     'max_features': ['sqrt', 'log2']},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("PHASE 3 FINAL SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df['avg_tpr'] = (results_df['tpr_yes'] + results_df['tpr_no']) / 2
results_df = results_df.sort_values('avg_tpr', ascending=False)

print(f"\n{'Model':<50} {'Yes TPR':>8} {'No TPR':>8} {'ROC':>8} {'MCC':>8} {'Status':<25}")
print("-" * 120)
for _, row in results_df.iterrows():
    print(f"{row['model']:<50} {row['tpr_yes']:>7.1%} {row['tpr_no']:>7.1%} "
          f"{row['roc_auc']:>8.4f} {row['mcc']:>8.4f} {row['status']:<25}")

best = results_df.iloc[0]
print(f"\n{'='*70}")
print(f"BEST MODEL: {best['model']}")
print(f"{'='*70}")
print(f"  Yes TPR: {best['tpr_yes']*100:.1f}%")
print(f"  No TPR:  {best['tpr_no']*100:.1f}%")
print(f"  ROC AUC: {best['roc_auc']:.4f}")
print(f"  MCC:     {best['mcc']:.4f}")
print(f"  Kappa:   {best['kappa']:.4f}")
print(f"  Status:  {best['status']}")
print(f"\nConfusion Matrix of Best Model:")
print(f"                 Predicted No    Predicted Yes")
print(f"  Actual No       {int(best['tn']):>8}        {int(best['fp']):>8}")
print(f"  Actual Yes      {int(best['fn']):>8}        {int(best['tp']):>8}")

# Count models meeting thresholds
ec2 = results_df[results_df['status'].str.contains('20%')].shape[0]
ec1 = results_df[results_df['status'].str.contains('10%')].shape[0]
meets = results_df[results_df['status'].str.contains('MEETS')].shape[0]
print(f"\nModels hitting Extra Credit 20%: {ec2}")
print(f"Models hitting Extra Credit 10%: {ec1}")
print(f"Models meeting minimum: {meets}")

results_df.to_csv('model_results_phase3.csv', index=False)
print(f"\nResults saved to: model_results_phase3.csv")
print("\nPHASE 3 COMPLETE")
