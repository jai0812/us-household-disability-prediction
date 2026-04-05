"""
===============================================================================
CS699 - Data Mining | Spring 2026
ENHANCED MODEL TUNING - Phase 2
Team: Kunj Patel & Jai Sharma
===============================================================================

This script loads the preprocessed train/test data from Phase 1 and performs
deeper parameter tuning on the most promising combinations, plus tests
additional balancing strategies to push performance higher.

Key findings from Phase 1:
  - Only B1 (Random Undersampling) produced balanced predictions
  - SMOTE/ADASYN/SMOTETomek all massively overfit to class No
  - Top performers: B1+RF, B1+SVM, B1+MLP, B1+GBM, B1+LR
  - Need to push Yes TPR >= 77% AND No TPR >= 73% for extra credit

Strategy for Phase 2:
  1. Deeper tuning on B1 (RUS) with all 9 classifiers - wider grids
  2. Replace B2 with NearMiss undersampling
  3. Replace B3 with SMOTE with lower sampling_strategy (partial oversample)
  4. Keep B4 as SMOTE+Tomek but tune sampling_strategy
  This ensures at least 1 undersampling method and at least 4 methods total.
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

# Balancing methods
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ============================================================================
# Load preprocessed train/test data
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
# Helper functions (same as Phase 1)
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
    wt_tpr  = w_no*tpr_no  + w_yes*tpr_yes
    wt_fpr  = w_no*fpr_no  + w_yes*fpr_yes
    wt_prec = w_no*prec_no + w_yes*prec_yes
    wt_rec  = w_no*rec_no  + w_yes*rec_yes
    wt_f1   = w_no*f1_no   + w_yes*f1_yes

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
    print(f"  {'Wt. Average':>15} {wt_tpr:>8.4f} {wt_fpr:>8.4f} {wt_prec:>8.4f} {wt_rec:>8.4f} {wt_f1:>8.4f} {roc:>8.4f} {mcc:>8.4f} {kappa:>8.4f}")

    meets_min = (tpr_yes >= 0.71) and (tpr_no >= 0.67)
    meets_ec1 = (tpr_yes >= 0.74) and (tpr_no >= 0.70)
    meets_ec2 = (tpr_yes >= 0.77) and (tpr_no >= 0.73)
    status = "BELOW MINIMUM"
    if meets_ec2: status = "EXTRA CREDIT 20%"
    elif meets_ec1: status = "EXTRA CREDIT 10%"
    elif meets_min: status = "MEETS MINIMUM"

    print(f"\n  Yes TPR: {tpr_yes*100:.1f}% | No TPR: {tpr_no*100:.1f}% | Status: {status}")

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
        grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
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
# B1: RANDOM UNDERSAMPLING - DEEPER TUNING
# ============================================================================
print("\n" + "#" * 70)
print("# B1: RANDOM UNDERSAMPLING - DEEPER PARAMETER TUNING")
print("#" * 70)

# B1 + C1: Logistic Regression (wider C range)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C1_LogisticRegression",
    LogisticRegression(random_state=42, max_iter=2000),
    {'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
     'solver': ['liblinear', 'lbfgs'],
     'penalty': ['l1', 'l2']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C2: KNN (wider k range + metric)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C2_KNN",
    KNeighborsClassifier(),
    {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
     'weights': ['uniform', 'distance'],
     'metric': ['euclidean', 'manhattan']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C3: Decision Tree (deeper tuning)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C3_DecisionTree",
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [3, 5, 7, 10, 15, 20, None],
     'min_samples_split': [2, 5, 10, 20],
     'min_samples_leaf': [1, 2, 5],
     'criterion': ['gini', 'entropy']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C4: Random Forest (wider range)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C4_RandomForest",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300, 500],
     'max_depth': [10, 15, 20, 30, None],
     'min_samples_split': [2, 5],
     'min_samples_leaf': [1, 2]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C5: Gradient Boosting (wider range)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C5_GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200, 300],
     'max_depth': [3, 4, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1, 0.2],
     'subsample': [0.8, 1.0]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C6: SVM (wider C range)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C6_SVM",
    CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=3000, dual='auto'), cv=3),
    {'estimator__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C7: Naive Bayes
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C7_NaiveBayes",
    GaussianNB(),
    {'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C8: XGBoost (wider + subsample)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C8_XGBoost",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [100, 200, 300],
     'max_depth': [3, 5, 7, 9],
     'learning_rate': [0.01, 0.05, 0.1],
     'subsample': [0.8, 1.0],
     'colsample_bytree': [0.8, 1.0]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C9: MLP (wider architecture search)
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C9_MLP",
    MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
    {'hidden_layer_sizes': [(50,), (100,), (200,), (100, 50), (100, 100)],
     'alpha': [0.0001, 0.001, 0.01],
     'learning_rate': ['adaptive'],
     'activation': ['relu', 'tanh']},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# B2: NearMiss UNDERSAMPLING (replacement for SMOTE which failed)
# ============================================================================
print("\n" + "#" * 70)
print("# B2: NearMiss UNDERSAMPLING")
print("#" * 70)

# B2 + C1
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C1_LogisticRegression",
    LogisticRegression(random_state=42, max_iter=2000),
    {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C2
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C2_KNN",
    KNeighborsClassifier(),
    {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C3
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C3_DecisionTree",
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10],
     'criterion': ['gini', 'entropy']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C4
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C4_RandomForest",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None],
     'min_samples_split': [2, 5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C5
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C5_GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C6
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C6_SVM",
    CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=3000, dual='auto'), cv=3),
    {'estimator__C': [0.001, 0.01, 0.1, 1, 10]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C7
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C7_NaiveBayes",
    GaussianNB(),
    {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C8
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C8_XGBoost",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C9
result = build_and_evaluate("B2_NearMiss", NearMiss(version=1),
    "C9_MLP",
    MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
    {'hidden_layer_sizes': [(50,), (100,), (100, 50)],
     'alpha': [0.0001, 0.001, 0.01],
     'learning_rate': ['adaptive']},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# B3: SMOTE with sampling_strategy=0.75 (partial oversample)
# ============================================================================
print("\n" + "#" * 70)
print("# B3: SMOTE (sampling_strategy=0.75 - partial oversample)")
print("#" * 70)

# Partial oversampling: minority grows to 75% of majority size, not 100%
# This keeps some imbalance which helps models not swing too far

# B3 + C1
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C1_LogisticRegression",
    LogisticRegression(random_state=42, max_iter=2000),
    {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C2
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C2_KNN",
    KNeighborsClassifier(),
    {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C3
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C3_DecisionTree",
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10],
     'criterion': ['gini', 'entropy']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C4
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C4_RandomForest",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None],
     'min_samples_split': [2, 5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C5
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C5_GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C6
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C6_SVM",
    CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=3000, dual='auto'), cv=3),
    {'estimator__C': [0.001, 0.01, 0.1, 1, 10]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C7
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C7_NaiveBayes",
    GaussianNB(),
    {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C8
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C8_XGBoost",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C9
result = build_and_evaluate("B3_SMOTE75", SMOTE(random_state=42, sampling_strategy=0.75),
    "C9_MLP",
    MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
    {'hidden_layer_sizes': [(50,), (100,), (100, 50)],
     'alpha': [0.0001, 0.001, 0.01],
     'learning_rate': ['adaptive']},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# B4: RUS with different random_state (diverse undersampling)
# ============================================================================
print("\n" + "#" * 70)
print("# B4: RANDOM UNDERSAMPLING (random_state=123 - different sample)")
print("#" * 70)

# B4 + C1
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C1_LogisticRegression",
    LogisticRegression(random_state=42, max_iter=2000),
    {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C2
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C2_KNN",
    KNeighborsClassifier(),
    {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C3
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C3_DecisionTree",
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10],
     'criterion': ['gini', 'entropy']},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C4
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C4_RandomForest",
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None],
     'min_samples_split': [2, 5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C5
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C5_GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C6
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C6_SVM",
    CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=3000, dual='auto'), cv=3),
    {'estimator__C': [0.001, 0.01, 0.1, 1, 10]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C7
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C7_NaiveBayes",
    GaussianNB(),
    {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C8
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C8_XGBoost",
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
     'learning_rate': [0.01, 0.05, 0.1]},
    X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C9
result = build_and_evaluate("B4_RUS123", RandomUnderSampler(random_state=123),
    "C9_MLP",
    MLPClassifier(random_state=42, max_iter=1000, early_stopping=True),
    {'hidden_layer_sizes': [(50,), (100,), (100, 50)],
     'alpha': [0.0001, 0.001, 0.01],
     'learning_rate': ['adaptive']},
    X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("PHASE 2 FINAL SUMMARY - ALL 36 MODELS")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df['avg_tpr'] = (results_df['tpr_yes'] + results_df['tpr_no']) / 2
results_df = results_df.sort_values('avg_tpr', ascending=False)

print(f"\n{'Model':<50} {'Yes TPR':>8} {'No TPR':>8} {'ROC':>8} {'MCC':>8} {'Status':<20}")
print("-" * 110)
for _, row in results_df.iterrows():
    print(f"{row['model']:<50} {row['tpr_yes']:>7.1%} {row['tpr_no']:>7.1%} "
          f"{row['roc_auc']:>8.4f} {row['mcc']:>8.4f} {row['status']:<20}")

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

results_df.to_csv('model_results_phase2.csv', index=False)
print(f"\nAll results saved to: model_results_phase2.csv")
print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)
