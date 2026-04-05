"""
===============================================================================
CS699 - Data Mining | Spring 2026
COMPLETE PROJECT CODE (Preprocessing + Model Building + Evaluation)
Team: Kunj Patel & Jai Sharma
Dataset: 2023 American Housing Survey (AHS) - project_data.csv
Target: Class (Yes = household has at least one person with disability)
===============================================================================

This single file includes:
  - All preprocessing steps
  - Train/test split
  - 4 balancing methods (including 1 undersampling)
  - 9 classification algorithms
  - Parameter tuning via GridSearchCV (3-fold)
  - 36 model combinations, each built and tested individually
  - Confusion matrices and full performance metrics for each model
  - Final best model selection

Balancing Methods:
  B1 = Random Undersampling
  B2 = SMOTE (Synthetic Minority Oversampling)
  B3 = ADASYN (Adaptive Synthetic Sampling)
  B4 = SMOTE + Tomek Links (combined over/undersampling)

Classification Algorithms:
  C1 = Logistic Regression
  C2 = K-Nearest Neighbors (KNN)
  C3 = Decision Tree
  C4 = Random Forest
  C5 = Gradient Boosting (GBM)
  C6 = Support Vector Machine (SVM - LinearSVC)
  C7 = Naive Bayes (Gaussian)
  C8 = XGBoost
  C9 = Multi-Layer Perceptron (Neural Network)
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import CalibratedClassifierCV

# Balancing methods
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ============================================================================
# ========================== PREPROCESSING ===================================
# ============================================================================

print("=" * 70)
print("PART 1: PREPROCESSING")
print("=" * 70)

# --- STEP 1: Load raw dataset ---
print("\n--- Step 1: Loading raw dataset ---")
df = pd.read_csv('project_data.csv')
print(f"Raw dataset shape: {df.shape}")

# --- STEP 2: Drop CONTROL column (unique ID) ---
print("\n--- Step 2: Dropping CONTROL column ---")
df = df.drop(columns=['CONTROL'])
print(f"Shape: {df.shape}")

# --- STEP 3: Drop rows with missing Class ---
print("\n--- Step 3: Dropping rows with missing Class label ---")
print(f"Missing Class rows: {df['Class'].isnull().sum()}")
df = df.dropna(subset=['Class'])
print(f"Shape after: {df.shape}")
print(f"Class No: {(df['Class'] == 'No').sum()} ({(df['Class'] == 'No').mean()*100:.1f}%)")
print(f"Class Yes: {(df['Class'] == 'Yes').sum()} ({(df['Class'] == 'Yes').mean()*100:.1f}%)")

# --- STEP 4: Separate features and target ---
print("\n--- Step 4: Separating features and target ---")
X = df.drop(columns=['Class'])
y = df['Class']

# --- STEP 5: Define variable types from codebook ---
print("\n--- Step 5: Variable classification (from AHS codebook) ---")

# Binary categorical (55 vars): Yes/No coded as 1/2
binary_cols = [
    'NOISE', 'DISHWASH', 'SOLAR', 'GARAGE', 'NOSTEP', 'CONDO',
    'NHQPCRIME', 'NHQPUBTRN', 'NHQRISK', 'NHQSCHOOL', 'NHQSCRIME',
    'NOWIRE', 'LANDLINE', 'PLUGS', 'PORCH', 'LEAKO', 'LEAKI', 'NOTOIL',
    'PAINTPEEL', 'HHSPAN', 'SAMESEXHH', 'GRANDHH',
    'HOA', 'KITCHENS', 'MONOXIDE', 'SMOKALRM', 'SPRNKSYSTM',
    'FRIDGE', 'KITCHSINK', 'WASHER',
    'HOT', 'NOWAT', 'WALLCRACK', 'FLOORHOLE',
    'FNDCRUMB', 'ROOFSHIN', 'ROOFHOLE', 'ROOFSAG',
    'WALLSIDE', 'WALLSLOPE', 'WINBOARD', 'WINBROKE', 'WINBARS',
    'MOLDKITCH', 'MOLDBATH', 'MOLDBEDRM', 'MOLDBASEM',
    'MOLDLROOM', 'MOLDOTHER',
    'HHSEE', 'HHMEMRY', 'HHCARE', 'HHERRND',
    'HHLDASTHMA', 'HHLDASTHMAER'
]

# Multi-category categorical (56 vars): codes with no natural ordering
categorical_cols = [
    'DINING', 'LAUNDY', 'TENURE', 'SOGIRESP', 'NEARABAND',
    'NEARBARCL', 'NEARTRASH', 'INTLANG', 'OMB13CBSA',
    'HHSEX', 'HHMAR', 'HHCITSHP', 'MILHH', 'HHRACE', 'HHGRAD',
    'HHNATVTY', 'PARTNER', 'HSHLDTYPE', 'MULTIGEN', 'SAMEHHLD',
    'HHFNTVTY', 'HHMNTVTY', 'HHPRNTHOME', 'HHGEN',
    'HHSOGILGBT', 'HHSOGISO', 'HHSOGIG',
    'UFINROOMS', 'STORIES', 'LOTSIZE', 'FINROOMS', 'YRBUILT',
    'FOUNDTYPE', 'UNITFLOORS', 'UNITSIZE',
    'COOKTYPE', 'COOKFUEL', 'DRYER', 'SEWTYPE',
    'HOTWATER', 'HEATFUEL', 'FIREPLACE', 'ACPRIMARY', 'ACSECNDRY',
    'HEATTYPE', 'SUPP1HEAT', 'SUPP2HEAT',
    'COLD', 'RODENT', 'ROACH', 'SEWBREAK', 'FUSEBLOW',
    'ADEQUACY', 'UPKEEP', 'CELLPHONE', 'WATSOURCE'
]

# True numeric (32 vars): real quantities
numeric_cols = [
    'TOTROOMS', 'PERPOVLVL', 'RATINGHS', 'RATINGNH',
    'HHMOVE', 'NUMELDERS', 'NUMADULTS', 'NUMNONREL',
    'HHYNGKIDS', 'HHOLDKIDS', 'NUMVETS', 'NUMYNGKIDS',
    'NUMOLDKIDS', 'NUMSUBFAM', 'NUMSECFAM', 'NUMPEOPLE',
    'HHADLTKIDS', 'BEDROOMS', 'BATHROOMS',
    'ELECAMT', 'GASAMT', 'OILAMT', 'OTHERAMT',
    'TRASHAMT', 'WATERAMT', 'UTILAMT',
    'NUMASTHMAST', 'POVLVLINC', 'INSURAMT',
    'HINCP', 'FINCP', 'TOTHCAMT'
]

print(f"Binary: {len(binary_cols)}, Categorical: {len(categorical_cols)}, Numeric: {len(numeric_cols)}")
print(f"Total: {len(binary_cols) + len(categorical_cols) + len(numeric_cols)} (should be 143)")

# --- STEP 6: Impute missing values ---
print("\n--- Step 6: Imputing missing values ---")
print(f"Missing values before: {X.isnull().sum().sum()}")

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
X[binary_cols] = cat_imputer.fit_transform(X[binary_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

print(f"Missing values after: {X.isnull().sum().sum()}")

# --- STEP 7: Outlier capping (IQR x 6) ---
print("\n--- Step 7: Outlier capping (IQR factor=6) ---")
total_capped = 0
for col in numeric_cols:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 6 * iqr
    upper = q3 + 6 * iqr
    capped = ((X[col] < lower) | (X[col] > upper)).sum()
    total_capped += capped
    X[col] = X[col].clip(lower=lower, upper=upper)
print(f"Total values capped: {total_capped}")

# --- STEP 8: Encode target ---
print("\n--- Step 8: Encoding target variable ---")
y = y.map({'Yes': 1, 'No': 0})
print(f"Class 0 (No): {(y==0).sum()}, Class 1 (Yes): {(y==1).sum()}")

# --- STEP 9: Encode features ---
print("\n--- Step 9: Encoding feature variables ---")
for col in binary_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = pd.get_dummies(X, columns=categorical_cols, dtype=int)
print(f"Shape after encoding: {X.shape}")

# --- STEP 10: Z-score standardization ---
print("\n--- Step 10: Z-score standardization ---")
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# --- STEP 11: Save preprocessed data ---
print("\n--- Step 11: Saving preprocessed dataset ---")
preprocessed_df = X.copy()
preprocessed_df['Class'] = y.values
preprocessed_df.to_csv('preprocessed_data.csv', index=False)
print(f"Saved: preprocessed_data.csv ({preprocessed_df.shape})")

# --- STEP 12: Train/Test split ---
print("\n--- Step 12: Train/Test split (70/30, stratified) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training: {X_train.shape}, Test: {X_test.shape}")
print(f"Train - No: {(y_train==0).sum()}, Yes: {(y_train==1).sum()}")
print(f"Test  - No: {(y_test==0).sum()}, Yes: {(y_test==1).sum()}")

# Save initial train and test
train_df = X_train.copy(); train_df['Class'] = y_train.values
train_df.to_csv('initial_train.csv', index=False)
test_df = X_test.copy(); test_df['Class'] = y_test.values
test_df.to_csv('initial_test.csv', index=False)
print("Saved: initial_train.csv, initial_test.csv")


# ============================================================================
# ========================= MODEL BUILDING ===================================
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 2: MODEL BUILDING AND EVALUATION")
print("=" * 70)


# ---------- HELPER: Evaluate and print metrics ----------
def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    """Print confusion matrix and full performance table. Return metrics dict."""
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


# ---------- HELPER: Build, tune, evaluate ----------
def build_and_evaluate(bal_name, bal_method, clf_name, clf, param_grid,
                       X_tr, y_tr, X_te, y_te):
    """Apply balancing, GridSearchCV tuning, evaluate on test set."""
    label = f"{bal_name} + {clf_name}"
    print(f"\n>>> Building: {label}")
    t0 = time.time()

    try:
        X_bal, y_bal = bal_method.fit_resample(X_tr, y_tr)
    except Exception as e:
        print(f"  Balancing failed ({e}), falling back to SMOTE")
        X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

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
# Define classifiers and their parameter grids (used across all balancing)
# ============================================================================

classifiers = {
    'C1_LogisticRegression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    ),
    'C2_KNN': (
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']}
    ),
    'C3_DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10],
         'criterion': ['gini', 'entropy']}
    ),
    'C4_RandomForest': (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [100, 200], 'max_depth': [10, 20],
         'min_samples_split': [2, 5]}
    ),
    'C5_GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [100, 200], 'max_depth': [3, 5],
         'learning_rate': [0.05, 0.1]}
    ),
    'C6_SVM': (
        CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, dual='auto'), cv=3),
        {'estimator__C': [0.01, 0.1, 1, 10]}
    ),
    'C7_NaiveBayes': (
        GaussianNB(),
        {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
    ),
    'C8_XGBoost': (
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
        {'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
         'learning_rate': [0.05, 0.1]}
    ),
    'C9_MLP': (
        MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
        {'hidden_layer_sizes': [(100,), (50,)],
         'alpha': [0.001, 0.01], 'learning_rate': ['adaptive']}
    ),
}

# ============================================================================
# B1: RANDOM UNDERSAMPLING
# ============================================================================
print("\n\n" + "#" * 70)
print("# BALANCING METHOD 1: RANDOM UNDERSAMPLING")
print("#" * 70)

# B1 + C1
clf, params = classifiers['C1_LogisticRegression']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C1_LogisticRegression", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C2
clf, params = classifiers['C2_KNN']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C2_KNN", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C3
clf, params = classifiers['C3_DecisionTree']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C3_DecisionTree", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C4
clf, params = classifiers['C4_RandomForest']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C4_RandomForest", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C5
clf, params = classifiers['C5_GradientBoosting']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C5_GradientBoosting", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C6
clf, params = classifiers['C6_SVM']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C6_SVM", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C7
clf, params = classifiers['C7_NaiveBayes']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C7_NaiveBayes", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C8
clf, params = classifiers['C8_XGBoost']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C8_XGBoost", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B1 + C9
clf, params = classifiers['C9_MLP']
result = build_and_evaluate("B1_RUS", RandomUnderSampler(random_state=42),
    "C9_MLP", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# B2: SMOTE
# ============================================================================
print("\n\n" + "#" * 70)
print("# BALANCING METHOD 2: SMOTE")
print("#" * 70)

# B2 + C1
clf, params = classifiers['C1_LogisticRegression']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C1_LogisticRegression", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C2
clf, params = classifiers['C2_KNN']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C2_KNN", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C3
clf, params = classifiers['C3_DecisionTree']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C3_DecisionTree", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C4
clf, params = classifiers['C4_RandomForest']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C4_RandomForest", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C5
clf, params = classifiers['C5_GradientBoosting']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C5_GradientBoosting", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C6
clf, params = classifiers['C6_SVM']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C6_SVM", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C7
clf, params = classifiers['C7_NaiveBayes']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C7_NaiveBayes", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C8
clf, params = classifiers['C8_XGBoost']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C8_XGBoost", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B2 + C9
clf, params = classifiers['C9_MLP']
result = build_and_evaluate("B2_SMOTE", SMOTE(random_state=42),
    "C9_MLP", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# B3: ADASYN
# ============================================================================
print("\n\n" + "#" * 70)
print("# BALANCING METHOD 3: ADASYN")
print("#" * 70)

# B3 + C1
clf, params = classifiers['C1_LogisticRegression']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C1_LogisticRegression", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C2
clf, params = classifiers['C2_KNN']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C2_KNN", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C3
clf, params = classifiers['C3_DecisionTree']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C3_DecisionTree", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C4
clf, params = classifiers['C4_RandomForest']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C4_RandomForest", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C5
clf, params = classifiers['C5_GradientBoosting']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C5_GradientBoosting", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C6
clf, params = classifiers['C6_SVM']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C6_SVM", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C7
clf, params = classifiers['C7_NaiveBayes']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C7_NaiveBayes", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C8
clf, params = classifiers['C8_XGBoost']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C8_XGBoost", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B3 + C9
clf, params = classifiers['C9_MLP']
result = build_and_evaluate("B3_ADASYN", ADASYN(random_state=42),
    "C9_MLP", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# B4: SMOTE + TOMEK LINKS
# ============================================================================
print("\n\n" + "#" * 70)
print("# BALANCING METHOD 4: SMOTE + TOMEK LINKS")
print("#" * 70)

# B4 + C1
clf, params = classifiers['C1_LogisticRegression']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C1_LogisticRegression", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C2
clf, params = classifiers['C2_KNN']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C2_KNN", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C3
clf, params = classifiers['C3_DecisionTree']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C3_DecisionTree", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C4
clf, params = classifiers['C4_RandomForest']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C4_RandomForest", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C5
clf, params = classifiers['C5_GradientBoosting']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C5_GradientBoosting", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C6
clf, params = classifiers['C6_SVM']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C6_SVM", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C7
clf, params = classifiers['C7_NaiveBayes']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C7_NaiveBayes", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C8
clf, params = classifiers['C8_XGBoost']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C8_XGBoost", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)

# B4 + C9
clf, params = classifiers['C9_MLP']
result = build_and_evaluate("B4_SMOTETomek", SMOTETomek(random_state=42),
    "C9_MLP", clf, params, X_train, y_train, X_test, y_test)
all_results.append(result)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 70)
print("PART 3: FINAL SUMMARY - ALL 36 MODELS")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df['avg_tpr'] = (results_df['tpr_yes'] + results_df['tpr_no']) / 2
results_df = results_df.sort_values('avg_tpr', ascending=False)

print(f"\n{'Model':<45} {'Yes TPR':>8} {'No TPR':>8} {'ROC':>8} {'MCC':>8} {'Status':<20}")
print("-" * 100)
for _, row in results_df.iterrows():
    print(f"{row['model']:<45} {row['tpr_yes']:>7.1%} {row['tpr_no']:>7.1%} "
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

results_df.to_csv('model_results_summary.csv', index=False)
print(f"\nAll results saved to: model_results_summary.csv")
print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)
