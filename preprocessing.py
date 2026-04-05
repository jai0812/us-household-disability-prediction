"""
===============================================================================
CS699 - Data Mining | Spring 2026
Preprocessing Script
Team: Kunj Patel & Jai Sharma
Dataset: 2023 American Housing Survey (AHS) - project_data.csv
Target: Class (Yes = household has at least one person with disability)
===============================================================================
This script performs all preprocessing steps on the raw dataset.
Output: preprocessed_data.csv, initial_train.csv, initial_test.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load the raw dataset
# ============================================================================
print("=" * 70)
print("STEP 1: Loading raw dataset")
print("=" * 70)

df = pd.read_csv('project_data.csv')
print(f"Raw dataset shape: {df.shape}")
print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

# ============================================================================
# STEP 2: Drop CONTROL column (unique ID - no predictive value)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Dropping CONTROL column (unique household ID)")
print("=" * 70)

df = df.drop(columns=['CONTROL'])
print(f"Shape after dropping CONTROL: {df.shape}")

# ============================================================================
# STEP 3: Drop rows with missing Class label
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Dropping rows with missing Class label")
print("=" * 70)

print(f"Class distribution before dropping:")
print(df['Class'].value_counts(dropna=False))
missing_class_count = df['Class'].isnull().sum()
print(f"\nRows with missing Class label: {missing_class_count}")

df = df.dropna(subset=['Class'])
print(f"\nShape after dropping missing Class rows: {df.shape}")
print(f"Class distribution after dropping:")
print(df['Class'].value_counts())
print(f"Class No: {(df['Class'] == 'No').sum()} ({(df['Class'] == 'No').mean()*100:.1f}%)")
print(f"Class Yes: {(df['Class'] == 'Yes').sum()} ({(df['Class'] == 'Yes').mean()*100:.1f}%)")

# ============================================================================
# STEP 4: Separate features and target
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Separating features and target variable")
print("=" * 70)

X = df.drop(columns=['Class'])
y = df['Class']
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# STEP 5: Define variable types based on codebook analysis
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Classifying variables based on AHS codebook")
print("=" * 70)

# --- BINARY CATEGORICAL (55 variables) ---
# Yes/No questions stored as 1/2 codes. No mathematical meaning.
binary_cols = [
    'NOISE', 'DISHWASH', 'SOLAR', 'GARAGE', 'NOSTEP', 'CONDO',
    'NHQPCRIME', 'NHQPUBTRN', 'NHQRISK', 'NHQSCHOOL', 'NHQSCRIME',
    'NOWIRE',  # has 3 unique but treated as binary per codebook
    'CELLPHONE',  # reclassified below if needed
    'LANDLINE', 'PLUGS', 'PORCH', 'LEAKO', 'LEAKI', 'NOTOIL',
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

# --- MULTI-CATEGORY CATEGORICAL (51 variables) ---
# Multiple category codes with no natural ordering.
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

# --- TRUE NUMERIC (37 variables) ---
# Real quantities where the number itself is meaningful.
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

# Verify all 143 features are accounted for
all_classified = set(binary_cols + categorical_cols + numeric_cols)
all_features = set(X.columns)
missing_from_classification = all_features - all_classified
extra_in_classification = all_classified - all_features

print(f"Binary categorical variables: {len(binary_cols)}")
print(f"Multi-category categorical variables: {len(categorical_cols)}")
print(f"True numeric variables: {len(numeric_cols)}")
print(f"Total classified: {len(all_classified)}")
print(f"Total features in dataset: {len(all_features)}")

if missing_from_classification:
    print(f"\nWARNING - Not classified: {missing_from_classification}")
if extra_in_classification:
    print(f"\nWARNING - Extra (not in dataset): {extra_in_classification}")

# ============================================================================
# STEP 6: Impute missing values
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Imputing missing values")
print("=" * 70)

total_missing_before = X.isnull().sum().sum()
print(f"Total missing values before imputation: {total_missing_before}")

# Numeric columns: median imputation (justified by skewness analysis)
num_imputer = SimpleImputer(strategy='median')
# Only impute columns that exist in the dataframe
numeric_cols_present = [c for c in numeric_cols if c in X.columns]
X[numeric_cols_present] = num_imputer.fit_transform(X[numeric_cols_present])

# Binary categorical columns: mode imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
binary_cols_present = [c for c in binary_cols if c in X.columns]
X[binary_cols_present] = cat_imputer.fit_transform(X[binary_cols_present])

# Multi-category categorical columns: mode imputation
categorical_cols_present = [c for c in categorical_cols if c in X.columns]
X[categorical_cols_present] = cat_imputer.fit_transform(X[categorical_cols_present])

total_missing_after = X.isnull().sum().sum()
print(f"Total missing values after imputation: {total_missing_after}")

# ============================================================================
# STEP 7: Outlier handling using IQR capping (factor = 6)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Outlier capping using IQR method (factor=6)")
print("=" * 70)

total_capped = 0
for col in numeric_cols_present:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 6 * iqr
    upper_bound = q3 + 6 * iqr

    capped_lower = (X[col] < lower_bound).sum()
    capped_upper = (X[col] > upper_bound).sum()
    capped_count = capped_lower + capped_upper

    if capped_count > 0:
        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        total_capped += capped_count

print(f"Total outlier values capped: {total_capped}")
print(f"Rows preserved: {X.shape[0]} (no rows removed)")

# ============================================================================
# STEP 8: Encode target variable
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Encoding target variable (Yes=1, No=0)")
print("=" * 70)

y = y.map({'Yes': 1, 'No': 0})
print(f"Class 0 (No): {(y == 0).sum()}")
print(f"Class 1 (Yes): {(y == 1).sum()}")
print(f"Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

# ============================================================================
# STEP 9: Encode feature variables
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Encoding feature variables")
print("=" * 70)

# Binary columns: label encode to 0/1
for col in binary_cols_present:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

print(f"Binary columns label-encoded: {len(binary_cols_present)}")

# Multi-category columns: one-hot encoding (dummy variables)
X = pd.get_dummies(X, columns=categorical_cols_present, dtype=int)
print(f"Shape after one-hot encoding: {X.shape}")

# ============================================================================
# STEP 10: Z-score standardization for numeric columns
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: Z-score standardization of numeric columns")
print("=" * 70)

scaler = StandardScaler()
X[numeric_cols_present] = scaler.fit_transform(X[numeric_cols_present])
print(f"Standardized {len(numeric_cols_present)} numeric columns (mean=0, std=1)")

# ============================================================================
# STEP 11: Combine features and target, save preprocessed dataset
# ============================================================================
print("\n" + "=" * 70)
print("STEP 11: Saving preprocessed dataset")
print("=" * 70)

preprocessed_df = X.copy()
preprocessed_df['Class'] = y.values
preprocessed_df.to_csv('preprocessed_data.csv', index=False)
print(f"Preprocessed dataset saved: preprocessed_data.csv")
print(f"Final shape: {preprocessed_df.shape}")

# ============================================================================
# STEP 12: Train/Test split (70/30, stratified)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 12: Train/Test split (70% train, 30% test, stratified)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"\nTraining set class distribution:")
print(f"  Class 0 (No): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Class 1 (Yes): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
print(f"\nTest set class distribution:")
print(f"  Class 0 (No): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"  Class 1 (Yes): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

# Save train and test datasets
train_df = X_train.copy()
train_df['Class'] = y_train.values
train_df.to_csv('initial_train.csv', index=False)
print(f"\nTraining set saved: initial_train.csv")

test_df = X_test.copy()
test_df['Class'] = y_test.values
test_df.to_csv('initial_test.csv', index=False)
print(f"Test set saved: initial_test.csv")

# ============================================================================
# PREPROCESSING SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE - SUMMARY")
print("=" * 70)
print(f"  Raw dataset:           9,241 rows x 145 columns")
print(f"  After dropping CONTROL: 9,241 rows x 144 columns")
print(f"  After dropping no-label: {df.shape[0]} rows x 144 columns")
print(f"  After imputation:       0 missing values remaining")
print(f"  After outlier capping:  {total_capped} values capped, all rows preserved")
print(f"  After encoding:         {preprocessed_df.shape[0]} rows x {preprocessed_df.shape[1]} columns")
print(f"  Training set:           {train_df.shape[0]} rows")
print(f"  Test set:               {test_df.shape[0]} rows")
print(f"\nFiles created:")
print(f"  1. preprocessed_data.csv")
print(f"  2. initial_train.csv")
print(f"  3. initial_test.csv")
