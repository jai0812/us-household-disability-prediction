for now  i just have put all this here ,, we will update the changes as per time ,, i havnet organised but all files are up 

# CS699 Data Mining - Spring 2026 | Final Project

## Predicting Disability Status in American Households
**Dataset:** 2023 American Housing Survey (AHS) — U.S. Census Bureau

**Team:** Kunj Patel & Jai Sharma

---

## Project Overview
Classification project to predict whether a household has at least one person with a disability, using 144 features from the AHS survey data. We build and compare 36+ models using 4 balancing methods × 9 classifiers with hyperparameter tuning.

## Repository Structure
```
CS699-DataMining-Project/
│
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── project_data.csv       # Original raw dataset (9,241 × 145)
│   └── AHS_Dictionary_2023.pdf # AHS codebook (313 pages)
│
├── code/
│   ├── preprocessing.py       # Standalone preprocessing script
│   └── project_code.py        # Complete project code (preprocessing + 36 models)
│
├── outputs/                   # Generated after running project_code.py
│   ├── preprocessed_data.csv  # Dataset after all preprocessing
│   ├── initial_train.csv      # Training set (70%)
│   ├── initial_test.csv       # Test set (30%)
│   └── model_results_summary.csv  # All 36 model results
│
├── reports/
│   ├── Patel_Kunj_IntermediateReport.pdf  # Intermediate report
│   └── Patel_Sharma_Report.pdf            # Final report (to be created)
│
└── presentation/
    └── Patel_Sharma_Slides.pptx           # Presentation slides (to be created)
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the complete project
```bash
cd code/
cp ../data/project_data.csv .
python project_code.py
```
This will:
- Preprocess the raw data
- Save preprocessed_data.csv, initial_train.csv, initial_test.csv
- Build and evaluate all 36 models
- Print confusion matrices and performance metrics for each
- Save model_results_summary.csv
- Identify the best model

### 3. (Optional) Run preprocessing only
```bash
cd code/
cp ../data/project_data.csv .
python preprocessing.py
```

## Methodology

### Preprocessing Pipeline
1. Drop CONTROL column (unique ID)
2. Drop 1,206 rows with missing Class labels (vacant units + incomplete interviews)
3. Classify 143 features into 3 types using AHS codebook:
   - 55 binary categorical → label encode (0/1)
   - 56 multi-category categorical → one-hot encode
   - 32 true numeric → median imputation + IQR capping + z-score standardization
4. 70/30 stratified train/test split

### Balancing Methods
| Code | Method | Type |
|------|--------|------|
| B1 | Random Undersampling | Undersampling |
| B2 | SMOTE | Oversampling |
| B3 | ADASYN | Oversampling |
| B4 | SMOTE + Tomek Links | Combined |

### Classification Algorithms
| Code | Algorithm |
|------|-----------|
| C1 | Logistic Regression |
| C2 | K-Nearest Neighbors |
| C3 | Decision Tree |
| C4 | Random Forest |
| C5 | Gradient Boosting |
| C6 | SVM (LinearSVC) |
| C7 | Gaussian Naive Bayes |
| C8 | XGBoost |
| C9 | MLP Neural Network |

### Performance Targets
| Level | Yes TPR | No TPR |
|-------|---------|--------|
| Minimum | ≥ 71% | ≥ 67% |
| Extra Credit 10% | ≥ 74% | ≥ 70% |
| Extra Credit 20% | ≥ 77% | ≥ 73% |

## Timeline
- [x] Intermediate report: 2/25
- [ ] Final report due: 4/8
- [ ] Presentation slides due: 4/15
- [ ] Presentation: 4/29
