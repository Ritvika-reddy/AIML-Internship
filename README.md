# 🤖 ElevateLabs AI & ML Internship — Task Portfolio
**Internship Duration:** 30 Days | **First 15 Days:** 6 Core ML Tasks | **Next 15 Days:** Project Phase

> This repository contains all completed tasks from the **ElevateLabs AI & ML Internship** program, covering the complete ML pipeline from data preprocessing to model building and evaluation.

---

## 📋 Table of Contents

| # | Task | Topic | Dataset | Result |
|---|------|--------|---------|--------|
| 1 | [Data Cleaning & Preprocessing](#task-1--data-cleaning--preprocessing) | Null handling, encoding, scaling | Titanic | 607 clean rows |
| 2 | [Exploratory Data Analysis](#task-2--exploratory-data-analysis-eda) | Statistics & visualizations | Titanic | 8 insight charts |
| 3 | [Linear Regression](#task-3--linear-regression) | Simple & Multiple LR | Housing Prices | R² = 0.65 |
| 4 | [Logistic Regression](#task-4--logistic-regression) | Binary classification | Breast Cancer | Accuracy = 96.49% |
| 5 | [Decision Trees & Random Forests](#task-5--decision-trees--random-forests) | Ensemble learning | Heart Disease | CV = 99.61% |
| 6 | [K-Nearest Neighbors](#task-6--k-nearest-neighbors-knn) | Instance-based learning | Iris | Accuracy = 96.67% |

---

## 🛠 Tools & Technologies Used

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly
- **Environment:** Jupyter Notebook

---

## Task 1 — Data Cleaning & Preprocessing

**Objective:** Learn how to clean and prepare raw data for Machine Learning.

**Dataset:** Titanic Dataset — 891 rows × 12 columns

### What Was Done
| Step | Action | Detail |
|------|--------|--------|
| 1 | Explored dataset | Checked shape, dtypes, nulls — found 3 columns with missing values |
| 2 | Handled missing values | Age → median (28.0), Embarked → mode ('S'), Cabin → dropped (77% missing) |
| 3 | Dropped irrelevant columns | Removed PassengerId, Name, Ticket |
| 4 | Encoded categorical features | Sex → Label Encoding, Embarked → One-Hot Encoding |
| 5 | Removed outliers | IQR method on Fare, SibSp, Parch |
| 6 | Feature scaling | StandardScaler on Age & Fare |

### Results
- **Final dataset:** 607 rows × 9 columns, zero missing values
- **Outlier removal:** 284 rows removed via IQR method

### Output Files
| File | Description |
|------|-------------|
| `Cleaning___Preprocessing.ipynb` | Main notebook |
| `titanic_cleaned.csv` | Final cleaned dataset |
| `boxplots_before.png` | Outlier visualization before removal |
| `boxplots_after.png` | Outlier visualization after removal |
| `correlation_heatmap.png` | Feature correlation heatmap |

---

## Task 2 — Exploratory Data Analysis (EDA)

**Objective:** Understand data using statistics and visualizations to uncover patterns, trends, and anomalies.

**Dataset:** Titanic Dataset (raw) — 891 rows × 12 columns

### What Was Done
| Chart File | What It Shows | Key Finding |
|-----------|---------------|-------------|
| `eda_01_survival_distribution.png` | Bar + Pie chart of survival | Only 38.4% survived |
| `eda_02_histograms.png` | Distributions of Age, Fare, SibSp, Parch | Fare skew = 4.79 (very right-skewed) |
| `eda_03_boxplots.png` | Numeric features split by survival | Survivors paid higher fares |
| `eda_04_categorical_survival.png` | Survival by Pclass, Sex, Embarkation | Female: 74.2% vs Male: 18.9% |
| `eda_05_age_analysis.png` | Age by survival + violin by class | 1st class passengers were older |
| `eda_06_correlation_heatmap.png` | Lower-triangle correlation matrix | Sex & Pclass most correlated with Survival |
| `eda_07_pairplot.png` | Pairwise feature relationships | Fare-Pclass strongly inversely correlated |
| `eda_08_fare_survival_class_gender.png` | Fare + gender survival breakdown | 1st class females: 96.8% survival |

### Key Insights
- **Gender effect:** Women had 74.2% survival vs 18.9% for men — "women and children first"
- **Class effect:** 1st class: 63% → 2nd: 47.3% → 3rd: 24.2%
- **Embarkation:** Cherbourg passengers had highest survival (55.4%)
- **Multicollinearity:** Pclass and Fare inversely correlated (−0.55)

---

## Task 3 — Linear Regression

**Objective:** Implement Simple and Multiple Linear Regression to predict house prices.

**Dataset:** Housing.csv — 545 rows × 13 columns | Target: `price` (₹)

### What Was Done
1. Preprocessed: binary yes/no → 1/0, furnishing → ordinal, StandardScaler applied
2. **Simple LR** — used only `area` as single feature
3. **Multiple LR** — used all 12 features after encoding and scaling
4. Evaluated with MAE, RMSE, R²; plotted residuals and checked multicollinearity

### Model Performance
| Metric | Simple LR | Multiple LR |
|--------|-----------|-------------|
| MAE (₹) | 1,474,748 | 979,680 |
| RMSE (₹) | 1,917,104 | 1,331,071 |
| **R² Score** | **0.2729** | **0.6495** |
| Features | 1 (area) | 12 (all) |

> Multiple LR improved R² by **+0.38** — adding more features significantly helps.

### Top Feature Coefficients (Multiple LR — Standardized)
| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| bathrooms | +523,153 | Strongest positive predictor |
| area | +519,288 | Larger area → higher price |
| airconditioning | +362,446 | AC adds significant premium |
| stories | +348,177 | More floors → higher price |
| bedrooms | +58,691 | Weakest effect |

### Output Files
`Linear_Regression.ipynb` + `lr_01_simple_regression.png` + `lr_02_multiple_regression.png` + `lr_03_feature_coefficients.png` + `lr_04_model_evaluation.png` + `lr_05_correlation_matrix.png`

---

## Task 4 — Logistic Regression

**Objective:** Build a binary classifier to predict whether a tumor is Malignant or Benign.

**Dataset:** Breast Cancer Wisconsin — 569 rows × 30 features | Target: `diagnosis` (M/B)

### What Was Done
1. Encoded: Malignant=1, Benign=0; applied StandardScaler on all 30 features
2. Trained `LogisticRegression` (max_iter=10000, solver=lbfgs)
3. Evaluated: confusion matrix, precision, recall, F1, ROC-AUC
4. Tuned threshold from 0.5 → 0.3 for medical use case (fewer missed cancers)
5. Plotted sigmoid function, ROC curve, PR curve, feature coefficients

### Model Performance
| Metric | Value |
|--------|-------|
| **Accuracy** | **96.49%** |
| **ROC-AUC** | **0.9960** |
| Precision (Malignant) | 0.9750 |
| Recall (Malignant) | 0.9286 |
| F1 Score | 0.9512 |
| FN at threshold 0.5 | 3 missed cancers ❌ |
| FN at threshold 0.3 | **1 missed cancer ✅** |

### Confusion Matrix (t=0.5)
|  | Predicted Benign | Predicted Malignant |
|--|-----------------|-------------------|
| Actual Benign | 71 (TN) | 1 (FP) |
| Actual Malignant | 3 (FN) | 39 (TP) |

### Output Files
`Logistic_Regression.ipynb` + 6 chart PNGs (`clf_01` to `clf_06`)

---

## Task 5 — Decision Trees & Random Forests

**Objective:** Implement tree-based models, understand overfitting, ensemble learning, and feature importance.

**Dataset:** Heart Disease Dataset — 1025 rows × 13 features | Target: `target` (0/1)

### What Was Done
1. Depth sweep (1–15): studied overfitting via train vs test accuracy gap
2. Trained final Decision Tree (max_depth=9, criterion=entropy)
3. Trained Random Forest (100 trees, bagging)
4. 5-fold stratified cross-validation for fair evaluation
5. Compared feature importances between DT and RF
6. Plotted: tree diagram, overfitting curve, confusion matrices, CV scores, Entropy vs Gini, n_trees vs accuracy

### Model Performance
| Metric | Decision Tree | Random Forest |
|--------|--------------|--------------|
| Test Accuracy | 100%* | 100%* |
| **CV Mean** | **99.02%** | **99.61%** |
| CV Std Dev | ±1.27% | **±0.78%** |

> *100% test accuracy due to duplicate rows. CV score is realistic. RF has lower variance = more stable.

### Top Features (Random Forest)
| Feature | Importance | Description |
|---------|-----------|-------------|
| cp | 0.1421 | Chest Pain Type — #1 predictor |
| thalach | 0.1173 | Max Heart Rate Achieved |
| ca | 0.1148 | Major Vessels (0-3) |
| oldpeak | 0.1126 | ST Depression |

### Output Files
`DecisionTrees___RandomForest.ipynb` + 7 chart PNGs (`dt_01` to `dt_07`)

---

## Task 6 — K-Nearest Neighbors (KNN)

**Objective:** Understand and implement KNN for multi-class classification using the Iris dataset.

**Dataset:** Iris Dataset — 150 rows × 4 features | Target: Species (3 classes, 50 each)

### What Was Done
1. Normalized features with StandardScaler (critical — KNN is distance-based)
2. K sweep (1–20): found optimal K = 1 (test), stable range K=7–13
3. Trained final KNN, evaluated accuracy, confusion matrix, classification report
4. Visualized decision boundaries on petal features
5. Analyzed feature variance as proxy for importance

### Model Performance
| Metric | Value |
|--------|-------|
| **Best K** | **1** (test accuracy), 7–13 (stable) |
| **Test Accuracy** | **96.67%** |
| Misclassifications | Only 1 (1 Virginica → Versicolor) |
| Iris-setosa | 100% precision & recall (perfectly separable) |

### Key Observations
- **Petal features** separate species far better than sepal features (clear in decision boundary plot)
- **PetalLengthCm** has highest variance (3.113) — most informative feature
- **Setosa** is completely linearly separable; Versicolor/Virginica have slight overlap

### Output Files
`KNN.ipynb` + `Iris.csv` + 5 chart PNGs (`plot1` to `plot5`)

---

## 📊 Overall Learning Summary

| Task | Core Concept | Key Takeaway |
|------|-------------|--------------|
| Task 1 | Data Preprocessing | Clean data is the foundation — preprocessing is 70% of ML work |
| Task 2 | Exploratory Data Analysis | EDA uncovers hidden patterns and guides all downstream decisions |
| Task 3 | Linear Regression | R² tells you variance explained; more features help but watch for multicollinearity |
| Task 4 | Logistic Regression | Accuracy alone is misleading — use Recall, F1, AUC for medical/imbalanced problems |
| Task 5 | Trees & Ensembles | Single trees overfit; ensembles (RF) generalize better with lower variance |
| Task 6 | KNN | Distance-based models are sensitive to scale — always normalize |

---
