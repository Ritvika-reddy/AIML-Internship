# Task 4 — Classification with Logistic Regression
**ElevateLabs AI/ML Internship**

## Objective
Build a binary classifier using Logistic Regression to predict whether a tumor is Malignant or Benign.

## Dataset
- **File:** `data.csv` — Breast Cancer Wisconsin Dataset
- **Rows:** 569 | **Features:** 30 numerical | **Target:** `diagnosis` (M/B)
- **Class distribution:** Benign=357 (62.7%), Malignant=212 (37.3%)
- **Missing values:** None ✅

## Tools Used
Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Steps Performed

### Step 1 — Load & Explore
- Loaded dataset, dropped `id` column (not useful)
- Checked class balance: slightly imbalanced (63% Benign, 37% Malignant)

### Step 2 — Preprocessing
- Encoded target: Malignant(M)=1, Benign(B)=0
- Train/test split: 80/20 with `stratify=y` to preserve class ratio
- Applied `StandardScaler` to all 30 features

### Step 3 — Train Logistic Regression
- Used `sklearn.linear_model.LogisticRegression` (max_iter=10000)
- Default solver: `lbfgs`

### Step 4 — Model Evaluation
Evaluated using confusion matrix, precision, recall, F1, and ROC-AUC.

### Step 5 — Threshold Tuning
Tested thresholds from 0.1 to 0.85. In a **medical context**, lowering the threshold reduces False Negatives (missed cancer) at the cost of more False Positives (unnecessary follow-ups).

### Step 6 — Visualizations (6 plots)

---

## Model Performance (Default Threshold = 0.5)

| Metric          | Value     |
|----------------|-----------|
| **Accuracy**    | **96.49%** |
| **ROC-AUC**     | **0.9960** |
| Precision (M)   | 0.9750    |
| Recall (M)      | 0.9286    |
| F1 Score (M)    | 0.9512    |
| True Positives  | 39 ✅     |
| False Negatives | 3 ❌ (Missed cancer — most critical) |
| False Positives | 1 ⚠️ (False alarm) |
| True Negatives  | 71 ✅     |

## Confusion Matrix Breakdown

|              | Predicted Benign | Predicted Malignant |
|-------------|-----------------|-------------------|
| **Actual Benign**    | 71 (TN) | 1 (FP) |
| **Actual Malignant** | 3 (FN)  | 39 (TP) |

## Threshold Tuning Results

| Threshold | Recall | FN (Missed Cancer) | F1     |
|-----------|--------|-------------------|--------|
| 0.5       | 0.9286 | 3                 | 0.9512 |
| **0.3 (medical)** | **0.9762** | **1** | **0.9762** |
| 0.25      | 0.9762 | 1                 | 0.9762 |

> **Key insight:** Lowering threshold from 0.5 → 0.3 reduces missed cancers from 3 → 1 with minimal cost.

## Top Influential Features

| Feature              | Effect on Malignant Probability |
|---------------------|---------------------------------|
| texture_worst        | Strong positive (+1.43)         |
| concave points_worst | Strong positive                 |
| radius_worst         | Strong positive                 |
| fractal_dimension_se | Negative (reduces probability)  |

---

## Charts

| File | Description |
|------|-------------|
| `clf_01_sigmoid_probdist.png` | Sigmoid function + predicted probability distribution |
| `clf_02_confusion_matrices.png` | Confusion matrices at threshold 0.5 and 0.3 |
| `clf_03_roc_curve.png` | ROC curve with AUC = 0.9960 |
| `clf_04_precision_recall.png` | Precision-Recall curve |
| `clf_05_threshold_tuning.png` | Precision/Recall/F1 vs Threshold |
| `clf_06_feature_coefficients.png` | Top 15 feature coefficients |

---


