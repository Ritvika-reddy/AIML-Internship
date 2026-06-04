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

## Interview Questions — Quick Answers

1. **Logistic vs Linear Regression:** Linear predicts a continuous value; Logistic predicts a probability (0–1) using the sigmoid function and is used for classification. Output is a class, not a number.

2. **Sigmoid function:** σ(z) = 1 / (1 + e⁻ᶻ). Maps any real number to (0,1). Used to convert the linear combination of features into a probability. At z=0, output=0.5.

3. **Precision vs Recall:**
   - **Precision** = TP / (TP+FP) → "Of all predicted Malignant, how many actually were?"
   - **Recall** = TP / (TP+FN) → "Of all actual Malignant, how many did we catch?"
   - In medical diagnosis, **Recall is more critical** (missing cancer = FN = dangerous).

4. **ROC-AUC curve:** ROC plots True Positive Rate vs False Positive Rate at every threshold. AUC = area under curve. AUC=0.9960 means the model almost perfectly separates classes. AUC=0.5 is random guessing.

5. **Confusion Matrix:** A 2×2 table showing TP, TN, FP, FN. Gives a complete picture of classification errors beyond just accuracy.

6. **Imbalanced classes:** Accuracy becomes misleading (e.g., 95% accuracy by always predicting majority). Solutions: use F1/AUC metrics, oversample minority (SMOTE), undersample majority, use class_weight='balanced'.

7. **Choosing threshold:** Depends on the cost of errors. In cancer detection, lower threshold (e.g., 0.3) is preferred to minimize FN (missed cancer). In spam detection, higher threshold avoids blocking legitimate emails. Use Precision-Recall or ROC curve to find optimal point.

8. **Multi-class Logistic Regression:** Yes — using One-vs-Rest (OvR) or Softmax (Multinomial). Scikit-learn handles this automatically with `multi_class='ovr'` or `'multinomial'`.
