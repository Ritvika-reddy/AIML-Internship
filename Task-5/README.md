# Task 5 — Decision Trees and Random Forests
**ElevateLabs AI/ML Internship**

## Objective
Implement Decision Tree and Random Forest classifiers to predict heart disease, understand overfitting, ensemble learning, and feature importance.

## Dataset
- **File:** `heart.csv` — Heart Disease Dataset
- **Rows:** 1025 | **Features:** 13 | **Target:** `target` (0=No Disease, 1=Heart Disease)
- **Class balance:** 51.3% Heart Disease (nearly balanced ✅)
- **Missing values:** None ✅

## Tools Used
Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Steps Performed

### Step 1 — Load & Explore
- Loaded dataset, confirmed balanced classes and zero nulls
- 13 features including age, chest pain type, cholesterol, max heart rate, etc.

### Step 2 — Preprocessing
- Stratified 80/20 train-test split (preserves class ratio)
- No scaling needed for tree-based models (trees are invariant to feature scale)

### Step 3 — Overfitting Analysis (Depth Sweep)
- Trained Decision Trees at depths 1–15
- Plotted train vs test accuracy to find the sweet spot before overfitting
- **Best depth: 9** (98.54% test accuracy in sweep)

### Step 4 — Decision Tree (max_depth=9, criterion=entropy)
- Trained final Decision Tree, visualized the full tree
- Printed top 3 levels of decision rules in text format

### Step 5 — Random Forest (100 trees)
- Trained Random Forest — an ensemble of 100 decision trees (bagging)
- Each tree trained on a random bootstrap sample with random feature subset

### Step 6 — Cross-Validation (5-Fold Stratified)
- Fair evaluation that uses all data for both training and testing
- Reveals true generalization performance

### Step 7 — Feature Importances
- Compared which features each model considers most important
- RF importances are averaged over 100 trees → more reliable

---

## Model Performance

| Metric            | Decision Tree   | Random Forest   |
|------------------|----------------|----------------|
| Test Accuracy     | 100.00%        | 100.00%        |
| **CV Mean**       | **99.02%**     | **99.61%**     |
| CV Std Dev        | ±1.27%         | **±0.78%**     |
| Tree Depth        | 9              | 100 trees      |

> Note: 100% test accuracy is due to some duplicate rows in this version of the dataset.
> Cross-validation (99%+) gives a more realistic performance estimate.
> **RF has lower variance (±0.78% vs ±1.27%)** — key advantage of ensemble methods.

## Feature Importances (Top 5 by Random Forest)

| Rank | Feature   | Description             | RF Importance |
|------|-----------|------------------------|---------------|
| 1    | cp        | Chest Pain Type         | 0.1421        |
| 2    | thalach   | Max Heart Rate Achieved | 0.1173        |
| 3    | ca        | Major Vessels (0-3)     | 0.1148        |
| 4    | oldpeak   | ST Depression           | 0.1126        |
| 5    | thal      | Thal                    | 0.0959        |

> Chest Pain Type is the most important predictor — makes strong clinical sense!

## Charts

| File | Description |
|------|-------------|
| `dt_01_tree_visualization.png` | Full decision tree diagram (depth=9) |
| `dt_02_overfitting_analysis.png` | Train vs test accuracy by tree depth |
| `dt_03_confusion_matrices.png` | Confusion matrices: DT vs RF |
| `dt_04_feature_importances.png` | Feature importances: DT vs RF (side by side) |
| `dt_05_cross_validation.png` | 5-fold CV scores: DT vs RF |
| `dt_06_comparison_entropy.png` | Accuracy comparison + Entropy vs Gini plot |
| `dt_07_rf_n_trees.png` | RF accuracy vs number of trees (diminishing returns) |

---

## Interview Questions — Quick Answers

1. **How does a decision tree work?** It splits data at each node by the feature that best separates the classes (highest information gain or lowest Gini). Continues until leaves are pure or max depth is reached. Follows a flowchart-like if-else structure.

2. **Entropy and Information Gain:**
   - **Entropy** = −Σ p·log₂(p) — measures impurity/uncertainty in a node (0=pure, 1=max mix)
   - **Information Gain** = Entropy(parent) − weighted average Entropy(children)
   - The split that maximizes IG is chosen at each node.

3. **Random Forest vs single tree:** RF trains many trees on random bootstrap samples (bagging) and random feature subsets. Final prediction = majority vote. This reduces variance, prevents overfitting, and gives much more stable predictions than a single tree.

4. **Overfitting & prevention:**
   - Overfitting = model learns training data too well, fails on new data
   - For trees: control `max_depth`, `min_samples_split`, `min_samples_leaf`
   - RF naturally reduces overfitting via ensemble averaging

5. **Bagging (Bootstrap Aggregating):** Train multiple models on different random subsets of training data (with replacement). Aggregate predictions by voting (classification) or averaging (regression). Reduces variance without increasing bias.

6. **Visualize a decision tree:** Use `sklearn.tree.plot_tree()` (matplotlib) or `export_graphviz()` + Graphviz. Shows splits, impurity, samples, and class distribution at each node.

7. **Feature importance:** Measures how much each feature reduces impurity across all splits it is used in, weighted by the number of samples. Higher = more important. RF averages this over all 100 trees → more reliable than single tree.

8. **Pros/Cons of Random Forests:**
   - ✅ Pros: High accuracy, handles missing values, no scaling needed, robust to outliers, gives feature importances, reduces overfitting
   - ❌ Cons: Slower to train/predict than single tree, less interpretable (black box), high memory usage, not ideal for very sparse data
