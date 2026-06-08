# Task 6: K-Nearest Neighbors (KNN) Classification
**Elevate Labs AI & ML Internship**

## Objective
Understand and implement KNN for classification problems using the Iris dataset.

## Tools Used
- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## Process

### 1. Load & Explore Dataset
- Loaded `Iris.csv` (150 rows × 6 columns)
- 3 classes: Iris-setosa, Iris-versicolor, Iris-virginica (50 samples each)
- Dropped the `Id` column (non-feature)

### 2. Preprocessing
- Defined features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- Label-encoded the target (Species → 0, 1, 2)
- **Normalized features using StandardScaler** — essential for KNN since it's distance-based

### 3. Train-Test Split
- 80% training (120 samples), 20% testing (30 samples)
- Stratified split to preserve class balance

### 4. Experimenting with K Values (K = 1 to 20)
| Best K | Test Accuracy |
|--------|--------------|
| 1      | **96.67%**   |

Multiple K values (1, 7, 9, 10, 11, 12, 13, 17, 18, 19) achieved 96.67% accuracy.

### 5. Model Evaluation
**Classification Report (K=1):**
| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Iris-setosa     | 1.00      | 1.00   | 1.00     |
| Iris-versicolor | 0.91      | 1.00   | 0.95     |
| Iris-virginica  | 1.00      | 0.90   | 0.95     |
| **Overall**     | **0.97**  | **0.97** | **0.97** |

**Confusion Matrix:**
```
[[10  0  0]
 [ 0 10  0]
 [ 0  1  9]]
```
Only 1 misclassification: 1 Virginica predicted as Versicolor.

### 6. Visualizations
| File | Description |
|------|-------------|
| `plot1_accuracy_vs_k.png` | Train vs Test Accuracy for K=1 to 20 |
| `plot2_confusion_matrix.png` | Heatmap of confusion matrix |
| `plot3_scatter_features.png` | Feature scatter plots by class |
| `plot4_decision_boundary.png` | KNN decision boundary (Petal features) |
| `plot5_feature_variance.png` | Feature variance (proxy for importance) |

---

## Key Concepts Learned
- **Instance-based learning**: KNN stores all training data and classifies based on proximity
- **Euclidean distance**: Default metric to measure closeness between points
- **Normalization**: Prevents features with larger scales from dominating distance calculations
- **K selection**: Use accuracy-vs-K plot to find the elbow/optimal K

---

## Files in This Repo
```
├── knn_iris.py                  # Main Python script
├── Iris.csv                     # Dataset
├── plot1_accuracy_vs_k.png
├── plot2_confusion_matrix.png
├── plot3_scatter_features.png
├── plot4_decision_boundary.png
├── plot5_feature_variance.png
└── README.md
```
