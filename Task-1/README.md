# Task 1 — Data Cleaning & Preprocessing
**ElevateLabs AI/ML Internship**

## Objective
Clean and prepare the Titanic dataset for Machine Learning using Python.

## Dataset
- **Source:** Titanic Dataset (891 rows × 12 columns)
- **File:** `Titanic-Dataset.csv`

## Steps Performed

### 1. Exploratory Data Analysis
- Loaded dataset and inspected shape, data types, and statistics
- Identified missing values: `Age` (19.87%), `Cabin` (77.1%), `Embarked` (0.22%)

### 2. Handling Missing Values
| Column | Strategy | Reason |
|--------|----------|--------|
| `Age` | Filled with **median** (28.0) | Skewed distribution; median is robust to outliers |
| `Embarked` | Filled with **mode** ('S') | Only 2 missing values |
| `Cabin` | **Dropped** | 77% missing — not recoverable |

### 3. Dropped Irrelevant Columns
Removed `PassengerId`, `Name`, `Ticket` — these have no predictive value for ML.

### 4. Encoding Categorical Features
- **Label Encoding** → `Sex`: male=1, female=0 (binary column)
- **One-Hot Encoding** → `Embarked`: created `Embarked_Q`, `Embarked_S` (drop_first=True to avoid dummy variable trap)

### 5. Outlier Detection & Removal (IQR Method)
- Visualized boxplots before and after removal
- Applied IQR rule: removed values below Q1−1.5×IQR or above Q3+1.5×IQR
- Columns treated: `Fare`, `SibSp`, `Parch`
- Rows after removal: **607** (from 891)

### 6. Feature Scaling
- **StandardScaler** (Z-score) applied to `Age` and `Fare` → mean≈0, std≈1
- **MinMaxScaler** (Normalization) also demonstrated for comparison

## Output Files
| File | Description |
|------|-------------|
| `task1_preprocessing.py` | Main Python script |
| `titanic_cleaned.csv` | Final cleaned dataset |
| `boxplots_before.png` | Outlier visualization before removal |
| `boxplots_after.png` | Outlier visualization after removal |
| `correlation_heatmap.png` | Feature correlation heatmap |

## Tools Used
- Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Final Dataset Shape
- **607 rows × 9 columns**, zero missing values

## Interview Questions — Quick Answers

1. **Types of missing data:** MCAR (Missing Completely At Random), MAR (Missing At Random), MNAR (Missing Not At Random)
2. **Handle categorical variables:** Label Encoding (ordinal/binary), One-Hot Encoding (nominal), Target Encoding (high cardinality)
3. **Normalization vs Standardization:** Normalization scales to [0,1] (MinMax); Standardization centers at mean=0, std=1 (Z-score). Use standardization when data has outliers; normalization when distribution is uniform.
4. **Detect outliers:** IQR method, Z-score, boxplots, scatter plots
5. **Preprocessing importance:** Garbage in = garbage out. Clean data leads to better model accuracy and faster convergence.
6. **One-Hot vs Label Encoding:** Label encoding assigns integers (can imply order); One-Hot creates binary columns per category (no order implied). Use OHE for nominal, Label for ordinal/binary.
7. **Data imbalance:** Oversample minority (SMOTE), undersample majority, use class weights in model, or use metrics like F1/AUC instead of accuracy.
8. **Preprocessing & accuracy:** Yes — encoding errors, unscaled features, and unhandled nulls can degrade accuracy significantly.
