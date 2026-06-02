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

