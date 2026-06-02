# Task 3 — Linear Regression (House Price Prediction)
**ElevateLabs AI/ML Internship**

## Objective
Implement Simple and Multiple Linear Regression to predict house prices and understand model evaluation metrics.

## Dataset
- **File:** `Housing.csv` — 545 rows × 13 columns
- **Target:** `price` (house price in ₹)
- **Features:** area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus
- **Missing values:** None ✅

## Tools Used
Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## Steps Performed

### Step 1 — Load & Explore
- Loaded dataset, checked shape (545, 13), data types, null values
- All 6 numeric + 7 categorical columns; zero missing values

### Step 2 — Preprocessing
- Binary columns (yes/no) → mapped to 1/0
- `furnishingstatus` → furnished=2, semi-furnished=1, unfurnished=0
- Features standardized using `StandardScaler` before multiple regression

### Step 3 — Simple Linear Regression (area → price)
- Single feature: `area`
- Train/test split: 80% / 20% (random_state=42)

### Step 4 — Multiple Linear Regression (all 12 features)
- All features used after encoding and scaling
- Same train/test split for fair comparison

---

## Model Performance

| Metric       | Simple LR (1 feature) | Multiple LR (12 features) |
|-------------|----------------------|--------------------------|
| MAE  (₹)    | 1,474,748            | 979,680                  |
| RMSE (₹)    | 1,917,104            | 1,331,071                |
| **R² Score**| **0.2729**           | **0.6495**               |
| Features    | 1 (area)             | 12 (all)                 |

> Multiple LR improved R² by **+0.38** — adding more features significantly helps.

---

## Feature Coefficients (Multiple LR — Standardized)

| Feature         | Coefficient | Interpretation                          |
|----------------|-------------|----------------------------------------|
| bathrooms       | +523,153    | More bathrooms → much higher price     |
| area            | +519,288    | Larger area → higher price             |
| airconditioning | +362,446    | AC → significant price premium         |
| stories         | +348,177    | More floors → higher price             |
| prefarea        | +266,661    | Preferred area → premium location      |
| bedrooms        |  +58,691    | Weakest effect (least influential)     |

---

## Charts

| File | Description |
|------|-------------|
| `lr_01_simple_regression.png` | Regression line + residual plot (Simple LR) |
| `lr_02_multiple_regression.png` | Actual vs Predicted + residual plot (Multiple LR) |
| `lr_03_feature_coefficients.png` | Horizontal bar chart of all feature coefficients |
| `lr_04_model_evaluation.png` | Model comparison bar chart + residual distribution |
| `lr_05_correlation_matrix.png` | Full correlation heatmap for multicollinearity check |

---

## Key Observations
- **Area alone** explains only 27% of price variance (R²=0.27) — price depends on many factors
- **All features together** explain 65% of variance — a strong improvement
- **Bathrooms and area** are the two most influential features
- **Residuals** are approximately normally distributed — a good sign for linear regression assumptions
- No severe multicollinearity detected (no feature pair has correlation > 0.8)

---

## Interview Questions — Quick Answers

1. **Assumptions of Linear Regression:** Linearity, independence of errors, homoscedasticity (constant variance), normality of residuals, no multicollinearity.
2. **Interpret coefficients:** Each coefficient = change in target per 1-unit change in that feature (when all others are held constant). Standardized coefficients allow fair comparison of feature importance.
3. **R² score:** Proportion of variance in target explained by the model. R²=0.65 means the model explains 65% of price variation. Closer to 1 is better.
4. **MSE vs MAE:** Prefer MSE when large errors are especially bad (it penalizes them more due to squaring). Prefer MAE for robustness to outliers since it treats all errors equally.
5. **Detect multicollinearity:** Use correlation matrix (|r| > 0.7 is a flag), or calculate VIF (Variance Inflation Factor). VIF > 5 or 10 indicates multicollinearity.
6. **Simple vs Multiple Regression:** Simple = one feature predicting the target; Multiple = two or more features. Multiple can capture more complex relationships.
7. **Linear Regression for classification?** Not recommended directly. Logistic Regression is the proper choice. However, you can threshold LR output but it's unreliable and violates assumptions.
8. **Violating assumptions:** Leads to biased or inefficient estimates — predictions may be systematically wrong, confidence intervals invalid, and p-values unreliable.
