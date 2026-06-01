# Task 2 — Exploratory Data Analysis (EDA)
**ElevateLabs AI/ML Internship**

## Objective
Understand the Titanic dataset using statistics and visualizations to uncover patterns, trends, and anomalies.

## Dataset
- **Source:** Titanic Dataset (891 rows × 12 columns)
- **File:** `Titanic-Dataset.csv` (raw data used for better visual readability)

## Tools Used
Python 3, Pandas, NumPy, Matplotlib, Seaborn, Plotly

---

## Steps & Charts

### Step 1 — Summary Statistics
- Shape, data types, describe(), missing value counts
- Skewness computed for all numeric features

### Step 2 — Survival Distribution (`eda_01_survival_distribution.png`)
- Bar chart + pie chart showing 38.4% survived vs 61.6% did not

### Step 3 — Histograms (`eda_02_histograms.png`)
- Distributions of Age, Fare, SibSp, Parch with mean & median lines
- Fare is heavily right-skewed (skew = 4.79) — most passengers paid low fares

### Step 4 — Boxplots vs Survival (`eda_03_boxplots.png`)
- Side-by-side boxplots for Age, Fare, SibSp, Parch by survival status
- Survivors paid higher fares on average

### Step 5 — Categorical Features vs Survival (`eda_04_categorical_survival.png`)
- Survival rates by Pclass, Sex, and Embarkation port
- Females: 74.2% survival | Males: 18.9% survival (huge gap)

### Step 6 — Age Analysis (`eda_05_age_analysis.png`)
- Age distribution by survival (overlapping histograms)
- Violin plot of age by passenger class

### Step 7 — Correlation Heatmap (`eda_06_correlation_heatmap.png`)
- Lower-triangle heatmap of all numeric features
- Sex and Pclass are most correlated with Survival

### Step 8 — Pairplot (`eda_07_pairplot.png`)
- Pairwise relationships between Age, Fare, Pclass, SibSp coloured by Survival

### Step 9 — Fare & Gender Analysis (`eda_08_fare_survival_class_gender.png`)
- Boxplot: Fare by class and survival side by side
- Grouped bar: Survival rate broken down by class AND gender

---

## Key Insights

| Insight | Finding |
|---------|---------|
| Overall Survival | Only 38.4% of passengers survived |
| Gender Effect | Women had 74.2% survival vs 18.9% for men ("women and children first") |
| Class Effect | 1st class: 63% → 2nd: 47.3% → 3rd: 24.2% (wealth = higher survival) |
| Fare Skewness | Skew of 4.79 — most passengers paid low fares, few paid extremely high |
| Embarkation | Cherbourg passengers had highest survival (55.4%) — more 1st class passengers there |
| Age | Children slightly more likely to survive; 1st class passengers were older on average |
| Multicollinearity | Pclass and Fare are inversely correlated (−0.55) — expected since richer = 1st class |

---

## Interview Questions — Quick Answers

1. **Purpose of EDA:** Understand data structure, detect anomalies, discover patterns, and decide feature engineering strategies before modeling.
2. **Boxplots:** Show median, IQR, and outliers. Help compare distributions across groups quickly.
3. **Correlation:** Measures linear relationship between two variables (−1 to +1). Useful for feature selection and detecting multicollinearity.
4. **Detect skewness:** Use `.skew()` in pandas, or visually via histogram. Positive skew = right tail; negative = left tail.
5. **Multicollinearity:** When two or more features are highly correlated with each other (e.g., Pclass and Fare). Problematic for linear models.
6. **EDA Tools:** Pandas (stats), Matplotlib/Seaborn (static plots), Plotly (interactive charts), Sweetviz/Pandas Profiling (automated EDA).
7. **EDA finding a problem:** In the Titanic dataset, EDA revealed the Cabin column was 77% missing — without EDA we'd have tried to use it and gotten a poor model.
8. **Visualization in ML:** Helps verify data quality, understand distributions, identify outliers, and communicate findings to non-technical stakeholders.
