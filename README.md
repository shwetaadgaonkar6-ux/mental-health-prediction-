# mental health prediction 
Can lifestyle and physiological signals predict mental health outcomes? Five ML models compared — Logistic Regression, SVM, Random Forest, Decision Tree, and Neural Network.

# 🧠 Mental Health Prediction — Machine Learning Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green?style=flat)
![pandas](https://img.shields.io/badge/pandas-Analysis-150458?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## Introduction

Mental health is one of the most under-measured areas in healthcare — not because the data doesn't exist, but because we haven't always known what to do with it.

Using a dataset of 2,978 anonymised records across 26 physiological, behavioural, and demographic features, I built and compared five classification models to predict mental health status (Excellent vs Poor) from lifestyle signals alone.

The question driving this project: *can everyday health indicators reliably predict mental health outcomes — and which model surfaces that signal best?*

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Jupyter Notebook | Development environment |
| pandas | Data loading and wrangling |
| NumPy | Numerical computation |
| scikit-learn | Modelling, tuning, evaluation |
| XGBoost | Gradient boosting classifier |
| Matplotlib & Seaborn | Visualisation |
| SciPy | Statistical testing (paired t-tests) |

---

## Features

- Binary classification — Excellent vs Poor mental health status
- ANOVA F-test for top feature selection
- Five ML models tuned with GridSearchCV
- Neural Network (MLP) with extensive hyperparameter search
- Evaluation across Accuracy, F1 Score and ROC AUC
- Paired t-tests to statistically validate model differences
- Confusion matrices and ROC curves for every model

---

## The Five Models

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Logistic Regression | 72.6% | 0.841 | 0.780 |
| SVM | 72.6% | 0.775 | 0.755 |
| Random Forest | 71.0% | — | 0.480 |
| Decision Tree | 66.1% | — | 0.376 |
| Neural Network (MLP) | Tuned via GridSearchCV | — | — |

---

## How I Built It

**1. Data pipeline**
Loaded and cleaned 2,978 records. Handled missing values, encoded categorical variables (Gender, Sleep Patterns, Education Level), and filtered for binary target classes.

**2. Feature selection**
Applied ANOVA F-test to rank all 26 features. Narrowed down to the top 5 most predictive: Gender, Cholesterol, BMI, Cognitive Function, and Sleep Patterns.

**3. Preprocessing**
Standardised all numerical features using StandardScaler for consistent model performance across distance-sensitive algorithms.

**4. Model training & tuning**
Trained four core classifiers on a 70/30 stratified split. Each tuned with GridSearchCV and 5-fold cross-validation.

**5. Neural network**
Built an MLP with hyperparameter search across layer sizes, activation functions, solvers, and learning rate strategies.

**6. Model comparison**
Compiled metrics across all models, visualised with grouped bar charts, and confirmed performance differences with paired t-tests.

---

## What I Learned

- How ANOVA feature selection shapes downstream model performance
- That Logistic Regression can match complex ensemble methods on clean, well-processed data
- Why standardisation matters differently across model types
- How to design fair comparisons using stratified splits and cross-validation
- The value of statistical testing — not just comparing numbers, but confirming differences are real
- Neural networks aren't always the answer

---

## How It Could Be Improved

- **Real-world data** — synthetic dataset limits generalisability; clinical data would strengthen findings
- **Class imbalance handling** — SMOTE or class weighting for more balanced F1 scores
- **Explainability** — SHAP values to surface which features drive individual predictions
- **Deployment** — wrap the best model in a FastAPI endpoint for real-time predictions
- **Multi-class target** — extend beyond binary to mental health severity levels

---

## How to Run

### Install dependencies
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy jupyter
```

### Clone and run
```bash
git clone https://github.com/shwetaadgaonkar6-ux/mental-health-prediction-ml.git
cd mental-health-prediction-ml
jupyter notebook mental_health_prediction.ipynb
```

Then: `Kernel → Restart & Run All`

### What to expect
- Feature importance bar chart (ANOVA F-scores)
- Confusion matrices and ROC curves per model
- Model comparison table and grouped bar chart
- Paired t-test results printed to console

---

## Dataset

[Kaggle — Human Age Prediction Synthetic Dataset](https://www.kaggle.com/datasets/abdullah0a/human-age-prediction-synthetic-dataset)
2,978 records · 26 features · Binary target: Excellent / Poor mental health

---

## Author

**Shweta Adgaonkar**
Master of Data Science — RMIT University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/shweta-adgaonkar-b852a6223)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?style=flat&logo=github)](https://github.com/shwetaadgaonkar6-ux)
