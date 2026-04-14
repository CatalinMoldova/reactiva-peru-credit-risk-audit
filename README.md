

# Replication & Critique: ML for Credit Risk in the Reactiva Peru Program

This repository contains a full replication and methodological critique of the study **"Machine Learning for Credit Risk in the Reactiva Peru Program: A Comparison of the Lasso and Ridge Regression Models"** by Geraldo-Campos et al. (2022).

---

## 📌 Project Highlights

* **Critical Discovery**: Diagnosed a fundamental **target leakage** flaw in the original experimental design where the credit risk target was mechanically derived from a primary regressor.
* **Statistical Rigor**: Validated the flaw using **Kruskal-Wallis tests** () and **Random Forest** models ().
* **Large-Scale Data**: Processed a dataset of **500,000+ SME loan records** from the Peruvian government's COVID-19 relief program.

---

## 🛠️ Technical Implementation

### Core Replication

* **Models**: Re-implemented **OLS, Lasso, and Ridge** regressions in Python.
* **Optimization**: Automated hyperparameter tuning using `LassoCV` and `RidgeCV`, comparing results against the fixed alphas used in the original paper.
* **Pipeline**: Built a robust preprocessing workflow including ordinal encoding for categorical variables (economic sector, lending entity) and `MinMaxScaler` normalization.

### Methodological Extension

The replication revealed that linear models achieved a weak . I extended the study to investigate this bottleneck:

1. **Leakage Diagnosis**: Proved the "Risk Level" was determined solely by "Covered Amount" quartiles.
2. **Non-Linear Validation**: Deployed **Random Forests** to test if the underlying rule could be recovered. The model achieved a perfect , confirming the deterministic relationship.
3. **Feature Importance**: Visualized that "Covered Amount" provided nearly 100% of the predictive power, rendering other regressors virtually obsolete.

---

## 📊 Key Results Summary

| Model | RMSE |  | Note |
| --- | --- | --- | --- |
| **Lasso (Optimal)** | 0.35660 | 0.08232 | Replicated Weak Performance |
| **Ridge (Optimal)** | 0.35661 | 0.08231 | Replicated Weak Performance |
| **Random Forest** | **0.00007** | **1.00000** | **Confirmed Target Leakage** |

---

## 📁 Repository Structure

* `MLCS_project.pdf`: Comprehensive academic paper detailing the replication and findings.
* `Notebooks/`: Contains the primary Jupyter Notebook with data cleaning, statistical tests, and modeling.
* `References/`: Links to the original study and Peruvian government datasets.

---

## 🚀 How to Run

1. **Clone the repo**:
```bash
git clone https://github.com/CatalinMoldova/Replication-and-Extension-of-Linear-Regressor-of-the-Reactiva-program-in-Peru.git

```


2. **Install dependencies**:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy

```


3. **Execute the notebook**:
Run the `.ipynb` file in any Jupyter environment to see the replication and statistical proofs.

---

## 🎓 Academic Context

This project was completed in December 2025 as part of the **Machine Learning and Computational Statistics** course at **New York University**.


