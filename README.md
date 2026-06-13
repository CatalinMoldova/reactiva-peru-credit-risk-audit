# Credit Risk Model Audit: Reactiva Perú

Replication and methodological audit of machine learning models used for credit-risk prediction in the Reactiva Perú government-backed loan program.

This project replicates and extends the study *“Machine Learning for Credit Risk in the Reactiva Peru Program: A Comparison of the Lasso and Ridge Regression Models”* by Geraldo-Campos et al. The original paper compares OLS, Lasso, and Ridge regression for predicting credit-risk levels among firms that received loans under the Reactiva Perú COVID-19 relief program.

My contribution goes beyond replication: I identify and validate a target-leakage issue showing that the risk target is mechanically linked to one of the input variables, making the prediction task less meaningful than it first appears.

## Project Summary

The original study reports that Lasso slightly outperforms Ridge for predicting credit-risk level. After reproducing the modelling pipeline, I extended the analysis with statistical tests and non-linear models to examine whether the target variable was truly being predicted from meaningful business features.

The key finding is that the target variable, `risk_level`, appears to be almost entirely determined by `covered_amount`. A Random Forest model recovers this relationship almost perfectly, while feature-importance analysis shows that `covered_amount` dominates the prediction signal.

## Why This Project Matters

Credit-risk models are used in banking, public policy, and financial decision-making. A model can appear accurate while learning a shortcut caused by data leakage or target construction. This project demonstrates why model validation must go beyond performance metrics.

The main lesson is:

> A model is not useful just because it predicts well. We must also verify that it is learning a real economic relationship rather than a mechanical rule embedded in the data.

## Research Question

Can Lasso and Ridge regression meaningfully predict credit-risk level in the Reactiva Perú dataset, or is the target variable mechanically determined by one of the regressors?

## Methods

### 1. Replication

I replicated the original modelling approach using:

* OLS regression
* Lasso regression
* Ridge regression
* MinMax scaling
* Ordinal encoding of categorical variables
* Train/test split evaluation
* Cross-validated alpha selection with `LassoCV` and `RidgeCV`

### 2. Methodological Audit

I then extended the analysis to test for leakage:

* Checked whether `risk_level` was derived from `covered_amount` thresholds
* Compared model performance across linear and non-linear models
* Used Random Forests to test whether the underlying target rule could be recovered
* Examined feature importance to identify dominant predictors
* Used statistical tests to compare the distribution of covered amounts across risk groups

## Dataset

The dataset contains firm-level records from the Reactiva Perú program, including:

* Economic sector
* Lending institution
* Department / region
* Covered loan amount
* Constructed credit-risk level

The full public dataset contains more than 500,000 SME loan records.

## Key Results

| Model         |    RMSE |      R² | Interpretation                             |
| ------------- | ------: | ------: | ------------------------------------------ |
| Lasso         | 0.35660 | 0.08232 | Weak predictive performance                |
| Ridge         | 0.35661 | 0.08231 | Similar to Lasso                           |
| Random Forest | 0.00007 | 1.00000 | Confirms deterministic target relationship |

## Main Finding

The Random Forest model achieves near-perfect performance because the target can be inferred almost entirely from `covered_amount`.

This suggests that the original prediction task may suffer from target leakage: the model is not discovering a complex credit-risk pattern, but instead recovering a rule used to construct the target.

## Visual Results

The repository includes:

* Model comparison chart
* Feature-importance plot
* Risk level vs. covered amount visualization
* Methodology diagram explaining the replication and leakage-audit pipeline

## Technical Stack

* Python
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* SciPy
* Jupyter Notebook

## Repository Structure

```text
src/            Reusable Python scripts for cleaning, training, and diagnostics
notebooks/      Main reproducible analysis notebook
data/           Data instructions and sample data
results/        Model metrics, statistical tests, and feature-importance outputs
assets/         Figures used in the README and portfolio case study
report/         Full written report
presentation/   Project presentation slides
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/reactiva_peru_credit_risk_audit.ipynb
```

Or run the scripts:

```bash
python src/train_linear_models.py
python src/leakage_diagnostics.py
python src/random_forest_validation.py
```

## Portfolio Takeaway

This project demonstrates:

* ML model replication
* Credit-risk modelling
* Data leakage diagnosis
* Regression and regularization
* Random Forest validation
* Statistical model criticism
* Reproducible data science workflow

## Project Origin

This project was developed as part of a Machine Learning and Computational Statistics course at New York University and later refactored into a portfolio-ready model validation case study.
