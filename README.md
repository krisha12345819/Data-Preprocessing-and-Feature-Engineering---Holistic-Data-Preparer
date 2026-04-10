# 🧠 Holistic Data Preparer — Customer Credit Risk

<div align="center">

![Data Science](https://img.shields.io/badge/Domain-Data%20Science-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Goal-ML%20Ready%20Dataset-green?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Duration](https://img.shields.io/badge/Duration-8%20Days-orange?style=for-the-badge)

> **"Turning raw, messy data into a clean, ML-ready masterpiece — one preprocessing step at a time."**

</div>

---

## 📌 Project Overview

You are a **Junior Data Scientist** at a fintech company. Your mission: take a raw **Customer Credit Risk Dataset** collected from multiple real-world sources and transform it into a **fully preprocessed, feature-engineered dataset** ready for machine learning.

The model will predict: **Will a customer default on a loan? (0 = No | 1 = Yes)**

---

## 🗂️ Repository Structure

```
📦 customer-credit-risk-preprocessing/
├── 📓 Customer_Credit_Risk_Data_Preprocessing.ipynb   # Main Jupyter Notebook
├── 📊 customer_credit_risk.csv                        # Main CSV dataset
├── 🗃️ credit_risk.db                                  # SQL database (repayment history)
├── 📋 customer_metadata.json                          # JSON metadata
├── 📈 data_profiling_report.html                      # Pandas Profiling Report
├── 📄 Holistic_Data_Preprocessing_Project_Theory_Report.pdf
└── 📝 README.md
```

---

## 🎯 Objective

> End-to-end data preprocessing and feature engineering on a Customer Credit Risk dataset collected from **CSV, JSON, SQL, and API** sources — producing a clean, consistent, and ML-ready dataset.

---

## 🗃️ Dataset Structure

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `customer_id` | String/Int | Unique customer identifier | No missing values |
| `age` | Integer | Age of customer (years) | Missing values injected |
| `gender` | Categorical | Male / Female / Other | Missing + category imbalance |
| `region` | Categorical | North / South / East / West | Good for One-Hot Encoding |
| `education_level` | Ordinal Cat. | Primary / Secondary / Graduate / Post-Graduate | For Ordinal Encoding |
| `employment_type` | Categorical | Salaried / Self-Employed / Unemployed | Missing values for imputation |
| `annual_income` | Float | Annual income (₹) | Outliers + missing values |
| `loan_amount` | Float | Loan amount requested (₹) | Outliers + skewed distribution |
| `loan_purpose` | Categorical | Home / Car / Education / Business / Other | For One-Hot Encoding |
| `credit_score` | Float | Credit score (300–850) | Outliers + missing values |
| `repayment_history` | Integer | Missed payments in last 12 months | For Binning / Outlier treatment |
| `transaction_count` | Integer | Total transactions in last 6 months | For K-Means Binning |
| `spending_ratio` | Float | Spending-to-Income ratio (%) | For log/Box-Cox/Yeo-Johnson |
| `join_date` | Date | Date customer joined bank | Extract Y/M/D/W |
| `default_flag` | Binary Int | 0 = No Default, 1 = Default | 🎯 ML Target Variable |

---

## 🚀 Project Workflow

### 🅐 Part A — Conceptual Foundation
- 📖 Short notes on **Data Analysis**, **Planning a Data Science Project**, **Framing an ML Problem**
- 🔢 **Tensors** explained with in-depth NumPy examples

### 🅑 Part B — Data Acquisition
Loaded data from **4 different sources**:
- 📄 CSV files → main transactions dataset
- 🗂️ JSON files → customer metadata
- 🗄️ SQL database → loan repayment history
- 🌐 Dummy API → external economic indicators

### 🅒 Part C — Data Understanding & Cleaning
- 🔍 Explored using `.info()` and `.describe()`
- 📊 Generated **Pandas Profiling** data quality report
- 🩹 Handled missing values with **6 strategies**:

| Strategy | Applied On |
|----------|-----------|
| Simple Imputer (mean/median) | `age`, numerical features |
| Simple Imputer (most frequent) | `employment_type` |
| Most Frequent Category Imputation | `gender` |
| Missing Indicator + Random Sample | `annual_income` |
| KNN Imputer (multivariate) | `annual_income`, `loan_amount`, `credit_score` |
| MICE Algorithm | Multivariate imputation |
| Complete Case Analysis | Dropping rows/columns |

### 🅓 Part D — Outlier Handling
Detected and treated outliers in `annual_income`, `loan_amount`, `credit_score` using:

| Method | Description |
|--------|-------------|
| 📐 Z-Score | Detect values > 3 standard deviations |
| 📦 IQR Method | Flag values outside 1.5×IQR |
| 📊 Percentile Method | Cap at 1st–99th percentile |
| 🪄 Winsorization | Limit extreme values without data loss |

### 🅔 Part E — Feature Engineering

**🔤 Encoding Categorical Variables:**
- Ordinal Encoding → `education_level`
- Label Encoding → `gender`
- One-Hot Encoding → `region`, `loan_purpose`

**🔢 Numerical Encoding / Binning:**
- Binning (discretize income into groups) → `annual_income`
- Binarization (flag if `credit_score > 700`) 
- Quantile Binning → `transaction_count`
- K-Means Binning → `transaction_count`

**📅 Date/Time Feature Extraction from `join_date`:**
- Extracted → `Year`, `Month`, `Day`, `Weekday`

### 🅕 Part F — Feature Scaling

| Scaler | Applied On |
|--------|-----------|
| ⚖️ Standardization (Z-score) | `annual_income`, `loan_amount` |
| 📏 Min-Max Scaling | All numeric columns |
| 🔒 MaxAbs Scaling | All numeric columns |
| 🛡️ Robust Scaling | All numeric columns |

### 🅖 Part G — Feature Construction & Transformation

**🛠️ Constructed New Features:**
- `debt_to_income_ratio` = `loan_amount / annual_income`
- `avg_monthly_transactions` = `transaction_count / 6`
- `spending_to_income_ratio` (engineered feature)

**🔄 Applied Transformations:**
- `FunctionTransformer` → log, reciprocal, square root on `spending_ratio`
- `PowerTransformer` → Box-Cox & Yeo-Johnson on `loan_amount`, `annual_income`
- `ColumnTransformer` → unified pipeline for categorical + numeric preprocessing

### 🅗 Part H — Final Deliverable
- ✅ Clean, fully transformed ML-ready dataset
- 📄 Theory report summarizing all preprocessing decisions
- 📊 Data profiling HTML report

---

## 📊 Summary of Techniques Applied

```
✅ Missing Value Imputation   → 6 strategies (Simple, KNN, MICE, etc.)
✅ Outlier Detection          → Z-score, IQR, Percentile, Winsorization
✅ Categorical Encoding       → Ordinal, Label, One-Hot
✅ Numerical Binning          → Equal-width, Quantile, K-Means, Binarization
✅ Feature Scaling            → Standard, MinMax, MaxAbs, Robust
✅ Transformations            → Log, Reciprocal, Sqrt, Box-Cox, Yeo-Johnson
✅ Feature Construction       → Debt-to-Income, Spending-to-Income ratios
✅ Date Engineering           → Year, Month, Day, Weekday extraction
✅ Pipeline                   → ColumnTransformer unified preprocessing
```

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 🎓 Learning Outcomes

By completing this project, you will be able to:

- 🗺️ Plan and execute a **complete data preprocessing workflow**
- 🧹 Perform **detailed data cleaning** using imputation and outlier handling
- 🔠 Apply **advanced encoding and scaling** techniques
- 🏗️ Construct and transform features to **improve ML readiness**
- 📦 Generate a **high-quality dataset** ready for ML model building

---

## 📁 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-credit-risk-preprocessing.git

# 2. Navigate to the project folder
cd customer-credit-risk-preprocessing

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Customer_Credit_Risk_Data_Preprocessing.ipynb
```

---

## 📬 Contact

> Feel free to reach out for questions, feedback, or collaboration!

<div align="center">

Made with ❤️ | Data Science Final Project 

</div>
