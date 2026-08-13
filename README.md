# 💳 Credit Card Fraud Detection System

A machine learning application built with **Python**, **Scikit-Learn**, and **Flask** designed to detect fraudulent credit card transactions in real-time while maintaining high classification precision and recall on highly imbalanced financial datasets.

---

## 📌 Resume Highlights
- **Built a fraud detection system** using Python and Scikit-Learn to classify fraudulent and legitimate transactions.
- **Applied data preprocessing, feature engineering, and imbalance handling techniques** to improve model performance.
- **Compared multiple machine learning models** and achieved high fraud detection accuracy using **Precision, Recall, and F1-score** metrics.

---

## 🌟 Key Features & Architecture

1. **Robust Feature Preprocessing & Scaling**:
   - `StandardScaler` integration via Scikit-Learn `ColumnTransformer` for scaling continuous transaction features (`Time` and `Amount`).
   - Automated serialization of the complete preprocessing and classifier workflow into a single Scikit-Learn `Pipeline`.

2. **Class Imbalance Mitigation**:
   - Tackled extreme class imbalance (~0.17% fraud prevalence) using controlled undersampling techniques to improve recall on fraudulent transactions while preserving pattern features.

3. **Multi-Model Evaluation**:
   - Systematic evaluation across four competitive machine learning algorithms:
     - **Logistic Regression**
     - **Decision Tree Classifier**
     - **Random Forest Classifier** *(Selected Production Model)*
     - **Gradient Boosting Classifier**

4. **Interactive Flask Web Application**:
   - Modern, card-based web interface built with clean HTML5/CSS3.
   - Real-time prediction endpoint returning classification status, probability score (%), and risk assessment (Low Risk vs High Risk).
   - **Quick-Fill Presets**: One-click test loading for Legitimate and Fraudulent transaction data.

---

## 📊 Model Comparison Results

Below are the evaluation metrics computed on the test set (`creditcard.csv`):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Production)** | **99.49%** | **0.2390** | **0.8878** | **0.3766** | **0.9737** |
| **Logistic Regression** | 98.92% | 0.1279 | 0.9082 | 0.2242 | 0.9715 |
| **Gradient Boosting** | 98.61% | 0.1002 | 0.8878 | 0.1801 | 0.9784 |
| **Decision Tree** | 98.18% | 0.0758 | 0.8571 | 0.1393 | 0.9004 |

---

## 📁 Project Directory Structure

```text
credit-card-fraud-detection/
├── app/
│   ├── app.py                 # Flask application server & prediction API
│   ├── static/
│   │   └── style.css          # Modern custom CSS layout & styles
│   └── templates/
│       └── index.html         # Responsive dashboard UI with quick-test presets
├── data/
│   └── creditcard.csv         # European cardholders dataset (Kaggle)
├── model/
│   └── fraud_model.pkl        # Serialized Scikit-Learn model pipeline
├── notebooks/
│   └── fraud_detection.ipynb  # End-to-end ML notebook (EDA, Preprocessing, Multi-Model Comparison)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Excluded environment and dataset files
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment Setup
Ensure Python 3.9+ is installed. Clone the repository and navigate to the project directory:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Train Model & Compare Metrics (Optional)
Run the Jupyter Notebook or execute the training pipeline script:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

### 3. Launch Flask Web Application

```bash
python app/app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

- Click **✅ Load Legitimate Sample** or **🚨 Load Fraudulent Sample** to populate test feature vectors instantly.
- Click **🔍 Classify Transaction** to receive the prediction result and fraud probability meter.

---

## 🛠️ Technology Stack
- **Language**: Python 3.13
- **Machine Learning**: Scikit-Learn, NumPy, Pandas
- **Web Backend**: Flask, Joblib
- **Frontend UI**: HTML5, Vanilla CSS3 (Inter Google Font)
- **Visualization**: Matplotlib, Seaborn
