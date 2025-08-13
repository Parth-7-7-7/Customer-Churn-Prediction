# 📊 Customer Churn Prediction

## 📌 Project Overview
This project predicts whether a bank customer will **churn** (leave the bank) based on demographic, account, and activity data.  
The main objective is to help banks **retain customers** by identifying those at high risk of leaving.

We used the **Bank Customer Churn Prediction** dataset from Kaggle, applied **data preprocessing**, trained multiple models, performed evaluation, and saved the best model using **Joblib** for future use.

---

## 🗂 Dataset
- **Source:** [Kaggle - Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shubh0799/bank-customer-churn-prediction)
- **Description:** Contains customer demographics, account details, and activity information.
- **Target Variable:** `Exited` (1 = Customer churned, 0 = Customer stayed)
- **Features Include:**
  - **Demographics:** Geography, Gender, Age
  - **Account Details:** Balance, Credit Score, Tenure, Products Held
  - **Customer Activity:** Number of transactions, active status

---

## 🚀 Features
- **Preprocessing**: 
  - Label Encoding & One-Hot Encoding for categorical variables
  - Feature scaling with `StandardScaler`
- **Model Training**:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Model Saving**:
  - Best model and scaler saved using **Joblib** for later deployment
- **Visualizations**:
  - Confusion Matrix
  - ROC Curve

---

## 🛠 Technologies Used
- **Python 3**
- **Pandas, NumPy** – Data handling
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Preprocessing, ML models, evaluation metrics
- **Joblib** – Model persistence

---

## 📊 Model Workflow
1. **Data Loading**
   - Load dataset from Kaggle.
2. **Exploratory Data Analysis (EDA)**
   - Summarize data, check class distribution, and feature correlations.
3. **Preprocessing**
   - Encode categorical variables.
   - Scale numerical features using `StandardScaler`.
4. **Model Training**
   - Train Logistic Regression, Random Forest, and Gradient Boosting.
5. **Model Evaluation**
   - Compare models using accuracy, classification report, and ROC curves.
6. **Model Saving**
   - Save the best model and scaler with:
     ```python
     joblib.dump(best_model, 'churn_model.joblib')
     joblib.dump(scaler, 'scaler.joblib')
     ```

---

## 📈 Results
- **Best Model:** Gradient Boosting Classifier
- **Accuracy:** ~86% (after preprocessing and tuning)
- Gradient Boosting outperformed Logistic Regression and Random Forest in ROC-AUC and recall.

---

## 📦 Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git

# Navigate to project directory
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook "Customer Churn Prediction.ipynb"
