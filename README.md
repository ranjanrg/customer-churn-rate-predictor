# 📉 Customer Churn Rate Predictor

A machine learning project that predicts whether a telecom customer will churn (leave the service) using **Logistic Regression**. Built on a real-world dataset to practice end-to-end ML workflows.

## 🧠 How It Works

1. Load and clean the telecom customer dataset (`train.csv`)
2. Encode categorical features (binary mapping + one-hot encoding)
3. Split data into train/test sets (80/20)
4. Scale features using **StandardScaler**
5. Train a **Logistic Regression** model with `class_weight='balanced'` to handle churn imbalance
6. Evaluate with accuracy score, confusion matrix, and classification report

## 🗂️ Project Structure

customer-churn-rate-predictor/
├── main.py       # Full ML pipeline
├── train.csv     # Dataset (not pushed to repo)
└── .gitignore

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas** — Data cleaning & preprocessing
- **Scikit-learn** — Logistic Regression, StandardScaler, evaluation metrics

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
   git clone https://github.com/ranjanrg/customer-churn-rate-predictor.git
   cd customer-churn-rate-predictor
```

2. **Install dependencies**
```bash
   pip install pandas scikit-learn
```

3. **Add the dataset**
   Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place `train.csv` in the root folder.

4. **Run the script**
```bash
   python main.py
```

## 📋 Features Used

The model uses customer attributes including tenure, monthly charges, total charges, internet service type, contract type, payment method, and various add-on services like online security, streaming, and tech support.
