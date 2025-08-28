# Credit Card Fraud Detection

This project implements a **machine learning pipeline** to detect fraudulent credit card transactions using logistic regression. It leverages an imbalanced real-world dataset and applies **under-sampling** techniques to balance the classes for accurate model training and evaluation.

---

## Overview

- Dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Model: Logistic Regression with feature scaling
- Techniques: Under-sampling, data splitting, accuracy evaluation
- Libraries: scikit-learn, pandas

---

## Features

- Binary classification to distinguish between legitimate and fraudulent transactions
- Data preprocessing and balancing using under-sampling
- Model pipeline using `StandardScaler` and `LogisticRegression`
- Accuracy evaluation on both training and testing sets

---

## Technologies Used

- Python 3.x
- pandas
- scikit-learn (LogisticRegression, train_test_split, StandardScaler, etc.)
- Jupyter Notebook / .py script

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Install required libraries
pip install pandas scikit-learn

3. Download the dataset
- Download the dataset from Kaggle
- Place creditcard.csv in the project root directory

4. Run the script
python fraud_detection.py

5. View output
- Accuracy scores for both training and test sets will be printed in the terminal.

