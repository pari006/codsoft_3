# Customer Churn Prediction

This project implements a machine learning model to predict whether a bank customer is likely to churn (i.e., leave the bank) based on their demographic and account information. It includes a complete pipeline from data preprocessing and model training to a user-friendly graphical interface for real-time predictions.

---

## Project Overview

- **Dataset:** Uses the Bank Customer Churn Prediction Dataset from Kaggle.
- **Goal:** Predict customer churn (binary classification: churn or not churn).
- **Model:** Random Forest Classifier (other models like Logistic Regression or Gradient Boosting can be used).
- **Preprocessing:**
  - Drops irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
  - Scales numerical features using `StandardScaler`.
  - Encodes categorical features (`Geography`, `Gender`) using `OneHotEncoder`.
- **Pipeline:** Combines preprocessing and model training into a single scikit-learn pipeline.
- **Evaluation:** Model accuracy and classification report are generated on a test split.
- **GUI:** A Tkinter-based graphical user interface allows users to input customer details and get churn predictions interactively.

---

## Features Used for Prediction

- Geography (categorical)
- Gender (categorical)
- Credit Score (numerical)
- Age (numerical)
- Tenure (numerical)
- Balance (numerical)
- Number of Products (numerical)
- Has Credit Card (binary: 1 or 0)
- Is Active Member (binary: 1 or 0)
- Estimated Salary (numerical)

---

## How It Works

1. **Data Loading:** Reads the CSV dataset and drops unnecessary columns.
2. **Data Splitting:** Splits data into training and testing sets.
3. **Preprocessing:** Applies scaling to numerical features and one-hot encoding to categorical features.
4. **Model Training:** Trains a Random Forest classifier on the processed training data.
5. **Prediction:** The trained model predicts churn on new input data.
6. **GUI Interface:**
   - Users enter customer details in the GUI form.
   - On clicking **Predict Churn**, the app processes inputs and displays whether the customer is likely to churn or stay.

---

## Usage

1. Run the script to train the model and launch the GUI.
2. Enter customer information in the input fields.
3. Click **Predict Churn** to see the prediction result.

---

## Dependencies

- Python 3.7+
- pandas
- scikit-learn
- joblib
- tkinter (usually included with Python)

Install dependencies via pip:

```bash
pip install pandas scikit-learn joblib
```
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spam-detection-svm.git
   cd spam-detection-svm
