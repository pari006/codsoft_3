# Customer Churn Prediction

This project predicts whether a bank customer is likely to **churn (exit)** or **stay** using machine learning models.  
It compares **Logistic Regression, Random Forest, and Gradient Boosting**, selects the best one based on accuracy, and saves the model for later use.  
The project also includes a **Tkinter GUI application** for user-friendly predictions.

---

## 📂 Repository Contents
1. **Churn_Modelling.csv**  
   - Dataset used for training and testing.  
   - Contains customer details like age, balance, tenure, geography, gender, credit score, etc.  

2. **churn_prediction_model.pkl**  
   - Saved best-performing trained model (via `joblib`).  
   - Automatically selected among Logistic Regression, Random Forest, and Gradient Boosting classifiers.  

3. **training_and_model.py**  
   - Main Python script for:  
     - Data preprocessing (scaling, encoding categorical features).  
     - Training multiple classifiers and selecting the best one.  
     - Saving the model.  
     - GUI app built with **Tkinter** to enter customer details and predict churn interactively.  

---

## ⚙️ How It Works
1. Loads the dataset and preprocesses features.  
2. Trains multiple ML models and evaluates them on accuracy & classification report.  
3. Saves the **best model** as `churn_prediction_model.pkl`.  
4. Launches a **Tkinter GUI** where users can input customer details.  
5. The GUI predicts if a customer will churn (`Exited = 1`) or stay (`Exited = 0`).  

---

## 🚀 Tech Stack
- **Python** (pandas, scikit-learn, joblib, tkinter)  
- **Machine Learning**: Logistic Regression, Random Forest, Gradient Boosting  
- **GUI**: Tkinter  

---

## 🎯 Example Use
- Enter customer details like **Age, Balance, Geography, Gender, Credit Score** etc.  
- The app instantly predicts if the customer is **likely to churn** or **likely to stay**.  

---

👉 This repo is useful for learning about **classification, preprocessing, model comparison, and GUI integration** in ML projects.

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
   git clone https://github.com/pari006/spam-detection-svm.git
   cd spam-detection-svm
