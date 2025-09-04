import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier #, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox
import tkinter.font as font

#Download dataset from here:
    #https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
# Step 1: Load the dataset
BASE_DIR = "D:\\Projects\\Customer Churn Prediction"  # Replace with your actual file path
file = os.path.join(BASE_DIR, "Churn_Modelling.csv")
data = pd.read_csv(file)

# Step 2: Data Preprocessing
# Drop unnecessary columns
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Step 3: Define features and target variable
X = data.drop(columns=['Exited'])  # Features
y = data['Exited']  # Target variable

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Preprocess the data
# Define categorical and numerical features
categorical_features = ['Geography', 'Gender']
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Step 6: Create a pipeline for the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())  # Change to LogisticRegression() with accuracy : 0.811 or GradientBoostingClassifier() with accuracy 0.864 as needed
])

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 10: Print results
print(f'Accuracy: {accuracy}')  # RandomForestClassifier() accuracy is 0.8655
print('Classification Report:')
print(report)

# Save the model
joblib.dump(model, os.path.join(BASE_DIR, 'churn_prediction_model.pkl'))

# Create the Tkinter interface
def predict_churn():
    # Get input values from the user
    geography = geography_var.get()
    gender = gender_var.get()
    credit_score = float(credit_score_entry.get())
    age = float(age_entry.get())
    tenure = float(tenure_entry.get())
    balance = float(balance_entry.get())
    num_of_products = int(num_of_products_entry.get())
    has_cr_card = int(has_cr_card_var.get())
    is_active_member = int(is_active_member_var.get())
    estimated_salary = float(estimated_salary_entry.get())

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Geography': [geography],
        'Gender': [gender],
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Show the result
    if prediction[0] == 1:
        messagebox.showinfo(" Prediction Result", "The customer is likely to churn.", icon='warning')
    else:
        messagebox.showinfo("Prediction Result", "The customer is likely to stay.", icon='info')

# Create the main window
root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("600x600")  # Set the window size
root.configure(bg="deepskyblue")  # Set background color

# Create a custom font
custom_font = font.Font(family="Comic Sans MS", size=15)

# Create input fields with labels
tk.Label(root, text="Customer Churn Prediction", font=("Lucida Calligraphy", 22,"bold"), bg="deepskyblue", fg = "blue2").grid(row=0, columnspan=2, pady=10)

# Centering the input fields
tk.Label(root, text="🌎 Geography:", font=custom_font, bg="pink", fg = "navy").grid(row=1, column=0, sticky='e', padx=10, pady=5)
geography_var = tk.StringVar()
tk.Entry(root, textvariable=geography_var, font=custom_font).grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="🚻 Gender:", font=custom_font, bg="pink", fg = "navy").grid(row=2, column=0, sticky='e', padx=10, pady=5)
gender_var = tk.StringVar()
tk.Entry(root, textvariable=gender_var, font=custom_font).grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="🎯 Credit Score:", font=custom_font, bg="pink", fg = "navy").grid(row=3, column=0, sticky='e', padx=10, pady=5)
credit_score_entry = tk.Entry(root, font=custom_font)
credit_score_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="👤 Age:", font=custom_font, bg="pink", fg = "navy").grid(row=4, column=0, sticky='e', padx=10, pady=5)
age_entry = tk.Entry(root, font=custom_font)
age_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="⏱ Tenure:", font=custom_font, bg="pink", fg = "navy").grid(row=5, column=0, sticky='e', padx=10, pady=5)
tenure_entry = tk.Entry(root, font=custom_font)
tenure_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="💰 Balance:", font=custom_font, bg="pink", fg = "navy").grid(row=6, column=0, sticky='e', padx=10, pady=5)
balance_entry = tk.Entry(root, font=custom_font)
balance_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="📦 Number of Products:", font=custom_font, bg="pink", fg = "navy").grid(row=7, column=0, sticky='e', padx=10, pady=5)
num_of_products_entry = tk.Entry(root, font=custom_font)
num_of_products_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="🧐 Has Credit Card (1/0):", font=custom_font, bg="pink", fg = "navy").grid(row=8, column=0, sticky='e', padx=10, pady=5)
has_cr_card_var = tk.IntVar()
tk.Entry(root, textvariable=has_cr_card_var, font=custom_font).grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="🤸 Is Active Member (1/0):", font=custom_font, bg="pink", fg = "navy").grid(row=9, column=0, sticky='e', padx=10, pady=5)
is_active_member_var = tk.IntVar()
tk.Entry(root, textvariable=is_active_member_var, font=custom_font).grid(row=9, column=1, padx=10, pady=5)

tk.Label(root, text="💲 Estimated Salary:", font=custom_font, bg="pink", fg = "navy").grid(row=10, column=0, sticky='e', padx=10, pady=5)
estimated_salary_entry = tk.Entry(root, font=custom_font)
estimated_salary_entry.grid(row=10, column=1, padx=10, pady=5)

# Create a button to make predictions
predict_button = tk.Button(root, text="🔮 Predict Churn", command=predict_churn, font=("Monotype Corsiva", 18), bg="pink", fg="navy")
predict_button.grid(row=11, columnspan=2, pady=20)

# Run the application
root.mainloop()
