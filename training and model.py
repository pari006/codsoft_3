import pandas as pd
import os,re, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score,classification_report
import tkinter as tk
from tkinter import ttk, messagebox

#Download dataset from here:
    #https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Set base directory and load data
BASE_DIR = r"D:\Projects\Spam SMS Detection"  # Replace with your actual file path
file_path = os.path.join(BASE_DIR, 'spam.csv')


# Load data with error handling
try:
    df = pd.read_csv(file_path, encoding='latin-1')
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at {file_path}")
except Exception as e:
    raise Exception(f"Error reading file: {str(e)}")
    
    
# Clean and preprocess data
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
df = df.drop_duplicates().dropna()  # Remove any NaN values and Duplicates

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

df['text'] = df['text'].apply(clean_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
x = df['text']
y = df['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)


# #model training with naive bayes
# model_nb = MultinomialNB()
# model_nb.fit(x_train_tfidf,y_train)

# #prediction and evaluation through naive bayes
# y_pred_nb = model_nb.predict(x_test_tfidf)
# print("Naive Bayes accuracy",accuracy_score(y_test,y_pred_nb))
# print(classification_report(y_test, y_pred_nb))


# #model training using logistic regression 
# model_lr = LogisticRegression(max_iter=1000)
# model_lr.fit(x_train_tfidf,y_train)

# #prediction and evaluation using logistic regression 
# y_pred_lr = model_lr.predict(x_test_tfidf)
# print("Logistic Regression accuracy: ", accuracy_score(y_test,y_pred_lr))
# print(classification_report(y_test,y_pred_lr))

#model training using svm 
model_svm = SVC()
model_svm.fit(x_train_tfidf,y_train)

#prediction and evaluation using svm 
# y_pred_svm = model_svm.predict(x_test_tfidf)
# print("SVM accuracy: ",accuracy_score(y_test,y_pred_svm))
# print(classification_report(y_test,y_pred_svm))
#After checking all the model accuracy, SVM model has the highest accuracy of 0.9796905222437138

#Save the model and vectorizer
joblib.dump(model_svm, os.path.join(BASE_DIR, 'svm_model.pkl'))
joblib.dump(vectorizer,os.path.join(BASE_DIR,'tfidf_vectorizer.pkl'))

   
# Create main window
root = tk.Tk()
root.title("Spam SMS Detector")
root.geometry("750x650")
root.resizable(False, False)

# Styling
style = ttk.Style()
style.configure('TButton', font=('Monotype Corsiva', 22) )

# Header
header_frame = ttk.Frame(root)
header_frame.pack(pady=10)

title_label = ttk.Label(header_frame, text="Spam SMS Detector", font=('Forte', 40), foreground="FireBrick4")
title_label.pack()

subtitle_label = ttk.Label(header_frame, text="Check if your message is spam or ham !?",font=('Lucida Calligraphy', 22), foreground="Tomato")
subtitle_label.pack(pady=15)

# Message Input
input_frame = ttk.Frame(root)
input_frame.pack(pady=20, padx=15)

message_label = ttk.Label(input_frame, text="Type your Message here: 👇",font=('Lucida Calligraphy', 20), foreground="DeepPink3")
message_label.pack(anchor="w")

message_text = tk.Text(input_frame, height=6, width=40, font=('Comic Sans MS', 15), 
                      borderwidth=6, relief="ridge",
                      highlightthickness=3, 
                      highlightbackground="hotpink",
                      highlightcolor="azure4")
message_text.pack()

# Check Button
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

# Load the model and vectorizer
model_svm = joblib.load(os.path.join(BASE_DIR, 'svm_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))

def check_spam():
    message = message_text.get("1.0", tk.END).strip()
    if not message:
        messagebox.showwarning("Warning", "Please enter a message")
        return
    
    # Clean the message and predict
    cleaned_message = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned_message])
    prediction = model_svm.predict(message_tfidf)
    
    if prediction[0] == 1:
        show_result("Spam Detected!", "This message is likely spam ⚠", "red")
    else:
        show_result("Ham Message 🎉 ", "This message appears legitimate ", "green")

check_button = ttk.Button(button_frame, text="Check Message", command=check_spam)
check_button.pack()

# Result Display
result_frame = ttk.Frame(root)
result_frame.pack(pady=10)

def show_result(title, description, color):
    # Clear previous result
    for widget in result_frame.winfo_children():
        widget.destroy()
    
    # Create new result
    result_box = ttk.Frame(result_frame)
    result_box.pack(fill="x", padx=20)
    
    # Configure frame based on result
    if color == "red":
        bg_color = "#fee2e2"
        border_color = "#ef4444"
    else:
        bg_color = "#dcfce7" 
        border_color = "#22c55e"
    
    result_box.configure(style="Result.TFrame")
    style.configure("Result.TFrame", background=bg_color)
    
    # Add colored border
    border = ttk.Frame(result_box, width=4, style="Border.TFrame")
    style.configure("Border.TFrame", background=border_color)
    border.pack(side="left", fill="y")
    
    # Result content
    content_frame = ttk.Frame(result_box)
    content_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    title_label = ttk.Label(content_frame, text=title, font=('Eras Demi ITC', 18))
    title_label.pack(anchor="w")
    
    desc_label = ttk.Label(content_frame, text=description, font=('High Tower Text', 15))
    desc_label.pack(anchor="w")

# Footer
footer_frame = ttk.Frame(root)
footer_frame.pack(side="bottom", pady=5)

footer_label = ttk.Label(footer_frame, text="Powered by SVM Model (Accuracy: 97.9%)",font=('MV Boli', 15), foreground="Sienna")
footer_label.pack()

root.mainloop()

