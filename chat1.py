# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from string import punctuation
import re

# +
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('Mental_Health_FAQ.CSV'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# -

import os
for dirname, _, filenames in os.walk('20000-Uttrance.CSV'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os
for dirname, _, filenames in os.walk('Bitext.CSV'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import os
for dirname, _, filenames in os.walk('full_Chat_data.CSV'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = None # specify 'None' if want to read whole file
data_mental = pd.read_csv('Mental_Health_FAQ.csv', delimiter=',', nrows = nRowsRead)
data_mental.dataframeName = 'Mental_Health_FAQ.csv'
nRow, nCol = data_mental.shape
print(f'There are {nRow} rows and {nCol} columns')

nRowsRead = None # specify 'None' if want to read whole file
data_20000_Uttrance = pd.read_csv('20000-Uttrance.csv', delimiter=',', nrows = nRowsRead)
data_20000_Uttrance.dataframeName = '20000-Uttrance.csv'
nRow, nCol = data_20000_Uttrance.shape
print(f'There are {nRow} rows and {nCol} columns')

nRowsRead = None # specify 'None' if want to read whole file
data_bitext = pd.read_csv('Bitext.csv', delimiter=',', nrows = nRowsRead)
data_bitext.dataframeName = 'Bitext.csv'
nRow, nCol = data_bitext.shape
print(f'There are {nRow} rows and {nCol} columns')

nRowsRead = None # specify 'None' if want to read whole file
data_alz = pd.read_csv('full_Chat_data.csv', delimiter=',', nrows = nRowsRead)
data_alz.dataframeName = 'full_Chat_data.csv'
nRow, nCol = data_alz.shape
print(f'There are {nRow} rows and {nCol} columns')

# +
#data = pd.concat([data_alz, data_mental,data_bitext,data_20000_Uttrance], ignore_index=True)
# -

data = pd.concat([data_alz, data_mental], ignore_index=True)

# +


# Specify the path where you want to save the CSV file
csv_filename = 'concatenated_data.csv'

# Save the concatenated data to a CSV file
data.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' has been created.")

# -

import os
for dirname, _, filenames in os.walk('train.CSV'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# +
# data = pd.concat([data_alz, data_mental], ignore_index=True)
# -

data

data.shape

data['Questions'] = data['Questions'].str.lower()

# +
# data_alz['Questions'] = data_alz['Questions'].str.lower()
# -

data_mental.head()

data_alz.head()

data_mental.head()

data.shape

# Remove unnecessary characters
data['Questions'] = data['Questions'].str.replace('[^\w\s]', '')

# +
# # Remove unnecessary characters
# data_mental['Questions'] = data_mental['Questions'].str.replace('[^\w\s]', '')

# +
# # Handle missing values if any
# data.dropna(inplace=True)
# -

data.head()

# +
# # Handle missing values if any
# data_mental.dropna(inplace=True)
# -

data['Intent'] = data['Questions']
questions_response_counts = data.groupby('Intent').size().reset_index(name='Count')

data.head()

# +
# data_mental['Intent'] = data_mental['Questions']
# questions_response_counts = data_mental.groupby('Intent').size().reset_index(name='Count')
# -

# Exploratory Data Analysis
intent_counts = data['Questions'].value_counts()

# +
# # Exploratory Data Analysis
# intent_counts = data_mental['Questions'].value_counts()

# +
# # Visualize intent distribution
# fig = px.bar(x=intent_counts.index, y=intent_counts.values, labels={'x': 'Intents', 'y': 'Count'})
# fig.show()

# +
# # Calculate average number of questions per intent
# avg_questions = data.groupby('Intent').size().mean()

# # Visualize average pattern count per intent
# fig = px.bar(x=questions_response_counts['Intent'], y=questions_response_counts['Count'],
#              labels={'x': 'Intents', 'y': 'Average Questions Count'})
# fig.show()
# -

# Intent Prediction Model
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Questions'], data['Intent'], test_size=0.2, random_state=42)

# +
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


# +

def train_model(X_train_vec, y_train):
    model = LinearSVC()
    model.fit(X_train_vec, y_train)
    return model

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    return report, y_pred

# Vectorize the text data
X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

# Train the model
model = train_model(X_train_vec, y_train)

# Evaluate the model
report, y_pred = evaluate_model(model, X_test_vec, y_test)
print(report)

# Visualize model performance
metrics = ['precision', 'recall', 'f1-score', 'support']
scores = classification_report(y_test, y_pred, output_dict=True)['weighted avg']

# +
# import joblib

# # Train your LinearSVC model
# model = LinearSVC()
# model.fit(X_train_vec, y_train)

# # Save the trained model to a file
# joblib.dump(model, 'trained_model.pkl')


# +
# import joblib

# # Train your LinearSVC model
# model = LinearSVC()
# model.fit(X_train_vec, y_train)

# # Save the trained model to a file
# joblib.dump(model, 'trained_model.pkl')

# # Save the vectorizer to a file
# joblib.dump(vectorizer, 'vectorizer.pkl')


# +
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Calculate confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=metrics, yticklabels=metrics)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# -

fig = px.bar(x=metrics, y=[scores[metric] for metric in metrics], labels={'x': 'Metrics', 'y': 'Score'})
fig.show()

# +
exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

print("Welcome to the Mental Health FAQ Chatbot!")
print("Ask a question or enter 'quit', 'pause', 'exit', 'goodbye', 'bye', 'later', or 'stop' to exit.")

while True:
    user_input = input("User: ")
    
    if user_input.lower() in exit_commands:
        print("Chatbot: Goodbye!")
        break
    
    # Vectorize user input
    user_input_vec = vectorizer.transform([user_input.lower()])
    
    # Predict the intent
    predicted_intent = model.predict(user_input_vec)[0]
    
    # Implement response generation mechanism based on predicted intent
    if predicted_intent in data['Questions'].values:
        response = data[data['Questions'] == predicted_intent]['Answers'].values[0]
    else:
        response = "Sorry, I don't have information about this topic."
    
    print("Chatbot:", response)
# -

# 	Answers
# 8	Just as there are different types of medications for physical illness, different treatment options are available for individuals with mental illness. Treatment works differently for different people. It is important to find what works best for you or your child.


