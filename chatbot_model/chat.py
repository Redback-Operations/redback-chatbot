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
for dirname, _, filenames in os.walk('full_Chat_data.CSV'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = None # specify 'None' if want to read whole file
data_mental = pd.read_csv('Mental_Health_FAQ.csv', delimiter=',', nrows = nRowsRead)
data_mental.dataframeName = 'Mental_Health_FAQ.csv'
nRow, nCol = data_mental.shape
print(f'There are {nRow} rows and {nCol} columns')

nRowsRead = None # specify 'None' if want to read whole file
data_alz = pd.read_csv('full_Chat_data.csv', delimiter=',', nrows = nRowsRead)
data_alz.dataframeName = 'full_Chat_data.csv'
nRow, nCol = data_alz.shape
print(f'There are {nRow} rows and {nCol} columns')

data = pd.concat([data_alz, data_mental], ignore_index=True)



# +
# data = pd.concat([data_alz, data_mental], ignore_index=True)
# -

data

data.shape

data['Questions'] = data['Questions'].str.lower()

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
# -

data['Intent'].isnull()

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

# #### old model

# + jupyter={"outputs_hidden": true}
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.metrics import classification_report

# def vectorize_text(X_train, X_test):
#     vectorizer = TfidfVectorizer()
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)
#     return X_train_vec, X_test_vec, vectorizer

# def train_model(X_train_vec, y_train):
#     model = LinearSVC()
#     model.fit(X_train_vec, y_train)
#     return model

# def evaluate_model(model, X_test_vec, y_test):
#     y_pred = model.predict(X_test_vec)
#     report = classification_report(y_test, y_pred)
#     return report, y_pred

# # Vectorize the text data
# X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

# # Train the model
# model = train_model(X_train_vec, y_train)

# # Evaluate the model
# report, y_pred = evaluate_model(model, X_test_vec, y_test)
# print(report)

# # Visualize model performance
# metrics = ['precision', 'recall', 'f1-score', 'support']
# scores = classification_report(y_test, y_pred, output_dict=True)['weighted avg']
# -

# #### New model 

# +
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Additional imports for Word Embeddings
from gensim.models import Word2Vec

# Additional imports for Transformer-based models
from transformers import BertTokenizer, BertModel


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

def vectorize_text(X_train, X_test, vectorizer_type='tfidf'):
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    elif vectorizer_type == 'count':
        vectorizer = CountVectorizer()
    elif vectorizer_type == 'word2vec':
        # Example code for Word2Vec vectorization
        X_train = [nltk.word_tokenize(doc) for doc in X_train]
        X_test = [nltk.word_tokenize(doc) for doc in X_test]
        model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)
        vectorizer = model.wv
    elif vectorizer_type == 'bert':
        # Example code for BERT vectorization
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        X_train_encoded = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
        X_test_encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
        return X_train_encoded, X_test_encoded, None

    X_train_preprocessed = [preprocess_text(text) for text in X_train]
    X_test_preprocessed = [preprocess_text(text) for text in X_test]

    X_train_vec = vectorizer.fit_transform(X_train_preprocessed)
    X_test_vec = vectorizer.transform(X_test_preprocessed)

    return X_train_vec, X_test_vec, vectorizer

def train_model(X_train_vec, y_train, model_type='svm'):
    if model_type == 'svm':
        model = LinearSVC()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()

    # Hyperparameter tuning using GridSearchCV
    if model_type == 'svm':
        param_grid = {'C': [0.1, 1, 10, 100]}
    elif model_type == 'random_forest':
        param_grid = {'n_estimators': [50, 100, 200]}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_vec, y_train)
    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    return best_model

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)
    return report, y_pred

# Data preprocessing
X_train_processed = [preprocess_text(text) for text in X_train]
X_test_processed = [preprocess_text(text) for text in X_test]

# Vectorize the text data
X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train_processed, X_test_processed, vectorizer_type='tfidf')

# Train the model
model = train_model(X_train_vec, y_train, model_type='svm')

# Evaluate the model
report, y_pred = evaluate_model(model, X_test_vec, y_test)
print(report)

# +
# Evaluate the model
report, y_pred = evaluate_model(model, X_test_vec, y_test)

# Extract the accuracy, macro avg, and weighted avg from the report
lines = report.split('\n')

accuracy_line = [line for line in lines if line.strip().startswith('accuracy')]
macro_avg_line = [line for line in lines if line.strip().startswith('macro avg')]
weighted_avg_line = [line for line in lines if line.strip().startswith('weighted avg')]

if macro_avg_line:
    macro_avg = macro_avg_line[0].split()[2:6]
    print("Macro avg:", " ".join(macro_avg))
else:
    print("Macro avg not found in the classification report.")

if weighted_avg_line:
    weighted_avg = weighted_avg_line[0].split()[2:6]
    print("Weighted avg:", " ".join(weighted_avg))
else:
    print("Weighted avg not found in the classification report.")


# +
# Evaluate the model
report, y_pred = evaluate_model(model, X_test_vec, y_test)

# Extract the lines containing precision, recall, F1-score, and support
lines = report.split('\n')

# Extract values for Macro avg
macro_avg_line = [line for line in lines if line.strip().startswith('macro avg')][0]
macro_avg_values = macro_avg_line.split()[2:6]

# Extract values for Weighted avg
weighted_avg_line = [line for line in lines if line.strip().startswith('weighted avg')][0]
weighted_avg_values = weighted_avg_line.split()[2:6]

# Print the precision, recall, and F1-score for Macro avg
print("Macro avg:")
print("Precision:", macro_avg_values[0])
print("Recall:", macro_avg_values[1])
print("F1-score:", macro_avg_values[2])
print("Support:", macro_avg_values[3])

# Print the precision, recall, and F1-score for Weighted avg
print("\nWeighted avg:")
print("Precision:", weighted_avg_values[0])
print("Recall:", weighted_avg_values[1])
print("F1-score:", weighted_avg_values[2])
print("Support:", weighted_avg_values[3])

# +
import numpy as np
import matplotlib.pyplot as plt

# Precision, recall, and F1-score values
precision = [0.13, 0.23]  # Macro avg and Weighted avg precision
recall = [0.15, 0.27]  # Macro avg and Weighted avg recall
f1_score = [0.14, 0.24]  # Macro avg and Weighted avg F1-score

# Class labels
labels = ['Macro avg', 'Weighted avg']

# Plotting
x = np.arange(len(labels))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(8, 6))

# Plotting precision
ax.bar(x - width, precision, width, label='Precision', color='blue')
# Plotting recall
ax.bar(x, recall, width, label='Recall', color='green')
# Plotting F1-score
ax.bar(x + width, f1_score, width, label='F1-score', color='orange')

# Adding labels, title, and legend
ax.set_xlabel('Averaging Method')
ax.set_ylabel('Score')
ax.set_title('Precision, Recall, and F1-score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()

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


