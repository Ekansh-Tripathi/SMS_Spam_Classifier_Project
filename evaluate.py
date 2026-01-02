import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load saved model and vectorizer
model = pickle.load(open('models/nb_model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))

# Load the correct dataset (since it's CSV)
df = pd.read_csv('archive/spam.csv', encoding='latin1')

# Display dataset info
print("✅ Loaded dataset: archive/spam.csv")
print(f"Total Rows: {df.shape[0]} | Columns: {list(df.columns)}\n")

# Rename columns if needed (some spam.csv files use 'v1' and 'v2')
if 'v1' in df.columns and 'v2' in df.columns:
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Transform test data
X_test_tfidf = tfidf.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
