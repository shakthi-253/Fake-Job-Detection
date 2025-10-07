# ==============================
# Import Libraries
# ==============================
import streamlit as st
import pandas as pd
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ==============================
# Streamlit App
# ==============================
st.title("🧠 Fake Job Posting Detection")
st.write("""
Detect whether a job posting is **real or fake** using Machine Learning (Logistic Regression Model).  
You can also give feedback (✅ Correct / ❌ Incorrect) to help retrain and improve the model.
""")


# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_dataset(path: str):
    df = pd.read_excel(path)
    return df

dataset_path = "Fakejob_dataset.xlsx"  # Ensure this file is in the same folder
df = load_dataset(dataset_path)

st.subheader("📘 Dataset Preview")
st.dataframe(df.head())


# ==============================
# Preprocessing
# ==============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(s):
    s = s.lower()
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocess(s):
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ==============================
# Model Training
# ==============================
@st.cache_resource
def train_model(df):
    text_cols = ['title', 'description', 'requirements', 'company_profile', 'benefits', 'industry']
    existing_text_cols = [c for c in text_cols if c in df.columns]
    df[existing_text_cols] = df[existing_text_cols].astype(str).fillna('')
    df['text'] = df[existing_text_cols].agg(' ||| '.join, axis=1)
    df['cleaned'] = df['text'].apply(clean_text)
    df['processed'] = df['cleaned'].apply(preprocess)

    X = df['processed']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return model, tfidf, results


with st.spinner("⏳ Training model..."):
    model, tfidf, results = train_model(df)

st.success("✅ Model trained successfully!")


# ==============================
# Display Results
# ==============================
st.subheader("📊 Model Performance")
st.write(f"- Accuracy:  {results['accuracy']*100:.2f}%")
st.write(f"- Precision: {results['precision']*100:.2f}%")
st.write(f"- Recall:    {results['recall']*100:.2f}%")
st.write(f"- F1 Score:  {results['f1']*100:.2f}%")


# ==============================
# Prediction Interface
# ==============================
st.subheader("📝 Test with Your Own Job Posting")

user_input = st.text_area("Enter job description/title/requirements:")

if st.button("Predict"):
    if user_input.strip():
        processed = preprocess(clean_text(user_input))
        vec = tfidf.transform([processed])
        pred = model.predict(vec)[0]
        label = "Real Job" if pred == 0 else "Fake Job"
        st.success(f"Prediction: {label}")

        st.session_state["last_input"] = user_input
        st.session_state["last_pred"] = int(pred)
    else:
        st.warning("Please enter job details for prediction.")


# ==============================
# Feedback System
# ==============================
feedback_file = "feedback.csv"
st.markdown("---")
st.subheader("🗣️ Was this prediction correct?")
feedback_choice = st.radio("Mark prediction as:", ["Correct", "Incorrect"])
if st.button("Save Feedback"):
    if "last_input" in st.session_state:
        df_fb = pd.DataFrame(
            [[st.session_state["last_input"], st.session_state["last_pred"], feedback_choice]],
            columns=["Text", "Prediction", "Feedback"]
        )
        if os.path.exists(feedback_file):
            old = pd.read_csv(feedback_file)
            df_fb = pd.concat([old, df_fb], ignore_index=True)
        df_fb.to_csv(feedback_file, index=False)
        st.info("Your feedback was saved to the feedback file.")
    else:
        st.warning("Please make a prediction before giving feedback.")


# ==============================
# Retraining with Local Excel + Feedback
# ==============================
st.markdown("---")
st.subheader("🔁 Retrain Model with Local Excel + Feedback")

if st.button("Retrain Model"):
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        incorrect = feedback_df[feedback_df["Feedback"] == "Incorrect"]
        if not incorrect.empty:
            incorrect = incorrect.rename(columns={"Text": "description"})
            incorrect["label"] = incorrect["Prediction"].apply(lambda x: 1 - x)
            new_df = pd.concat([df, incorrect], ignore_index=True)

            with st.spinner("Retraining model using local Excel + feedback CSV..."):
                model, tfidf, results = train_model(new_df)
            st.success("✅ Model retrained using local Excel + feedback CSV!")
        else:
            st.info("No incorrect feedback found — model remains unchanged.")
    else:
        st.warning("No feedback data found to retrain.")


# ==============================
# View Feedback
# ==============================
if os.path.exists(feedback_file):
    st.subheader("🗂️ Recent Feedback")
    fb = pd.read_csv(feedback_file)
    st.dataframe(fb.tail(10))


# ==============================
# Cache Control
# ==============================
if st.button("🧹 Clear Cache & Restart"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
