import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset and train model
df = pd.read_csv('spam.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['cleaned'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})

model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("📧 Email Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

user_input = st.text_area("Enter email or SMS content:")

if st.button("Check"):
    cleaned_input = clean_text(user_input)
    vector_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vector_input)[0]
    result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
    st.subheader(result)
