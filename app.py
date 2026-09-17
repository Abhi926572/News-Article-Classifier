import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Page Configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

# App Title & Description
st.title("📰 News Article Classification App")
st.write("An intelligent application that automatically categorizes news articles into **Technology, Business, Sports, Entertainment, or Politics** using Machine Learning.")

# Load Dataset & Train Model (Cached for performance)
@st.cache_resource
def load_and_train_model():
    csv_path = 'news_article_classification_dataset.csv'
    if not os.path.exists(csv_path):
        st.error(f"Dataset file '{csv_path}' not found in the directory. Please add it.")
        return None, None
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Combine title and content for better training context
    df['text'] = df['title'].fillna('') + " " + df['content'].fillna('')
    X = df['text']
    y = df['category']
    
    # Build a text classification pipeline (TF-IDF + Naive Bayes)
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', max_features=1000),
        MultinomialNB()
    )
    
    # Train the model on the full dataset
    model.fit(X, y)
    return model, df

model, df = load_and_train_model()

if model is not None:
    st.sidebar.header("📊 Dataset Overview")
    st.sidebar.write(f"Total Articles Loaded: **{len(df)}**")
    st.sidebar.write("Categories available:")
    for cat in df['category'].unique():
        st.sidebar.text(f"• {cat}")

    # Main Interface for User Input
    st.subheader("🔍 Classify a New Article")
    
    input_title = st.text_input("Article Title / Headline:", placeholder="e.g., Central bank announces unexpected rate cut...")
    input_content = st.text_area("Article Content/Body:", placeholder="Enter full news text or summary here...")

    if st.button("Predict Category", type="primary"):
        if input_title.strip() == "" and input_content.strip() == "":
            st.warning("⚠️ Please enter a title or content to classify.")
        else:
            # Combine inputs
            sample_text = [input_title + " " + input_content]
            
            # Predict category
            prediction = model.predict(sample_text)[0]
            probabilities = model.predict_proba(sample_text)[0]
            classes = model.classes_
            
            # Display Result
            st.success(f"### Predicted Category: **{prediction}**")
            
            # Display Confidence Breakdown
            st.subheader("📈 Model Confidence Breakdown")
            prob_df = pd.DataFrame({
                'Category': classes,
                'Confidence': [f"{p*100:.2f}%" for p in probabilities]
            }).sort_values(by='Confidence', ascending=False)
            
            st.dataframe(prob_df, hide_index=True, use_container_width=True)