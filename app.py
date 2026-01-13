import streamlit as st
import pickle
import numpy as np

import joblib
@st.cache_resource
def load_model():
    tfidf = joblib.load('tfidf.joblib')
    model = joblib.load('model.joblib')
    return model, tfidf


st.set_page_config(page_title="Emotion Detector", layout="wide")
st.title("Emotion Detector")
st.markdown("**Enter any sentence → Get emotion instantly**")


text_input = st.text_area(
    "Type your sentence here:", 
    placeholder="e.g., I feel so happy today!",
    height=120,
    label_visibility="collapsed"
)


if st.button(" Detect Emotion", type="primary", use_container_width=True):
    if text_input.strip():
        model, tfidf = load_model()
        
        
        vec = tfidf.transform([text_input])
        emotion = model.predict(vec)[0]
        confidence = np.max(model.predict_proba(vec)) * 100
        
       
        st.markdown(f"""
        ##  **{emotion.upper()}**
        **Confidence: {confidence:.1f}%**
        
        *Your text:* "{text_input}"
        """)
        
        
        emotion_list = [
    'happiness', 'love', 'relief', 'fun', 'enthusiasm', 'neutral', 
    'empty', 'surprise', 'anger', 'hate', 'sadness', 'worry', 'boredom'
]
        st.markdown(f"### {emotion} detected!")
        
    else:
        st.warning("Please enter some text first!")

st.markdown("---")
st.markdown("Developed by Hassaan Yousaf | [GitHub](https://github.com/hassaanyousaf) | [LinkedIn](https://www.linkedin.com/in/hassaan-yousaf1/)")