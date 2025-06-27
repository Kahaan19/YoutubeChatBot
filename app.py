# app.py
import streamlit as st
from main import get_answer

st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("🎥 YouTube RAG Chatbot using LangChain + Gemini")

video_url = st.text_input("Enter YouTube Video URL")
question = st.text_input("Ask a question about the video")

if st.button("Get Answer"):
    if video_url and question:
        with st.spinner("Processing..."):
            answer = get_answer(video_url, question)
            st.success("Answer:")
            st.write(answer)
    else:
        st.error("Please enter both video URL and question.")
