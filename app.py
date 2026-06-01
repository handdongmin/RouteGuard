"""Streamlit entry point for EscapeRoute Vision."""

import streamlit as st


st.set_page_config(page_title="EscapeRoute Vision", layout="wide")
st.title("EscapeRoute Vision")
st.write("Indoor evacuation-route risk analysis from a short smartphone video.")

video = st.file_uploader("Upload a route video", type=["mp4", "mov", "avi"])

if video is None:
    st.info("Upload a short video to start the safety audit.")
else:
    st.video(video)
    st.warning("Analysis pipeline scaffolded. Detection will be connected next.")
