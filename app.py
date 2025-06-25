"""Run with: streamlit run app.py"""
import streamlit as st
from inference import summarize

st.set_page_config(page_title="VN News Summarizer", page_icon="📰")
st.title("Tóm tắt báo tiếng Việt")

url_or_text = st.text_area("Dán URL bài báo **hoặc** đoạn văn bản")
num_sent = st.slider("Số câu tóm tắt", 1, 10, 3)
run_btn = st.button("Tóm tắt")

if run_btn and url_or_text.strip():
    with st.spinner("Đang tóm tắt …"):
        summary, idx, full = summarize(url_or_text.strip(), num_sent)
    st.subheader("Tóm tắt")
    st.success(" ".join(summary))
    st.subheader("Bài gốc (đánh dấu câu được chọn)")
    for i, s in enumerate(full):
        if i in idx:
            st.markdown(f"<mark>{s}</mark>", unsafe_allow_html=True)
        else:
            st.write(s)