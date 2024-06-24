import streamlit as st
import langchain_helper as lch
import textwrap
st.title("YOUTUBE ASSISTANT")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.text_area(label="ENTER YOUTUBE URL", max_chars=100)
        query = st.text_area(label="Ask me about the video", max_chars=100, key="query")
        submit_button = st.form_submit_button(label="SUBMIT")

if submit_button and youtube_url and query:
    with st.spinner("Processing..."):
        try:
            db = lch.create_vector_db_from_youtube_url(youtube_url)
            response, docs = lch.get_response_from_query(db, query, k=4)
            st.subheader('Answer:')
            st.text(textwrap.fill(response))
        except Exception as e:
            st.error(f"An error occurred: {e}")


