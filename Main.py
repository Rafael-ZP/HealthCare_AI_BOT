
import streamlit as st
from Helper import create_vector,get_qa_chain


st.title("Rafael's Q&A Bot Welcomes you.....🔥")

bttn = st.button("Create a Knowledge Base")

if bttn:
    pass

question=st.text_input("Question Please: ")

if question:
    chain = get_qa_chain()
    response= chain(question) # type: ignore

    st.header("Answer: ")
    st.write(response["result"])