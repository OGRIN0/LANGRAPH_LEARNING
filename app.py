import streamlit as st
from transformers import pipeline
from bot import chatbot

mybot=chatbot()
workflow=mybot()

st.title("ChatBot with LangGraph")
st.write("Ask any question, and I'll try to answer it!")

question = st.text_input("Enter your question here:")
input={"messages": [question]}

if st.button("Get Answer"):
    if input:
        response=workflow.invoke(input)
        st.write("**Answer:**", response['messages'][-1].content)
    else:
        st.warning("Please enter a question to get an answer.")

st.markdown("---")
st.caption("Powered by Streamlit and Transformers")
