import streamlit as st

st.title("✅ Streamlit is Working!")

name = st.text_input("Enter your name:", "")

if name:
    st.write(f"Hello, **{name}** 👋 Streamlit is running properly!")
else:
    st.write("Type something above to test input handling.")
