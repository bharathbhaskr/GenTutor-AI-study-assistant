import os
import sys
import streamlit as st

# Ensure this folder (src/) is on sys.path so we can import rag_chat.py directly
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from rag_chat import RAGChat

st.set_page_config(page_title="GenTutor", page_icon="🎓")
st.title("🎓 GenTutor – RAG Chatbot")

# Initialize bot and history
if "bot" not in st.session_state:
    st.session_state.bot = RAGChat()
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str,str]] = []

# Render past messages
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)

# Accept new user input
if user_input := st.chat_input("Ask a question…"):
    # Display user
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and display answer
    with st.chat_message("assistant"):
        try:
            answer = st.session_state.bot.ask(user_input)
        except Exception as e:
            answer = f"**Error**: {e}"
        st.markdown(answer)

    # Save answer
    st.session_state.history.append(("assistant", answer))
