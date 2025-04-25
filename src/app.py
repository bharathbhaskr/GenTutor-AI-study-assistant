import streamlit as st
from rag_chat import RAGChat

st.set_page_config(page_title="GenTutor", page_icon="🎓")
st.title("🎓 GenTutor – RAG Chatbot")

if "bot" not in st.session_state:
    st.session_state.bot = RAGChat()
if "history" not in st.session_state:
    st.session_state.history = []

for sender, msg in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(msg)

if prompt := st.chat_input("Ask a question…"):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            answer = st.session_state.bot.ask(prompt)
        except Exception as e:
            answer = f":red[Error] – {e}"
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))