import os
import sys
import streamlit as st

# ─── Make sure we can import your RAG logic ─────────────────────────────────
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from rag_chat import RAGChat

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenTutor AI & Prompt Engineering",
    page_icon="🎓",
    layout="wide",
)

# ─── Sidebar: enhanced UI ───────────────────────────────────────────────────
logo_path = os.path.join(THIS_DIR, "northeasternuniversity.png")
st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("## GenTutor AI\n_v1.0.0_")

st.sidebar.markdown(
    "**Your interactive RAG chatbot**\n"
    "– Academic assistant built with prompt engineering."
)


st.sidebar.markdown("### How to use")
st.sidebar.markdown(
    "1. Ask any question about generative AI concepts.\n"
    "2. (Optional) Upload a `.md` file to rebuild index.\n"
    "3. Clear chat to start over."
)

# —–– Optionally rebuild the index from a new markdown doc —––––
upload = st.sidebar.file_uploader(
    "📄 Upload .md to rebuild index", type=["md"]
)
if upload:
    # write to temp file so build_index can read it
    tmp = os.path.join(THIS_DIR, "temp_upload.md")
    with open(tmp, "wb") as f:
        f.write(upload.getbuffer())
    st.session_state.bot.build_index(tmp)
    st.sidebar.success("✅ Vector index rebuilt!")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.history = []
    st.experimental_rerun()

st.sidebar.markdown(
    "**Built with:** Streamlit · LangChain · Hugging Face · FAISS\n\n"
    "[🛠️ GitHub](https://github.com/your-username/GenTutor-RAG)"
)

# ─── Main header & subheader ────────────────────────────────────────────────
st.title("🎓 GenTutor – AI & Prompt Engineering Assistant")
st.subheader("Ask me anything about generative AI concepts and prompt engineering.")

# ─── Initialize RAG bot & chat history ──────────────────────────────────────
if "bot" not in st.session_state:
    st.session_state.bot = RAGChat()
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = []

# ─── Render previous messages ────────────────────────────────────────────────
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)

# ─── Accept new input & display response ────────────────────────────────────
if user_input := st.chat_input("Your question…"):
    # Store & show the user’s message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the assistant’s reply
    with st.chat_message("assistant"):
        try:
            answer = st.session_state.bot.ask(user_input)
        except Exception as e:
            answer = f"**Error**: {e}"
        st.markdown(answer)

    # Save reply in history
    st.session_state.history.append(("assistant", answer))





