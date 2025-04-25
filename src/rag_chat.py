from __future__ import annotations
import os
import pathlib
import functools
from typing import List, Union

import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

# ─── Load HF token from Streamlit secrets or ENV ──────────────────────────
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise RuntimeError(
        "Hugging Face token missing! Add it to .streamlit/secrets.toml "
        "or set HUGGINGFACEHUB_API_TOKEN in your shell."
    )
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# ─── Constants ──────────────────────────────────────────────────────────────
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = pathlib.Path("vectorstore")
CHUNK, OVERLAP = 512, 128

SYSTEM_PROMPT = (
    "You are GenTutor, a concise academic assistant.\n"
    "Answer the user’s question using ONLY the provided context.\n"
    "Do NOT repeat the context—just give a single short answer."
)

# ─── Cached embedding loader ────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _emb(device: str = "cpu") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

class RAGChat:
    def __init__(
        self,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        device: str = "cpu",
    ):
        # 1) LLM
        self.llm = HuggingFaceHub(
            repo_id=model,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 256},
        )
        # 2) Embeddings
        self.emb = _emb(device)
        # 3) Ensure index directory
        INDEX_DIR.mkdir(exist_ok=True)

        # 4) Retriever
        vs = FAISS.load_local(
            INDEX_DIR,
            self.emb,
            allow_dangerous_deserialization=True
        )
        self.retriever = vs.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        # 5) QA chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}")
        ])
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)

    def build_index(self, md_file: str) -> None:
        text = pathlib.Path(md_file).read_text(encoding="utf-8")
        docs = [Document(page_content=text, metadata={"source": md_file})]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK, chunk_overlap=OVERLAP
        )
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, self.emb)
        vs.save_local(INDEX_DIR)
        print("[OK] Vector store built.")

    def ask(self, question: str) -> str:
        docs = self.retriever.get_relevant_documents(question)
        out = self.qa_chain.invoke({"input": question, "context": docs})

        # Normalize to a string
        if isinstance(out, dict):
            raw = out.get("output") or out.get("answer") or ""
        else:
            raw = str(out)

        # Strip off everything before "Answer:"
        if "Answer:" in raw:
            return raw.split("Answer:", 1)[1].strip()

        # If no marker, just return the raw text
        return raw.strip()
