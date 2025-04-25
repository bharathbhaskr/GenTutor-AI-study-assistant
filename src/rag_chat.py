from __future__ import annotations
import os, pathlib, functools
from typing import List, Union

import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Load Hugging Face API token securely from .streamlit/secrets.toml
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = pathlib.Path("vectorstore")
CHUNK, OVERLAP = 512, 128

SYSTEM_PROMPT = """You are GenTutor, a concise academic assistant.
Use the retrieved context to answer. If you don’t know, say NA.
{context}"""

@functools.lru_cache(maxsize=1)
def _emb(device="cpu"):
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

class RAGChat:
    def __init__(self, model:str="HuggingFaceH4/zephyr-7b-beta", device:str="cpu"):
        # Set up HuggingFaceHub LLM
        self.llm = HuggingFaceHub(
            repo_id=model,
            model_kwargs={
                "temperature": 0.3,
                "max_new_tokens": 512
            }
        )
        self.emb = _emb(device)
        INDEX_DIR.mkdir(exist_ok=True)

    def build_index(self, md_file:str):
        text = pathlib.Path(md_file).read_text(encoding="utf-8")
        docs = [Document(page_content=text, metadata={"source": md_file})]
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK, chunk_overlap=OVERLAP)
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, self.emb)
        vs.save_local(INDEX_DIR)
        print("[OK] Vector store built.")

    def _load_vs(self):
        return FAISS.load_local(
               INDEX_DIR,
               self.emb,
               allow_dangerous_deserialization=True
        )

    def _chain(self):
        retriever = self._load_vs().as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        comp = LLMChainExtractor.from_llm(self.llm)
        comp_retriever = ContextualCompressionRetriever(
            base_compressor=comp, base_retriever=retriever
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(comp_retriever, qa_chain)

    def ask(self, question: str) -> str:
        out = self._chain().invoke({"input": question})
        return out["answer"]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--build-index")
    p.add_argument("--question")
    a = p.parse_args()
    chat = RAGChat()
    if a.build_index:
        chat.build_index(a.build_index)
    elif a.question:
        print(chat.ask(a.question))
    else:
        p.print_help()
