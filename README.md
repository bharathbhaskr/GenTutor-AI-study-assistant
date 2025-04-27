# GenTutor – RAG + Prompt Engineering Study Assistant

**GenTutor** is a Retrieval-Augmented Generation (RAG) and Prompt Engineering based AI assistant that provides high-quality, context-grounded answers about Generative AI concepts.  
It combines advanced retrieval techniques with customized prompt templates to minimize hallucinations and deliver reliable, domain-specific responses.

---

## Built Using
- **LangChain** for retrieval orchestration and prompt management
- **Hugging Face Hub** for Large Language Model access
- **sentence-transformers/all-MiniLM-L6-v2** for high-quality embeddings
- **FAISS** for efficient vector similarity search
- **Streamlit** for a clean, interactive front-end

---

## Repository Structure
```
gen-tutor-project/
├── data/
│   ├── raw/               # (Optional) Source documents
│   └── processed/         # Combined knowledge base files
├── src/
│   ├── data_scraper.py     # (Optional) Script for fetching source content
│   ├── rag_chat.py         # RAGChat class + CLI for indexing and querying
│   └── app.py              # Streamlit-based web application
├── vectorstore/            # FAISS vector index
├── .streamlit/
│   └── secrets.toml        # Hugging Face API token storage
├── requirements.txt
└── README.md               # This file
```

---

## Quickstart Guide

### 1. Clone the Repository and Set Up Virtual Environment
```bash
git clone <your-repo-url> gen-tutor-project
cd gen-tutor-project
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
**macOS M1/M2 Note:**  
If `faiss-cpu` installation fails, install FAISS via Homebrew:
```bash
brew install faiss
```
Then update your `requirements.txt` accordingly.

---

### 3. Configure Hugging Face API Token
Create or edit `.streamlit/secrets.toml`:
```toml
HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
```
**Note**: Keep your token private; do not push it to public repositories.

Message us if you wnat to use the token

---

### 4. Prepare the Knowledge Base
Prepare or scrape documents and save them into:
```
data/processed/combined_data.md
```

---

## Building and Running the System

### Build the FAISS Index
```bash
python -m src.rag_chat --build-index data/processed/combined_data.md
```
Creates a FAISS vector store under `/vectorstore/`.

---

### Launch the Streamlit App
```bash
streamlit run src/app.py
```
Access the app at: [http://localhost:8501](http://localhost:8501)

**Example Queries:**
- "Describe Retrieval-Augmented Generation (RAG)."
- "What strategies improve prompt engineering results?"
- "Why is fine-tuning important for specialized tasks?"

---

## Development and Testing

- **Rebuild Index After Updating Data:**
  ```bash
  python -m src.rag_chat --build-index data/processed/combined_data.md
  ```
- **Run One-off CLI Queries:**
  ```bash
  python -m src.rag_chat --question "Explain the purpose of positional encoding in Transformers."
  ```

- **Customize Prompt Behavior:**
  Edit the `SYSTEM_PROMPT` inside `src/rag_chat.py` to modify the model’s answering style (e.g., make it more detailed, critical, or instructional).

---

## Optional: Docker Deployment

**Dockerfile:**
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
```

**Build and Run:**
```bash
docker build -t gentutor .
docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN=hf_your_token_here gentutor
```

---

## Key Features

- **Retrieval-Augmented Generation (RAG):**  
  Leverages dense vector search over a curated knowledge base to ground every response.

- **Prompt Engineering:**  
  Uses structured, context-aware prompts to maximize answer relevance and minimize hallucination.

- **Fine-Tuned LLM Access:**  
  Utilizes tuned models for better alignment with domain-specific queries.

- **Dynamic Index Management:**  
  Quickly rebuilds or updates FAISS indexes when new documents are introduced.

- **Streamlit User Interface:**  
  An intuitive and easy-to-deploy web application for interacting with the system.

---

## Contributing

We welcome contributions to improve GenTutor!

1. Fork this repository.
2. Create a new feature branch.
3. Make your changes and commit.
4. Open a Pull Request with a description of your changes.

---

## License

This project is licensed under the MIT License.
