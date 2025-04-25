# GenTutor – RAG Chatbot

**GenTutor** is a Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that answers user questions by retrieving and combining context from a custom Wikipedia corpus. It uses:

- **LangChain** for RAG pipelines  
- **Hugging Face Hub** (via `HuggingFaceHub`) as the LLM  
- **sentence-transformers/all-MiniLM-L6-v2** for embeddings  
- **FAISS** for vector search  
- **Streamlit** for the UI  

---

## 📂 Repository Layout

gen-tutor-project/ ├── data/ │ ├── raw/ # Wikipedia pages saved as CSV │ └── processed/ # combined_data.csv + combined_data.md ├── src/ │ ├── data_scraper.py # Fetch & process Wikipedia pages │ ├── rag_chat.py # RAGChat class + CLI for building index & one-off queries │ └── app.py # Streamlit UI ├── vectorstore/ # FAISS index files ├── .streamlit/ │ └── secrets.toml # Your Hugging Face API token ├── requirements.txt └── README.md # This file

yaml
Copy

---

## 🚀 Quickstart

### 1. Clone & create a venv

```bash
git clone <your-repo-url> gen-tutor-project
cd gen-tutor-project
python3 -m venv .venv
source .venv/bin/activate
2. Install dependencies
bash
Copy
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
macOS M1/M2 note:
If faiss-cpu fails to build, install via Homebrew:

bash
Copy
brew install faiss
then remove or loosen the faiss-cpu pin in requirements.txt.

3. Configure your Hugging Face token
Create or edit .streamlit/secrets.toml:

toml
Copy
HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
Do not commit your token to version control.

4. Scrape & prepare the data
Edit pages.txt to list one Wikipedia page title per line, for example:

text
Copy
Massachusetts
History of Massachusetts
Massachusetts Bay Colony
Run the scraper:

bash
Copy
python src/data_scraper.py --pages pages.txt --out data
You’ll end up with:

data/raw/ → one CSV per page

data/processed/combined_data.csv

data/processed/combined_data.md

5. Build the FAISS index
bash
Copy
python -m src.rag_chat --build-index data/processed/combined_data.md
This writes your vector store files into vectorstore/.

6. Launch the Streamlit app
bash
Copy
streamlit run src/app.py
Open your browser at http://localhost:8501 (or the URL printed in your terminal) and start chatting:

User:
massachusetts capital

GenTutor:
Boston

🛠️ Development Workflow
Rebuild index after updating data/processed/combined_data.md:

bash
Copy
python -m src.rag_chat --build-index data/processed/combined_data.md
One-off query without running Streamlit:

bash
Copy
python -m src.rag_chat --question "Who is the mayor of Boston?"
Adjust prompt in src/rag_chat.py under SYSTEM_PROMPT to customize behavior.

🐳 Docker (optional)
dockerfile
Copy
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
Build and run:

bash
Copy
docker build -t gentutor .
docker run -p 8501:8501 -e HUGGINGFACEHUB_API_TOKEN=hf_your_token_here gentutor
🤝 Contributing
Fork the repo

Create a feature branch

Submit a PR

📜 License
This project is licensed under the MIT License.
