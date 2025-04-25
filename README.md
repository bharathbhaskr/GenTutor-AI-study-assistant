# GenTutor Project

Quick‑start:
```bash
git clone <this‑repo>
cd gen-tutor-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# scrape data
echo "Massachusetts" > pages.txt
python src/data_scraper.py --pages pages.txt --out data
# build index
python -m src.rag_chat --build-index data/processed/combined_data.md
# launch UI
streamlit run src/app.py
```
Set your OpenAI key either via `.streamlit/secrets.toml` or env‑var `OPENAI_API_KEY`.