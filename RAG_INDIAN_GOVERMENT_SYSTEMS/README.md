# ⚖️ RAG Indian Government Systems

A **Multilingual Voice-Enabled Retrieval-Augmented Generation (RAG)** system for Indian legal documents. Built to make Indian law accessible to every citizen in their native language — through voice or text.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🗂️ Project Structure

```
RAG_INDIAN_GOVERMENT_SYSTEMS/
├── app/
│   ├── main.py               # Streamlit chat UI with voice & multilingual support
│   └── dashboard.py          # Evaluation dashboard (RAG vs Non-RAG metrics)
├── src/
│   ├── ingest.py             # PDF ingestion → chunking → ChromaDB
│   └── query.py              # CLI query interface
├── evaluation/
│   ├── dataset.csv           # Evaluation Q&A dataset
│   └── run_experiments.py    # Baseline vs RAG experiment runner
├── data/
│   └── raw/                  # Place your Indian legal PDFs here
├── results/                  # Experiment outputs saved here
├── chroma_db/                # Auto-generated vector database (gitignored)
├── .env                      # API keys (gitignored)
└── requirements.txt
```

---

## 🏗️ System Architecture

```
         👤 USER
    🎤 Voice / ⌨️ Text
              ↓
  ┌───────────────────────┐
  │      BHASHINI         │
  │  ASR · Lang Detect    │
  │  Translation (NMT)    │
  └──────────┬────────────┘
             ↓
       User Query (English)
             ↓
  ┌──────────────────────┐
  │     Hybrid RAG       │
  │  Vector + BM25       │
  │  Cross-Encoder       │
  │  Reranking           │
  └──────────┬───────────┘
             ↓
           LLM
             ↓
   Evidence Verification
       ↙          ↘
  ✅ Supported   ⚠️ Abstain
       ↓
  BHASHINI NMT + TTS
       ↓
      🔊 USER
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎤 **Voice Input** | Speak your legal question using your microphone |
| 🌐 **Multilingual** | Supports English, Hindi, Tamil, Telugu, Marathi, Bengali & more via BHASHINI |
| 🔍 **Hybrid Retrieval** | Combines semantic vector search (ChromaDB) + keyword search (BM25) |
| 🎯 **Cross-Encoder Reranking** | Re-scores retrieved chunks for maximum precision |
| 🛡️ **Evidence Verification** | AI explicitly abstains instead of hallucinating when evidence is weak |
| 📚 **Source Citations** | Every answer shows the source PDF, page number, and retrieved evidence |
| 🔊 **Text-to-Speech** | Answers are read aloud in your native language |
| 📊 **Evaluation Dashboard** | Compares RAG vs Non-RAG on faithfulness, hallucination & abstention rates |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/RAG_INDIAN_GOVERMENT_SYSTEMS.git
cd RAG_INDIAN_GOVERMENT_SYSTEMS/RAG_INDIAN_GOVERMENT_SYSTEMS
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your legal documents
Place Indian legal PDF documents (e.g., IPC, IT Act, Constitution) into:
```
data/raw/
```

### 5. Ingest documents into the vector database
```bash
python src/ingest.py
```

### 6. Run the app
```bash
streamlit run app/main.py
```

---

## 📊 Running the Evaluation

```bash
# Step 1: Run experiments (Baseline vs RAG)
cd evaluation
python run_experiments.py

# Step 2: View the results dashboard
cd ..
streamlit run app/dashboard.py
```

The dashboard shows:
- **Abstention Accuracy** — Did the model correctly refuse unanswerable questions?
- **Faithfulness** — Did it correctly answer answerable questions?
- **Hallucination Rate** — How often did the model make things up?

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **UI** | Streamlit |
| **Vector DB** | ChromaDB |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Keyword Search** | BM25 (`rank_bm25`) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | `google/flan-t5-base` (local, no API key needed) |
| **Multilingual** | BHASHINI (ASR, NMT, TTS) |
| **PDF Parsing** | LangChain + PyPDF |

---

## 📄 Supported Legal Documents

Currently tested with:
- Indian Penal Code (IPC)
- Information Technology Act, 2000

You can add any Indian legal PDF to `data/raw/` and re-run `python src/ingest.py`.

---

## 🔮 Roadmap

- [x] Hybrid Retrieval (Vector + BM25)
- [x] Cross-Encoder Reranking
- [x] Evidence Verification & Abstention
- [x] Multilingual Support (English, Hindi, Tamil, Telugu, Marathi, Bengali)
- [x] Voice Input & Text-to-Speech
- [x] RAG vs Non-RAG Evaluation Dashboard
- [ ] BHASHINI API full integration
- [ ] Expand evaluation dataset to 300+ questions
- [ ] BERTScore / ROUGE faithfulness metrics
- [ ] Add more legal documents (Constitution, RTI Act, Consumer Protection Act)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

MIT License — free to use for research and educational purposes.

---

## 🙏 Acknowledgements

- [BHASHINI](https://bhashini.gov.in) — Government of India's language AI platform
- [LangChain](https://langchain.com)
- [ChromaDB](https://www.trychroma.com)
- [Hugging Face](https://huggingface.co)
