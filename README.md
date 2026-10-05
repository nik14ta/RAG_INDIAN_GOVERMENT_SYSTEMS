# ⚖️ RAG Indian Government Systems

<div align="center">

**A Multilingual Voice-Enabled RAG System for Indian Legal Documents**

*Making Indian law accessible to every citizen — in their own language, through voice or text.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-00C7B7?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 🎯 What is this?

This system allows **any Indian citizen** to ask questions about Indian law — in their own language, by voice or text — and get verified, sourced answers backed by real legal documents.

Instead of reading 500-page PDFs, just ask:
> *"Cyber terrorism ki saza kya hai?"* 🎤

And get a cited, verified answer in Hindi — spoken aloud.

---

## 🏗️ Architecture

```
         👤 USER
    🎤 Voice / ⌨️ Text
              ↓
  ┌────────────────────────┐
  │       🇮🇳 BHASHINI      │
  │  ASR  ·  Lang Detect   │
  │  Translation (NMT)     │
  └──────────┬─────────────┘
             ↓
      User Query (English)
             ↓
  ┌──────────────────────────┐
  │       Hybrid RAG         │
  │  Vector Search + BM25    │──→ ChromaDB
  │  Cross-Encoder Reranker  │
  └──────────┬───────────────┘
             ↓
       LLM Generation
       (Flan-T5 Local)
             ↓
    Evidence Verification
        ↙          ↘
  ✅ Supported   ⚠️ Abstain
   + Citations
        ↓
  🇮🇳 BHASHINI NMT + TTS
        ↓
       🔊 USER
```

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🎤 | **Voice Input** | Speak your legal question using your microphone |
| 🌐 | **Multilingual** | Supports English, Hindi, Tamil, Telugu, Marathi, Bengali & more |
| 🔍 | **Hybrid Retrieval** | Vector search (ChromaDB) + BM25 keyword search combined |
| 🎯 | **Cross-Encoder Reranking** | Re-scores results for maximum precision |
| 🛡️ | **Evidence Verification** | Abstains instead of hallucinating when evidence is weak |
| 📚 | **Source Citations** | Source PDF · Page number · Retrieved evidence for every answer |
| 🔊 | **Text-to-Speech** | Answers read aloud in the user's native language |
| 📊 | **Evaluation Dashboard** | RAG vs Non-RAG: faithfulness, hallucination & abstention metrics |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/RAG_INDIAN_GOVERMENT_SYSTEMS.git
cd RAG_INDIAN_GOVERMENT_SYSTEMS/RAG_INDIAN_GOVERMENT_SYSTEMS

# 2. Setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your legal PDFs to data/raw/
# e.g. Indian Penal Code, IT Act, Constitution PDFs

# 5. Ingest documents
python src/ingest.py

# 6. Launch the app
streamlit run app/main.py
```

---

## 📊 Evaluation

```bash
cd evaluation && python run_experiments.py && cd ..
streamlit run app/dashboard.py
```

The dashboard compares **RAG vs Non-RAG baseline** on:
- ✅ Abstention Accuracy
- 📖 Faithfulness
- 🚫 Hallucination Rate

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Vector DB | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` |
| Keyword Search | BM25 (`rank_bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `google/flan-t5-base` *(local, no API key needed)* |
| Multilingual | 🇮🇳 BHASHINI (ASR · NMT · TTS) |
| PDF Parsing | LangChain + PyPDF |

---

## 📄 Supported Legal Documents

- ✅ Indian Penal Code (IPC)
- ✅ Information Technology Act, 2000
- ➕ Add any Indian legal PDF to `data/raw/` and re-run ingestion

---

## 🔮 Roadmap

- [x] Hybrid Retrieval (Vector + BM25)
- [x] Cross-Encoder Reranking
- [x] Evidence Verification & Abstention
- [x] Multilingual Voice Input & TTS
- [x] RAG vs Non-RAG Evaluation Dashboard
- [ ] Full BHASHINI API integration
- [ ] Expand evaluation dataset to 300+ questions
- [ ] BERTScore / ROUGE faithfulness scoring
- [ ] Add Constitution, RTI Act, Consumer Protection Act

---

## 🙏 Acknowledgements

- [BHASHINI](https://bhashini.gov.in) — Government of India's AI language platform
- [LangChain](https://langchain.com) · [ChromaDB](https://trychroma.com) · [Hugging Face](https://huggingface.co)

---

<div align="center">

Made with ❤️ for Indian citizens · MIT License

</div>
