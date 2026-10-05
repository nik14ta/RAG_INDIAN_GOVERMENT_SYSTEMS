import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import tempfile

# New Audio & Translation Imports
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langdetect import detect
from gtts import gTTS
import base64

st.set_page_config(page_title="Multilingual Legal RAG", page_icon="⚖️", layout="wide")

CHROMA_PATH = "chroma_db"

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=150)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embeddings, llm, reranker

@st.cache_resource
def load_database_and_bm25(_embeddings):
    if not os.path.exists(CHROMA_PATH):
        return None, None, None
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=_embeddings)
    all_docs = db.get()
    docs_text = all_docs['documents']
    metadatas = all_docs['metadatas']
    tokenized_corpus = [doc.lower().split(" ") for doc in docs_text]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    return db, bm25, (docs_text, metadatas)

def hybrid_search(query, db, bm25, docs_data, k=5):
    docs_text, metadatas = docs_data
    vector_results = db.similarity_search_with_score(query, k=k)
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_idx = np.argsort(bm25_scores)[::-1][:k]
    
    seen_content = set()
    combined_docs = []
    
    for doc, _ in vector_results:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            combined_docs.append(doc)
            
    for i in top_n_idx:
        text = docs_text[i]
        if text not in seen_content:
            seen_content.add(text)
            mock_doc = type('Document', (object,), {'page_content': text, 'metadata': metadatas[i]})()
            combined_docs.append(mock_doc)
            
    return combined_docs

def get_audio_player(text, lang='en'):
    """Generates an HTML audio player for TTS"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
                return md
    except Exception as e:
        return ""

def main():
    st.title("⚖️ Multilingual Voice Legal Assistant")
    
    embeddings, llm, reranker = load_models()
    db, bm25, docs_data = load_database_and_bm25(embeddings)
    
    with st.sidebar:
        st.header("⚙️ Pipeline Settings")
        use_hybrid = st.checkbox("Hybrid Search", value=True)
        use_reranker = st.checkbox("Cross-Encoder", value=True)
        st.markdown("---")
        st.success("Multilingual TTS UI Active")
        
    if db is None:
        st.error(f"Vector database not found. Please run `python src/ingest.py` first.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Speak or type your legal question in English, Hindi, Tamil, Telugu, Marathi, Bengali, or your preferred language.", "lang": "en"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "verification" in message:
                if message["verification"] == "supported":
                    st.success("✅ Supported by retrieved evidence")
                else:
                    st.warning("⚠️ I couldn't verify this answer from the available legal documents.")
            if "audio" in message and message["audio"]:
                st.markdown(message["audio"], unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                with st.expander("📚 Proper Source Citations"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}: {source['file']}**")
                        st.write(f"**Page:** {source['page']}")
                        st.write(f"**Evidence:**\n{source['text']}")
                        st.markdown("---")

    # Display audio recorder right above the chat input
    col1, col2 = st.columns([1, 15])
    with col1:
        audio_bytes = audio_recorder(text="", icon_size="2x")
    with col2:
        st.caption("Click the microphone to record a voice message 🎤")

    # Determine prompt from either Voice or Text
    prompt = None
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            tmp_path = f.name
        
        r = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = r.record(source)
        try:
            prompt = r.recognize_google(audio_data)
            st.toast(f"Transcribed: {prompt}")
        except Exception as e:
            st.error("Could not understand the audio. Please try again or type.")
            
    text_prompt = st.chat_input("E.g., Cyber terrorism ki saza kya hai?")
    if text_prompt:
        prompt = text_prompt

    if prompt:
        # Detect Language and Translate to English if needed
        try:
            detected_lang = detect(prompt)
        except:
            detected_lang = 'en'
            
        original_prompt = prompt
        if detected_lang != 'en':
            with st.spinner(f"Translating from '{detected_lang}' to English..."):
                prompt = GoogleTranslator(source='auto', target='en').translate(prompt)
                
        st.session_state.messages.append({"role": "user", "content": f"{original_prompt} *(Translated: {prompt})*" if detected_lang != 'en' else original_prompt, "lang": detected_lang})
        
        # We need to trigger rerun to show user message immediately, or just write it:
        with st.chat_message("user"):
            st.markdown(st.session_state.messages[-1]["content"])

        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing..."):
                
                # --- 1. RETRIEVAL ---
                if use_hybrid and bm25 is not None:
                    retrieved_docs = hybrid_search(prompt, db, bm25, docs_data, k=5)
                else:
                    retrieved_docs = db.similarity_search(prompt, k=5)

                # --- 2. RERANKING ---
                if use_reranker and len(retrieved_docs) > 0:
                    pairs = [[prompt, doc.page_content] for doc in retrieved_docs]
                    scores = reranker.predict(pairs)
                    ranked_indices = np.argsort(scores)[::-1]
                    retrieved_docs = [retrieved_docs[i] for i in ranked_indices][:2]
                else:
                    retrieved_docs = retrieved_docs[:2]
                
                if not retrieved_docs:
                    st.warning("No relevant context found.")
                    return

                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # --- 3. VERIFICATION ---
                llm_prompt = f"""You are a strict legal assistant. 
If the context below does not clearly contain the answer to the question, reply EXACTLY with "I abstain: The provided legal documents do not contain the answer." 
Do not guess.

Context:
{context_text}

Question: {prompt}

Answer:"""
                
                response = llm(llm_prompt)[0]['generated_text']

                # --- 4. EVIDENCE VERIFICATION BADGE ---
                ABSTAIN_PHRASES = ["abstain", "cannot answer", "do not contain the answer", "not mentioned", "i abstain"]
                is_abstained = any(p in response.lower() for p in ABSTAIN_PHRASES)
                verification_status = "unsupported" if is_abstained else "supported"

                # --- 5. OUTPUT TRANSLATION & TTS ---
                final_response = response
                audio_html = ""
                if detected_lang != 'en':
                    with st.spinner(f"Translating answer back to '{detected_lang}'..."):
                        try:
                            final_response = GoogleTranslator(source='en', target=detected_lang).translate(response)
                        except:
                            pass  # fallback to english

                with st.spinner("Generating audio..."):
                    try:
                        audio_html = get_audio_player(final_response, lang=detected_lang if detected_lang in ['hi', 'ta', 'te', 'mr', 'ur', 'bn', 'gu', 'kn', 'ml', 'en'] else 'en')
                    except:
                        audio_html = get_audio_player(final_response, lang='en')

                # --- 6. BUILD SOURCE CITATIONS ---
                source_data = []
                for doc in retrieved_docs:
                    source_data.append({
                        "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                        "page": doc.metadata.get("page", "Unknown"),
                        "text": doc.page_content
                    })

                # --- 7. DISPLAY ---
                st.markdown(final_response)

                # Verification badge
                if verification_status == "supported":
                    st.success("✅ Supported by retrieved evidence")
                else:
                    st.warning("⚠️ I couldn't verify this answer from the available legal documents.")

                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)

                with st.expander("📚 Proper Source Citations"):
                    for i, source in enumerate(source_data):
                        st.markdown(f"**Source {i+1}: `{source['file']}`**")
                        st.markdown(f"📄 **Page:** {source['page']}")
                        st.markdown(f"🔍 **Evidence:**\n\n_{source['text']}_")
                        st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "verification": verification_status,
                    "audio": audio_html,
                    "sources": source_data,
                    "lang": detected_lang
                })

if __name__ == "__main__":
    main()

