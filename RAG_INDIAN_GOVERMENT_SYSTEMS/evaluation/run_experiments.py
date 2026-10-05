import os
import pandas as pd
import time
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
import warnings

warnings.filterwarnings('ignore')

CHROMA_PATH = "../chroma_db"

def load_systems():
    print("Loading models and databases (this takes a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=150)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    all_docs = db.get()
    docs_text = all_docs['documents']
    tokenized_corpus = [doc.lower().split(" ") for doc in docs_text]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    
    return llm, reranker, db, bm25, docs_text

def run_hybrid_search(query, db, bm25, docs_text, k=3):
    vector_results = db.similarity_search(query, k=k)
    
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_idx = np.argsort(bm25_scores)[::-1][:k]
    
    seen = set()
    combined_docs = []
    
    for doc in vector_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined_docs.append(doc.page_content)
            
    for i in top_n_idx:
        text = docs_text[i]
        if text not in seen:
            seen.add(text)
            combined_docs.append(text)
            
    return combined_docs

def evaluate_abstention(response):
    abstain_phrases = ["abstain", "cannot answer", "do not contain the answer", "not mentioned"]
    return any(phrase in response.lower() for phrase in abstain_phrases)

def main():
    if not os.path.exists("dataset.csv"):
        print("Error: dataset.csv not found in evaluation/ folder.")
        return
        
    df = pd.read_csv("dataset.csv")
    llm, reranker, db, bm25, docs_text = load_systems()
    
    results = []
    
    print(f"Running experiments on {len(df)} questions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        q = row['question']
        is_answerable = row['is_answerable']
        
        # 1. BASELINE (No RAG)
        baseline_prompt = f"Answer the following question about Indian Law.\nQuestion: {q}\nAnswer:"
        baseline_ans = llm(baseline_prompt)[0]['generated_text']
        
        # 2. RAG PIPELINE
        retrieved_texts = run_hybrid_search(q, db, bm25, docs_text, k=5)
        
        # Rerank
        pairs = [[q, text] for text in retrieved_texts]
        scores = reranker.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:2]
        final_context = "\n\n".join([retrieved_texts[i] for i in ranked_indices])
        
        rag_prompt = f"""You are a strict legal assistant. 
If the context below does not clearly contain the answer to the question, reply EXACTLY with "I abstain: The provided legal documents do not contain the answer." 
Do not guess.

Context:
{final_context}

Question: {q}

Answer:"""
        rag_ans = llm(rag_prompt)[0]['generated_text']
        
        # 3. METRICS
        baseline_abstained = evaluate_abstention(baseline_ans)
        rag_abstained = evaluate_abstention(rag_ans)
        
        # Faithfulness (Heuristic: If answerable and didn't abstain, assume faithful for now. LLM-as-a-judge would replace this)
        # Abstention evaluation
        correct_abstention_rag = (rag_abstained == (not is_answerable))
        correct_abstention_baseline = (baseline_abstained == (not is_answerable))
        
        results.append({
            "Question": q,
            "Is_Answerable": is_answerable,
            "Baseline_Answer": baseline_ans,
            "RAG_Answer": rag_ans,
            "RAG_Correct_Abstention": correct_abstention_rag,
            "Baseline_Correct_Abstention": correct_abstention_baseline
        })
        
    results_df = pd.DataFrame(results)
    os.makedirs("../results", exist_ok=True)
    results_df.to_csv("../results/experiment_results.csv", index=False)
    print("Experiments complete! Results saved to results/experiment_results.csv")

if __name__ == "__main__":
    main()
