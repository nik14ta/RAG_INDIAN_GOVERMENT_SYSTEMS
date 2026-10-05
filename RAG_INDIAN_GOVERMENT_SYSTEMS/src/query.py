import os
import argparse
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Uncomment these if you plan to use Langchain's Google GenAI integration
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA

CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
You are a legal assistant specializing in Indian Law. 
Answer the question based ONLY on the following legal context. 
If the context does not contain the answer, say "I cannot answer this based on the provided legal documents."

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Query the Indian Legal RAG system.")
    parser.add_argument("query", type=str, help="The legal question you want to ask.")
    args = parser.parse_args()

    # 1. Prepare Vector DB
    print("Loading vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 2. Search DB
    print(f"Searching for relevant legal documents for: '{args.query}'\n")
    results = db.similarity_search_with_score(args.query, k=3)
    
    if len(results) == 0:
        print("No matching legal context found in the database.")
        return

    # 3. Construct Prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    prompt = prompt_template.format(context=context_text, question=args.query)

    print("--- CONTEXT RETRIEVED ---")
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"[{i+1}] Source: {source} (Page {page}) - Match Score: {score:.4f}")
    
    print("\n--- PROMPT READY FOR LLM ---")
    print(prompt)
    
    print("\n[!] To generate a response, uncomment the LLM setup in this script and pass this prompt to Gemini or your preferred model.")

if __name__ == "__main__":
    main()
