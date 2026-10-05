import os
import argparse
import pandas as pd
from transformers import pipeline

def load_qa_pairs():
    # In a real scenario, you'd load this from a CSV or JSON dataset
    return [
        {"question": "What is the penalty for cyber terrorism under the IT Act?", "expected": "Life imprisonment."},
        {"question": "What is Section 420 of the IPC about?", "expected": "Cheating and dishonestly inducing delivery of property."},
        {"question": "Does the IT Act cover physical theft?", "expected": "No, it focuses on cyber crimes."}
    ]

def evaluate_baseline(llm, qa_pairs):
    print("\n--- Running Non-RAG (Baseline) Evaluation ---")
    results = []
    for qa in qa_pairs:
        prompt = f"Answer the following question about Indian Law.\nQuestion: {qa['question']}\nAnswer:"
        response = llm(prompt)[0]['generated_text']
        print(f"Q: {qa['question']}")
        print(f"Expected: {qa['expected']}")
        print(f"Generated (No RAG): {response}\n")
        results.append({
            "Question": qa['question'],
            "Expected": qa['expected'],
            "Non-RAG Answer": response
        })
    return pd.DataFrame(results)

def main():
    print("Loading LLM (flan-t5-base) for evaluation...")
    llm = pipeline("text-generation", model="google/flan-t5-base", max_new_tokens=50)
    
    qa_pairs = load_qa_pairs()
    
    # Run Baseline Evaluation (Non-RAG)
    df_baseline = evaluate_baseline(llm, qa_pairs)
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    df_baseline.to_csv("results/evaluation_baseline.csv", index=False)
    print("Saved baseline evaluation to results/evaluation_baseline.csv")
    print("To run the RAG evaluation, you would pass these questions through the full pipeline in app/main.py!")

if __name__ == "__main__":
    main()
