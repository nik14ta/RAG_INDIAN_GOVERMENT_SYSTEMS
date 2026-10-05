import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
DATA_DIR = "data/raw"
CHROMA_PATH = "chroma_db"

def main():
    parser = argparse.ArgumentParser(description="Ingest legal PDFs and create a vector database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing Database")
        import shutil
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

    # 1. Load Documents
    print(f"Loading PDFs from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
         os.makedirs(DATA_DIR)
    
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print(f"No PDFs found in '{DATA_DIR}'. Please add some Indian legal documents (PDFs) and try again.")
        return

    print(f"Loaded {len(documents)} document pages.")

    # 2. Split Texts
    print("Splitting texts into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Create Embeddings & Store in Vector DB
    print("Initializing embedding model (Sentence Transformers)...")
    # Using a popular, lightweight open-source embedding model suitable for general text/legal text
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Storing chunks in ChromaDB...")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Done! Data successfully ingested into Chroma vector store at '{CHROMA_PATH}'.")

if __name__ == "__main__":
    main()
