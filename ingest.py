import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_database():
    # 1. Point to the directory containing your data
    DATA_PATH = "./data"
    DB_PATH = "./chroma_db"
    
    print("Executing: Loading PDF files...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Success: Loaded {len(raw_documents)} document pages.")
    
    # 2. Break the massive text pages down into manageable context chunks
    print("Executing: Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Maximum characters per text block
        chunk_overlap=150     # Overlap prevents cutting a concept perfectly in half
    )
    text_chunks = text_splitter.split_documents(raw_documents)
    print(f"Success: Created {len(text_chunks)} semantic text chunks.")
    
    # 3. Choose a free, high-quality embedding model from HuggingFace
    print("Executing: Initializing local embedding calculations...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Generate the math vectors and save them onto your hard drive
    print("Executing: Storing vectors into local ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"Pipeline Complete: Database built successfully at '{DB_PATH}'!")

if __name__ == "__main__":
    build_vector_database()