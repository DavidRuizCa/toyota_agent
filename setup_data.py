import os
import glob
import duckdb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration paths
DATA_DIR = "data"
DOCS_DIR = "docs"
DB_DIR = "db"
CHROMA_PATH = os.path.join(DB_DIR, "chroma_db")
DUCKDB_PATH = os.path.join(DB_DIR, "duckdb")

def setup_rag_db():
    # Get all subdirectories in the docs folder
    folders = glob.glob(os.path.join(DOCS_DIR, "*"))
    documents = []
    
    for folder in folders:
        doc_type = os.path.basename(folder)
        folder_docs = glob.glob(os.path.join(folder, "*.pdf"))
        print(f"Loading {doc_type}...")
        
        for doc in folder_docs:
            print(f"Loading {doc}...")
            loader = PyPDFLoader(doc)
            docs = loader.load()
            
            # Add metadata to each page for filtering
            for page in docs:
                page.metadata["doc_type"] = doc_type # Include doc_type in metadata
            documents.extend(docs)

    # Split documents into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    embeddings = OpenAIEmbeddings()
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

def setup_sql_db():
    print("Setting up DuckDB...")
    con = duckdb.connect(DUCKDB_PATH)
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for csv_file in csv_files:
        # Use the filename as the table name
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"Loading {table_name} from {csv_file}...")
        
        # Create table from CSV content
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")
    
    # Verify created tables
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"Tables created: {tables}")
    con.close()
    print("DuckDB setup complete.")

    setup_rag_db()
    setup_sql_db()