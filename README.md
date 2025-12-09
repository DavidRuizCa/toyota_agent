# Toyota Agentic Assistant

An intelligent agentic assistant designed to answer questions about Toyota vehicles, sales data, warranty terms, and owner manuals. It uses a hybrid approach combining SQL querying for structured data and RAG (Retrieval Augmented Generation) for unstructured documents.

![alt text](Arquitecture_diagram.png)


## Features

- **Natural Language Interface**: Ask questions in plain English.
- **SQL Agent**: Queries a DuckDB database for sales, model, and regional data.
- **RAG Agent**: Retrieves information from PDF documents (contracts, owner manuals) using ChromaDB and OpenAI embeddings.
- **Streamlit UI**: A user-friendly web interface.
- **Intelligent Routing**: The agent decides whether to use SQL, RAG, or both based on the user's question.

## Prerequisites

- Python 3.8+
- OpenAI API Key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd toyota_assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your OpenAI API key and paths:
   ```env
   OPENAI_API_KEY=your_api_key_here
   CHROMA_PATH=db/chroma_db
   DUCKDB_PATH=db/toyota.db
   ```

## Setup

1. **Data Preparation**:
   - Place your CSV data files in the `data/` directory.
   - Place your PDF documents (contracts, manuals) in the `docs/` directory.

2. **Initialize Database and Vector Store**:
   Run the setup script to ingest data and build the indexes:
   ```bash
   python setup_data.py
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

Open your browser and navigate to the URL provided (usually `http://localhost:8501`).

## Project Structure

- `agent/`: Contains the `Agent` class and orchestration logic.
- `tools/`: Contains the tool definitions (`retrieve` for RAG, `run_sql` for SQL).
- `app.py`: The main Streamlit application file.
- `setup_data.py`: Script to ingest data into DuckDB and ChromaDB.
- `data/`: Directory for source CSV files.
- `docs/`: Directory for source PDF files.
- `db/`: Directory where the DuckDB database and ChromaDB vector store are persisted.

## Technologies Used

- **LangChain**: For agent orchestration and tool management.
- **OpenAI GPT-4o-mini**: For natural language understanding and SQL generation.
- **DuckDB**: High-performance analytical database for structured data.
- **ChromaDB**: Vector database for document retrieval.
- **Streamlit**: For the web user interface.
