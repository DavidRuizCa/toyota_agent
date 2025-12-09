import os
from dotenv import load_dotenv
import duckdb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool

load_dotenv()
CHROMA_PATH = os.getenv('CHROMA_PATH')
DUCKDB_PATH = os.getenv('DUCKDB_PATH')

@tool
def retrieve(question, doc_type):
    """Retrieve relevant documents to answer the user's question.
    Use this tool to answer questions about warranty terms, policy clauses, or owner manual content.
    You can search for two types of documents: 'contracts' or 'user_manuals'.
    
    Args:
        question: The user's question.
        doc_type: The type of document ('contracts' or 'user_manuals')
        
    Returns:
        dict: A dictionary containing the context and sources.
    """
    # Initialize the Chroma vector store
    vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=OpenAIEmbeddings())
    
    # Create a retriever to fetch the top 3 most relevant documents
    # Filter by the specified document type (contracts or user_manuals)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"doc_type": f"{doc_type}"}})

    # Retrieve documents relevant to the question
    docs = retriever.invoke(question)

    # Deduplicate retrieved documents
    unique_docs = []
    retrieved_docs = set()
    for doc in docs:
        if doc.page_content not in retrieved_docs:
            unique_docs.append(doc)
            retrieved_docs.add(doc.page_content)

    # Format the retrieved documents into a single  string and add sources
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    sources = [doc.metadata.get("source", "Unknown") for doc in unique_docs]

    return {"context": context, "sources": list(set(sources))}

@tool
def run_sql(question):
    """Run a SQL query on the database.
    Use this tool for questions about sales, time, country/region, model, powertrain, etc.
    
    Args:
        question: The user's question.
        
    Returns:
        dict: A dictionary containing the query and result.
    """
    # Generate the SQL query based on the natural language question
    sql_query = generate_sql(question)

    # Execute the query and convert result to a list of dicts
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        result = con.execute(sql_query).df().to_dict(orient="records")

    except Exception as e:
        return f"Error executing SQL: {e}"
    finally:
        con.close()
    return {"query": sql_query, "sql_result": result}

def get_schema():
    """
    Retrieves the database schema, including table names, column names, and types.
    For dimension tables (starting with DIM_), it also fetches distinct values for text columns
    to understand the available data.

    Returns:
        str: A formatted string describing the schema.
    """
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    tables = con.execute("SHOW TABLES").fetchall()
    schema_info = []

    # Iterate over all tables in the database
    for table in tables:
        # Get column details for the current table
        table_name = table[0]
        columns = con.execute(f"DESCRIBE {table_name}").fetchall()
        col_descriptions = []
        
        for col in columns:
            col_name = col[0]
            col_type = col[1]
            
            # Check if table is a dimension table (starts with DIM_) and column is text-like
            # For dimension tables, fetch sample values
            distinct_values_str = ""
            if table_name.startswith("DIM_") and col_type in ['VARCHAR', 'STRING']:
                distinct_values = con.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 50").fetchall()
                values_list = [str(v[0]) for v in distinct_values]
                distinct_values_str = f" (Values: {', '.join(values_list)})"
        
            col_descriptions.append(f"{col_name} ({col_type}){distinct_values_str}")

        col_str = ", ".join(col_descriptions)
        schema_info.append(f"Table: {table_name}\nColumns: {col_str}")

    con.close()
    return "\n\n".join(schema_info)

def generate_sql(question):
    """
    Generates a SQL query using an LLM based on the user's question and the database schema.

    Args:
        question (str): The user's natural language question.

    Returns:
        str: The generated SQL query.
    """
    schema = get_schema()
    # Construct the prompt with schema and rules
    prompt = f"""You are a SQL expert. Given the following database schema, write a SQL query to answer the user's question.
    
    Follow these rules:
    1. Return only SQL—no explanations, comments, or natural-language text.
    2. Use only the tables and columns provided in the schema.
    3. Do not invent fields, tables, or values.
    4. Prefer simple, readable SQL.
    5. If the question is ambiguous, choose the safest reasonable interpretation based strictly on the schema.
    6. Never modify the database; generate only SELECT queries unless explicitly instructed otherwise.
    7. When filters involve text, use case-insensitive matching when appropriate.
    8. If the user asks for something impossible with the available schema, generate the closest valid SQL query.

    Schema:
    {schema}

    Question: {question}
    
    Return ONLY the SQL query. Do not include markdown formatting (```sql ... ```) or explanations.
    """
    # Initialize the LLM
    llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
    
    # Generate the SQL query
    sql_query = llm.invoke(prompt)
    return sql_query.content