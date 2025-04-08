
import os
import logging
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF document.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        A string containing all text from the PDF
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")

    try:
        # Open the PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text = ''.join([doc.page_content for doc in docs])
        return text

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


def split_text_into_chunks(text: str, chunk_size: int = 1300, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size.

    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    logger.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")

    return chunks


def create_vector_store(chunks: List[str], collection_name: str = "pdf_embeddings") -> chromadb.Collection:
    """
    Create embeddings for chunks and store them in a Chroma vector database.

    Args:
        chunks: List of text chunks to embed
        collection_name: Name of the collection in the vector database

    Returns:
        Chroma collection with the embeddings
    """
    logger.info("Creating vector store with sentence-transformers embedding model")

    # Initialize the client
    chroma_client = chromadb.Client()

    # Use sentence-transformers model for embeddings
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Free, lightweight model
    )

    # Create or get the collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        logger.info(f"Created new collection: {collection_name}")

    # Generate IDs for chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Add documents and their embeddings to the collection
    collection.add(
        documents=chunks,
        ids=ids
    )

    logger.info(f"Added {len(chunks)} documents to vector store")
    return collection


def process_pdf(pdf_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a PDF file: extract text, chunk it, embed chunks, and store in vector DB.

    Args:
        pdf_path: Path to the PDF file
        options: Dictionary of options (chunk_size, chunk_overlap, collection_name)

    Returns:
        Dictionary with results
    """
    # Set default options if none provided
    if options is None:
        options = {}

    chunk_size = options.get("chunk_size", 1000)
    chunk_overlap = options.get("chunk_overlap", 200)
    collection_name = options.get("collection_name", "pdf_embeddings")

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)

    # Create vector store
    collection = create_vector_store(chunks, collection_name)

    return {
        "text_length": len(text),
        "chunk_count": len(chunks),
        "collection": collection
    }


def perform_search(collection, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search the vector store for chunks similar to the query.

    Args:
        collection: Chroma collection to search
        query: Search query
        n_results: Number of results to return

    Returns:
        List of most similar chunks with metadata
    """
    logger.info(f"Searching for: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results

def call_openrouter_llm(context: str, query: str, api_key: str) -> str:
    """
    Use OpenRouter API to generate an answer using provided context.

    Args:
        context: Retrieved document chunks
        query: User query
        api_key: OpenRouter API key

    Returns:
        LLM-generated answer
    """
    import requests

    system_prompt = (
        "You are a helpful assistant. Only answer questions using the provided context. "
        "If the question is not related to computers or the given content, say: "
        "'I only know about computers.'"
    )

    payload = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    response_json = response.json()

    try:
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        return f"Failed to get response from LLM: {response_json}"


def main():
    """Main function to demonstrate PDF processing and searching."""
    # Path to your PDF file
    pdf_path = "data/Ch.01_Introduction_ to_computers.pdf"

    # Processing options
    options = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "collection_name": "my_pdf_collection"
    }

    # Process the PDF
    result = process_pdf(pdf_path, options)
    logger.info(f"Processed PDF with {result['chunk_count']} chunks")

    ## Perform a search
    query = input("Enter your query: ")
    search_results = perform_search(result["collection"], query, n_results=3)

    # Get top chunks as context
    retrieved_chunks = search_results.get("documents", [[]])[0]
    context = "\n\n".join(retrieved_chunks)

    # Call OpenRouter LLM
    api_key = os.getenv("API_KEY")
    answer = call_openrouter_llm(context, query, api_key)

    print("\n🧠 Answer:\n", answer)

main()

