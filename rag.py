import os
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_pgvector_connection(db_params: Dict[str, str]):
    conn = psycopg2.connect(**db_params)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    return conn

def embed_and_store_products(db_params: Dict[str, str],
                             table_name: str = "product_embeddings",
                             embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
    logger.info(f"Embedding product data and storing in {table_name}")
    conn = create_pgvector_connection(db_params)
    model = SentenceTransformer(embedding_model_name)

    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                product_id INT,
                content TEXT,
                embedding vector(384)
            );
        """)
        conn.commit()

        # Pull data from 'products' table
        cur.execute("SELECT id, name, category, color, price, img_url FROM products;")
        products = cur.fetchall()

    logger.info(f"Fetched {len(products)} products")

    # Prepare text and embed
    data = []
    for product in products:
        pid, name, category, color, price, img_url = product
        text = f"{name}, category: {category}, color: {color}, price: {price}, image: {img_url}"
        embedding = model.encode(text).tolist()
        data.append((pid, text, embedding))

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"INSERT INTO {table_name} (product_id, content, embedding) VALUES %s",
            data,
            template="(%s, %s, %s)"
        )
        conn.commit()

    conn.close()
    logger.info("Embeddings stored successfully")

def perform_search(db_params: Dict[str, str],
                   query: str,
                   table_name: str = "product_embeddings",
                   n_results: int = 3,
                   embedding_model_name: str = "all-MiniLM-L6-v2") -> Dict[str, List[Any]]:
    logger.info(f"Searching for: '{query}'")

    model = SentenceTransformer(embedding_model_name)
    query_embedding = model.encode(query).tolist()
    conn = create_pgvector_connection(db_params)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT product_id, content, embedding <-> %s::vector AS distance
            FROM {table_name}
            ORDER BY distance ASC
            LIMIT %s;
        """, (query_embedding, n_results))
        results = cur.fetchall()
    conn.close()

    return {
        "ids": [r[0] for r in results],
        "documents": [r[1] for r in results],
        "distances": [float(r[2]) for r in results]
    }

def call_openrouter_llm(context: str, query: str, api_key: str) -> str:
    system_prompt = (
        "You are a smart assistant that helps users find and resolve queries about products. "
        "You can only answer using the provided product context. Your job is to:\n\n"
        "1. Understand what product or information the user is looking for.\n"
        "2. Use only the given context to find relevant products.\n"
        "3. Respond with matching product data in JSON format.\n"
        "4. If no relevant product is found, respond with exactly:\n"
        "   'sorry we don't have any product'.\n"
        "5. If the user asks about anything unrelated to products, reply with:\n"
        "   'I only know about products, nothing else.'\n"
        "6. If no product context is provided or available for the query, say:\n"
        "   'sorry i can not help u with it'.\n\n"
        "Stay helpful, focused on products, and do not answer beyond product-related information.\n\n"
        "Your response must be a JSON object with these keys:\n"
        "- 'response': little bit of description of what u found or not found'\n"
        "- 'products': a list of matching product objects (with filtered keys depending on user query).\n"
        "- 'follow-up_question': a helpful suggestion (e.g., 'Would you like to see some books?') dont suggest products out of our scope\n"
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

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers,
                                 json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return "LLM call failed."

def start_conversation():
    db_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "ecom"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "admin")
    }
    print("Rag Bot: Hello there what you wanna know about products? \n" )
    embed_and_store_products(db_params)

    query = input("You: ")
    search_results = perform_search(db_params, query)
    context = "\n\n".join(search_results["documents"])

    api_key = os.getenv("OPEN_ROUTER_API_KEY")  # Replace with your actual key
    answer = call_openrouter_llm(context, query, api_key)
    print("\n🧠 Answer:\n", answer)

start_conversation()
