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

# --- PGVector DB Connection ---
def create_pgvector_connection(db_params: Dict[str, str]):
    conn = psycopg2.connect(**db_params)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    return conn

# --- Embedding + Storage ---
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

        cur.execute("SELECT id, name, category, color, price, img_url, description FROM products;")
        products = cur.fetchall()

    logger.info(f"Fetched {len(products)} products")

    data = []
    for product in products:
        pid, name, category, color, price, img_url, description = product
        text = f"{name}, category: {category}, color: {color}, price: {price}$, image: {img_url}, description: {description}"
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

# --- Vector Search ---
def perform_search(db_params: Dict[str, str],
                   query: str,
                   table_name: str = "product_embeddings",
                   n_results: int = 5,
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

# --- Classify Query Intent ---
def classify_query(query: str, api_key: str) -> str:
    system_prompt = (
        "Classify this user query into one of the following categories:\n"
        "- product_search: when the query clearly asks for specific product attributes, price, or types\n"
        "- niche_advice: when the user asks for recommendations, ideas, or trends, not specific products\n"
        "- mixed: when the query includes both an idea and also hints at product interest\n\n"
        "Only return the class name as output. Nothing else."
    )

    payload = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
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
        return response.json()['choices'][0]['message']['content'].strip().lower()
    except Exception as e:
        logger.error(f"Query classification error: {e}")
        return "product_search"  # fallback

# --- Call LLM for Final Response ---
def call_openrouter_llm(context: str, query: str, api_key: str) -> str:
    system_prompt = (
        "You are a helpful assistant for a premium branded merchandise company. "
        "You handle two types of queries:\n\n"
        "1. **Product-specific queries**: Use the provided product context to return matching products in JSON.\n"
        "2. **Niche advice queries**: Share creative, brand-aligned insights — but **do not return any product data**.\n\n"
        "**Important rules:**\n"
        "- If the context is advisory (e.g., trends, ideas), never return products in the 'products' field.\n"
        "- Only return products if the context includes actual product data.\n"
        "- Regardless of query type, your 'follow-up_question' must guide the user toward product discovery.\n"
        "- Your response must always follow this strict JSON format:\n\n"
        "{\n"
        "  'response': 'Short helpful message',\n"
        "  'products': [only if applicable],\n"
        "  'follow-up_question': 'Prompt encouraging the user to explore actual merchandise or request specific items.'\n"
        "}\n\n"
        "Stay creative, insightful, and always on-brand."
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

# --- Main Chat Loop ---
def start_conversation():
    db_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "ecom"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "admin")
    }
    api_key = os.getenv("OPEN_ROUTER_API_KEY")

    print("🧢 RAG Bot: Hello! What are you looking for today?\n")
    query = input("You: ")

    # 1. Classify Query
    intent = classify_query(query, api_key)
    logger.info(f"Detected intent: {intent}")

    # 2. Get Context (if needed)
    if intent in ["product_search", "mixed"]:
        search_results = perform_search(db_params, query)
        context = "\n\n".join(search_results["documents"])
    else:
        context = "This is a niche advisory question. Do not include products in the response."

    # 3. LLM Call
    answer = call_openrouter_llm(context, query, api_key)

    print("\n🧠 Answer:\n", answer)

start_conversation()

