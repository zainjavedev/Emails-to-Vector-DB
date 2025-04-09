import psycopg2
import os

db_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "ecom"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "admin")
    }

conn = psycopg2.connect(
   **db_params
)

cur = conn.cursor()

with open('data/dummy-products.csv', 'r') as f:
    next(f)  # skip header
    cur.copy_expert("COPY products(id, name, price, img_url, color, category, description) FROM STDIN WITH CSV", f)

conn.commit()
cur.close()
conn.close()