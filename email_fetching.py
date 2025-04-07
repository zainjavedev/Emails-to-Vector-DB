import imaplib
import email
import yaml
import chromadb
import os

# Load credentials
with open('credentials.yml') as f:
    content = f.read()

my_credentials = yaml.load(content, Loader=yaml.FullLoader)
username = my_credentials['username']
password = my_credentials['password']
imap_url = 'imap.gmail.com'

# Connect to email server
my_email = imaplib.IMAP4_SSL(imap_url)
my_email.login(username, password)
my_email.select('inbox')

# Initialize ChromaDB in the /chromadb directory
chroma_path = "./chromadb"
if not os.path.exists(chroma_path):
    os.makedirs(chroma_path)

client = chromadb.PersistentClient(path=chroma_path)
collection = client.create_collection("email_texts")


def fetch_and_store_emails(num_emails=10):
    result, data = my_email.search(None, 'ALL')
    email_ids = data[0].split()
    email_ids = email_ids[-num_emails:]  # Get the last 10 emails
    email_ids = reversed(email_ids)  # Process in reverse chronological order
    texts = []
    ids = []

    for email_id in email_ids:
        result, msg_data = my_email.fetch(email_id, '(RFC822)')
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        email_text = ""

        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                email_text += part.get_payload(decode=True).decode() + "\n"
            elif part.get_content_type() == "text/html":
                pass
        if email_text:  # Only add if there is text
            texts.append(email_text)
            ids.append(email_id.decode())  # Use email_id as string ID

    # Add texts to ChromaDB
    collection.add(
        documents=texts,
        ids=ids
    )

    print(f"Successfully processed and stored {len(texts)} emails.")


fetch_and_store_emails()

# Example usage to query ChromaDB (optional)
# results = collection.query(
#     query_texts=["summary of the email"],
#     n_results=2
# )
# print(results)

my_email.close()
my_email.logout()