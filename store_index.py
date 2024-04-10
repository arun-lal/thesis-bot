from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
index_name = "thesisbot"

load_dotenv()

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_api_env = os.environ.get('PINECONE_API_ENV')

# Extract pdf
logging.info(f"Exrtact data from PDF ...")
extracted_data = load_pdf(data="data/")

# Create text chunks
logging.info(f"Create text chunks ...")
text_chunks = text_split(extracted_data)
print(f"Length of my chunks: {len(text_chunks)}")

# download and create embedding model
logging.info(f"download and create embedding model ...")
embeddings = download_hugging_face_embeddings()

# Initialize pinecone
logging.info(f"Initialize pinecone ...")
pinecone.Pinecone(api_key=pinecone_api_key,
              environment=pinecone_api_env)

logging.info(f"Create Vector Store in Pinecone ...")
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)

logging.info(f"Vector Store created ...")