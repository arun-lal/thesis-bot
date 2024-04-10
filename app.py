import os
import pinecone
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import *

index_name = "thesisbot"

load_dotenv()

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_api_env = os.environ.get('PINECONE_API_ENV')

# Load Embedding model
embeddings = download_hugging_face_embeddings()

# Initialize pinecone
pinecone.Pinecone(api_key=pinecone_api_key,
              environment=pinecone_api_env)

# Load the index
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

retriever = vectorstore.as_retriever()

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": prompt}

llm = CTransformers(model='../model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens':512,
                            'temperature': 0.8} 
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs=chain_type_kwargs
)

while True:
    user_input=input(f"Input Prompt:")
    result=qa({"query": user_input})
    print("Response:", result["result"])