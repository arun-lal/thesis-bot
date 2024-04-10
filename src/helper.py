from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


def load_pdf(data):
    """Extract data from the PDF"""
    loader = DirectoryLoader(data, 
                             glob="*.pdf", 
                             loader_cls=PyPDFLoader)
    document = loader.load()

    return document


def text_split(extracted_data):
    """Create text chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=20
        )    
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


def download_hugging_face_embeddings():
    """download embedding model from Hugging Face"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings