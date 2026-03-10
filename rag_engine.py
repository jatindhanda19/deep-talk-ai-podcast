"""This module handles the document processing part of the project.
   It takes a pdf file , breaks it into chunks , converts those chunks
   into  embeddings, and stores them inside Chroma Vector database.

   The vectorstore database is later used by RAG pipeline to retrieve.
   relevant context when generating podcast script or answering question.
"""
#Library import
import time
import hashlib
import logging
import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Load environment variable 
load_dotenv()

#Basic logger for debugging and inforamation messages
logger = logging.getLogger(__name__)

# Embedding model used to context text chunks into vectors
_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

#Text splitter 
#This breaks large documents into smaller ovelapping chunks
# So that retriever can find relevant information
_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def _file_hash(file_path: str)-> str:
    """Creates a short hash for this file.
       This helps us generate a unique folder name for vector database
       based on content of this pdf file
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:16]
    

def build_vectorstore(file_path: str)-> Chroma:
    """ Main function that builds the vector database
       
    1. Load the PDF document
    2. split it into smaller chunks
    3. convert those chunks into embeddings
    4. Store them in Chroma vector database
    5. return a retriver object for semantic search
    """
    # Make sure the PDf file exists before processing
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found:{file_path}")
    
    #Load the PDF file
    loader = PyPDFLoader(file_path)
    documents= loader.load()
 
    # if nothing was extracted from PDF, stop the process
    if not documents:
        raise ValueError(f"No Content extracted:{file_path}")

    logger.info("loaded PDF with %d pages from: %s", len(documents), file_path)

    # Split the document into smaller pieces
    chunks = _splitter.split_documents(documents)
    logger.info("split into %d chunks", len(chunks))

    # Generate a unique ID for file
    file_id = _file_hash(file_path)
    persist_dir = f"./chroma_db_{file_id}"

    # Buid the chroma vector database and store the embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings,
        persist_directory=persist_dir
    )

   # Return a retriever object that will be used in RAG pipeline
    return vectorstore

if __name__ == "__main__":
    file_path = "ERP PPT.pdf"
    vectorstore = build_vectorstore(file_path)
    print(f"Vectorstore built successfully with collection: {vectorstore._collection.name}")
