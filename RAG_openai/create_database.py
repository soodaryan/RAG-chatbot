import os
import shutil
import openai

from typing import List
from dotenv import load_dotenv
from langchain.schema import Document 
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

DATA_PATH = "/Users/aryansood/Github/RAG-chatbot/RAG_openai/chroma"
CHROMA_PATH = "/Users/aryansood/Github/RAG-chatbot/RAG_openai/chroma"

def main() : 
    generate_data_store()

def generate_data_store() : 
    documents = load_doc()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_doc ():
    loader = DirectoryLoader(DATA_PATH, glob = "*.md")
    documents = loader.load()
    return documents

def split_text(documents: List[Document]) : 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks [10]
    print(document.page_content)
    print(document.metadata)

    return chunks 

def save_to_chroma(chunks: List[Document]) : 
    # clear the db first 
    if os.path.exists(CHROMA_PATH) : 
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    db.persist() # to force the data to save / LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    print(f"Number of documents in Chroma: {db._collection.count()}")


if __name__ == "__main__" : 
    main()