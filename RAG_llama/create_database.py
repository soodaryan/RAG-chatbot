import os
import shutil
import argparse

from typing import List
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

from get_embedding_func import get_embedding_function
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "/Users/aryansood/Github/RAG-chatbot/RAG_llama/data"


def load_documents() : 
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents : List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)
