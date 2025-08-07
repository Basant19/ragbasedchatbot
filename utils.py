import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from typing import List
from exception import CustomException
import sys
from logging_config import logger

def save_uploaded_file(uploaded_file) -> str:
    try:
        os.makedirs("temp_files", exist_ok=True)
        file_ext = os.path.splitext(uploaded_file.name)[1]
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join("temp_files", file_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        logger.error(str(e))
        raise CustomException(e, sys)

def load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.error(str(e))
        raise CustomException(e, sys)

def load_url(url: str) -> List[Document]:
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        logger.error(str(e))
        raise CustomException(e, sys)

def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(str(e))
