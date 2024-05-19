from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

def importdata(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    document=loader.load()
    return document


def datachunking(data_extracted):
    chunker=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
    chunks=chunker.split_documents(data_extracted)
    return chunks


def embeddingmodel():
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding