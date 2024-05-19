from src.helper import importdata,datachunking,embeddingmodel
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv() 
extrected_data=importdata("dat/")
docs=datachunking(extrected_data)
embedding=embeddingmodel()
PineconeVectorStore(pinecone_api_key=os.getenv("PINECONE_API_KEY"))
PineconeVectorStore.from_texts([p.page_content for p in docs],embedding=embedding,index_name="med-bot")

