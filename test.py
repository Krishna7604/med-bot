from src.helper import HuggingFaceEmbeddings
from src.helper import modelResult
from src.helper import embeddingmodel
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
API_VERSION = "2024-02-01"
MODEL_NAME = "gpt-35-turbo"
ENDPOINT = os.getenv("ENDPOINT")
API_KEY = os.getenv("openai_API_KEY")
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)
os.environ['PINECONE_API_KEY'] =os.getenv("PINECONE_API_KEY")
embedding=embeddingmodel()
index="med-bot"
docsearch=PineconeVectorStore.from_existing_index(index_name=index,embedding=embedding)
# print(docsearch.similarity_search("acne",k=2))
print(modelResult(client,docsearch,"who is prime minister"))