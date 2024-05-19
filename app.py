from flask import Flask,render_template,jsonify,request
from langchain.chains import RetrievalQA
from src.helper import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from src.prompts import *
from langchain.llms import CTransformers
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
os.environ['PINECONE_API_KEY'] =os.getenv("PINECONE_API_KEY")
embedding=HuggingFaceEmbeddings()
index="med-bot"
docsearch=PineconeVectorStore.from_existing_index(index_name=index,embedding=embedding)



prompt=PromptTemplate(template=prompttemplate,input_variables=["context", "question"])
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={"max_new_tokens":512,
                          "temperature":0.5})
medbot=RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=docsearch.as_retriever(kwargs={"k":2}),
                              
                            chain_type_kwargs={"prompt": prompt})


@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=medbot({"query":input})
    print("Response : hello")
    return result["result"]


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)