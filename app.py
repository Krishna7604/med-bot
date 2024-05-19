from flask import Flask,render_template,jsonify,request
from langchain.chains import RetrievalQA
from src.helper import embeddingmodel
from langchain.prompts import PromptTemplate
from src.prompts import *
from langchain.llms import CTransformers
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from openai import AzureOpenAI
load_dotenv()
app = Flask(__name__)
os.environ['PINECONE_API_KEY'] =os.getenv("PINECONE_API_KEY")
embedding=embeddingmodel()
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

API_VERSION = "2024-02-01"
MODEL_NAME = "gpt-35-turbo"
ENDPOINT = os.getenv("ENDPOINT")
API_KEY = os.getenv("openai_API_KEY")
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)
from src.helper import modelResult
def modelResult(client,docsearch,question):
    res=docsearch.similarity_search(question,k=4)
    result=""
    from src.prompts import prompttemplate2
    MODEL_NAME = "gpt-35-turbo"
    
    for i in range(4):
        prompt=prompttemplate2.format(context=res[i].page_content)
        MESSAGES = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES)
        result+=completion.choices[0].message.content
    prompt2="""use the following context to answer the user's question in more summerised way
    don't make the answer
    if question is not related to the context just return "not related  to me "
    context={result}
    ensure answer should be based on context and more detailed way
    """
    MESSAGES2 = [
            {"role": "system", "content": prompt2},
            {"role": "user", "content": question}]
    response=client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES2)
    return response.choices[0].message.content
        



@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=medbot({"query":input})
    print("Response : ")
    return result["result"]


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)