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
def modelResult(client,docsearch,question):
    res=docsearch.similarity_search(question,k=2)
    result=""
    from src.prompts import prompttemplate2
    MODEL_NAME = "gpt-35-turbo"
    
    for i in range(2):
        prompt=prompttemplate2.format(context=res[i].page_content)
        MESSAGES = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES)
        result+="\n\n"+completion.choices[0].message.content
        print(result)
    prompt2="""use the following context to answer the user's question in more summerised way
    don't make the answer
    if question is not related to the context just return "not related  to me "
    context={result}
    ensure answer should be based on context and more detailed way
    """.format(result=result)
    MESSAGES = [
            {"role": "system", "content": prompt2},
            {"role": "user", "content": question}]
    response=client.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES)
    return response.choices[0].message.content