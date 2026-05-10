import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#frontend
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

faiss_path="/Users/arindambairagi/Desktop/Rag ChatBot project/FAISS_INDEX"
API_KEY="paste your google api key here" #google api key for using the gemini model
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(faiss_path,embeddings,allow_dangerous_deserialization=True)
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=API_KEY)
retriever=db.as_retriever(search_kwargs={"k":3})
system_prompt="""
You are a helpful assistant.
use the context to answer the question in three sentences maximum.
If you dont know the answer, say you dont know.
Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
qa_chain=create_stuff_documents_chain(llm,prompt)#combines retrieved documents into 1 prompt
rag_chain=create_retrieval_chain(retriever,qa_chain)#creates a full rag pipeline
app=FastAPI()
#middleware is used to allow requests from the frontend part.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

@app.post("/query")
def query_rag(query: Query):
    response = rag_chain.invoke({"input": query.text})
    return {"answer": response.get("answer", "No answer found")}