import os#for system files validation 
from langchain_community.document_loaders import PyPDFLoader,TextLoader,DirectoryLoader#for loading files
from langchain_text_splitters import RecursiveCharacterTextSplitter#for splitting the text into chunks
from langchain_huggingface import HuggingFaceEmbeddings#for creating embeddings
from langchain_community.vectorstores import FAISS#for storing the embeddings in a vector database(Facebook au similarity search)

data_path="/Users/arindambairagi/Desktop/Rag ChatBot project/Knowledge_base"#path to the knowledge base
faiss_path="/Users/arindambairagi/Desktop/Rag ChatBot project/FAISS_INDEX"#path to store the faiss index
print("Loading Text files...")
txt_loader=DirectoryLoader(data_path,glob="**/*.txt",loader_cls=TextLoader)#loading text files
txt_docs=txt_loader.load()#loading the documents
print("Loading PDF files...")
pdf_loader=DirectoryLoader(data_path,glob="**/*.pdf",loader_cls=PyPDFLoader)
pdf_docs=pdf_loader.load()#loading the documents
print("Text and PDF files loaded successfully.")
docs=pdf_docs+txt_docs
print("chunking the documents...")
chunks=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
docs=chunks.split_documents(docs)
print("Creating Embeddings")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.from_documents(docs,embeddings)
db.save_local(faiss_path)
print("embeddings are stored in database")
