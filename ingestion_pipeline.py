import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
  print(f"loading the file from {docs_path} ...")

  #check if docs directory exist
  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"the directory {docs_path} does not exist ")

  #load the text file
  loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader
  )

  #this will give us five different langchain documents
  documents = loader.load()

  if len(documents) == 0:
    raise FileNotFoundError(f"file not found in this path {docs_path}")
  for i, doc in enumerate(documents[:2]):
    print(f"\n Document {i+1}")
    print(f" Source: {doc.metadata['source']}")
    print(f" Content length: {len(doc.page_content)} characters")
    print(f" Content preview: {doc.page_content[:100]}...")
    print(f" metadata: {doc.metadata}")
  return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
  print("Spliting document into chunks....")

  text_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
  )

  chunks = text_splitter.split_documents(documents)

  if chunks:
    for i, chunk in enumerate(chunks[:5]):
      print(f"\n Chunk {i+1}")
      print(f" Source: {chunk.metadata['source']}")
      print(f" Content length: {len(chunk.page_content)} characters")
      print(f" Content preview: {chunk.page_content}")
      print("-" * 50)
    if len(chunks) > 5:
      print(f"\n ..... and {len(chunks)} more chunks")

  return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
  print("creating a vector for these chunks and store them in chroma db")

  embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

  #creating vector store
  print("___ Creating Vector Store ___")
  vectorStore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
  )
  print("finished")
  print(f"vectors saved in to {persist_directory}")
  return vectorStore

def main():
  print("hello world")

  #so inside this function we are ging to have sub functions
  # 1. loading the file
  documents = load_documents(docs_path="docs")

  # 2. chunking the file
  chunks = split_documents(documents)

  # 3. embedding and store it to vector db
  vectorStore = create_vector_store(chunks)

if __name__ == "__main__":
  main()