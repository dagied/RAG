from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query = "In what year Tesla did begin production of the Roadster?"

retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"Query: {query}")
print("---- Context ------")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} (from {doc.metadata['source']}):\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model
model = ChatOllama(model="llama3.2")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)