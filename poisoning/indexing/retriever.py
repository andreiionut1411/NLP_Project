from langchain_community.vectorstores import Qdrant as QdrantStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# === Init Qdrant and embedding ===
collection_name = "nq_docs"
embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

qdrant_client = QdrantClient(host="localhost", port=6333)

# === Load QdrantStore ===
store = QdrantStore(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_function,
    vector_name="embedding"
)

# === Create retriever ===
retriever = store.as_retriever(search_kwargs={"k": 5})

# === Query ===
query = "query: what type of fertilisation takes place in humans"
results = retriever.get_relevant_documents(query)

# === Print results ===
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:\n{doc.page_content}")
