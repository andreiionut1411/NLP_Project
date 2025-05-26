import sys
import json
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore as QdrantStore
from transformers import AutoTokenizer

# === CLI Argument ===
if len(sys.argv) < 2:
    print("❌ Usage: python index.py <num_contexts_per_line>")
    sys.exit(1)

try:
    contexts_per_line = int(sys.argv[1])
except ValueError:
    print("❌ Please provide a valid integer for number of contexts per line.")
    sys.exit(1)

# === Config ===
input_path = "poisoned.jsonl"
collection_name = "nq_docs"
chunk_size = 1024
chunk_overlap = 20
batch_size = 1000

# === Embedding model ===
embedding_model_name = "intfloat/e5-small-v2"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

# === Init Qdrant ===
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=15.0)
collections = qdrant_client.get_collections().collections
if collection_name not in [c.name for c in collections]:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={"embedding": VectorParams(size=384, distance=Distance.COSINE)}
    )

store = QdrantStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_function,
    vector_name="embedding"
)

# === Text Splitter ===
splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# === Index poisoned contexts with limit per line ===
buffer = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc=f"🧪 Indexing first {contexts_per_line} contexts per line"):
        record = json.loads(line)
        all_contexts = record.get("contexts", [])
        selected_contexts = all_contexts[:contexts_per_line]  # Take only first N contexts

        for ctx in selected_contexts:
            chunks = splitter.split_text(ctx)
            docs = [Document(page_content=chunk, metadata={"poison": True}) for chunk in chunks]
            buffer.extend(docs)

            if len(buffer) >= batch_size:
                store.add_documents(buffer)
                buffer = []

if buffer:
    store.add_documents(buffer)

print("✅ Poisoned contexts indexed!")