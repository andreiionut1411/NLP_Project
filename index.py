from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore as QdrantStore
from transformers import AutoTokenizer
import json
from nltk import sent_tokenize

# === Config ===
input_path = "wiki-pages/wiki-026.jsonl"
collection_name = "new_emb_base"
chunk_size = 150          # 1024 tokens, not characters
chunk_overlap = 20         # 20 token overlap
batch_size = 1000

# === Embedding model ===
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

# === Init Qdrant ===
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=15.0)

# Create collection only if it doesn't exist
collections = qdrant_client.get_collections().collections
if collection_name not in [c.name for c in collections]:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "embedding": VectorParams(size=768, distance=Distance.COSINE)
        }
    )

# LangChain Qdrant wrapper
store = QdrantStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_function,
    vector_name="embedding"
)

# === Token-based text splitter ===
splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

def smart_chunk(text: str, token_threshold: int = 150):
    token_count = len(tokenizer.encode(text, add_special_tokens=False))

    if token_count <= token_threshold:
        return sent_tokenize(text)
    else:
        return splitter.split_text(text)


# === Chunk, embed and upload ===
with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

    buffer = []

    for line in tqdm(lines, total=len(lines), desc="📦 Smart Chunking & Uploading"):
        text_obj = json.loads(line)
        text = text_obj.get("text", "")
        if not text:
            continue

        chunks = smart_chunk(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        buffer.extend(docs)

        if len(buffer) >= batch_size:
            store.add_documents(buffer)
            buffer = []

    if buffer:
        store.add_documents(buffer)

print("✅ Done indexing!")
