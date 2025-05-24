from langchain_community.vectorstores import Qdrant as QdrantStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm
import asyncio
from collections import Counter
import matplotlib.pyplot as plt
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NUM_OF_RETRIEVED_CHUNKS = 40
NUM_CHUNKS = 5

# === Init Qdrant and embedding ===
collection_name = "new_emb_base"
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

qdrant_client = QdrantClient(host="localhost", port=6333)

# === Load QdrantStore ===
store = QdrantStore(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_function,
    vector_name="embedding"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    api_key="api_key"
    )

retriever = store.as_retriever(search_kwargs={"k": NUM_OF_RETRIEVED_CHUNKS})
compressor = FlashrankRerank(top_n=NUM_CHUNKS, model='rank-T5-flan')
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

prompt = """You are a helpful assistant.
You will receive some contexts which will help you decide if a claim is correct or not.
You will answer with "SUPPORTS" if the claim is backed up by the context.
You will answer with "REFUTES" if, from the context, you find that the claim is false.
You will answer with "NOT ENOUGH INFO" if there is not enough information in the context to decide if the claim is true or false.
You MUST answer only with these options. Any other option will be ignored.
When deciding if the claim is true or false based on the context, you MUST only use the information provided by the context, NOT your general knowledge.
Contexts: {context}
Claim: {query}
Answer:"""


def get_unique_documents(query: str, n: int, k: int = 20):
    """
    Retrieve documents using the retriever and return the first `n` unique ones.

    Args:
        query (str): The input query string.
        n (int): Number of unique documents to return.
        k (int): Number of documents to retrieve initially (to allow room for deduplication).

    Returns:
        List[Document]: A list of unique documents based on page_content.
    """
    raw_results = retriever.get_relevant_documents(query)

    unique_docs = []
    seen_contents = set()

    for doc in raw_results:
        content = doc.page_content.strip()
        if content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(content)
        if len(unique_docs) >= n:
            break

    return unique_docs

async def answer_question(query, n=5, reranker=False):
    if reranker:
        compressed_docs = compression_retriever.invoke(query)
        unique_chunks = ([doc.page_content for doc in compressed_docs])
    else:
        compressed_docs = get_unique_documents(query='query: ' + query, n=n)
        unique_chunks = [doc.page_content for doc in compressed_docs]

    chunk_str = '\n\n'.join(f"Chunk {i+1}\n{str(chunk)}" for i, chunk in enumerate(unique_chunks))
    message = prompt.format(context=chunk_str, query=query)
    answer = await llm.ainvoke(message)
    return answer, unique_chunks

async def run_all():
    y_true = []
    y_pred = []
    output_records = []

    with open("filtered_statements_good.jsonl") as file:
        lines = [json.loads(line) for line in file]

    queries = [line["claim"] for line in lines]
    results = await asyncio.gather(*[
        answer_question(claim, n=5, reranker=True) for claim in queries
    ])

    for line, (result, context_chunks) in zip(lines, results):
        predicted = result.content.strip().upper()
        gold = line["label"].strip().upper()
        claim = line["claim"]

        y_true.append(gold)
        y_pred.append(predicted)

        context = "\n\n".join(context_chunks)

        output_records.append({
            "claim": claim,
            "context": context,
            "answer": predicted,
            "GT": gold
        })

    with open("jsons/4o_5_flan_40.jsonl", "w") as outfile:
        for record in output_records:
            json.dump(record, outfile)
            outfile.write("\n")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n✅ Accuracy:  {accuracy:.3f}")
    print(f"🎯 Precision: {precision:.3f}")
    print(f"📥 Recall:    {recall:.3f}")
    print(f"📊 F1-score:  {f1:.3f}")


def run_all_with_reranker_rank_tracking():
    rerank_positions = []

    with open("filtered_statements_good.jsonl") as file:
        lines = [json.loads(line) for line in file]

    for line in tqdm(lines, desc="Analyzing reranker impact"):
        query = line["claim"]

        # Step 1: Initial retrieval
        initial_docs = retriever.get_relevant_documents(query)
        original_texts = [doc.page_content.strip() for doc in initial_docs]

        # Step 2: Reranking
        reranked_docs = compression_retriever.invoke(query)
        reranked_texts = [doc.page_content.strip() for doc in reranked_docs]

        # Step 3: Record the original index of the top reranked chunk(s)
        for top_doc in reranked_texts[:NUM_CHUNKS]:
            try:
                rank = original_texts.index(top_doc)
                rerank_positions.append(rank)
            except ValueError:
                continue  # In case of slight text mismatch

    # Step 4: Plot histogram
    count = Counter(rerank_positions)
    x = list(range(NUM_OF_RETRIEVED_CHUNKS))
    y = [count.get(i, 0) for i in x]

    plt.bar(x, y)
    plt.xlabel("Original Rank in Retrieval")
    plt.ylabel("Frequency as Top Reranked Chunk")
    plt.title("Impact of Reranking: Original Positions of Top Chunks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


asyncio.run(run_all())

# run_all_with_reranker_rank_tracking()