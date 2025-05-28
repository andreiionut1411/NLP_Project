from langchain_community.vectorstores import Qdrant as QdrantStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
import json
from tqdm import tqdm
import asyncio
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple
from langchain.schema import Document
from ragatouille import RAGPretrainedModel

NUM_OF_RETRIEVED_CHUNKS = 20
NUM_CHUNKS = 1




class ColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):

        print(f"ColBERT model: {model_name}")
        self.model = RAGPretrainedModel.from_pretrained(model_name)
            
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []
        
        doc_texts = [doc.page_content for doc in documents]
        
        try:
            results = self.model.rerank(query=query, documents=doc_texts, k=top_k)
            
            # Add this check to handle None results
            if results is None:
                print("Warning: Reranker returned None, falling back to original order")
                return documents[:top_k]
            
            reranked_docs = []
            for result in results:
                # result contains {'content', 'score', 'rank'}
                doc_idx = doc_texts.index(result['content'])
                reranked_docs.append(documents[doc_idx])
            
            print(f"ColBERT reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            print(f"Error in ColBERT reranking: {e}")
            print("Falling back to original document order")
            return documents[:top_k]



collection_name = "new_emb_base"
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

qdrant_client = QdrantClient(host="localhost", port=6333)

store = QdrantStore(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_function,
    vector_name="embedding"
)

import os


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    api_key=os.getenv("OPENAI_API_KEY")  
)

retriever = store.as_retriever(search_kwargs={"k": NUM_OF_RETRIEVED_CHUNKS})

RERANKER_TYPE = "colbert" # Change to "none" for no reranking, or "colbert" for ColBERT reranking

if RERANKER_TYPE == "colbert":
    reranker = ColBERTReranker(model_name="colbert-ir/colbertv2.0")



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


def get_unique_documents(query: str, n: int) -> List[Document]:
    """
    Retrieve documents using the retriever and return the first `n` unique ones.
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


async def answer_question(query: str, n: int = 5, use_reranker: bool = False) -> Tuple:

    if use_reranker:

        initial_docs = retriever.get_relevant_documents(query)
        
        unique_docs = []
        seen_contents = set()
        for doc in initial_docs:
            content = doc.page_content.strip()
            if content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content)
        
        reranked_docs = reranker.rerank(query, unique_docs, top_k=n)
        unique_chunks = [doc.page_content for doc in reranked_docs]
        
    else:
        
        docs = get_unique_documents(query=f'query: {query}', n=n)
        unique_chunks = [doc.page_content for doc in docs]

    chunk_str = '\n\n'.join(f"Chunk {i+1}\n{str(chunk)}" for i, chunk in enumerate(unique_chunks))
    message = prompt.format(context=chunk_str, query=query)
    
    answer = await llm.ainvoke(message)
    return answer, unique_chunks


async def run_evaluation(use_reranker: bool = True, output_file: str = "results.jsonl"):

    y_true = []
    y_pred = []
    output_records = []

    with open("filtered_statements_good.jsonl") as file:
        lines = [json.loads(line) for line in file]

    queries = [line["claim"] for line in lines]
    
    method = f"{RERANKER_TYPE} reranking" if use_reranker else "simple retrieval"
    print(f"Running evaluation with {method}")
    print(f"Processing {len(queries)} claims...")
    
    results = await asyncio.gather(*[
        answer_question(claim, n=NUM_CHUNKS, use_reranker=use_reranker) for claim in queries
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
            "GT": gold,
            "used_reranker": use_reranker,
            "reranker_type": RERANKER_TYPE if use_reranker else "none"
        })

    with open(output_file, "w") as outfile:
        for record in output_records:
            json.dump(record, outfile)
            outfile.write("\n")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"Results with {method.title()}:")
    print(f"{'='*50}")
    print(f"✅ Accuracy:  {accuracy:.3f}")
    print(f"🎯 Precision: {precision:.3f}")
    print(f"📥 Recall:    {recall:.3f}")
    print(f"📊 F1-score:  {f1:.3f}")
    print(f"💾 Results saved to: {output_file}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":

    asyncio.run(run_evaluation(use_reranker=True, output_file=f"jsons/4o_5_flan_40_{RERANKER_TYPE}.jsonl"))   