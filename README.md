# NLP_Project

## Indexing the data

For this project we used a subset of the FEVER dataset that contains the Wikipedia pages that start with the letter C.

We used Qdrant as vector DB. First you need to start the docker:

`docker run -p 6333:6333 -p 6334:6334   -v $(pwd)/qdrant_data:/qdrant/storage   -e QDRANT__LOG_LEVEL=ERROR   qdrant/qdrant`

After the docker is running you can start indexing each file individually. You should update the `input_path` with the path to the file you want to index and then run:

`python3 index.py`

## Retrieving the data

For testing the RAG, you can use the `rag.py` script. You should add your api key to the script and modify the variables from the script to make different testing setups.

The most important part in terms of how to run the code is the line:

`results = await asyncio.gather(*[answer_question(claim, n=5, reranker=True) for claim in queries])`

If the reranker is False, then we don't use the reranker. By modifying n, you decide how many chunks to include in the context for the RAG. Here you can play with it.

If reranker == True, then you can modify NUM_OF_RETRIEVED_CHUNKS and NUM_CHUNKS defined at the begining of the script. By modifying NUM_OF_RETRIEVED_CHUNKS you decide how many chunks to go in the reranking process. With NUM_CHUNKS you decide how many chunks to keep and send to the LLM's context.


## RAG Poisoning

For testing RAG posioning we have the poisoning folder. In the poison subfolder you have multiple useful scripts.

`python3 generate_poison.py` -> You should have a questions.jsonl file with the questions. This script will create 5 fake contexts that are created as in the PoisonRAG paper.

`python3 index_poison.py N` -> You index N chunks in the database. N should be smaller or equal to 5.

`python3 delete_poison` -> This removes the poisoned chunks

`python3 rag.py` -> This script runs the actual tests. You can modify the variables in the code to make different scenarios.


In the indexing subfolder of the poisoning part you have multiple interesting scripts. The first one is select_pairs.py that randomly selects pairs of questions and answers from the NQ dataset pool of questions.

The clean_html.py script cleans the raw Wikipedia pages. After they are cleaned, you should run split.py which splits the texts into files of 15000 Wikipedia pages. Then you can use the index.py script to start indexing the pages. You should run this script for each file with Wikipedia pages. The original dataset was split into multiple parts so that in case the indexing fails, you don't need to start from the begining.


## References

@article{zou2024poisonedrag,
  title={Poisonedrag: Knowledge corruption attacks to retrieval-augmented generation of large language models},
  author={Zou, Wenxuan and Geng, Rui and Wang, Boxin and Jia, Jiawei},
  journal={arXiv preprint arXiv:2402.07867},
  year={2024}
}