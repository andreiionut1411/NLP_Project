# NLP_Project

## Indexing the data

For this project we used a subset of the FEVER dataset that contains the Wikipedia pages that start with the letter C.

We used Qdrant as vector DB. First you need to start the docker:

`docker run -p 6333:6333 -p 6334:6334   -v $(pwd)/qdrant_data:/qdrant/storage   -e QDRANT__LOG_LEVEL=ERROR   qdrant/qdrant`

After the docker is running you can start indexing each file individually. You should update the `input_path` with the path to the file you want to index and then run:

`python3 index.py`

## Retrieving the data

For testing the RAG, you can use the `rag.py` script. You should add your api key to the script and modify the variables from the script to make different testing setups.