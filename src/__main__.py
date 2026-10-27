from tqdm import tqdm
from indexing import indexing
import bm25s
import fire
import sys
import os

if __name__ == '__main__':
    indexing()
    retriever = bm25s.BM25.load("./data/processed", load_corpus=True)

    query = "does the fish purr like a cat?"
    query_tokens = bm25s.tokenize(query)
    try:
        result, scores = retriever.retrieve(query_tokens, k=5)
        for i in range(result.shape[1]):
            dict_ = result[0, i]
            score = scores[0, i]

            print(f"Fichier source : {dict_['file_path']}")
            print(f"Extrait : {dict_['text'][:100]}...")
            print(f"Score : {score:.2f}\n")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")
