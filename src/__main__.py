from tqdm import tqdm
from indexing import indexing
import bm25s
import fire
import sys
import os

if __name__ == '__main__':
    retriever: bm25s.BM25 = indexing()

    query = "does the fish purr like a cat?"
    query_tokens = bm25s.tokenize(query)
    try:
        docs, scores = retriever.retrieve(query_tokens, k=1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")
