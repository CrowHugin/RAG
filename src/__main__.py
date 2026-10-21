from indexing import indexing
import os
import bm25s
import json

def _saveInFile(content: str, filepath: str) -> None:
# def add_to_file(filepath: str, new_dict: dict[str, Any]) -> None:
    """Appends a chunk to a JSON list in a file."""
    data = []
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(content)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    file: str = "test.json"
    chunks: list[str] = indexing()
    for i in range(len(chunks)):
        _saveInFile(chunks[i], file)
    with open(file, 'r', encoding="UTF-8") as f:
        data = json.load(f)

    for i in range(len(data)):
        chunks_tokens = bm25s.tokenize(data[i])
        retriever = bm25s.BM25(corpus=data[i])
        retriever.index(chunks_tokens)

        query = "does the fish purr like a cat?"
        query_tokens = bm25s.tokenize(query)
        docs, scores = retriever.retrieve(query_tokens, k=1)
        print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")
