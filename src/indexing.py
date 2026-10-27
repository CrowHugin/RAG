import os
import sys
import bm25s
import tree_sitter_python
from semantic_text_splitter import MarkdownSplitter, CodeSplitter

def _getExt(filepath: str) -> str | None:
    if os.path.exists(filepath):
        return (filepath.split(".")[-1])
    else:
        return None

def indexing() -> bm25s.BM25:
    file_py: str = "./data/raw/vllm-0.10.1/find_cuda_init.py"
    file_md: str = "./data/raw/vllm-0.10.1/CODE_OF_CONDUCT.md"
    ext: str | None = _getExt(file_md)
    if ext not in ("md", "py"):
        print("Error: file format not recognised")
        # si dans une boucle, mettre un continue later on
        sys.exit(1)
    max_characters = 2000
    if ext == "md":
        splitter = MarkdownSplitter(max_characters)
        file: str = file_md
    elif ext == "py":
        splitter = CodeSplitter(tree_sitter_python.language(), max_characters)
        file = file_py
    try:
        with open(file, 'r', encoding="UTF-8") as f:
            data = f.read()
    except Exception as e:
        print(f"Error while reading file {file_md}: {e}")

    chunks = splitter.chunks(data)
    chunks_tokens = bm25s.tokenize(chunks)
    retriever = bm25s.BM25(corpus=chunks)
    retriever.index(chunks_tokens)
    retriever.corpus = chunks
    retriever.save("./data/processed")
    return (retriever)
