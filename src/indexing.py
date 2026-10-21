import os
import sys
import fire
import tree_sitter_python
from semantic_text_splitter import MarkdownSplitter, CodeSplitter

def _getExt(filepath: str) -> str | None:
    if os.path.exists(filepath):
        return (filepath.split(".")[-1])
    else:
        return None

def indexing() -> list[str]:
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
        with open(file_md, 'r', encoding="UTF-8") as f:
            data = f.read()
    elif ext == "py":
        splitter = CodeSplitter(tree_sitter_python.language(), max_characters)
        with open(file_py, 'r', encoding="UTF-8") as f:
            data = f.read()
    chunks = splitter.chunks(data)
    return (chunks)
    # for i in range(len(chunks)):
    #     print(f"---\n{chunks[i]}")

