import os
import sys
import fire
from semantic_text_splitter import MarkdownSplitter, CodeSplitter

def _getExt(filepath: str) -> str | None:
    if os.path.exists(filepath):
        return (filepath.split(".")[-1])
    else:
        return None


if __name__ == '__main__':
    file: str = "./data/raw/vllm-0.10.1/CODE_OF_CONDUCT.md"
    ext: str | None = _getExt(file)
    if ext not in ("md", "py"):
        print("Error: file format not recognised")
        # si dans une boucle, mettre un continue later on
        sys.exit(1)
    print(ext)
    max_characters = 2000
    if ext == "md":
        splitter = MarkdownSplitter(max_characters)
    elif ext == "py":
        splitter = CodeSplitter(max_characters)
    with open(file, 'r', encoding="UTF-8") as f:
        data = f.read()
    chunks = splitter.chunks(data)
    for i in range(len(chunks)):
        print(f"---\n{chunks[i]}")
        print(len(chunks[i]))
