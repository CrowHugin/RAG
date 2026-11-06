import os
import sys
import bm25s
import tree_sitter_python
from typing import Any, List, Dict
from tqdm import tqdm
from semantic_text_splitter import MarkdownSplitter, CodeSplitter


def _getExt(filepath: str) -> str | None:
    """
    Function to get the extention of a file
    """
    if os.path.exists(filepath):
        return (filepath.split(".")[-1])
    else:
        return None


def indexing(files: List[str], max_characters: int) -> None:
    """
    Function to index files with bm25s
    """
    corpus_metadata: List[Dict[str, Any]] = []
    splitter: CodeSplitter | MarkdownSplitter

    for file in tqdm(files, desc="indexing files"):
        ext: str | None = _getExt(file)

        if ext not in ("md", "py"):
            continue

        if "tests/" in file or "benchmarks/" in file:
            continue

        if ext == "py":
            splitter = CodeSplitter(tree_sitter_python.language(),
                                    max_characters)
        else:
            splitter = MarkdownSplitter(max_characters)
        try:
            with open(file, 'r', encoding="UTF-8") as f:
                data = f.read()
        except Exception as e:
            print(f"Error while reading file {file}: {e}",
                  file=sys.stderr)
            continue

        if not data.strip():
            continue

        chunk_infos = splitter.chunk_indices(data)

        for index, (start_char, text) in enumerate(chunk_infos):
            end_char = start_char + len(text)
            enhanced_text = f"File:{file}\nContent:\n{text}"
            corpus_metadata.append(
                {
                    "file_path": file,
                    "chunk_id": index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": enhanced_text
                }
            )

    if corpus_metadata:
        textToTokenize: List[str] = [item["text"] for item in corpus_metadata]
        chunks_tokens = bm25s.tokenize(textToTokenize)

        retriever = bm25s.BM25(corpus=corpus_metadata)
        retriever.index(chunks_tokens)
        retriever.save("./data/processed", corpus=corpus_metadata)
