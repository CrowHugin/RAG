import os
import sys
import bm25s
import tree_sitter_python
from tqdm import tqdm
from semantic_text_splitter import MarkdownSplitter, CodeSplitter

def _getExt(filepath: str) -> str | None:
    if os.path.exists(filepath):
        return (filepath.split(".")[-1])
    else:
        return None

def indexing(files: list[str], max_characters) -> None:
    corpus_metadata: list[dict[str, str]] = []

    for file in tqdm(files, desc="indexing files"):
        ext: str | None = _getExt(file)
        if ext not in ("md", "py"):
            continue
        if ext == "py":
            splitter = CodeSplitter(tree_sitter_python.language(), max_characters)
        else:
            splitter = MarkdownSplitter(max_characters)
        try:
            with open(file, 'r', encoding="UTF-8") as f:
                data = f.read()
        except Exception as e:
            print(f"Error while reading file {file}: {e}",
                  file=sys.stderr)

        chunk_infos = splitter.chunk_indices(data)
        chunks = splitter.chunks(data)

        for index, (start_char, text) in enumerate(chunk_infos):
            end_char = start_char + len(text)
            corpus_metadata.append(
                {
                    "file_path": file,
                    "chunk_id": index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": text
                }
            )

    if corpus_metadata:
        textToTokenize: list[str] = [item["text"] for item in corpus_metadata]
        chunks_tokens = bm25s.tokenize(textToTokenize)

        retriever = bm25s.BM25(corpus=corpus_metadata)
        retriever.index(chunks_tokens)
        retriever.save("./data/processed", corpus=corpus_metadata)
