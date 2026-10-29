from tqdm import tqdm
from indexing import indexing
from transformers import pipeline
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
        result, scores = retriever.retrieve(query_tokens, k=4)
        for i in range(result.shape[1]):
            dict_ = result[0, i]
            score = scores[0, i]

            print(f"{os.path.relpath(dict_['file_path'])} \
[{dict_["start_char"]}:{dict_["end_char"]}]")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    pipe = pipeline(task="text-generation",
                    model="Qwen/Qwen3-0.6B",
                    device_map = "auto")
    chunks_texts = [result[0, i]['text'] for i in range(result.shape[1])]
    context = "\n\n---\n".join(chunks_texts)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    print("\n--- Génération de la réponse ---")
    outputs = pipe(
        messages, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

    response = outputs[0]['generated_text'][-1]['content']
    print(response)
