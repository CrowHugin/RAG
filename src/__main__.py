from indexing import indexing
from transformers import pipeline
from pydantic import BaseModel, Field
import bm25s
import json
import glob
import fire
import sys
import os


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


class RAGCLI:
    def __init__(self):
        self.context: str | None = None

    def index(self, max_chunk_size: int = 2000):
        files = glob.glob(os.path.join("data/raw", "**", "*"), recursive=True)
        indexing(files, max_chunk_size)

    def search(self,
               query: str = None,
               k: int = 1,
               verbose: bool = False) -> list[dict[str, str]]:
        retriever = bm25s.BM25.load("data/processed", load_corpus=True)
        query_tokens = bm25s.tokenize(query)
        data_dict: dict[str, str] = {}
        list_data_dict: list[dict[str, str]] = []
        try:
            result, scores = retriever.retrieve(query_tokens, k=k)
            for i in range(result.shape[1]):
                dict_ = result[0, i]
                score = scores[0, i]
                chunks_texts = [result[0, i]['text'] for i in range(result.shape[1])]
                self.context = "\n\n---\n".join(chunks_texts)

                data_dict = {
                    "file_path": os.path.relpath(dict_['file_path']),
                    "first_character_index": dict_["start_char"],
                    "last_character_index": dict_["end_char"]
                }
                if verbose:
                    print(data_dict)
                list_data_dict.append(data_dict)

        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        return list_data_dict


    def search_dataset(self, 
                       dataset_path: str,
                       k: int = 1,
                       save_directory: str = "res"):
        if os.path.exists(dataset_path):
            list_data: list[dict[str, str]] = []
            data: dict[str, str] = {}
            all_data: list[dict[str, str]] = []
            results: dict[list[dict[str, str]], dict[str, int]] = {}
            with open(dataset_path, "r", encoding="UTF-8") as f:
                data_folder = json.load(f)
            for i in data_folder["rag_questions"]:
                data = {
                    "question_id": i["question_id"],
                    "question": i["question"],
                    "retrieved_sources": self.search(i["question"], k)
                }
                all_data.append(data)
            results = {
                "search_results": all_data,
                "k": k
            }
            print(results)


                
        # question_id: str
        # question: str
        # retrieved_sources: List[MinimalSource]

        # search_results: List[MinimalSearchResults]
        # k: int

            
    # def answer(self, query: str, k: int = 1):
    #     pass

    # def answer_dataset(self,
    #                    student_search_results_path: str,
    #                    save_directory: str):
    #     pass

    # def evaluate(self,
    #              student_search_results_path: str,
    #              dataset_path: str):
    #     pass


    # pipe = pipeline(task="text-generation",
    #                 model="Qwen/Qwen3-0.6B",
    #                 device_map = "auto")
    # chunks_texts = [result[0, i]['text'] for i in range(result.shape[1])]
    # context = "\n\n---\n".join(chunks_texts)

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
    #     },
    #     {
    #         "role": "user",
    #         "content": f"Context:\n{context}\n\nQuestion: {query}"
    #     }
    # ]
    # print("\n--- Génération de la réponse ---")
    # outputs = pipe(
    #     messages, 
    #     max_new_tokens=256,
    #     do_sample=True,
    #     temperature=0.7
    # )

    # response = outputs[0]['generated_text'][-1]['content']
    # print(response)

if __name__ == '__main__':
    fire.Fire(RAGCLI)
