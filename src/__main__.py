from indexing import indexing
from transformers import pipeline
from pydantic import BaseModel, Field
from typing import List, Dict
from tqdm import tqdm
import numpy
import uuid
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
        self.retriever: bm25s.B25 | None = None 
        self.pipe = None 

    def _SetRetriever(self) -> None:
        """
        Set the model in memory
        """
        if self.retriever == None:
            self.retriever = bm25s.BM25.load("data/processed",
                                             load_corpus=True)

    def _SetContext(self, res: numpy.ndarray) -> None:
        """
        Set the Context
        """
        if self.context == None:
            for i in range(res.shape[1]):
                chunks_texts = [res[0, i]['text']]
                self.context = "\n\n---\n".join(chunks_texts)

    def _SetPipe(self) -> None:
        """
        Set the Hugging face pipeline
        """
        if self.pipe == None:
            self.pipe = pipeline(
                task="text-generation",
                model="Qwen/Qwen3-0.6B",
                device_map="auto",
                torch_dtype="auto"
            )


    def index(self, max_chunk_size: int = 2000):
        """
        Ingest data/raw and build the index in data/processed
        """
        files = glob.glob(os.path.join("data/raw", "**", "*"), recursive=True)
        indexing(files, max_chunk_size)

    def search(self,
               query: str = None,
               k: int = 2,
               verbose: bool = False) -> list[dict[str, str]]:
        """
        Will search inside the index the k better results for the query
        """
        query_tokens = bm25s.tokenize(query)
        data_dict: Dict[str, str] = {}
        list_data_dict: List[Dict[str, str]] = []
        try:
            result, scores = self.retriever.retrieve(query_tokens, k=k)
            self._SetContext(result)
            for i in range(result.shape[1]):
                dict_ = result[0, i]
                score = scores[0, i]

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
                       save_directory: str = "./res"
                       ) -> None:
        """
        Will search over a dataset and write a StudentSearchResults json file
        Will use the search function
        """
        self._SetRetriever()
        if os.path.exists(dataset_path):
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
            os.makedirs(save_directory, exist_ok=True)
            with open(f"{save_directory}/res.json", "w", encoding="UTF-8") as f:
                json.dump(results, f, indent = 4)
        else:
            print(f"Error: dataset {dataset_path} not found")
            sys.exit(1)

        try:
            StudentSearchResults(search_results=all_data, k=k)
        except Exception as e:
            print(f"Error: {e}")
                

    def answer(self, query: str, k: int = 2, verbose: bool = False) -> str:
        """
        Will generate an answer using the context and the pipeline
        """
        query_tokens = bm25s.tokenize(query)
        try:
            self._SetRetriever()
            result, scores = self.retriever.retrieve(query_tokens, k=k)
            self._SetContext(result)
            self._SetPipe()
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say 'I don't know'."
            },
            {
                "role": "user",
                "content": f"Context:\n{self.context}\n\nQuestion: {query}"
            }
        ]
        outputs = self.pipe(
            messages, 
            max_new_tokens=256,
            do_sample=False,
            truncation=True
        )
        response = outputs[0]['generated_text'][-1]['content']
        if "</think>" in response:
            _, _, clean_reponse = response.partition("</think>")
            res = clean_reponse.strip()
        else:
            res = response.strip()
        if verbose:
            print(res)
        return res

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str):
        """
        Will generate some answer over a dataset and write a 
        StudentSearchResultsAndAnswer json file.
        Will use the answer function
        """
        if not os.path.exists(student_search_results_path):
            print(f"Error: dataset {dataset_path} not found")
            sys.exit(1)

        with open(student_search_results_path, "r", encoding="UTF-8") as f:
            data_folder = json.load(f)

        search_results = data_folder.get("search_results", [])
        all_data = []
        k = 4
        for item in tqdm(search_results, desc="generating answers"):
            query = item["question"]
            answer_text = self.answer(query, k)
            retrieved_sources = item.get("retrieved_sources", [])

            data = {
                "question_id": item["question_id"],
                "question": query,
                "retrieved_sources": retrieved_sources,
                "answer": answer_text
            }
            all_data.append(data)
        results = {
            "search_results": all_data,
            "k": k
        }
        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/answerDataSet.json", "w", encoding="UTF-8") as f:
            json.dump(results, f, indent = 4)
        try:
            StudentSearchResultsAndAnswer(search_results=all_data, k=k)
        except Exception as e:
            print(f"Error: {e}")

    # def evaluate(self,
    #              student_search_results_path: str,
    #              dataset_path: str):
    #     pass

if __name__ == '__main__':
    fire.Fire(RAGCLI)
