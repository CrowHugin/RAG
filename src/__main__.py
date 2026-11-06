from tqdm import tqdm
from .indexing import indexing
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from .recall import recall
from transformers import pipeline, Pipeline
import re
import os
import sys
import json
import glob
import fire
import uuid
import bm25s
import numpy


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
    def __init__(self) -> None:
        self.context: str | None = None
        self.retriever: bm25s.BM25 | None = None
        self.pipe: Pipeline | None = None

    def _SetRetriever(self) -> bm25s.BM25:
        """
        Set the model in memory
        """
        if self.retriever is None:
            self.retriever = bm25s.BM25.load("data/processed",
                                             load_corpus=True)
        return self.retriever

    def _SetContext(self, res: numpy.ndarray) -> None:
        """
        Set the Context
        """
        chunks_texts = []
        for i in range(res.shape[1]):
            chunks_texts.append(res[0, i]['text'])
        self.context = "\n\n---\n".join(chunks_texts)

    def _SetPipe(self) -> Pipeline:
        """
        Set the Hugging face pipeline
        """
        if self.pipe is None:
            self.pipe = pipeline(
                task="text-generation",
                model="Qwen/Qwen3-0.6B",
                device_map="auto",
                dtype="auto"
            )
        return self.pipe

    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingest data/raw and build the index in data/processed
        """
        files = glob.glob(os.path.join("data/raw", "**", "*"), recursive=True)
        indexing(files, max_chunk_size)

    def search(self,
               query: str = "",
               k: int = 2) -> List[Dict[str, str]]:
        """
        Will search inside the index the k better results for the query
        """
        query_tokens = bm25s.tokenize(query)
        data_dict: Dict[str, str] = {}
        list_data_dict: List[Dict[str, str]] = []
        try:
            retriever = self._SetRetriever()
            result, scores = retriever.retrieve(query_tokens, k=k)
            self._SetContext(result)
            for i in range(result.shape[1]):
                dict_ = result[0, i]

                data_dict = {
                    "file_path": os.path.relpath(dict_['file_path']),
                    "first_character_index": dict_["start_char"],
                    "last_character_index": dict_["end_char"]
                }
                list_data_dict.append(data_dict)

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        return list_data_dict

    def search_dataset(self,
                       dataset_path: str,
                       save_directory: str,
                       k: int = 1
                       ) -> None:
        """
        Will search over a dataset and write a StudentSearchResults json file
        Will use the search function
        """
        if os.path.exists(dataset_path):
            data: Dict[str, Any] = {}
            all_data: List[Dict[str, Any]] = []
            results: Dict[str, Any] = {}
            with open(dataset_path, "r", encoding="UTF-8") as f:
                data_folder = json.load(f)
            for i in tqdm(data_folder["rag_questions"],
                          desc=f"Searching through dataset: \
({dataset_path.split("/")[-1]})"):
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
            datasetScope: str = dataset_path.split("/")[-1].split(".")[0]
            path: str = save_directory + "/" + datasetScope
            os.makedirs(path, exist_ok=True)
            with open(f"{path}/searchResults.json",
                      "w",
                      encoding="UTF-8") as f:
                json.dump(results, f, indent=4)
        else:
            print(f"Error: dataset {dataset_path} not found", file=sys.stderr)
            sys.exit(1)

        try:
            StudentSearchResults.model_validate({
                "search_results": all_data,
                "k": k,
            })
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def answer(self, query: str,
               k: int = 4,
               context: str | None = None) -> str:
        """
        Will generate an answer using the context and the pipeline
        """
        if context is None:
            query_tokens = bm25s.tokenize(query)
            try:
                retriever = self._SetRetriever()
                result, scores = retriever.retrieve(query_tokens, k=k)
                self._SetContext(result)
            except Exception as e:
                print(f"Error during retrieval: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            self.context = context

        try:
            pipe = self._SetPipe()
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. \
Answer the user's question using ONLY the provided context. \
Do not use <think> tags."

                },
                {
                    "role": "user",
                    "content": f"Context:\n{self.context}\n\nQuestion: {query}"
                }
            ]
            outputs = pipe(
                messages,
                max_new_tokens=1024,
                do_sample=False,
            )
            response = outputs[0]['generated_text'][-1]['content']
            res = re.sub(r'<think>.*?</think>',
                         '',
                         response,
                         flags=re.DOTALL).strip()
            if "<think>" in res:
                res = res.split("<think>")[0].strip()
            if not res:
                res = "I don't know"
            return res
        except Exception as e:
            print(f"Error during generation: {e}")
            sys.exit(1)

    def answer_dataset(self,
                       student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Will generate some answer over a dataset and write a
        StudentSearchResultsAndAnswer json file.
        Will use the answer function
        """
        if not os.path.exists(student_search_results_path):
            print(f"Error: dataset {student_search_results_path} not found")
            sys.exit(1)

        with open(student_search_results_path, "r", encoding="UTF-8") as f:
            data_folder = json.load(f)

        search_results = data_folder.get("search_results", [])
        all_data = []
        k = data_folder.get("k", 4)

        for item in tqdm(search_results, desc="generating answers"):
            query = item["question"]
            retrieved_sources = item.get("retrieved_sources", [])
            context_parts = []
            for source in retrieved_sources:
                try:
                    with open(source["file_path"], "r", encoding="UTF-8") as f:
                        content = f.read()
                        chunks_text = (content[source
                                       ["first_character_index"]:source
                                       ["last_character_index"]])
                        context_parts.append(chunks_text)
                except Exception as e:
                    print(f"Error {e}", file=sys.stderr)
                    continue

            context = "\n\n---\n\n".join(context_parts)
            answer_text = self.answer(query, context=context)

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
        datasetScope: str = (student_search_results_path.split("/")
                             [-1].split(".")[0])
        path: str = save_directory + "/" + datasetScope + "AndAnswer"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/answerDataset.json", "w", encoding="UTF-8") as f:
            json.dump(results, f, indent=4)
        try:
            StudentSearchResultsAndAnswer.model_validate({
                "search_results": all_data,
                "k": k,
            })
        except Exception as e:
            print(f"Error: {e}")

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        recall(student_search_results_path, dataset_path)


if __name__ == '__main__':
    fire.Fire(RAGCLI)
