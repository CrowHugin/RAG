*This activity has been created as part of the 42 curriculum by bkuntz*

___
## **Description** 

This project is about using what's RAG and does it works within an AI context.

RAG (Retrieved-augmented generation) is a way to get the context of a question without using AI. The resources are then less costly. You then give the context to the AI and it will give you a answer using the context.

___
## **Instructions** 

| Command         | use                                  |
| --------------- | ------------------------------------ |
| all             | default to install                   |
| install         | install the env using uv             |
| run             | run the script                       |
| debug           | run the script in debug mode         |
| clean           | Clean the unnecessary files / folder |
| lint            | run a basic lint                     |
| lint-strict<br> | run a more stricter lint             |
___
## **Ressources**

resources used:
- https://bm25s.github.io/
- https://pypi.org/project/semantic-text-splitter/
- https://github.com/huggingface/transformers/blob/main/docs/source/en/pipeline_tutorial.md
### AI usage

AI was used for comprehension when necessary and for debugging some issues.
___
## **System architecture**

*Pipeline:*
chunking (semantic_text_splitter) -> indexing (bm25s) -> retrieval(bm25s) -> build_context -> generate (Qwen3-0.6B) -> answer json

recall can be use on the result on the retrieval to verify if the top-k sources are the good one.

The chunking / indexing part create a folder data/processed with necessary infos on every chunk.

The retrieval part will use the infos on this folder to get the top-K sources for a query.

The script will build the context, give it to the AI and put the answer on a json file, with other infos.

___
## **Chunking strategy**

Python files and Markdown are different on a fundamental level, so the chunking strategies are meant to be different as well. From `semantic_text_splitter` we use `MarkdownSplitter` and `CodeSplitter`.

The chunking size cannot be negative, and cannot exceed 2000 character.
bm25s will then index and save the data on `data/processed`
___
## **Retrieval method**
bm25s will load the data on `data/processed`.
The `.retrieve` will then rank sources and then keep the top-k sources.
___
## **Performance Analysis**

A recall was made to check if the retrieval is working, we need more than 50% on code and more than 80% on doc.
___
## **Design Decisions**

`Search_dataset` use the `search` function and the Answer_dataset use the `answer` function.
To make things way faster, sending the question by batch was investigated but needed the `answer function` to return different answer. This was forbidden inside the subject so the idea was given up on. 
___
## **Challenges faced**

Usually the most difficult things to do is to find the good module to use, but a friend gave me some hints on what to do.

The other complicating thing was the motivation, I don't like working on AI project.
___
## **Example usage**
```bash
uv run python -m src index --max_chunk_size 1500
uv run python -m src search_dataset --dataset_path \ data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 -save_directory \ data/output/search_results
uv run python -m src answer_dataset --student_search_results_path \ data/output/search_results/dataset_code_public/searchResults.json \ 
--save_directory data/output/search_results_and_answer

```