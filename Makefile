DATASETSCOPE = dataset_code_public
K = 4

all: install

install:
	uv sync

run: install
	uv run python -m src index --max_chunk_size 1000
	uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/$(DATASETSCOPE).json -k $(K) -save_directory data/output/search_results
	uv run python -m src answer_dataset --student_search_results_path data/output/search_results/$(DATASETSCOPE)/searchResults.json --save_directory data/output/search_results_and_answer

debug:
	uv run python -m pdb src index --max_chunk_size 1000
	uv run python -m pdb src search_dataset --dataset_path data/datasets/AnsweredQuestions/$(DATASETSCOPE).json -k $(K) -save_directory data/output/search_results
	uv run python -m pdb src answer_dataset --student_search_results_path data/output/search_results/$(DATASETSCOPE)/searchResults.json --save_directory data/output/search_results_and_answer

clean:
	rm -rf .venv
	rm -rf *__pycache__* *mypy_cache*
	rm -rf data/output/search_results/$(DATASETSCOPE)
	rm -rf data/output/search_results_and_answer/$(DATASETSCOPE)
	rm -rf data/processed/*.json data/processed/*jsonl data/processed/*.npy

lint:
	uv run flake8 src  && \
	uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	uv run flake8 src && \
	uv run mypy --strict src --ignore-missing-imports


.PHONY: all install run debug lint lint-strict clean fclean
