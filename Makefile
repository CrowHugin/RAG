FILE = dataset_code_public.json
SCOPE = AnsweredQuestions
MAX_CHUNK_SIZE = 2000
K = 5

DATASET_PATH    = data/datasets/$(SCOPE)/$(FILE)
SEARCH_DIR      = data/output/search_results/
ANSWER_DIR      = data/output/search_results_and_answer/
SEARCH_FILE     = $(SEARCH_DIR)/$(FILE)

all: install

install:
	uv sync

run: install
	uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)
	uv run python -m src search_dataset --dataset_path $(DATASET_PATH) --k $(K) -save_directory $(SEARCH_DIR)
	uv run python -m src answer_dataset --student_search_results_path $(SEARCH_FILE) --save_directory $(ANSWER_DIR)

debug:
	uv run python -m pdb src index --max_chunk_size $(MAX_CHUNK_SIZE)
	uv run python -m pdb src search_dataset --dataset_path data/datasets/AnsweredQuestions/$(DATASET_PATH).json --k $(K) -save_directory data/output/search_results
	uv run python -m pdb src answer_dataset --student_search_results_path data/output/search_results/$(FILE)/searchResults.json --save_directory data/output/search_results_and_answer

clean:
	rm -rf .venv
	rm -rf *__pycache__* *mypy_cache*
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
