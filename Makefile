all: install

install:
	if [ ! -f ./pyproject.toml ]; then uv init .; fi
	uv venv ./.venv
	# uv pip install -e attachments_project_ai/llm_sdk --python ./.venv
	# uv pip install accelerate pydantic

run:
	uv run python -m src

debug:
	uv run python -m pdb src

lint:
	uv run flake8 src  && \
	uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	uv run flake8 src && \
	uv run mypy --strict src --ignore-missing-imports

clean:
	rm -rf ./venv
	rm -rf *__pycache__* *mypy_cache*
	rm -rf attachments_project_ai/data/output

.PHONY: all install run debug lint lint-strict clean fclean
