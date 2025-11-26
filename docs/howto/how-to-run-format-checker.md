# How to run format checker

We highly recommand to use pre-commit tool to run format checker automatically before commit

## Install pre-commit

- Install `uv` to use `pre-commit` hook to automatically format code before commit
  - `uv`: https://docs.astral.sh/uv/
- Install `pre-commit` hook by `uv run pre-commit install`

## Run pre-commit hook manually

- You can use `uv run pre-commit run` command to manually run pre-commit hook to staged files
- You can use `uv run pre-commit run -a` command to run pre-commit hook on all files
  - You can use pre-commit hook to apply C/C++ and python to entire project
