# Test Suite Guide

## Suite Shape

Tests mirror package and entrypoint ownership:

- Core: `test_neis.py`, `test_cleaning.py`, `test_recommender.py`, `test_service.py`, `test_ui.py`.
- Mokpo: `test_mokpo_data.py`, `test_mokpo_analytics.py`, `test_mokpo_ui.py`, `test_mokpo_service.py`.
- Profiles/models: `test_student_profiles.py`, `test_student_profile_ui.py`, `test_matrix_factorization.py`.
- Integrations: `test_text_vectors.py`, `test_nim_chat.py`, `test_live_data.py`.
- Course artifacts: `test_jupyter_textbook.py`, `test_jupyter_verifier.py`, `test_frozen_notebook_outputs.py`, `test_jupyter_installation.py`, `test_jupyter_support.py`, `test_notebook.py`.

## Conventions

- Use pytest function tests. `conftest.py` only adds the project root and `src/` to `sys.path`; there are no shared fixtures to assume.
- Use `tmp_path` for SQLite, CSV, JSON, notebooks, verification outputs, and key-path behavior.
- Use `monkeypatch` or injected callables/clients for environment, HTTP, embedding, NIM, and Gradio boundaries.
- Prefer explicit DataFrame fixtures with the smallest schema that exercises the contract.
- Assert user-facing Korean guidance where it is part of safety, validation, or fallback behavior.
- Use `pytest.raises` for invalid boundary inputs; do not weaken production validation to simplify a test.
- Keep randomness deterministic and numerical assertions tolerant only where the algorithm is genuinely approximate.
- Assert complete schemas/order and meaningful `DataFrame.attrs` when they are part of the analytics contract.
- For persistence, cover resume plus rejected, partial, stale, and concurrent writes where relevant.

## Isolation Rules

Tests must not:

- call live NEIS or NVIDIA services;
- download models or depend on a GPU;
- read workspace credential files;
- open `runtime_data/student_profiles.sqlite3`;
- bind a public/LAN server;
- mutate tracked data or generated notebooks in place;
- print or assert literal real credentials, passwords, or student personal data.

## Commands

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_<area>.py
.\.venv\Scripts\python.exe -m pytest -q
```

The configured `testpaths` is `tests` and `addopts` includes `-ra`. Notebook builder tests use temporary artifacts and do not replace fresh-kernel verification. For textbook changes also run `scripts\verify_jupyter_textbook.py`; for Colab changes run `scripts\verify_colab.py`.
