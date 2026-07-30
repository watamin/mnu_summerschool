# Scripts Guide

## Ownership

| Script | Owns |
| --- | --- |
| `build_jupyter_textbook.py` | Generates chapters `00`-`08` and appendices `A`/`B` under `jupyter_course/chapters/` |
| `freeze_jupyter_outputs.py` | Executes chapters and stores the tracked teaching outputs |
| `verify_jupyter_textbook.py` | Fresh-kernel textbook verification and disposable reports |
| `build_colab.py` | Generates the student notebook under `notebooks/` |
| `verify_colab.py` | Validates the generated Colab notebook |
| `collect_live_meals.py`, `collect_mokpo_meals.py`, `refresh_sample_data.py` | Explicit live/sample data refresh boundaries |

Generator scripts are handwritten source. Generated notebooks are artifacts of these scripts.

## Textbook Workflow

Run from the project root, in order:

```powershell
.\.venv\Scripts\python.exe scripts\build_jupyter_textbook.py
.\.venv\Scripts\python.exe scripts\freeze_jupyter_outputs.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_jupyter_textbook.py
```

For Colab changes:

```powershell
.\.venv\Scripts\python.exe scripts\build_colab.py
.\.venv\Scripts\python.exe scripts\verify_colab.py
```

## Rules

- Edit the builder when chapter structure or cells change; do not make a builder-owned change only in an `.ipynb`.
- Keep chapter filenames/order aligned with `CHAPTER_FILES` and the tests.
- Frozen notebooks must retain execution counts, visible outputs, and the expected stored-output metadata.
- Verification writes disposable files under `verification/`; never hand-maintain those outputs.
- Keep scripts callable with the project-local interpreter from the project root.
- Data collectors may call external services and may require a credential file path. Run them only for an explicit data-refresh task, never print the credential, and validate output before replacing checked-in samples.
- Builders and verifiers must be deterministic and must not depend on the developer's current working directory outside the project root.

## Side-Effect Boundaries

- Both builders overwrite generated notebooks; `freeze_jupyter_outputs.py` overwrites every tracked chapter after fresh-kernel execution and path redaction.
- Verifiers create disposable reports/executed notebooks under ignored verification locations; those files are evidence, not source.
- The three collector/refresh scripts use the network and replace JSON snapshots. Run them only for an explicit refresh and inspect the diff afterward.
- If a collector temporarily sets `NEIS_API_KEY`, restore the previous environment value and never place the key in payloads, logs, exceptions, or tests.

## Validation

Tests for generator structure, frozen outputs, installation order, and fresh-kernel execution live in `tests/test_jupyter_*.py`, `tests/test_frozen_notebook_outputs.py`, and `tests/test_notebook.py`. Update those contracts when the generated format intentionally changes.
