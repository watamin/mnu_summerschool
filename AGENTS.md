# MNU Summer School Repository Guide

## Overview

Korean-language meal-data teaching project and Gradio service for Mokpo-area schools. It combines a reusable Python package, checked-in NEIS snapshots, classroom preference experiments, and a generated Jupyter textbook. Python 3.11+ is required.

## Structure

| File | Use |
| --- | --- |
| `mokpo_service.py` | Primary local/LAN Mokpo service, authentication, dataset/profile/client wiring |
| `web_app.py` | Local web app backed by the textbook support loader |
| `app.py` | Compact/Colab-compatible recommendation demo |

`create_service_app()` wires validated Mokpo data, the 45-food survey pool, `StudentProfileStore`, and the two-day cafeteria menu into the three-step `create_mokpo_app()` flow.

### Directory Layout

- `src/neis_meal_ai/`: domain, analytics, persistence, integrations, and Gradio composition.
- `data/`: checked-in school, meal, cafeteria, and teaching datasets.
- `runtime_data/`: local SQLite state; never treat as source.
- `scripts/`: live-data collection plus notebook build/freeze/verification tools.
- `tests/`: 24 module- and workflow-oriented pytest files.
- `jupyter_course/`: generated, output-bearing textbook chapters plus handwritten support/docs.
- `notebooks/`: generated student Colab notebook.
- `docs/`: teacher/student material and dated specs/plans.

## Where to Look

| Task | Primary location | Contract tests |
| --- | --- | --- |
| NEIS request and validation | `src/neis_meal_ai/neis.py`, `cleaning.py` | `test_neis.py`, `test_cleaning.py` |
| Compact recommendations | `recommender.py`, `service.py`, `ui.py` | matching module tests |
| Mokpo data and food value | `mokpo_data.py`, `mokpo_analytics.py` | `test_mokpo_data.py`, `test_mokpo_analytics.py` |
| Student profiles and factorization | `student_profiles.py`, `matrix_factorization.py`, `student_profile_ui.py` | matching profile/model tests |
| Core Gradio service | `core_meal_ui.py`, `mokpo_service.py` | `test_core_meal_ui.py`, `test_mokpo_service.py` |
| Legacy extended analytics UI | `mokpo_ui.py` | `test_mokpo_ui.py` |
| Textbook and Colab content | `scripts/build_*.py` | notebook/textbook tests and verifiers |

## Code Map

- Compact path: `neis.py` fetches rows -> `cleaning.py` shapes meals -> `recommender.py` scores -> `service.py` applies live/fallback policy -> `ui.py` presents results.
- Mokpo path: `mokpo_data.py` validates snapshots -> `student_profiles.py` restores 30 ratings -> `core_meal_ui.py` predicts the selected lunch and compares the actual rating.
- Profile path: `StudentProfileStore` persists ratings -> `matrix_factorization.py` predicts blanks -> `student_profile_ui.py` adapts callbacks.
- App wiring: `mokpo_service.py:create_service_app()` owns dataset, profile store, embedder, NIM client, authentication, and launch mode.
- Textbook path: builder writes ordered chapters -> freezer executes and stores outputs -> verifier reruns every chapter in a fresh kernel.

## Commands

Run from this directory with the project virtual environment. If `.venv` does not exist, first follow the Python 3.11+ environment setup in `README.md`.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe mokpo_service.py
.\.venv\Scripts\python.exe mokpo_service.py --lan
```

Notebook work additionally needs `pip install -e ".[dev,jupyter]"`. Follow `scripts/AGENTS.md` for the ordered textbook and Colab build/verify commands.

## Conventions

- Python is 3.11+; package imports are relative inside `src/neis_meal_ai`.
- Top-level entrypoints may add the project and `src/` roots to `sys.path`; keep that boundary out of package modules.
- Use pandas DataFrames at analytics/service boundaries and preserve their documented columns and `.attrs` notices.
- Keep network, embedding, NIM, persistence, and fetcher dependencies injectable so tests remain offline.
- Import Gradio and optional model dependencies lazily where the current module does so.
- Preserve local-only launch defaults. `mokpo_service.py --lan` is the explicit classroom-LAN boundary.
- The core recommender/text path must continue to work without scikit-learn; the direct Python/NumPy path avoids known Windows DLL failures.

## Anti-Patterns

- Never commit or print API keys, passwords, `.env*`, key text files, real student identifiers, runtime databases, or exported rating CSVs.
- Do not treat recommendations, clusters, or similarity scores as medical, nutrition, or allergy guarantees.
- Do not call live NEIS/NVIDIA services or download embedding models in ordinary tests.
- Do not hand-edit builder-owned notebook cells or frozen outputs; change the generator and regenerate.
- Do not import Gradio into pure analytics/model layers or launch an app at import time.

## Unique Styles

- UI explanations use clear introductory Korean; analytics expose formulas and source evidence for guided review.
- External integrations degrade to deterministic TF-IDF/sample-data behavior and report the backend/source used.
- Runtime profile names may exist in the ignored local store but must never enter committed fixtures, notebooks, logs, or exports.

## Change Ownership

- Change behavior in handwritten Python or generator scripts first.
- Regenerate tracked notebooks instead of patching generated cells or frozen outputs manually.
- Do not edit `runtime_data/student_profiles.sqlite3`, caches, or `verification/` outputs as source.
- Live collectors and external clients are opt-in boundaries. Do not run them during ordinary tests.
- Add or update the mirrored `tests/test_<module>.py` contract when changing a protected behavior.

Read the child guide in `src/neis_meal_ai/`, `scripts/`, `tests/`, or `jupyter_course/` before changing files there.

## Notes

- `.gitignore` is the authority for credentials, caches, runtime state, exports, and verification artifacts.
- `freeze_jupyter_outputs.py`, builders, and collectors overwrite files; run them only when those artifacts are in scope.
- No CI workflow is present. Local pytest plus the applicable notebook verifier is the merge gate.
