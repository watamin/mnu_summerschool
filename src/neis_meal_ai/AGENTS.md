# Package Guide: `neis_meal_ai`

## Module Boundaries

| Area | Modules | Responsibility |
| --- | --- | --- |
| NEIS core | `neis.py`, `cleaning.py`, `service.py`, `recommender.py`, `ui.py` | Fetch, normalize, score, present the compact recommender |
| Mokpo data | `mokpo_data.py` | Validate and load school/meal/cafeteria payloads |
| Mokpo analytics | `mokpo_analytics.py` | Food value, maps, collaborative prediction, feedback, evaluation |
| Profiles | `student_profiles.py`, `matrix_factorization.py`, `student_profile_ui.py` | SQLite ratings, factorization/evaluation, profile UI callbacks |
| Integrations | `text_vectors.py`, `nim_chat.py` | Injectable embeddings and grounded NVIDIA NIM chat |
| Composition | `mokpo_ui.py` | Gradio tabs, callbacks, and `create_mokpo_app()` |

## Dependency Direction

- `neis.py` and `mokpo_data.py` are external-data boundaries; validate before returning domain data.
- `cleaning.py`, `recommender.py`, `text_vectors.py`, and `matrix_factorization.py` must not depend on Gradio.
- `service.py` shapes compact UI results but delegates scoring to `recommender.py`.
- `mokpo_ui.py` may compose analytics, profile, text, and NIM modules; those modules must not import it.
- Keep NIM/model initialization lazy. Importing the package or building offline tests must not trigger downloads or network calls.

## Coding Conventions

- Use Python 3.11 type syntax, dataclasses for domain values, and `Protocol` for replaceable clients/embedders.
- Inject HTTP functions, fetchers, embedders, NIM clients, profile stores, and file paths at boundaries.
- Raise the existing typed/domain error (`NeisApiError`, `NimChatError`) or `ValueError`; UI modules translate expected errors into Korean guidance.
- Preserve DataFrame schemas, sort order, deterministic seeds, and metadata stored in `.attrs`.
- Keep package imports relative. Do not add entrypoint-style `sys.path` mutation here.
- Do not reintroduce scikit-learn into the core text/recommendation path; use the established NumPy/direct TF-IDF implementation.

## Protected Invariants

- `mokpo_data.py` validates office codes, dates, counts, lunch rows, and metadata before constructing `MokpoDataset`.
- `student_profiles.py` owns SQLite transactions: exactly 45 unique pool foods, 30 assignments per profile, and integer ratings 1-5.
- Blank values from a stale browser view must not erase newer saved ratings.
- Preserve tested Korean DataFrame schemas, deterministic order/seeds, and `.attrs` such as backend, device, notice, and school name.
- Feedback exports retain `FEEDBACK_COLUMNS` and spreadsheet formula-injection protection.
- `mokpo_ui.py` keeps teacher-only export/configuration separate from student views and translates expected errors into Korean guidance.

## Hotspots

`mokpo_ui.py` and `mokpo_analytics.py` are the largest integration files. Before editing, identify the existing callback/function and its mirrored tests; avoid unrelated cleanup. `student_profiles.py` owns SQLite behavior. `matrix_factorization.py` owns fitting/evaluation. `text_vectors.py` owns deterministic TF-IDF fallback and the optional lazy SentenceTransformer adapter.

## Tests

Every module has a matching `tests/test_<module>.py`. Run the narrow test first, then the suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_mokpo_analytics.py
.\.venv\Scripts\python.exe -m pytest -q
```

Use injected fakes and temporary paths. Tests here must not contact NEIS/NVIDIA, download embedding models, or open the real profile database.
