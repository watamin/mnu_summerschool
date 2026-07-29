# Student Profiles and Matrix Factorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a password-gated classroom web service that persistently stores each student's 30 food ratings and predicts the real missing student-food cells with explainable matrix factorization.

**Architecture:** Keep durable SQLite behavior in a focused `student_profiles.py` repository, numerical training and evaluation in a pure `matrix_factorization.py` module, and Gradio composition in `mokpo_ui.py`. `mokpo_service.py` owns command-line LAN selection and reads the shared password from an external file without exposing it to source control.

**Tech Stack:** Python 3.12, SQLite, NumPy, pandas, matplotlib, Gradio 6.20, pytest

## Global Constraints

- Preserve every existing service feature and its current local-only default.
- Use 45 real foods from `data/mokpo_meals_live.json`; assign 15 common plus 15 rotating foods to each student.
- Accept only integer ratings from 1 through 5 and support partial saves.
- Never hard-code, print, return, commit, or render the real shared password.
- Keep `share=False`; LAN access is enabled only by the explicit `--lan` flag.
- Use observed ratings only for training and held-out observed ratings only for MAE/RMSE evaluation.
- Every production behavior starts with a failing test.

---

### Task 1: Persistent student profile repository

**Files:**
- Create: `src/neis_meal_ai/student_profiles.py`
- Create: `tests/test_student_profiles.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `validate_student_name(name) -> str`, `StudentProfileStore(db_path, food_pool)`, `load_survey(name) -> DataFrame`, `save_ratings(name, rows) -> SaveResult`, `status() -> DataFrame`, `rating_matrix() -> DataFrame`, and `export_ratings(path) -> Path`.
- Persists: `foods`, `profiles`, `assignments`, and `ratings` tables with WAL, foreign keys, and busy timeout.

- [ ] Write failing tests for valid/invalid names, 15+15 deterministic assignment, refresh persistence, partial upsert, rating validation, status, matrix shape, CSV export, and concurrent writers.
- [ ] Run `pytest tests/test_student_profiles.py -q` and confirm import/test failures.
- [ ] Implement the schema, parameterized operations, and immutable assignments with the interfaces above.
- [ ] Add `runtime_data/`, `mokpo_password.txt`, and generated rating exports to `.gitignore`.
- [ ] Run `pytest tests/test_student_profiles.py -q` and confirm all repository tests pass.
- [ ] Commit `feat: persist classroom food ratings`.

### Task 2: Bias-aware matrix factorization and honest evaluation

**Files:**
- Create: `src/neis_meal_ai/matrix_factorization.py`
- Create: `tests/test_matrix_factorization.py`

**Interfaces:**
- Consumes: a pandas rating matrix with student index, food columns, and `NaN` missing values.
- Produces: `fit_matrix_factorization(matrix, rank=3, seed=42) -> MatrixFactorizationModel`, `evaluate_matrix_factorization(matrix, seed=42) -> EvaluationResult`, and `analyze_rating_matrix(matrix) -> MatrixAnalysis` containing completed scores, prediction mask, per-student Best/Worst rows, heatmap data, and two-dimensional user coordinates.

- [ ] Write failing tests for deterministic predictions, 1..5 clipping, observed-cell preservation, missing-cell completion, insufficient data, deterministic holdout, finite MAE/RMSE, and improvement over a synthetic low-rank baseline.
- [ ] Run `pytest tests/test_matrix_factorization.py -q` and confirm import/test failures.
- [ ] Implement global/user/item biases plus rank-3 SGD with regularization and seeded initialization.
- [ ] Implement a holdout splitter that retains training observations and compares against the training global-mean baseline.
- [ ] Implement completed matrices, labels, Best/Worst predictions, and two-dimensional student factors.
- [ ] Run `pytest tests/test_matrix_factorization.py -q` and confirm all numerical tests pass.
- [ ] Commit `feat: predict student food matrix gaps`.

### Task 3: Profile survey and matrix experiment UI

**Files:**
- Create: `src/neis_meal_ai/student_profile_ui.py`
- Create: `tests/test_student_profile_ui.py`
- Modify: `src/neis_meal_ai/mokpo_ui.py`
- Modify: `tests/test_mokpo_ui.py`

**Interfaces:**
- Consumes: `StudentProfileStore`, `gr.Request.username`, and the Task 2 analysis functions.
- Produces: callbacks `load_profile_callback`, `save_profile_callback`, `matrix_dashboard_callback`, `export_ratings_callback`, `matrix_heatmap_figure`, and `student_map_figure`.
- Changes: `create_mokpo_app(..., profile_store=None)` injects persistence and places the two new tabs before existing tabs.

- [ ] Write failing callback tests for authenticated-name loading, resume, validation messages, saved progress, insufficient-class data, completed dashboard tables, CSV output, and both figures.
- [ ] Run the focused UI tests and confirm failures.
- [ ] Implement pure callback formatting and plots in `student_profile_ui.py`.
- [ ] Wire `내 프로필·30개 평가` and `학생 행렬분해 실험` into `mokpo_ui.py` without changing existing callback outputs.
- [ ] Run `pytest tests/test_student_profile_ui.py tests/test_mokpo_ui.py -q` and confirm both suites pass.
- [ ] Commit `feat: add student matrix experiment screens`.

### Task 4: Password gate and explicit classroom LAN launch

**Files:**
- Modify: `mokpo_service.py`
- Create: `tests/test_mokpo_service.py`
- Modify: `tests/test_mokpo_ui.py`
- Create outside repository: `../mokpo_password.txt`

**Interfaces:**
- Produces: `load_shared_password(path) -> str`, `build_authenticator(password) -> Callable[[str, str], bool]`, `parse_args(argv)`, and `launch_options(lan, password) -> dict`.
- Launches: local mode on `127.0.0.1` without authentication; `--lan` mode on `0.0.0.0` with authentication, `share=False`, and a Korean login message.

- [ ] Write failing tests for missing/blank password files, constant-time password comparison behavior, invalid usernames, local defaults, LAN options, and injected profile store.
- [ ] Run `pytest tests/test_mokpo_service.py tests/test_mokpo_ui.py -q` and confirm failures.
- [ ] Implement CLI parsing, external password loading, auth callable, DB creation, and launch options.
- [ ] Create the external password file requested by the teacher and verify its non-empty content without printing it.
- [ ] Run the focused service tests and confirm they pass.
- [ ] Commit `feat: gate classroom LAN service`.

### Task 5: Teacher operations documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/교사용_목포_급식_AI_서비스.md`
- Create: `docs/교사용_학생평점_행렬분해_실험.md`
- Test: `tests/test_mokpo_service.py`

**Interfaces:**
- Documents: password file preparation, `--lan` command, `ipconfig`, private-network firewall access, student workflow, save/resume, evaluation interpretation, backup/export, and common-password limitations.

- [ ] Add a failing documentation assertion for the exact safe launch command and required headings without asserting the real password.
- [ ] Run the documentation test and confirm failure.
- [ ] Write the Korean teacher guide and link it from README and the existing service guide.
- [ ] Run the focused documentation test and confirm it passes.
- [ ] Commit `docs: explain classroom rating experiment`.

### Task 6: End-to-end verification and pull request update

**Files:**
- Verify only unless a discovered defect requires a focused test and fix.

**Interfaces:**
- Verifies: six named test profiles, 180 saved observations, a 6x45 matrix, 90 predicted gaps, auth login, real HTTP serving, secret exclusion, package health, and all regressions.

- [ ] Populate a temporary database with six synthetic student response patterns and assert 180 observations and 90 missing cells.
- [ ] Run the real matrix analysis and assert finite metrics, bounded predictions, Best/Worst outputs, and two plot objects.
- [ ] Launch on an unused local port with authentication, verify unauthenticated denial and authenticated HTTP access, then stop the process.
- [ ] Search tracked files for the exact external secret and confirm zero matches without printing the value.
- [ ] Run `python -m pytest -q` and `python -m pip check` with the project virtual environment.
- [ ] Request independent code review, fix any valid issues test-first, and rerun the complete verification.
- [ ] Push the branch and update the existing pull request.
