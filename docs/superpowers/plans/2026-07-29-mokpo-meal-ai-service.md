# Mokpo Meal AI Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 목포시 중·고교 실제 NEIS 급식으로 학생 설문·실제 만족도 피드백, 모둠 분석, 학교 비교, 식단 시뮬레이션을 제공하는 완성형 Gradio 웹 서비스를 만든다.

**Architecture:** 교사용 수집기가 인증키를 일시적으로 사용해 공개 JSON 스냅샷을 만들고, 런타임 웹 앱은 네트워크나 인증키 없이 이 스냅샷만 읽는다. 분석 계층은 TF-IDF와 선택적 Sentence Transformer 임베딩을 같은 인터페이스로 제공하며, Gradio 콜백은 Pandas 표와 설명 문구만 반환한다.

**Tech Stack:** Python 3.11+, Pandas, NumPy, Gradio, requests, pytest, 선택적 PyTorch CUDA와 Sentence Transformers

## Global Constraints

- 실제 이름·학번·반·연락처·질병명을 입력하거나 저장하지 않는다.
- 나이 대신 중1~고3 학년 구간을 사용한다.
- 인증키는 수집 프로세스에서만 사용하며 JSON·로그·오류 메시지에 포함하지 않는다.
- 현재 공개 스냅샷은 목포시 중·고교 31개를 2026-06-01~2026-07-29로 조회한 결과이며, 실제 수록 식단일은 2026-06-24~2026-07-29이다.
- 임베딩 모델은 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`이며 사용할 수 없으면 TF-IDF로 자동 전환한다.
- 만족도·Food MBTI·가성비·잔반·조합 점수는 실제 조사값이 아니라 수업용 예측 또는 대체지표라고 표시한다.
- 학생은 코드를 작성하지 않고 최대 6명의 익명 설문 수집과 피드백 분석에 집중한다.
- 응답은 브라우저 세션에만 두고 사용자가 요청할 때 익명 CSV로 내려받는다.
- 2026-07-30·31 목포대 학생식당 메뉴는 사용자 제공 원문을 별도 JSON으로 보존하고 NEIS라고 표시하지 않는다.
- 콘텐츠 기반, 유저 기반, 혼합 추천을 각각 계산해 실제 1~5점 리뷰와 비교한다.
- 기존 남악고 웹 앱과 Jupyter 교과서 테스트를 깨뜨리지 않는다.

---

### Task 1: 목포 학교·급식 공개 스냅샷 경계

**Files:**
- Create: `src/neis_meal_ai/mokpo_data.py`
- Create: `scripts/collect_mokpo_meals.py`
- Create: `tests/test_mokpo_data.py`
- Create after tests pass: `data/mokpo_schools.json`
- Create after tests pass: `data/mokpo_meals_live.json`

**Interfaces:**
- Consumes: `SchoolInfo`, `_request_json`, `_extract_rows`, `fetch_meals`, `neis_api_key_from_file`, `meals_to_frame`
- Produces: `load_mokpo_dataset(school_path: Path, meal_path: Path) -> MokpoDataset`, `collect_mokpo_snapshot(...) -> tuple[dict, dict]`

- [ ] **Step 1: Write failing catalog and validation tests**

```python
def test_catalog_accepts_only_mokpo_middle_and_high_schools(tmp_path):
    dataset = load_mokpo_dataset(write_catalog_and_meals(tmp_path))
    assert set(dataset.schools["school_kind"]) == {"중학교", "고등학교"}

def test_dataset_rejects_wrong_office_school_or_meal_type(tmp_path):
    with pytest.raises(RuntimeError):
        load_mokpo_dataset(write_invalid_snapshot(tmp_path))
```

- [ ] **Step 2: Run tests and confirm missing module/function failures**

Run: `python -m pytest tests/test_mokpo_data.py -q`

- [ ] **Step 3: Implement immutable dataset loader and allowlists**

`MokpoDataset` contains `schools`, cleaned `meals`, and metadata. Validation checks Q10, catalog membership, middle/high kind, address containing 목포시, lunch-only rows, eight-digit dates, ISO UTC collection time, and row counts.

- [ ] **Step 4: Add failing collection boundary tests**

Test school filtering, per-school meal collection, public field allowlist, partial-school reporting, environment restoration, and safe CLI failure text without the fake key.

- [ ] **Step 5: Implement `collect_mokpo_snapshot` and safe CLI**

The collector queries all Q10 schools once, filters the catalog, fetches meals per school, continues past schools with no meals, and fails when no school produced meals. It writes only after the complete payload passes the same validator used by the app.

- [ ] **Step 6: Run Task 1 tests**

Run: `python -m pytest tests/test_mokpo_data.py tests/test_live_data.py tests/test_neis.py -q`

### Task 2: TF-IDF and optional GPU embedding engine

**Files:**
- Create: `src/neis_meal_ai/text_vectors.py`
- Create: `tests/test_text_vectors.py`
- Create: `requirements-embedding.txt`

**Interfaces:**
- Produces: `VectorResult(matrix: np.ndarray, backend: str, device: str, notice: str)`, `encode_texts(texts, method="tfidf", embedder=None)`, `cosine_scores(query, documents, method="tfidf", embedder=None)`

- [ ] **Step 1: Write failing deterministic TF-IDF tests**

Assert matrix dimensions, unit-vector normalization, self-similarity, and zero handling for empty Korean text.

- [ ] **Step 2: Run tests and confirm missing implementation**

Run: `python -m pytest tests/test_text_vectors.py -q`

- [ ] **Step 3: Implement character n-gram TF-IDF vectors**

Reuse the educational 2~4 character n-gram formula and return normalized dense NumPy arrays suitable for small classroom datasets.

- [ ] **Step 4: Add failing fake-embedder and fallback tests**

The fake embedder returns known vectors and exposes `device="cuda"`. Another fake raises `ImportError`; the expected backend is TF-IDF with a Korean fallback notice.

- [ ] **Step 5: Implement lazy `SentenceTransformerEmbedder`**

Select `cuda` when `torch.cuda.is_available()`, otherwise `cpu`; call `encode(..., normalize_embeddings=True)`. Import torch and sentence-transformers inside the loader so the normal course requirements remain lightweight.

- [ ] **Step 6: Run vector tests**

Run: `python -m pytest tests/test_text_vectors.py tests/test_recommender.py -q`

### Task 3: Multi-school analytics

**Files:**
- Create: `src/neis_meal_ai/mokpo_analytics.py`
- Create: `tests/test_mokpo_analytics.py`

**Interfaces:**
- Consumes: cleaned meal frame and `cosine_scores`
- Produces: `predict_satisfaction`, `best_worst_menus`, `user_based_prediction`, `evaluate_recommenders`, `school_statistics`, `signature_terms`, `recommend_high_schools`, `food_mbti`, `meal_buddies`, `analyze_feedback`, `cluster_feedback`, `pareto_candidates`, `menu_pair_scores`

- [ ] **Step 1: Write failing satisfaction tests**

Use a three-menu frame and assert scores remain in 0~100, liked menus outrank avoided menus, Best and Worst do not overlap, and the backend/device explanation is returned.

- [ ] **Step 2: Implement satisfaction and Best/Worst calculation**

Use 70 similarity points, up to 15 type points, spice penalty, and 20-point avoid penalties exactly as the design specifies.

- [ ] **Step 3: Write failing school statistics and signatures tests**

Assert school counts, max/min menu frequency, average calories, and that a school-unique word receives its signature rank.

- [ ] **Step 4: Implement school statistics, signature TF-IDF, and high-school ranking**

High-school ranking averages each school's top five menu similarities and returns a warning that the result only compares meals.

- [ ] **Step 5: Write failing Food MBTI and buddy tests**

Assert all four axes, deterministic tie handling, malformed line rejection, pairwise cosine ranking, and maximum six anonymous profiles.

- [ ] **Step 6: Implement Food MBTI and meal buddies**

Parse `별명|좋아하는 메뉴|피하는 메뉴` lines without persisting them and return a similarity matrix plus closest pairs.

- [ ] **Step 7: Write failing Pareto and menu-pair tests**

Assert dominated rows are excluded, non-dominated rows retained, same-day pairs counted once, and every output includes the proxy-data warning.

- [ ] **Step 8: Implement Pareto and pair scoring**

Use predicted group satisfaction, dish-count value proxy, calorie deviation, and co-occurrence counts. Never call proxy output actual cost or actual leftovers.

- [ ] **Step 9: Run analytics tests**

Run: `python -m pytest tests/test_mokpo_analytics.py -q`

- [ ] **Step 10: Add failing anonymous feedback tests**

Assert participant-code uniqueness, grade and 1~5 rating validation, maximum six rows, exact CSV round trip, predicted-vs-actual error, small-sample warning, and deterministic two-cluster labels.

- [ ] **Step 11: Implement feedback validation, CSV round trip, analysis, and clustering**

Use the fixed schema from the design. Convert actual rating to a 20~100 scale only for error comparison and preserve the original 1~5 rating in output.

- [ ] **Step 12: Add failing collaborative-filter tests**

Use a literal three-student, two-menu rating table. Assert target-rating exclusion, similarity-weighted prediction, item-mean and global-mean fallback, 0~100 scaling, hybrid weights, and hand-computed MAE.

- [ ] **Step 13: Implement user-based and hybrid evaluation**

Return prediction value, common-rating coverage, fallback label, and per-row errors. Do not read a target row's actual rating while predicting that row.

### Task 4: Service callbacks and four-tab Gradio app

**Files:**
- Create: `src/neis_meal_ai/mokpo_ui.py`
- Create: `mokpo_service.py`
- Create: `tests/test_mokpo_ui.py`

**Interfaces:**
- Consumes: `MokpoDataset` and Task 3 analysis functions
- Produces: `create_mokpo_app(dataset, embedder=None) -> gr.Blocks`, `gr.State` survey callbacks, CSV export/import, pure analysis callback helpers

- [ ] **Step 1: Write failing callback tests**

Call callbacks directly and assert Korean explanations, Best/Worst tables, survey add/duplicate rejection, CSV export/import, content/user/hybrid prediction comparison, predicted-vs-actual summary, school ranking, Food MBTI, buddy matrix, Pareto candidates, and safe error responses.

- [ ] **Step 2: Implement pure callback helpers**

Callbacks accept primitives and return Markdown strings, DataFrames, or Matplotlib figures. They do not read files, keys, environment variables, or network resources.

- [ ] **Step 3: Write failing Gradio configuration test**

Assert the four tab labels `학생 설문·개인 결과`, `모둠 피드백 분석`, `학교 급식 지도`, `AI 식단 실험실`, 31-school source label, privacy warning, six-person small-sample warning, model limitation, proxy warning, and local-only launch options appear in the config.

- [ ] **Step 4: Implement the Gradio layout**

Build the four feedback-centered tabs from the design. Keep results empty until a button is pressed so full menu data and survey rows are not embedded in initial HTML.

- [ ] **Step 5: Implement `mokpo_service.py` entry point**

Load public snapshots once at startup, build the app, and launch on `127.0.0.1` with `share=False`.

- [ ] **Step 6: Run UI tests**

Run: `python -m pytest tests/test_mokpo_ui.py tests/test_ui.py tests/test_web_app.py -q`

### Task 5: Documentation, GPU setup, and actual public data

**Files:**
- Modify: `README.md`
- Modify: `jupyter_course/00_설치_준비.md`
- Create: `docs/교사용_목포권_급식_AI_서비스.md`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `requirements-jupyter.txt`

**Interfaces:**
- Consumes: final commands and verified model/device output
- Produces: teacher/student installation and execution instructions

- [ ] **Step 1: Collect actual public snapshots**

Run `scripts/collect_mokpo_meals.py` with the external key file, date range 20260601~20260729, and output paths under `data/`. Scan the repository for the exact key afterward.

- [ ] **Step 2: Install the optional embedding stack**

Install the current PyTorch CUDA build appropriate for RTX 5080, then `requirements-embedding.txt`. Verify `torch.cuda.is_available()` and GPU name before downloading the model.

- [ ] **Step 3: Run a real model smoke test**

Encode three Korean menu sentences, assert shape `(3, 384)`, finite values, normalized rows, and CUDA device use. Record elapsed time without recording local usernames or cache paths.

- [ ] **Step 4: Write textbook-style instructions**

Explain the survey workflow, 목포대 7월 30·31일 검증 절차, CSV schema, 콘텐츠 기반과 유저 기반의 차이, actual-vs-predicted comparison, vector dimensions, cosine similarity, prediction formula, tabs, limitations, one-command service run, optional GPU installation, and TF-IDF fallback in Korean.

- [ ] **Step 5: Verify repository hygiene**

Run exact-key search, generic secret-pattern search, `git diff --check`, and ensure model caches and key files are ignored.

### Task 6: Full verification and delivery

**Files:**
- Modify: `jupyter_course/verification-report.md`

**Interfaces:**
- Consumes: every prior deliverable
- Produces: reproducible final verification evidence

- [ ] **Step 1: Run the full automated suite**

Run: `python -m pytest -q`

- [ ] **Step 2: Run dependency and syntax checks**

Run: `python -m pip check` and compile all new entry points/modules.

- [ ] **Step 3: Launch the real app and call each button API**

Launch locally with no share URL, use a Gradio client for at least the personal recommendation callback, and directly exercise other pure callbacks.

- [ ] **Step 4: Perform independent code review**

Review privacy, key failure paths, public allowlists, deterministic scores, misleading statistical claims, and UI safety text. Fix findings through red-green tests.

- [ ] **Step 5: Update verification report and commit**

Record actual counts and timings, rerun changed checks, commit, push `codex/json-tutorial-appendix`, and update PR #3 without including either token.
