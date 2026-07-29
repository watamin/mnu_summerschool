# Rating Matrix, Food Map, and NVIDIA NIM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explainable school-level TF-IDF food value rankings, a 30-food regularized pseudoinverse recommender, a two-dimensional food map, and a grounded Korean NVIDIA NIM data assistant to the completed Mokpo meal service.

**Architecture:** Keep numerical work in pure functions under `mokpo_analytics.py`, isolate external NVIDIA HTTP behavior in `nim_chat.py`, and use `mokpo_ui.py` only to validate Gradio values and compose outputs. The new recommender always uses normalized Korean character n-gram TF-IDF and a regularized Moore-Penrose pseudoinverse; NIM receives only compact computed evidence.

**Tech Stack:** Python 3.12, NumPy, pandas, matplotlib, requests, Gradio 6, pytest

## Global Constraints

- Preserve the existing content-based, user-based, hybrid, MNU review, Food MBTI, meal-buddy, and Pareto features.
- Use only actual dishes found in `data/mokpo_meals_live.json` for the 30-food survey.
- Define “data value” as school-document TF-IDF, not nutrition, health, price, or personal preference.
- Use `TF = count / school dish count`, `IDF = ln((1+school count)/(1+document frequency))+1`, and `TF-IDF = TF × IDF`.
- Use `lambda = 0.1` and `w = X.T @ pinv(X @ X.T + lambda * I) @ (y - 3)` for the rating model.
- Never print, return, commit, or place the NVIDIA API key in a browser component.
- The NIM default is `https://integrate.api.nvidia.com/v1/chat/completions` with model `meta/llama-3.1-8b-instruct`.
- The long privacy paragraph named in the approved design must be removed from the app header.
- Every production behavior is introduced by a failing test and verified green before the next task.

---

### Task 1: Make Korean source fingerprints newline-independent

**Files:**
- Modify: `scripts/build_jupyter_textbook.py`
- Regenerate: `jupyter_course/chapters/03_TFIDF_글자를_숫자로.ipynb`
- Test: `tests/test_jupyter_textbook.py`
- Test: `tests/test_jupyter_verifier.py`

**Interfaces:**
- Consumes: UTF-8 document bytes checked out with LF, CRLF, or CR line endings.
- Produces: canonical LF text and the existing manifest SHA-256 result.

- [ ] **Step 1: Use the existing four failing tests as the red case**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_jupyter_textbook.py::test_chapter_executes_independently_with_expected_result tests/test_jupyter_verifier.py::test_verify_textbook_executes_all_chapters_in_fresh_kernels -q
```

Expected: FAIL with `원문 확인 실패: 01_신경망_소개.md` on a CRLF checkout.

- [ ] **Step 2: Canonicalize line endings before hashing and analysis**

Change the generated code cell to:

```python
document_text = document_path.read_text(encoding="utf-8")
document_text = document_text.replace("\r\n", "\n").replace("\r", "\n")
document_bytes = document_text.encode("utf-8")
actual_sha256 = sha256(document_bytes).hexdigest()
```

Update the nearby textbook explanation to say the UTF-8 text is normalized to LF before `sha256` is computed.

- [ ] **Step 3: Regenerate the checked-in textbook**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\build_jupyter_textbook.py
```

- [ ] **Step 4: Verify the original failure is green**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_jupyter_textbook.py tests/test_jupyter_verifier.py -q
```

Expected: all tests in both files pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/build_jupyter_textbook.py jupyter_course/chapters
git commit -m "fix: normalize textbook source fingerprints"
```

---

### Task 2: Add detailed school food TF-IDF value calculations

**Files:**
- Modify: `src/neis_meal_ai/mokpo_analytics.py`
- Modify: `tests/test_mokpo_analytics.py`

**Interfaces:**
- Produces: `school_food_values(frame: pd.DataFrame, school_name: str | None = None, top_n: int | None = None) -> pd.DataFrame`.
- Produces columns: `학교`, `학교급`, `순위`, `음식`, `등장 횟수`, `학교 전체 음식 수`, `TF`, `등장 학교 수`, `전체 학교 수`, `IDF`, `데이터 가치 점수`.
- Produces: `school_food_value_explanation(values: pd.DataFrame) -> str`.
- Produces: `school_food_frequencies(frame: pd.DataFrame, school_name: str, top_n: int = 15) -> pd.DataFrame`.

- [ ] **Step 1: Write failing hand-calculated tests**

Add tests using `_meal_frame()` where `목포가람고등학교` has five dish occurrences and `투움바스파게티` appears at one of three schools:

```python
def test_school_food_values_show_each_tfidf_factor() -> None:
    values = school_food_values(_meal_frame(), "목포가람고등학교")
    row = values.loc[values["음식"] == "투움바스파게티"].iloc[0]
    assert row["등장 횟수"] == 1
    assert row["학교 전체 음식 수"] == 5
    assert row["TF"] == pytest.approx(0.2, abs=1e-4)
    assert row["등장 학교 수"] == 1
    assert row["전체 학교 수"] == 3
    assert row["IDF"] == pytest.approx(1.6931, abs=1e-4)
    assert row["데이터 가치 점수"] == pytest.approx(0.3386, abs=1e-4)

def test_school_food_value_explanation_substitutes_real_numbers() -> None:
    values = school_food_values(_meal_frame(), "목포가람고등학교")
    message = school_food_value_explanation(values)
    assert "÷ 5" in message
    assert "ln((1 + 3)" in message
    assert "TF-IDF 데이터 가치 점수" in message
```

- [ ] **Step 2: Run the tests and verify missing imports fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_analytics.py -q
```

Expected: collection fails because the three functions do not exist.

- [ ] **Step 3: Implement the pure calculations**

Build a `Counter` for every school, a document-frequency `Counter`, and records using:

```python
tf = count / total_items
idf = math.log((1 + school_count) / (1 + document_frequency[food])) + 1.0
value = tf * idf
```

Round displayed values to four decimals only after calculations. Rank within each school using data value descending, count descending, and food name ascending. The explanation must use values from the first ranked row.

- [ ] **Step 4: Run focused and analytics tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_analytics.py -q
```

Expected: all analytics tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/neis_meal_ai/mokpo_analytics.py tests/test_mokpo_analytics.py
git commit -m "feat: explain school food tfidf value"
```

---

### Task 3: Add the 30-food sample and regularized pseudoinverse recommender

**Files:**
- Modify: `src/neis_meal_ai/mokpo_analytics.py`
- Modify: `tests/test_mokpo_analytics.py`

**Interfaces:**
- Produces: `sample_school_foods(frame, school_name, *, sample_size=30, seed=0) -> pd.DataFrame` with `음식`, `평점`.
- Produces: `inverse_matrix_recommendations(frame, school_name, ratings, *, regularization=0.1) -> pd.DataFrame`.
- Recommendation columns: `음식`, `예상 평점`, `가장 영향 준 평가 음식`, `그 음식 평점`, `유사도`, `등장 횟수`.
- Recommendation attrs: `rated_count`, `feature_count`, `gram_shape`, `regularization`, `formula`.

- [ ] **Step 1: Write failing sampling tests**

```python
def test_sample_school_foods_is_reproducible_and_uses_real_unique_foods() -> None:
    first = sample_school_foods(_meal_frame(), "목포가람고등학교", sample_size=3, seed=7)
    second = sample_school_foods(_meal_frame(), "목포가람고등학교", sample_size=3, seed=7)
    assert first.equals(second)
    assert len(first) == 3
    assert first["음식"].is_unique
    assert set(first["음식"]) <= {
        "투움바스파게티", "오이피클", "요구르트", "매운 닭갈비덮밥", "배추김치"
    }
    assert set(first["평점"]) == {3}
```

- [ ] **Step 2: Write failing inverse prediction tests**

Use a fixture with distinct `치즈파스타`, `토마토파스타`, `고등어구이`, and `갈치구이` dishes. Rate `치즈파스타=5` and `고등어구이=1`, then assert `토마토파스타` ranks above `갈치구이`. Also assert all-neutral ratings produce only 3.0 predictions and invalid ratings raise `ValueError`.

- [ ] **Step 3: Run and verify the new tests fail for missing functions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_analytics.py -q
```

- [ ] **Step 4: Implement sampling and matrix prediction**

Use `np.random.default_rng(seed).choice` without replacement. Vectorize all unique school foods once with `encode_texts(foods, method="tfidf")`, select rated rows, and calculate:

```python
centered = rating_values - 3.0
gram = rated_vectors @ rated_vectors.T + regularization * np.eye(len(rated_vectors))
coefficients = np.linalg.pinv(gram) @ centered
weights = rated_vectors.T @ coefficients
predictions = np.clip(3.0 + all_vectors @ weights, 1.0, 5.0)
```

For each un-rated food, calculate per-rated-food influence as `(all_vector @ rated_vectors.T) * coefficients`; select the largest absolute contribution and expose its cosine similarity and original rating.

- [ ] **Step 5: Verify analytics tests pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_analytics.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add src/neis_meal_ai/mokpo_analytics.py tests/test_mokpo_analytics.py
git commit -m "feat: add inverse matrix food recommender"
```

---

### Task 4: Add deterministic two-dimensional food coordinates and plot

**Files:**
- Modify: `src/neis_meal_ai/mokpo_analytics.py`
- Modify: `src/neis_meal_ai/mokpo_ui.py`
- Modify: `tests/test_mokpo_analytics.py`
- Modify: `tests/test_mokpo_ui.py`

**Interfaces:**
- Produces: `food_map_coordinates(frame, school_name, ratings, recommendations, *, max_items=50) -> pd.DataFrame`.
- Coordinate columns: `번호`, `음식`, `X`, `Y`, `구분`, `평점`, `등장 횟수`.
- Produces: `food_map_figure(coordinates: pd.DataFrame) -> matplotlib.figure.Figure`.
- Produces callbacks: `sample_foods_callback(...)` and `matrix_recommendation_callback(...)`.

- [ ] **Step 1: Write failing coordinate behavior tests**

Assert that coordinates contain finite X/Y values, no more than 50 unique foods, the same input produces equal coordinates, directly rated foods have `구분 == "직접 평가"`, and recommended foods retain their predicted ratings.

- [ ] **Step 2: Write failing callback tests**

Call `sample_foods_callback` with a fixed seed and assert the table size and message. Pass edited ratings into `matrix_recommendation_callback` and assert that it returns the formula explanation, disjoint Best/Worst tables, a matplotlib `Figure`, and a coordinate table.

- [ ] **Step 3: Run focused tests and observe missing behavior failures**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_analytics.py tests/test_mokpo_ui.py -q
```

- [ ] **Step 4: Implement deterministic SVD coordinates**

Create the selected food list from rated foods, recommendation leaders, and frequency leaders. Center the TF-IDF matrix and calculate:

```python
u, singular_values, vt = np.linalg.svd(centered_matrix, full_matrices=False)
coordinates = u[:, :2] * singular_values[:2]
```

Pad a missing second axis with zero. Stabilize each axis sign by finding the loading with the largest absolute value and flipping the axis when that loading is negative. Reject non-finite results.

- [ ] **Step 5: Implement the matplotlib plot and callbacks**

Use ASCII plot labels to avoid missing Korean font warnings, number each point, color from 1 to 5, scale by occurrence count, use a black edge for direct ratings, and a star marker for the top five predictions. Keep Korean food names in the adjacent dataframe.

- [ ] **Step 6: Verify focused tests pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_analytics.py tests/test_mokpo_ui.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add src/neis_meal_ai/mokpo_analytics.py src/neis_meal_ai/mokpo_ui.py tests/test_mokpo_analytics.py tests/test_mokpo_ui.py
git commit -m "feat: visualize food similarity in two dimensions"
```

---

### Task 5: Add the grounded NVIDIA NIM client and chatbot callback

**Files:**
- Create: `src/neis_meal_ai/nim_chat.py`
- Create: `tests/test_nim_chat.py`
- Modify: `src/neis_meal_ai/mokpo_ui.py`
- Modify: `tests/test_mokpo_ui.py`

**Interfaces:**
- Produces: `load_nvidia_api_key(*, key_path: str | Path | None = None, environ: Mapping[str, str] | None = None) -> str`.
- Produces: `build_grounded_messages(question: str, context: str, history: Sequence[Mapping[str, str]]) -> list[dict[str, str]]`.
- Produces class `NvidiaNimClient(key_path=None, base_url=..., model=..., timeout=30.0, session=None)` with `ask(question, context, history=()) -> str`.
- Produces: `meal_chat_callback(dataset, *, school_name, question, history, nim_client, matrix_recommendations=None) -> tuple[list[dict[str, str]], str]`.

- [ ] **Step 1: Write failing key parsing tests**

Use `tmp_path` files containing raw `nvapi-example` and `NVIDIA_API_KEY=nvapi-example`; assert both return `nvapi-example`. Assert missing, empty, or non-`nvapi-` values raise a Korean `NimChatError` without echoing the bad value.

- [ ] **Step 2: Write failing request/response boundary tests**

Use a recording fake session whose `post` returns this complete response shape:

```python
{
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "meta/llama-3.1-8b-instruct",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "근거를 보면 돈까스입니다.", "refusal": None},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}
```

Assert the returned answer, endpoint, Bearer header, model, `temperature=0.2`, `top_p=0.7`, `max_tokens=700`, and that the system prompt includes the supplied TF-IDF context. The fake replaces only the external HTTP operation; key parsing and payload construction remain real.

- [ ] **Step 3: Run tests and verify missing module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nim_chat.py -q
```

- [ ] **Step 4: Implement `nim_chat.py`**

Parse keys without logging. Validate questions at 1–500 characters. Keep only the last 12 history messages. Send `requests.Session.post(..., timeout=30)` with JSON:

```python
{
    "model": self.model,
    "messages": messages,
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 700,
    "stream": False,
}
```

Catch `requests.RequestException`, non-2xx status, missing choices, and empty content; convert them to concise Korean `NimChatError` messages without including response bodies or keys.

- [ ] **Step 5: Write and verify callback tests**

Use a fake client returning a fixed Korean explanation. Assert the callback appends one user and one assistant message, clears the question box, and that the context passed to the fake contains the selected school, TF, IDF, and data value. Assert empty questions do not call the client.

- [ ] **Step 6: Implement the grounded callback**

Build context from `school_statistics`, `school_food_frequencies`, `school_food_values`, and `school_food_value_explanation`. Include up to five matrix Best/Worst rows only when supplied and belonging to the selected school.

- [ ] **Step 7: Verify NIM and UI tests pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nim_chat.py tests/test_mokpo_ui.py -q
```

- [ ] **Step 8: Commit**

```powershell
git add src/neis_meal_ai/nim_chat.py src/neis_meal_ai/mokpo_ui.py tests/test_nim_chat.py tests/test_mokpo_ui.py
git commit -m "feat: add grounded NVIDIA NIM meal assistant"
```

---

### Task 6: Integrate the new Gradio tabs and service key source

**Files:**
- Modify: `mokpo_service.py`
- Modify: `src/neis_meal_ai/mokpo_ui.py`
- Modify: `tests/test_mokpo_ui.py`

**Interfaces:**
- `create_mokpo_app(..., nim_client: NvidiaNimClient | None = None)` accepts a lazy client.
- `create_service_app()` supplies `PROJECT_ROOT.parent / "nvidia_nim.txt"` without reading it during import.

- [ ] **Step 1: Write failing app configuration tests**

Build the app with a fake NIM client and inspect `demo.get_config_file()` labels. Assert the configuration exposes `30개 음식 역행렬 추천`, `학교별 가치 음식`, `NVIDIA NIM 데이터 해설`, `음식 30개 뽑기`, `역행렬 추천 계산`, `TF-IDF 데이터 가치 점수`, and `대화 지우기`. Assert the removed long privacy paragraph is absent from component values.

- [ ] **Step 2: Run the UI tests and verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_ui.py -q
```

- [ ] **Step 3: Build the three UI areas**

Add the editable 30-row dataframe and matrix output state; replace the school map layout with core frequency, detailed TF-IDF factors, real substitution formula, global leaderboard, and the existing high-school ranking; add the NIM chat tab with school dropdown, `gr.Chatbot(type="messages")`, question box, submit button, and clear button.

- [ ] **Step 4: Wire service startup to the external key file**

In `mokpo_service.py`, define:

```python
NIM_KEY_PATH = PROJECT_ROOT.parent / "nvidia_nim.txt"
```

Pass `NvidiaNimClient(key_path=NIM_KEY_PATH)` into `create_mokpo_app`. The constructor must not read or validate the file until `ask()`.

- [ ] **Step 5: Verify UI tests pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mokpo_ui.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add mokpo_service.py src/neis_meal_ai/mokpo_ui.py tests/test_mokpo_ui.py
git commit -m "feat: integrate meal survey insights and chat tabs"
```

---

### Task 7: Update instructions and run end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `docs/교사용_목포_급식_AI_서비스.md`
- Modify: `.gitignore`

**Interfaces:**
- Documents explain the exact run command, 30-food workflow, matrix formula, TF-IDF value formula, two-dimensional map, NIM key lookup, and chatbot limits.

- [ ] **Step 1: Update instructions and secret ignore rules**

Add `nvidia_nim.txt` and `.env` key variants to `.gitignore`. Document that the default local file belongs one directory above the repository and may contain either the raw key or `NVIDIA_API_KEY=...`. Never include an example containing a real key prefix beyond the literal placeholder `nvapi-your-key`.

- [ ] **Step 2: Run formatting and secret scans**

```powershell
git diff --check
rg -n --hidden --glob '!.git/**' "nvapi-[A-Za-z0-9_-]{8,}" .
```

Expected: the secret scan returns no matches.

- [ ] **Step 3: Run the complete automated suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Expected: all tests pass and pip reports no broken requirements.

- [ ] **Step 4: Run an actual local Gradio HTTP and callback smoke test**

Launch on an unused local port without opening a browser, request `/`, call the sample/value/matrix callbacks with real Mokpo data, and close the server. Expected: HTTP 200, 30 unique foods where available, finite predictions, and finite map coordinates.

- [ ] **Step 5: Run one real NVIDIA NIM smoke request**

Read `C:\Users\user\Documents\New project\nvidia_nim.txt` through `NvidiaNimClient`, ask a short Korean question using one selected school's computed context, and print only `NIM_OK`, response character count, and model name. Never print the key, headers, prompt, or full response.

- [ ] **Step 6: Commit documentation**

```powershell
git add .gitignore README.md docs/교사용_목포_급식_AI_서비스.md
git commit -m "docs: explain food matrix and NIM analysis workflow"
```

- [ ] **Step 7: Review final diff and push the existing PR branch**

```powershell
git status --short
git log --oneline -8
git push origin codex/json-tutorial-appendix
```
