"""중학생용 NEIS 급식 AI Jupyter 교과서 9개 장과 부록을 생성한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "jupyter_course" / "chapters"
CHAPTER_FILES = (
    "00_시작하기.ipynb",
    "A_JSON_기초_튜토리얼.ipynb",
    "01_NEIS_API_요청하기.ipynb",
    "02_급식데이터_정리와_그래프.ipynb",
    "03_TFIDF_글자를_숫자로.ipynb",
    "04_유사도와_식단군집.ipynb",
    "05_개인추천_점수설계.ipynb",
    "06_Jupyter_추천화면.ipynb",
    "07_테스트와_모델카드.ipynb",
    "08_발표와_체험.ipynb",
)


def _markdown(
    source: str,
    *,
    role: str | None = None,
    activity_number: int | None = None,
) -> dict:
    metadata: dict[str, object] = {}
    if role is not None:
        metadata["textbook_role"] = role
    if activity_number is not None:
        metadata["activity_number"] = activity_number
    return {
        "cell_type": "markdown",
        "metadata": metadata,
        "source": source.strip() + "\n",
    }


def _code(source: str, *, tags: list[str] | None = None) -> dict:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.strip() + "\n",
    }


def _lesson(
    *,
    number: str,
    title: str,
    question: str,
    session: str,
    minutes: int,
    objectives: list[str],
    connection: str,
    keywords: list[tuple[str, str]],
    concept: str,
    hand_example: str,
    prediction: str,
    code_sections: list[tuple[str, str, str, str]],
    exercise_text: str,
    exercise_code: str,
    check_questions: list[str],
    check_answer: str,
    summary: list[str],
    next_text: str,
) -> dict:
    chapter_label = f"{number}장" if number.isdigit() else f"부록 {number}"
    keyword_table = "\n".join(
        ["| 용어 | 뜻 |", "|---|---|"]
        + [f"| **{word}** | {meaning} |" for word, meaning in keywords]
    )
    cells: list[dict] = [
        _markdown(
            f"""
# {chapter_label}. {title}

| 오늘의 질문 | 예상 시간 |
|---|---:|
| {question} | {session} · 약 {minutes}분 |
"""
            ,
            role="chapter-opener",
        ),
        _markdown(
            "## 이 장에서 배울 내용\n\n"
            + "\n".join(f"- {objective}" for objective in objectives),
            role="objectives",
        ),
        _markdown(f"## 생각 열기\n\n{connection}", role="opener"),
        _markdown(f"## 핵심 용어\n\n{keyword_table}", role="terms"),
        _markdown(f"## 개념 익히기\n\n{concept}", role="concept"),
        _markdown(f"## 활동 전 생각\n\n{hand_example}", role="pre-activity"),
        _markdown(
            f"""
## 예상하기

{prediction}
""",
            role="prediction",
        ),
    ]
    for activity_number, section in enumerate(code_sections, 1):
        heading, code, reading, code_guide = section
        cells.extend(
            [
                _markdown(
                    f"## 활동 {activity_number}. {heading}",
                    role="activity",
                    activity_number=activity_number,
                ),
                _markdown(
                    f"### 코드 살펴보기\n\n{code_guide}",
                    role="code-guide",
                    activity_number=activity_number,
                ),
                _code(code),
                _markdown(
                    f"### 결과 해석하기\n\n{reading}",
                    role="activity-result",
                    activity_number=activity_number,
                ),
            ]
        )
    cells.extend(
        [
            _markdown(
                f"""
## 탐구 활동

{exercise_text}

먼저 기본값으로 한 번 실행하세요. 그다음 표시된 값 하나만 바꾸고, 달라진 결과를 아래에 적습니다.
""",
                role="inquiry",
            ),
            _code(exercise_code, tags=["student-exercise"]),
            _markdown(
                "### 내가 본 변화\n\n"
                "- 내가 바꾼 값:  \n"
                "- 화면에서 달라진 것:  \n"
                "- 내 설명:  ",
                role="observation",
            ),
            _markdown(
                "## 확인 문제\n\n"
                + "\n".join(
                    f"{index}. {question}"
                    for index, question in enumerate(check_questions, 1)
                ),
                role="check",
            ),
            _markdown(
                f"""
## 정답과 해설

{check_answer}
""",
                role="answer",
            ),
            _markdown(
                "## 핵심 정리\n\n"
                + "\n".join(f"- {item}" for item in summary)
                + f"\n\n### 다음 장\n\n{next_text}",
                role="summary",
            ),
            _code(
                """
import json
print("__CHAPTER_RESULT__=" + json.dumps(chapter_result, ensure_ascii=False))
""",
                tags=["chapter-result"],
            ),
        ]
    )
    for index, cell in enumerate(cells):
        cell["id"] = f"chapter-{number}-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "jupyter_course": {
                "chapter": number,
                "session": session,
                "estimated_minutes": minutes,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CODE = """
import sys
from pathlib import Path

current_folder = Path.cwd().resolve()
for candidate in (current_folder, *current_folder.parents):
    if (candidate / "jupyter_course" / "notebook_support.py").is_file():
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError(
        "프로젝트 폴더를 찾지 못했습니다. 프로젝트 최상위 폴더에서 "
        r".\\.venv\\Scripts\\python.exe -m notebook 명령으로 다시 시작하세요."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jupyter_course.notebook_support import course_setup

setup = course_setup(PROJECT_ROOT)
PROJECT_ROOT = setup["root"]
raw_rows = setup["rows"]
meal_df = setup["frame"]
data_source = setup["source"]
print("프로젝트 폴더:", PROJECT_ROOT)
print("데이터 출처:", data_source)
print("급식 행 수:", len(raw_rows))
"""


def _chapter_00() -> dict:
    return _lesson(
        number="00",
        title="설치 확인과 Jupyter 시작하기",
        question="가상환경과 Jupyter가 제대로 준비되었는지 어떻게 확인할까?",
        session="1회차 전반",
        minutes=70,
        objectives=[
            "가상환경이 필요한 까닭을 설명할 수 있다.",
            "requirements의 패키지를 설치하고 Jupyter를 실행하는 명령 순서를 말할 수 있다.",
            "Markdown 셀과 Code 셀의 차이를 말할 수 있다.",
            "현재 Python·Notebook·핵심 패키지 버전을 확인할 수 있다.",
        ],
        connection="`00_설치_준비.md`의 설치를 마쳤다면 이 노트북이 프로젝트의 `.venv`에서 실행되고 있는지 확인합니다. Python과 패키지 버전이 맞아야 뒤의 여덟 장도 같은 결과를 낼 수 있습니다.",
        keywords=[
            ("가상환경", "이 프로젝트에 필요한 Python 패키지만 따로 담는 폴더"),
            ("requirements", "설치할 패키지와 허용 버전을 한 줄씩 적은 목록"),
            ("pip", "Python 패키지를 내려받고 설치하는 도구"),
            ("셀", "설명이나 코드를 담는 Jupyter의 한 칸"),
            ("커널", "Python 코드를 실제로 실행하고 변수를 기억하는 엔진"),
        ],
        concept="""
가상환경은 프로젝트마다 따로 쓰는 공구함입니다. `.venv` 안의 Python을 사용하면 다른 수업의 패키지와 섞이지 않습니다. `requirements-jupyter.txt`에는 필요한 공구가 한 줄에 하나씩 적혀 있습니다.

이 명령들은 노트북 Code 셀이 아니라 **프로젝트 폴더의 PowerShell**에서 한 단계씩 실행합니다.

### 설치 1단계 — Python 확인

`py -3 --version`으로 Python 3.11 이상인지 먼저 확인합니다. 버전이 낮으면 다음 단계로 가지 않습니다.

### 설치 2단계 — 가상환경 만들기

`py -3 -m venv .venv`는 현재 프로젝트 안에 `.venv` 공구함을 만듭니다. 이 명령은 프로젝트마다 처음 한 번만 실행합니다.

### 설치 3단계 — pip 갱신

`.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip`로 가상환경 안의 설치 도구만 갱신합니다.

### 설치 4단계 — requirements 설치

`.\\.venv\\Scripts\\python.exe -m pip install -r requirements-jupyter.txt`는 목록을 위에서 아래로 읽어 필요한 패키지를 설치합니다.

### 설치 5단계 — Jupyter Notebook 실행

`.\\.venv\\Scripts\\python.exe -m notebook`을 실행하고 PowerShell 창을 열어 둡니다. 브라우저가 열리면 이 0장으로 들어옵니다.

`notebook`은 Jupyter 화면을 열고, `ipykernel`은 Code 셀을 실행합니다. `ipywidgets`는 06장의 입력 상자와 버튼을 만들고, `gradio`는 완성된 추천기를 웹 화면으로 엽니다. `pandas`와 `numpy`는 급식 표와 숫자를 다루며, `matplotlib`은 그래프를 그리고 `requests`는 NEIS에 데이터를 요청합니다. 이것이 **각 패키지가 하는 일**입니다.

Jupyter Notebook은 **설명 페이지와 실험실이 한 화면에 붙어 있는 전자 교과서**입니다. Markdown 셀은 교과서의 설명이고, Code 셀은 직접 눌러 보는 실험 장치입니다.

커널은 책상 위의 작업 기억과 비슷합니다. 위 셀을 실행하면 변수를 기억하지만, 커널을 다시 시작하면 기억이 지워집니다. 그래서 좋은 노트북은 위에서 아래로 다시 실행해도 작동해야 합니다.

우리가 만들 흐름은 **익명 취향 입력 → NEIS 급식 데이터 → 글자 비교 AI → 추천 이유가 있는 표**입니다.
""",
        hand_example="""
종이에 큰 학교 공구함과 작은 프로젝트 공구함을 그립니다. Python, Jupyter, Pandas, 다른 수업 패키지를 어느 공구함에 넣을지 표시해 보세요. 이번 수업 패키지는 `.venv`라고 쓴 작은 공구함에 모입니다.
""",
        prediction="- Python은 3.11 이상으로 표시된다.\n- Notebook과 핵심 패키지 8개의 버전이 표시된다.\n- 예비 급식 행 수가 5로 표시된다.",
        code_sections=[
            (
                "Python·Jupyter·핵심 패키지 확인",
                SETUP_CODE
                + """
from importlib.metadata import version
from jupyter_course.notebook_support import evaluate_jupyter_environment

python_version = sys.version.split()[0]
package_names = [
    "notebook", "ipykernel", "ipywidgets", "gradio", "pandas",
    "numpy", "matplotlib", "requests",
]
package_versions = {name: version(name) for name in package_names}
environment_check = evaluate_jupyter_environment(
    PROJECT_ROOT,
    python_version=(sys.version_info.major, sys.version_info.minor),
    notebook_version=package_versions["notebook"],
    executable=sys.executable,
)

print("현재 Python:", python_version)
print("현재 Python 경로:", sys.executable)
for name, installed_version in package_versions.items():
    print(f"{name:12} {installed_version}")
if not environment_check["ready"]:
    raise RuntimeError(
        "설치 확인 실패:\\n- " + "\\n- ".join(environment_check["issues"])
    )
print("설치 확인: 준비 완료")

chapter_result = {
    "chapter": "00",
    "environment_ready": environment_check["ready"],
    "sample_rows": len(raw_rows),
    "python_version": python_version,
    "notebook_version": package_versions["notebook"],
    "packages_checked": len(package_versions),
}
""",
                "`설치 확인: 준비 완료`가 보이면 Python 3.11 이상, Notebook 7.x, 프로젝트의 `.venv` 커널이 모두 확인된 것입니다. 조건이 맞지 않으면 고칠 내용을 표시하고 셀이 바로 멈춥니다. 행 수 5는 개인정보가 없는 수업용 예비 급식입니다.",
                """
1. `sys.version`과 `sys.executable`은 현재 커널이 쓰는 Python 버전과 경로를 보여 줍니다.<br>
2. `package_names`는 이 장에서 확인할 핵심 패키지 이름표입니다.<br>
3. `version(name)`은 실제 설치된 버전을 읽습니다.<br>
4. `evaluate_jupyter_environment`는 버전과 현재 Python 경로를 실제 조건과 비교합니다.<br>
5. 마지막 `chapter_result`는 자동 검증기가 확인할 설치 증거를 모읍니다.
""",
            ),
            (
                "표의 첫 두 행 살펴보기",
                """
columns_to_see = ["date", "menu_text", "calories"]
print(meal_df[columns_to_see].head(2).to_string(index=False))
""",
                "한 행은 하루 급식을 뜻합니다. date는 날짜, menu_text는 합친 메뉴, calories는 NEIS가 제공한 열량입니다.",
                """
1. `columns_to_see`에는 화면에서 확인할 세 열의 이름을 적습니다.<br>
2. `meal_df[columns_to_see]`는 전체 표에서 그 세 열만 고릅니다.<br>
3. `head(2)`는 위에서부터 두 행만 보여 줍니다.
""",
            ),
        ],
        exercise_text="package_to_check를 `pandas`, `matplotlib`, `ipywidgets` 중 하나로 바꾸어 설치 버전과 이 프로젝트에서 하는 일을 말해 보세요.",
        exercise_code="""
package_to_check = "pandas"
print("확인할 패키지:", package_to_check)
print("설치 버전:", version(package_to_check))
""",
        check_questions=[
            "가상환경을 만드는 이유는 무엇인가요?",
            "requirements-jupyter.txt는 어떤 역할을 하나요?",
            "Markdown 셀과 Code 셀은 무엇이 다른가요?",
            "커널을 다시 시작하면 왜 위에서부터 다시 실행해야 하나요?",
        ],
        check_answer="""
1. 이번 프로젝트의 패키지를 다른 수업이나 학교 PC 전체의 Python과 섞지 않기 위해서입니다.<br>
2. 설치할 패키지와 허용 버전을 한 줄씩 기록해 모두 같은 환경을 만들게 합니다.<br>
3. Markdown 셀은 설명을 보여 주고 Code 셀은 Python을 실행합니다.<br>
4. 커널의 변수 기억이 지워지기 때문입니다.
""",
        summary=[
            "가상환경은 프로젝트 전용 Python 공구함이다.",
            "requirements는 필요한 패키지를 한 줄씩 기록한 설치 목록이다.",
            "Jupyter는 설명과 코드를 한 흐름에서 다루는 전자 교과서다.",
            "좋은 노트북은 새 커널에서도 위에서 아래로 실행된다.",
        ],
        next_text="부록 A에서는 아주 작은 급식 예제로 JSON의 모양을 한 단계씩 연습합니다.",
    )


def _appendix_json() -> dict:
    return _lesson(
        number="A",
        title="JSON 기초 튜토리얼",
        question="작은 급식 정보가 어떻게 JSON 자료로 자랄까?",
        session="1회차 중반",
        minutes=55,
        objectives=[
            "글자·숫자·참과 거짓의 자료형을 화면에서 확인할 수 있다.",
            "목록에서는 위치 번호로, 사전에서는 이름표로 값을 찾을 수 있다.",
            "사전과 목록이 겹친 자료에서 바깥쪽부터 한 칸씩 이동할 수 있다.",
            "JSON 문자열을 Python 자료로 바꾸고 실제 급식 파일에서 메뉴를 찾을 수 있다.",
        ],
        connection="처음부터 긴 NEIS 응답을 보면 괄호와 영어 이름이 한꺼번에 보여 어렵습니다. 메뉴 하나에서 시작해 메뉴 목록, 이름표가 붙은 급식 카드, 상자 속 상자로 한 계단씩 넓혀 봅시다. 앞 활동에서 만든 자료가 다음 활동의 재료가 됩니다.",
        keywords=[
            ("값", "프로그램이 기억하는 글자·숫자·참과 거짓 같은 내용"),
            ("자료형", "값을 어떤 방법으로 다룰지 알려 주는 종류"),
            ("list", "여러 값을 순서대로 담은 Python 목록"),
            ("dict", "이름표인 키와 값을 짝지어 담은 Python 사전"),
            ("JSON", "데이터를 저장하거나 주고받기 위한 글자 형식"),
            ("중첩", "목록이나 사전 안에 다른 목록이나 사전이 들어간 구조"),
        ],
        concept="""
### 먼저 기억할 한 문장

JSON은 어려운 계산식이 아니라 **자료의 모양을 글자로 적는 약속**입니다. Python에서 먼저 값·목록·사전을 다뤄 본 뒤 JSON을 보면 기호의 역할이 훨씬 잘 보입니다.

### 1계단. 값 하나

`"김치"`는 글자, `500.5`는 소수, `True`는 참과 거짓을 나타내는 값입니다. `type()`을 사용하면 Python이 각 값을 어떤 종류로 기억하는지 확인할 수 있습니다.

### 2계단. 순서가 있는 목록

여러 메뉴를 차례대로 담을 때는 `list`를 사용합니다. 목록은 대괄호 `[ ]`로 만들고 위치 번호는 0부터 시작합니다. 첫 번째 값은 `[0]`, 두 번째 값은 `[1]`입니다.

### 3계단. 이름표가 붙은 사전

학교명·날짜·열량처럼 역할이 다른 값은 `dict`에 담습니다. 사전은 중괄호 `{ }`로 만들고 `"school": "남악고등학교"`처럼 **키와 값**을 짝지어 적습니다. 값은 위치가 아니라 `meal_card["school"]`처럼 키로 찾습니다.

### 4계단. 사전과 목록 겹치기

급식 카드의 `dishes` 값으로 메뉴 목록을 넣을 수 있습니다. 이때 `meal_card["dishes"][1]`은 먼저 `dishes` 키로 목록을 꺼내고, 그 목록의 두 번째 값을 고른다는 뜻입니다. 긴 경로는 왼쪽부터 한 칸씩 읽습니다.

### 5계단. Python 자료와 JSON 문자열 구분하기

겉모양이 비슷해도 Python 사전은 계산에 바로 쓸 수 있는 자료이고, JSON은 저장하거나 전송할 때 사용하는 글자입니다. `json.loads()`는 JSON 글자를 Python 자료로 바꾸고, `json.dumps()`는 Python 자료를 JSON 글자로 바꿉니다.

| JSON 모양 | 뜻 | Python으로 바뀐 뒤 |
|---|---|---|
| `{ }` | 키와 값을 묶은 객체 | `dict` |
| `[ ]` | 순서가 있는 배열 | `list` |
| `"김치"` | 글자 | `str` |
| `500.5` | 숫자 | `float` |
| `true` | 참 | `True` |

### 6계단. 실제 파일 읽기

마지막에는 저장소의 `data/namak_meals_sample.json`을 읽습니다. 이 파일은 `metadata`와 `rows`라는 두 이름표가 있는 사전입니다. `rows`의 값은 급식 다섯 건을 담은 목록입니다. 처음부터 긴 경로를 쓰지 않고 `파일 전체 → rows → 첫 행 → DDISH_NM`으로 나누어 이동합니다.
""",
        hand_example="""
다음 세 줄을 보고 값을 찾는 방법을 말해 봅시다.

```python
menu = "김치볶음밥"
dishes = ["밥", "국", "김치"]
meal = {"school": "남악고등학교", "dishes": dishes}
```

| 찾을 값 | 먼저 고를 것 | 코드 |
|---|---|---|
| 메뉴 한 개 | 변수 이름 | `menu` |
| 목록의 첫 번째 값 | 위치 0 | `dishes[0]` |
| 학교명 | `school` 이름표 | `meal["school"]` |
| 급식 카드의 두 번째 메뉴 | `dishes` 이름표, 위치 1 | `meal["dishes"][1]` |

대괄호 안에 숫자가 있으면 **위치**, 따옴표가 있는 글자가 있으면 **이름표**를 고른다고 생각하면 됩니다.
""",
        prediction="- 값 세 개의 자료형은 차례로 `str`, `float`, `bool`이다.\n- 메뉴 목록의 위치 1에는 ‘미트볼로제파스타’가 있다.\n- JSON 글자를 `json.loads()`로 바꾸면 Python의 `dict`가 된다.\n- 실제 예비 파일의 `rows`에는 급식 다섯 건이 있다.",
        code_sections=[
            (
                "값 하나와 자료형 확인하기",
                """
menu_name = "김치"
calories = 500.5
is_lunch = True

value_types = [
    type(menu_name).__name__,
    type(calories).__name__,
    type(is_lunch).__name__,
]

print("메뉴:", menu_name, "→", value_types[0])
print("열량:", calories, "→", value_types[1])
print("중식인가요?:", is_lunch, "→", value_types[2])
""",
                "글자는 `str`, 소수는 `float`, 참과 거짓은 `bool`로 나타났습니다. 자료형을 알면 더하기, 비교하기처럼 그 값으로 할 수 있는 일을 예상할 수 있습니다.",
                """
1. `menu_name`, `calories`, `is_lunch`는 값 하나씩을 기억하는 이름입니다.<br>
2. `type(menu_name)`은 값의 자료형을 확인합니다.<br>
3. `.__name__`은 `<class 'str'>` 대신 `str`처럼 짧은 이름만 꺼냅니다.<br>
4. 세 자료형을 목록에 모아 차례대로 비교합니다.
""",
            ),
            (
                "메뉴를 목록에 차례대로 담기",
                """
dishes = ["양송이스프", "미트볼로제파스타", "피자"]
first_dish = dishes[0]
second_dish = dishes[1]

print("메뉴 목록:", dishes)
print("메뉴 수:", len(dishes))
print("위치 0:", first_dish)
print("위치 1:", second_dish)
""",
                "메뉴가 세 개 들어 있어 목록 길이는 3입니다. 사람에게 두 번째인 메뉴의 위치 번호는 1입니다. Python의 목록 위치가 0부터 시작하기 때문입니다.",
                """
1. `[ ]` 안에 쉼표로 나눈 값 세 개를 넣어 목록을 만듭니다.<br>
2. `len(dishes)`는 목록에 든 값의 수를 셉니다.<br>
3. `dishes[0]`은 첫 번째 메뉴를 고릅니다.<br>
4. `dishes[1]`은 두 번째 메뉴를 골라 `second_dish`에 저장합니다.
""",
            ),
            (
                "이름표가 붙은 급식 카드 만들기",
                """
meal_card = {
    "school": "남악고등학교",
    "date": "20260624",
    "calories": 500.5,
}

school_name = meal_card["school"]
print("급식 카드:", meal_card)
print("이름표 목록:", list(meal_card.keys()))
print("학교명:", school_name)
print("날짜:", meal_card["date"])
""",
                "사전은 위치 번호 대신 이름표로 값을 찾습니다. `school`, `date`, `calories`는 키이고, 각 키의 오른쪽에 있는 내용이 값입니다.",
                """
1. `{ }` 안에 `키: 값` 쌍을 세 개 적어 사전을 만듭니다.<br>
2. `meal_card["school"]`은 `school` 이름표에 붙은 값을 꺼냅니다.<br>
3. `meal_card.keys()`는 사전에 있는 이름표를 보여 줍니다.<br>
4. 날짜는 계산할 수가 아니라 여덟 자리 표기이므로 글자로 저장했습니다.
""",
            ),
            (
                "사전 안에 메뉴 목록 넣기",
                """
meal_card["dishes"] = dishes

meal_part = meal_card["dishes"]
nested_date = meal_card["date"]
nested_second_dish = meal_part[1]

print("dishes의 자료형:", type(meal_part).__name__)
print("한 칸씩 찾은 날짜:", nested_date)
print("한 칸씩 찾은 두 번째 메뉴:", nested_second_dish)
print("한 줄로 쓴 같은 경로:", meal_card["dishes"][1])
""",
                "`meal_card`는 사전이고 그 안의 `dishes` 값은 목록입니다. 한 칸씩 나누어 찾은 결과와 한 줄 경로로 찾은 결과가 같습니다.",
                """
1. `meal_card["dishes"] = dishes`는 급식 카드에 새 이름표와 목록을 붙입니다.<br>
2. 먼저 `meal_card["dishes"]`만 실행해 안쪽 목록을 꺼냅니다.<br>
3. 꺼낸 목록에서 `[1]`로 두 번째 메뉴를 찾습니다.<br>
4. 익숙해지면 두 동작을 `meal_card["dishes"][1]`로 이어 쓸 수 있습니다.
""",
            ),
            (
                "JSON 글자를 Python 자료로 바꾸기",
                """
import json

json_text = '''{
  "school": "남악고등학교",
  "date": "20260624",
  "dishes": ["양송이스프", "미트볼로제파스타"]
}'''

loaded_data = json.loads(json_text)
json_is_dict = isinstance(loaded_data, dict)
restored_json_text = json.dumps(loaded_data, ensure_ascii=False, indent=2)

print("바꾸기 전:", type(json_text).__name__)
print("바꾼 뒤:", type(loaded_data).__name__)
print("두 번째 메뉴:", loaded_data["dishes"][1])
print("다시 JSON 글자로 바꾼 결과:\\n", restored_json_text)
""",
                "따옴표 세 개 사이의 JSON은 처음에는 긴 글자 하나인 `str`입니다. `json.loads()`를 실행한 뒤에야 키와 위치로 값을 찾을 수 있는 `dict`가 됩니다.",
                """
1. `json_text`는 JSON 모양을 하고 있지만 자료형은 아직 `str`입니다.<br>
2. `json.loads(json_text)`는 JSON 객체와 배열을 Python 사전과 목록으로 바꿉니다.<br>
3. 바뀐 자료에서는 `loaded_data["dishes"][1]`처럼 값을 찾을 수 있습니다.<br>
4. `json.dumps(...)`는 Python 자료를 저장하거나 전송할 JSON 글자로 되돌립니다.<br>
5. `ensure_ascii=False`는 한글을 그대로 보이게 하고 `indent=2`는 들여쓰기를 넣습니다.
""",
            ),
            (
                "실제 예비 JSON 파일에서 첫 급식 찾기",
                SETUP_CODE
                + """
import json

sample_path = PROJECT_ROOT / "data" / "namak_meals_sample.json"
sample_data = json.loads(sample_path.read_text(encoding="utf-8"))

sample_rows_data = sample_data["rows"]
sample_first_row = sample_rows_data[0]
sample_rows = len(sample_rows_data)
sample_menu = sample_first_row["DDISH_NM"]

print("파일 전체 자료형:", type(sample_data).__name__)
print("가장 바깥 이름표:", list(sample_data.keys()))
print("rows 자료형:", type(sample_rows_data).__name__)
print("급식 행 수:", sample_rows)
print("첫 급식 날짜:", sample_first_row["MLSV_YMD"])
print("첫 메뉴 일부:", sample_menu[:100] + "...")

chapter_result = {
    "chapter": "A",
    "value_types": value_types,
    "second_dish": second_dish,
    "school_name": school_name,
    "nested_date": nested_date,
    "json_is_dict": json_is_dict,
    "sample_rows": sample_rows,
    "sample_menu": sample_menu,
    "tutorial_steps": 6,
}
""",
                "연습용 작은 자료와 같은 방법으로 실제 파일도 읽었습니다. 바깥 사전에서 `rows` 목록을 고르고, 위치 0의 첫 급식 사전에서 날짜와 메뉴 키를 찾았습니다.",
                """
1. `sample_path`는 저장소 안의 실제 예비 JSON 파일 위치입니다.<br>
2. `read_text(...)`로 파일 글자를 읽고 `json.loads(...)`로 Python 사전으로 바꿉니다.<br>
3. `sample_data["rows"]`로 급식 목록을 꺼냅니다.<br>
4. `sample_rows_data[0]`으로 첫 급식 사전을 고릅니다.<br>
5. 마지막 사전에서 `MLSV_YMD`와 `DDISH_NM` 키의 값을 읽습니다.<br>
6. 긴 경로에서 막히면 한 줄로 합치지 말고 이 코드처럼 중간 변수로 나눕니다.
""",
            ),
        ],
        exercise_text="목록 연습으로 돌아가 `practice_dishes`의 값 하나만 바꾸세요. 위치 번호 0과 1의 결과를 확인한 뒤, 사람의 순서와 Python 위치 번호가 어떻게 다른지 한 문장으로 적습니다.",
        exercise_code="""
practice_dishes = ["밥", "국", "김치"]

print("첫 번째 메뉴, 위치 0:", practice_dishes[0])
print("두 번째 메뉴, 위치 1:", practice_dishes[1])
print("전체 메뉴 수:", len(practice_dishes))
""",
        check_questions=[
            "`\"김치\"`, `500.5`, `True`의 자료형은 각각 무엇인가요?",
            "목록의 첫 번째 값과 두 번째 값은 각각 몇 번 위치인가요?",
            "사전에서 값은 위치 번호와 이름표 가운데 무엇으로 찾나요?",
            "`meal_card[\"dishes\"][1]`을 두 단계로 나누어 설명해 보세요.",
            "JSON 글자와 Python 사전은 어떻게 다르며 `json.loads()`는 무엇을 하나요?",
            "긴 중첩 경로에서 오류가 났을 때 중간 변수를 만드는 까닭은 무엇인가요?",
        ],
        check_answer="""
1. 차례로 `str`, `float`, `bool`입니다.<br>
2. 위치 번호는 0부터 시작하므로 첫 번째 값은 0, 두 번째 값은 1입니다.<br>
3. 사전은 `school` 같은 이름표인 키로 값을 찾습니다.<br>
4. 먼저 `dishes` 키로 메뉴 목록을 꺼내고, 그 목록의 위치 1에서 두 번째 메뉴를 찾습니다.<br>
5. JSON은 저장·전송할 때 쓰는 글자이고 Python 사전은 코드에서 바로 다룰 자료입니다. `json.loads()`는 JSON 글자를 Python 자료로 바꿉니다.<br>
6. 각 단계의 자료형과 값을 따로 확인하면 어느 이름표나 위치에서 잘못되었는지 찾기 쉽기 때문입니다.
""",
        summary=[
            "값 하나는 자료형을 가지며 `type()`으로 확인할 수 있다.",
            "목록은 위치 번호로 찾고 첫 위치는 0이다.",
            "사전은 이름표인 키로 값을 찾는다.",
            "중첩 자료는 바깥쪽부터 한 칸씩 이동한다.",
            "JSON은 글자 형식이고 `json.loads()`를 거쳐야 Python 자료로 다룰 수 있다.",
            "실제 급식 파일도 작은 예제와 같은 규칙으로 읽을 수 있다.",
        ],
        next_text="01장에서는 JSON 문법 연습을 반복하지 않고, 공식 NEIS 주소에 조건을 붙여 GET 요청을 보내는 과정에 집중합니다.",
    )


def _chapter_01() -> dict:
    return _lesson(
        number="01",
        title="NEIS API 요청하기",
        question="우리 학교 급식을 공식 서버에 어떻게 요청할까?",
        session="1회차 후반",
        minutes=55,
        objectives=[
            "API의 요청과 응답을 식당 주문에 비유해 설명할 수 있다.",
            "요청 주소와 조건을 확인한 뒤 NEIS API에 GET 요청을 보낼 수 있다.",
            "상태 코드와 받은 행 수로 요청 결과를 확인할 수 있다.",
            "학교명보다 학교 코드가 정확한 식별값인 이유를 말할 수 있다.",
            "연결에 실패했을 때 같은 기간의 예비 자료로 전환할 수 있다.",
        ],
        connection="부록 A에서 JSON 파일을 여는 방법을 익혔습니다. 이제 자료를 이미 저장해 둔 파일에서 읽는 대신, NEIS 서버의 정해진 창구에 학교 코드와 날짜를 보내 직접 받아 봅시다.",
        keywords=[
            ("API", "다른 서비스에 정해진 규칙으로 데이터를 요청하는 창구"),
            ("요청", "주소와 조건을 서버에 보내는 일"),
            ("응답", "서버가 요청을 처리한 뒤 돌려주는 결과"),
            ("params", "어느 학교·기간·형식의 자료를 원하는지 적은 요청 조건"),
            ("GET", "서버에 자료를 달라고 요청하는 대표적인 HTTP 방식"),
            ("상태 코드", "서버가 요청을 어떻게 처리했는지 나타내는 숫자"),
            ("학교 코드", "같은 이름의 학교를 구분하는 공식 식별값"),
        ],
        concept="""
API는 급식실 주문 창구와 비슷합니다. ‘남악고등학교, 2026년 6월 24일부터 30일까지 중식’을 정해진 양식으로 요청하면 서버가 정해진 양식의 응답을 줍니다.

### 1. 주소와 조건을 나누어 적는다

요청 주소는 `https://open.neis.go.kr/hub/mealServiceDietInfo`입니다. 교육청 코드, 학교 코드, 날짜 같은 조건은 `params` 사전에 넣습니다. `requests.get()`은 주소와 조건을 합쳐 GET 요청을 보내고, 서버는 상태 코드와 JSON 응답을 돌려줍니다.

| 구분 | 이번 수업의 값 | 뜻 |
|---|---|---|
| URL | `.../mealServiceDietInfo` | 어느 API 창구로 갈지 |
| `ATPT_OFCDC_SC_CODE` | `Q10` | 어느 교육청인지 |
| `SD_SCHUL_CODE` | `7140272` | 어느 학교인지 |
| `MLSV_FROM_YMD` | `20260624` | 조회 시작일 |
| `MLSV_TO_YMD` | `20260630` | 조회 종료일 |

### 2. 보내기 전에 완성된 주소를 확인한다

코드는 `requests.Request(...).prepare()`로 요청 미리보기를 만듭니다. 이 단계에서는 아직 서버에 요청하지 않습니다. 주소에 `SD_SCHUL_CODE=7140272`와 날짜가 제대로 들어갔는지 먼저 확인합니다.

### 3. 상태 코드를 보고 응답을 읽는다

`requests.get(..., timeout=15)`가 실제 요청을 보냅니다. 상태 코드 200은 서버가 요청을 정상적으로 처리했다는 뜻입니다. `raise_for_status()`는 404나 500 같은 HTTP 오류를 찾아내고, `response.json()`은 받은 JSON 글자를 Python 자료로 바꿉니다. JSON의 기호와 중첩 경로가 낯설면 부록 A로 돌아가 다시 연습합니다.

### 4. 실패할 때 사용할 자료도 준비한다

교실 인터넷이나 NEIS 서버가 잠시 불안정할 수 있습니다. 그래서 같은 학교·같은 기간의 예비 자료를 저장소에 함께 둡니다. 요청이 실패했다고 아무 날짜의 자료를 대신 쓰지 않고, 요청 기간과 겹칠 때만 전환합니다.

공식 안내에 따르면 인증키를 쓰지 않은 호출은 샘플 자료 5건으로 제한됩니다. 수업에서는 먼저 짧은 샘플 요청을 보내고, 별도 인증키를 사용할 때는 코드에 적지 않고 `NEIS_API_KEY` 환경 변수에서 읽습니다.
""",
        hand_example="""
식당에 주문서를 낸다고 생각하고 빈칸을 채워 보세요.

| 주문서 항목 | 적을 내용 |
|---|---|
| 주문 창구 | 급식식단정보 API 주소 |
| 학교 | 학교명이 아니라 공식 학교 코드 `________` |
| 시작일 | `________` |
| 종료일 | `________` |
| 받고 싶은 형식 | `json` |

URL은 주문 창구이고 `params`는 주문서의 조건입니다. 같은 ‘남악고’라는 글자가 있더라도 공식 코드를 사용해야 정확한 학교를 구별할 수 있습니다.
""",
        prediction="- 요청 미리보기 주소에는 학교 코드 7140272가 들어간다.\n- 요청을 보내기 전 기본값에서는 `live_request_sent`가 `False`이다.\n- 실시간 조회 실패 실험에서는 같은 기간의 남악고 예비 자료 5행을 사용한다.",
        code_sections=[
            (
                "NEIS 요청 주소를 만들고 GET 함수 준비하기",
                SETUP_CODE
                + """
import os
import requests

NEIS_MEAL_URL = "https://open.neis.go.kr/hub/mealServiceDietInfo"
request_params = {
    "Type": "json",
    "pIndex": 1,
    "pSize": 5,
    "ATPT_OFCDC_SC_CODE": "Q10",
    "SD_SCHUL_CODE": "7140272",
    "MLSV_FROM_YMD": "20260624",
    "MLSV_TO_YMD": "20260630",
}

api_key = os.getenv("NEIS_API_KEY", "").strip()
if api_key:
    request_params["KEY"] = api_key

preview_params = {
    key: value for key, value in request_params.items() if key != "KEY"
}
prepared_request = requests.Request(
    "GET", NEIS_MEAL_URL, params=preview_params
).prepare()
prepared_request_url = prepared_request.url

def request_neis_meals(params, *, http_get=requests.get):
    response = http_get(NEIS_MEAL_URL, params=params, timeout=15)
    response.raise_for_status()
    safe_params = {key: value for key, value in params.items() if key != "KEY"}
    safe_url = requests.Request(
        "GET", NEIS_MEAL_URL, params=safe_params
    ).prepare().url
    return response.json(), safe_url, response.status_code

first_keys = sorted(raw_rows[0].keys())

print("요청 방식: GET")
print("요청 주소:", NEIS_MEAL_URL)
print("요청 조건 수:", len(request_params))
print("실제로 전송될 주소:", prepared_request_url)
print("인증키:", "환경 변수에서 읽음" if api_key else "샘플 호출(최대 5건)")
""",
                "아직 서버에 보내지는 않았지만, 주소와 조건이 합쳐진 모습을 확인했습니다. 실제 요청 함수는 15초 안에 응답이 없으면 멈추고, HTTP 오류가 있으면 그대로 알려 줍니다.",
                """
1. `NEIS_MEAL_URL`은 급식식단정보 API의 고정 주소입니다.<br>
2. `request_params`는 JSON 형식, 페이지, 학교, 기간을 키와 값으로 묶습니다.<br>
3. `preview_params`는 화면에 보여 줄 조건에서 인증키를 제외합니다.<br>
4. `requests.Request(...).prepare()`는 요청을 보내지 않고 안전한 미리보기 URL만 만듭니다.<br>
5. `requests.get(..., timeout=15)`가 실제 GET 요청을 보냅니다.<br>
6. `raise_for_status()`는 404나 500 같은 HTTP 오류를 성공으로 착각하지 않게 합니다.<br>
7. `response.json()`은 응답 JSON을 Python 사전과 목록으로 바꿉니다.<br>
8. 함수 안의 `safe_params`도 인증키를 뺀 응답 주소만 화면에 돌려줍니다.<br>
9. 인증키가 있더라도 URL이나 화면에 출력하지 않습니다.
""",
            ),
            (
                "실시간 우선·예비 자료 전환 연습",
                """
from jupyter_course.notebook_support import load_classroom_frame
from neis_meal_ai.neis import NeisApiError

def classroom_offline_demo(_school, _start, _end):
    raise NeisApiError("교실용 연결 실패 실험")

classroom_frame, classroom_source = load_classroom_frame(
    PROJECT_ROOT,
    fetcher=classroom_offline_demo,
)
print("선택된 데이터 출처:", classroom_source)
print("분석 가능한 급식 행 수:", len(classroom_frame))

try:
    load_classroom_frame(
        PROJECT_ROOT,
        start="20260101",
        end="20260102",
        fetcher=classroom_offline_demo,
    )
except NeisApiError as error:
    print("날짜가 겹치지 않을 때의 안내:", error)

chapter_result = {
    "chapter": "01",
    "source": classroom_source,
    "raw_rows": len(classroom_frame),
    "first_keys": first_keys,
    "prepared_request_url": prepared_request_url,
    "live_request_sent": False,
}
""",
                "실시간 조회가 실패해도 같은 학교·같은 기간의 예비 자료가 있을 때만 전환됩니다. 기간이 겹치지 않으면 조용히 다른 날짜를 쓰지 않고 정확한 안내를 보여 줍니다.",
                """
1. `classroom_offline_demo`는 교실에서 연결 실패 상황을 재현합니다.<br>
2. `load_classroom_frame`은 실시간 조회가 실패하면 같은 기간의 예비 자료를 찾습니다.<br>
3. `try`와 `except NeisApiError`는 날짜가 맞지 않는 경우의 안내를 확인합니다.
""",
            ),
        ],
        exercise_text="아래의 SEND_LIVE_REQUEST를 True로 바꾸면 방금 만든 함수가 공식 NEIS 서버에 GET 요청을 한 번 보냅니다. 먼저 False 상태에서 안내를 읽고, 요청 주소에 학교 코드와 날짜가 맞는지 확인한 뒤 True로 바꾸세요. 인증키가 없다면 공식 포털의 샘플 호출 제한에 따라 최대 5건만 받을 수 있습니다.",
        exercise_code="""
SEND_LIVE_REQUEST = False
live_request_sent = False

if SEND_LIVE_REQUEST:
    try:
        live_payload, response_url, status_code = request_neis_meals(request_params)
        live_request_sent = True
        print("응답 상태 코드:", status_code)
        print("보낸 주소:", response_url)
        if "mealServiceDietInfo" in live_payload:
            live_rows = live_payload["mealServiceDietInfo"][1]["row"]
            print("받은 급식 행 수:", len(live_rows))
            print("첫 급식 날짜:", live_rows[0]["MLSV_YMD"])
            print("첫 메뉴:", live_rows[0]["DDISH_NM"][:100] + "...")
        else:
            print("급식 행 대신 안내 응답이 왔습니다:", live_payload)
    except (requests.RequestException, ValueError, KeyError, IndexError) as error:
        print("요청 또는 JSON 읽기 안내:", error)
else:
    print("기본값은 전송하지 않습니다. 주소와 조건을 확인한 뒤 True로 바꾸세요.")

chapter_result["live_request_sent"] = live_request_sent
""",
        check_questions=[
            "NEIS 요청에서 URL과 params는 각각 어떤 역할을 하나요?",
            "학교명 대신 학교 코드를 요청 조건에 넣는 까닭은 무엇인가요?",
            "요청을 보내기 전에 미리보기 주소에서 무엇을 확인해야 하나요?",
            "`status_code`, `raise_for_status()`, `response.json()`은 차례로 무엇을 확인하거나 바꾸나요?",
            "실시간 API가 잠시 멈춰도 수업을 이어 갈 수 있는 이유는 무엇인가요?",
        ],
        check_answer="""
1. URL은 어느 API 창구로 갈지, params는 어느 학교의 어느 기간 자료를 달라고 할지 정합니다.<br>
2. 비슷하거나 같은 학교 이름이 있어도 공식 코드는 학교를 정확히 구별하는 식별값이기 때문입니다.<br>
3. 학교 코드 `7140272`, 교육청 코드 `Q10`, 시작일과 종료일이 맞는지 확인합니다.<br>
4. 상태 코드는 서버 처리 결과를 나타내고, `raise_for_status()`는 HTTP 오류를 확인하며, `response.json()`은 JSON 응답을 Python 자료로 바꿉니다.<br>
5. 같은 구조의 공식 NEIS 예비 데이터 5행을 프로젝트에 포함했기 때문입니다.
""",
        summary=[
            "GET 요청은 API 주소와 params 조건을 합쳐 서버에 자료를 요청한다.",
            "학교명 대신 공식 학교 코드로 정확한 학교를 구별한다.",
            "보내기 전에 미리보기 주소의 학교 코드와 날짜를 확인한다.",
            "응답은 상태 코드를 확인한 뒤 JSON으로 읽는다.",
            "실시간 조회에 실패하면 같은 학교·같은 기간의 예비 자료로 전환한다.",
            "남악고의 공식 코드는 Q10 / 7140272다.",
        ],
        next_text="02장에서는 원본 문자열을 분석 가능한 표로 바꾸고 그래프로 읽습니다.",
    )


def _chapter_02() -> dict:
    return _lesson(
        number="02",
        title="급식 데이터 정리와 그래프",
        question="복잡한 메뉴 글자를 어떻게 표로 바꿀까?",
        session="2회차",
        minutes=180,
        objectives=[
            "HTML 줄바꿈과 알레르기 번호를 메뉴명에서 분리할 수 있다.",
            "열량과 영양 문자열을 숫자 열로 바꾸는 이유를 설명할 수 있다.",
            "표와 막대그래프에서 상대적인 차이를 읽을 수 있다.",
        ],
        connection="NEIS에서 받은 메뉴에는 `<br/>`, 괄호 속 번호, `kcal` 같은 표시가 함께 들어 있습니다. 사람이 보기에는 뜻이 분명하지만 그대로는 계산하기 어렵습니다. 분석에 필요한 부분을 나누고 숫자를 꺼내는 과정이 필요합니다.",
        keywords=[
            ("전처리", "분석 전에 데이터를 정리하고 변환하는 과정"),
            ("결측값", "기록되지 않아 비어 있는 값"),
            ("DataFrame", "행과 열로 이루어진 Pandas 표"),
            ("시각화", "숫자의 관계를 그래프로 표현하는 일"),
        ],
        concept="""
전처리는 요리 전 재료 손질과 같습니다. 메뉴 문자열에는 줄바꿈 표시, 괄호 속 알레르기 번호, 단위가 붙은 숫자가 함께 있습니다. 사람은 눈으로 구분하지만 Python은 규칙을 알려 주어야 합니다.

숫자가 비어 있다고 0이라고 단정하면 안 됩니다. ‘기록 없음’과 ‘영양소 0’은 다른 뜻이므로 결측값은 데이터 부족으로 다룹니다.
""",
        hand_example="""
‘미트볼파스타 (1.2.5.6)&lt;br/&gt;피클’을 종이에 써서 메뉴명, 번호, 줄바꿈 표시를 서로 다른 색으로 표시해 보세요.
""",
        prediction="- 메뉴 원문에서 HTML 표시가 사라진다.\n- 날짜별 열량 막대가 5개 나타난다.",
        code_sections=[
            (
                "한 행을 손질하기",
                SETUP_CODE
                + """
from neis_meal_ai.cleaning import (
    extract_allergy_codes,
    parse_calories,
    split_dishes,
)

sample_menu = raw_rows[0]["DDISH_NM"]
print("원문 일부:", sample_menu[:120])
print("메뉴 목록:", split_dishes(sample_menu))
print("알레르기 번호:", extract_allergy_codes(sample_menu))
print("열량 숫자:", parse_calories(raw_rows[0]["CAL_INFO"]))
""",
                "메뉴 이름은 읽기 쉬운 목록이 되고, 알레르기 번호는 별도 튜플로 보존됩니다. 삭제가 아니라 분리입니다.",
                """
1. `sample_menu`에는 첫 급식의 메뉴 원문을 저장합니다.<br>
2. `split_dishes`는 줄바꿈 표시를 기준으로 메뉴를 나눕니다.<br>
3. `extract_allergy_codes`와 `parse_calories`는 번호와 열량을 각각 숫자로 바꿉니다.
""",
            ),
            (
                "정제 표와 결측값 확인",
                """
learning_columns = [
    "date", "menu_text", "calories", "carbs_g",
    "protein_g", "fat_g", "dish_count",
]
print(meal_df[learning_columns].to_string(index=False))
print()
print("열별 빈 값 개수:")
print(meal_df[learning_columns].isna().sum().to_string())
""",
                "한 행은 하루, 한 열은 같은 종류의 특징입니다. 빈 값 개수가 0이면 이번 예비 자료의 영양 수치가 모두 기록된 것입니다.",
                """
1. `learning_columns`는 학습에 사용할 열의 순서를 정합니다.<br>
2. `meal_df[learning_columns]`는 선택한 열만 같은 순서로 보여 줍니다.<br>
3. `isna().sum()`은 열마다 비어 있는 값의 개수를 셉니다.
""",
            ),
            (
                "날짜별 열량 막대그래프",
                """
import os
import io
import matplotlib
if os.getenv("NEIS_JUPYTER_VERIFY") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

ax = meal_df.plot.bar(x="date", y="calories", legend=False, color="#4C78A8")
ax.set_title("Namak High School lunch calories (relative view)")
ax.set_xlabel("date")
ax.set_ylabel("kcal")
plt.tight_layout()
if os.getenv("NEIS_JUPYTER_VERIFY") != "1":
    plt.show()
else:
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format="png", bbox_inches="tight")
    plt.close()
    display(Image(data=image_buffer.getvalue()))

chapter_result = {
    "chapter": "02",
    "clean_rows": len(meal_df),
    "columns": list(meal_df.columns),
    "chart_ready": True,
}
""",
                "막대의 높이는 날짜 사이의 상대 차이를 보여 줍니다. 이 그래프만으로 건강함이나 학생에게 맞는 식단을 판단할 수 없습니다.",
                """
1. `meal_df.plot.bar`는 `date`를 가로축, `calories`를 세로축으로 그립니다.<br>
2. `set_title`, `set_xlabel`, `set_ylabel`은 그래프의 제목과 축 이름을 붙입니다.<br>
3. 자동 검증에서는 그래프를 PNG 그림으로 바꾸어 노트북 출력에 저장합니다. 그래서 학생이 셀을 다시 실행하지 않아도 완성된 그래프를 먼저 볼 수 있습니다.
""",
            ),
        ],
        exercise_text="graph_column을 protein_g 또는 carbs_g로 바꾸어 가장 높은 날짜가 달라지는지 확인하세요.",
        exercise_code="""
graph_column = "protein_g"
highest_row = meal_df.loc[meal_df[graph_column].idxmax()]
print(graph_column, "값이 가장 큰 날짜:", highest_row["date"])
print("값:", highest_row[graph_column])
""",
        check_questions=[
            "알레르기 번호를 메뉴명에서 지우기만 하지 않고 별도 열에 보존하는 이유는 무엇인가요?",
            "결측값과 숫자 0은 왜 다른가요?",
            "열량 그래프를 건강 순위라고 부르면 안 되는 이유는 무엇인가요?",
        ],
        check_answer="""
1. 표시와 분석에 다시 사용해야 하는 정보이기 때문입니다.<br>
2. 결측값은 기록이 없다는 뜻이고 0은 실제 값이 0이라는 뜻입니다.<br>
3. 열량 한 가지 수치만으로 개인의 건강 적합성을 판단할 수 없기 때문입니다.
""",
        summary=[
            "전처리는 원본을 메뉴 목록, 번호, 숫자 열로 나눈다.",
            "결측값을 0으로 단정하지 않는다.",
            "그래프는 상대 차이를 관찰하는 도구이지 건강 판정표가 아니다.",
        ],
        next_text="03장에서는 메뉴 글자를 숫자 벡터로 바꾸는 TF-IDF를 배웁니다.",
    )


def _chapter_03() -> dict:
    return _lesson(
        number="03",
        title="TF-IDF — 글자를 숫자로",
        question="컴퓨터는 메뉴 글자를 어떻게 숫자로 바꿀까?",
        session="3회차 전반",
        minutes=90,
        objectives=[
            "한 단어를 2-gram과 3-gram으로 직접 나눌 수 있다.",
            "TF, DF, IDF가 각각 무엇을 세는지 표로 설명할 수 있다.",
            "TF와 IDF를 곱해 한 글자 조각의 TF-IDF를 계산할 수 있다.",
            "출처가 기록된 한국어 문서 세 편에서 문서별 특징어를 찾을 수 있다.",
            "취향 문장과 메뉴를 숫자로 비교할 준비를 할 수 있다.",
        ],
        connection="‘토마토파스타를 좋아한다’라는 취향과 ‘미트볼파스타’라는 메뉴는 비슷해 보입니다. 그러나 컴퓨터는 문장의 느낌을 그대로 비교하지 못합니다. 두 문장에 함께 나타나는 글자 조각을 세어 숫자로 표현해 봅시다.",
        keywords=[
            ("n-gram", "연속된 n개의 글자 조각"),
            ("TF", "한 문서 안에서 글자 조각이 나타난 정도"),
            ("DF", "그 글자 조각이 들어 있는 문서의 수"),
            ("IDF", "여러 문서 중 드물게 나타나 구별에 도움 되는 정도"),
            ("TF-IDF", "한 문서 안의 중요도와 전체 문서에서의 희소성을 곱한 값"),
            ("말뭉치", "같은 방법으로 비교하려고 모아 둔 여러 문서"),
            ("벡터", "여러 숫자를 순서대로 모은 표현"),
        ],
        concept="""
### 1. n-gram은 움직이는 창으로 만드는 글자 조각

컴퓨터는 문장을 뜻으로 바로 이해하지 못하므로 먼저 비교할 수 있는 작은 조각을 만듭니다. n-gram의 n은 **한 조각에 들어가는 글자 수**입니다.

`파스타` 위에 두 칸짜리 창을 올리고 한 칸씩 옮기면 `파스`, `스타`가 나옵니다. 이것이 2-gram입니다. 세 칸짜리 창을 쓰면 `파스타` 하나가 나오며, 이것은 3-gram입니다.

이 프로젝트는 단어 앞뒤의 경계도 구별하려고 공백을 붙입니다. 눈에 보이지 않는 공백을 `·`로 표시하면 `파스타`의 2-gram은 `·파`, `파스`, `스타`, `타·`입니다. 2글자 조각은 짧은 공통점을 잘 찾고, 3~4글자 조각은 더 구체적인 이름을 잡습니다. 그래서 한 크기만 고르지 않고 2~4글자를 함께 사용합니다.

한국어 메뉴는 `치즈 파스타`, `치즈파스타`처럼 띄어쓰기가 달라질 수 있습니다. 단어 전체가 정확히 같아야 하는 방식보다 문자 조각 방식이 작은 표기 차이에서도 공통 부분을 찾기 쉽습니다.

### 2. TF는 한 문장 안을 본다

TF는 **Term Frequency**, 즉 한 문장 안에서 특정 조각이 차지하는 비율입니다.

    TF = 그 조각이 나온 횟수 ÷ 그 문장의 전체 n-gram 수

같은 조각이 많이 나오면 TF가 커집니다. 단순 횟수 대신 전체 조각 수로 나누는 까닭은 짧은 메뉴와 긴 메뉴를 조금 더 공정하게 비교하기 위해서입니다.

### 3. DF와 IDF는 모든 문장을 함께 본다

DF는 **Document Frequency**입니다. 어떤 조각이 총 몇 번 반복됐는지가 아니라, 그 조각을 포함한 문서가 몇 개인지를 셉니다. 한 문서에서 열 번 반복되어도 DF에는 문서 한 개로 셉니다.

반 학생 모두가 ‘급식’이라는 말을 쓰면 그 말만으로 누가 쓴 문장인지 구별하기 어렵습니다. 반대로 한 학생만 ‘파스타’라는 말을 썼다면 강한 단서가 됩니다. IDF는 이런 차이를 숫자로 나타냅니다.

    IDF = log((전체 문서 수 + 1) ÷ (DF + 1)) + 1

DF가 작으면 나누는 수가 작아져 IDF가 커지고, DF가 크면 IDF가 작아집니다. `log`는 드문 조각의 값이 지나치게 커지지 않도록 눌러 줍니다. 식의 `+1`은 문서가 적거나 처음 보는 조각이 있어도 안전하게 계산하기 위한 장치입니다.

### 4. TF-IDF는 두 관점을 합친다

    TF-IDF = TF × IDF

어떤 조각이 **이 메뉴 안에서는 자주 나오고(TF가 큼)**, **다른 메뉴에는 드물면(IDF가 큼)** TF-IDF가 커집니다. 모든 조각의 TF-IDF 값을 같은 순서로 늘어놓은 숫자 목록이 벡터입니다. 다음 장에서는 두 벡터가 가리키는 방향을 비교합니다.

### 5. 실제 문서는 먼저 같은 기준으로 단어를 골라야 한다

문서 한 편만 읽으면 어떤 단어가 그 글만의 특징인지 판단하기 어렵습니다. 주제가 다른 여러 문서를 한데 모은 **말뭉치**에서 TF와 IDF를 함께 계산해야 합니다. 이 장에서는 공개 라이선스로 배포된 한국어 AI 입문 문서 세 편을 직접 파일로 읽습니다.

한국어는 `이미지`, `이미지를`, `이미지는`처럼 조사가 붙습니다. 전문 형태소 분석기를 쓰면 이를 더 정교하게 나눌 수 있지만, 이번 활동에서는 원리를 눈으로 확인하는 것이 먼저입니다. `[가-힣]{2,}`라는 규칙으로 한글 두 글자 이상을 찾고, `있습니다`, `그리고`처럼 주제를 구별하기 어려운 자주 쓰는 말은 작은 불용어 목록에서 뺍니다. 따라서 결과는 완벽한 국어 분석이 아니라 **같은 단순 규칙으로 세 문서를 공정하게 비교한 결과**입니다.

### 이 장의 TF-IDF 튜토리얼 순서

공식만 읽고 넘어가지 않습니다. 아래 여섯 단계를 차례로 실행하면서 **어느 문서에서 어떤 단어가 몇 점을 받았는지** 끝까지 추적합니다.

1. **1단계. 비교할 문서 세 편 확인하기** — 제목, 글자 수, 출처를 확인합니다.
2. **2단계. 문서에서 분석할 한글 단어 고르기** — 단어를 뽑고 흔한 표현을 제외합니다.
3. **3단계. 단어 하나의 계산 과정 따라가기** — 등장 횟수, 전체 단어 수, TF, DF, IDF, TF-IDF를 차례로 계산합니다.
4. **4단계. 문서별 상위 TF-IDF 결과 비교하기** — 세 문서의 상위 단어와 실제 점수를 한 표에서 봅니다.
5. **5단계. 결과를 문서 내용과 연결해 해석하기** — 점수가 높은 까닭과 분석의 한계를 문장으로 설명합니다.
6. **6단계. 같은 방법을 급식 메뉴에 적용하기** — 가상 취향과 다섯 메뉴의 순위, 유사도, 공통 글자 조각 수를 확인합니다.

표의 소수는 읽기 쉽도록 반올림합니다. 계산에는 반올림하기 전 값을 사용하므로 손으로 계산한 값과 마지막 자리에서 조금 다를 수 있습니다.
""",
        hand_example="""
종이에 `파스타`를 쓰고 단어 앞뒤에 경계 표시 `·`를 붙여 `·파스타·`로 만드세요.

| 창 크기 | 창을 한 칸씩 옮긴 결과 |
|---|---|
| 2-gram | `·파`, `파스`, `스타`, `타·` |
| 3-gram | `·파스`, `파스타`, `스타·` |

이제 A=`파스타 피자`, B=`파스타`, C=`밥 국` 세 문서에서 `파스`와 `피자`를 비교합니다. 추천기에 쓰는 2~4글자 조각을 모두 세면 A는 15개, B는 9개, C는 6개입니다.

| 문서 | 조각 | 등장 횟수 | 전체 조각 수 | TF | DF | IDF | TF-IDF |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `파스` | 1 | 15 | 0.067 | 2 | 1.288 | 0.086 |
| A | `피자` | 1 | 15 | 0.067 | 1 | 1.693 | 0.113 |
| B | `파스` | 1 | 9 | 0.111 | 2 | 1.288 | 0.143 |
| B | `피자` | 0 | 9 | 0.000 | 1 | 1.693 | 0.000 |
| C | `파스` | 0 | 6 | 0.000 | 2 | 1.288 | 0.000 |
| C | `피자` | 0 | 6 | 0.000 | 1 | 1.693 | 0.000 |

`피자`는 한 문서에만 있어 IDF는 크지만 B와 C에는 실제로 등장하지 않아 TF-IDF가 0입니다. 반대로 B의 `파스`는 IDF가 조금 작아도 짧은 B 문서에서 차지하는 비율인 TF가 커서 0.143이 됩니다. TF-IDF는 **드문 단어이면 무조건 높은 값**이 아니라 TF와 IDF가 모두 있어야 만들어지는 점수입니다.
""",
        prediction="- `파스타`의 경계를 포함한 2-gram은 네 개이다.\n- 세 문서 중 한 문서에만 있는 `피자`의 IDF가 두 문서에 있는 `파스`의 IDF보다 크다.\n- 신경망 문서에는 `뉴런`, 컴퓨터 비전 문서에는 `이미지`, 책임 있는 AI 문서에는 `책임`이 높은 특징어로 나타날 것이다.\n- ‘파스타, 피자’를 좋아하는 가상 취향은 파스타·피자 메뉴와 가장 높은 유사도를 보인다.",
        code_sections=[
            (
                "움직이는 창으로 n-gram 만들기",
                SETUP_CODE
                + """
from neis_meal_ai.recommender import _char_ngrams, _tfidf_similarity
from IPython.display import display
import pandas as pd

ngram_word = "파스타"
padded_word = f" {ngram_word} "
two_grams = [
    padded_word[index : index + 2].replace(" ", "·")
    for index in range(len(padded_word) - 1)
]
three_grams = [
    padded_word[index : index + 3].replace(" ", "·")
    for index in range(len(padded_word) - 2)
]

print("경계를 표시한 단어:", padded_word.replace(" ", "·"))
print("2-gram:", two_grams)
print("3-gram:", three_grams)

example = "치즈 파스타"
grams = _char_ngrams(example)
print("\\n추천기가 만든 2~4글자 조각 일부:")
print(list(grams.items())[:15])
""",
                "두 칸짜리 창을 한 칸씩 옮겨 네 개의 2-gram을 만들었습니다. 추천기는 각 단어에 같은 방법을 적용하고, 2·3·4글자 조각을 모두 모읍니다.",
                """
1. `padded_word`는 단어 앞뒤에 공백을 붙여 경계를 표시합니다.<br>
2. `range(len(padded_word) - 1)`은 두 칸짜리 창이 시작할 위치를 만듭니다.<br>
3. `padded_word[index : index + 2]`는 현재 위치부터 두 글자를 자릅니다.<br>
4. 화면에서 경계가 보이도록 공백을 `·`로 바꿉니다. 실제 계산에서는 공백을 그대로 씁니다.<br>
5. `_char_ngrams(example)`은 같은 과정을 2·3·4글자 창으로 반복하고 조각별 횟수를 셉니다.
""",
            ),
            (
                "작은 세 문서의 TF-IDF 계산표 만들기",
                """
import math

tiny_document_rows = [
    {"문서": "A", "내용": "파스타 피자"},
    {"문서": "B", "내용": "파스타"},
    {"문서": "C", "내용": "밥 국"},
]
tiny_documents = [row["내용"] for row in tiny_document_rows]
tiny_counts = [_char_ngrams(text) for text in tiny_documents]
terms_to_compare = ["파스", "피자"]

print("[비교할 작은 문서]")
display(pd.DataFrame(tiny_document_rows))

toy_tfidf_table = []
for row, counts in zip(tiny_document_rows, tiny_counts):
    total_ngrams = sum(counts.values())
    for compared_term in terms_to_compare:
        count = counts[compared_term]
        tf = count / total_ngrams
        df = sum(compared_term in other for other in tiny_counts)
        idf = math.log((len(tiny_documents) + 1) / (df + 1)) + 1
        tfidf = tf * idf
        toy_tfidf_table.append(
            {
                "문서": row["문서"],
                "조각": compared_term,
                "등장 횟수": count,
                "전체 조각 수": total_ngrams,
                "TF": round(tf, 3),
                "DF": df,
                "IDF": round(idf, 3),
                "TF-IDF": round(tfidf, 3),
                "계산": (
                    f"{count} ÷ {total_ngrams} × {idf:.3f} "
                    f"= {tfidf:.3f}"
                ),
            }
        )

print("[TF-IDF 계산표]")
display(pd.DataFrame(toy_tfidf_table))
""",
                "A·B·C의 두 조각을 모두 계산해 여섯 줄의 표로 만들었습니다. 등장하지 않은 조각은 TF가 0이므로 IDF가 커도 TF-IDF는 0입니다. B의 `파스`는 짧은 문서 안에서 차지하는 비율이 커서 A의 `파스`보다 높은 값을 받습니다.",
                """
1. `tiny_document_rows`는 문서 이름과 내용을 함께 저장해 어느 결과가 어느 문서에서 나왔는지 잃지 않게 합니다.<br>
2. `tiny_counts`는 각 문서의 2~4글자 조각과 등장 횟수를 셉니다.<br>
3. 바깥쪽 `for`는 문서 A·B·C를, 안쪽 `for`는 `파스`·`피자`를 차례로 살핍니다.<br>
4. `count / total_ngrams`가 TF이고, `sum(compared_term in other ...)`가 DF입니다.<br>
5. `tf * idf`로 최종 TF-IDF를 구한 뒤 계산식까지 `계산` 열에 남깁니다.<br>
6. `pd.DataFrame(...)`은 여섯 줄의 계산 결과를 읽기 쉬운 표로 보여 줍니다.
""",
            ),
            (
                "A 문서의 ‘피자’ 점수를 한 줄씩 따라가기",
                """
term = "피자"
term_count_in_a = tiny_counts[0][term]
all_ngram_count_in_a = sum(tiny_counts[0].values())
term_tf_in_a = term_count_in_a / all_ngram_count_in_a
term_df = sum(term in counts for counts in tiny_counts)
rare_idf = math.log(
    (len(tiny_documents) + 1) / (term_df + 1)
) + 1
rare_tfidf_in_a = term_tf_in_a * rare_idf

common_df = sum("파스" in counts for counts in tiny_counts)
common_idf = math.log(
    (len(tiny_documents) + 1) / (common_df + 1)
) + 1

print("1. A 문서의 '피자' 등장 횟수:", term_count_in_a)
print("2. A 문서의 전체 2~4글자 조각 수:", all_ngram_count_in_a)
print(
    "3. TF =",
    f"{term_count_in_a} ÷ {all_ngram_count_in_a}",
    "=",
    round(term_tf_in_a, 4),
)
print("4. DF = '피자'를 포함한 문서 수 =", term_df)
print(
    "5. IDF =",
    f"log((3+1)/({term_df}+1))+1",
    "=",
    round(rare_idf, 4),
)
print(
    "6. TF-IDF =",
    f"{term_tf_in_a:.4f} × {rare_idf:.4f}",
    "=",
    round(rare_tfidf_in_a, 4),
)
print("\\n비교: '파스'의 DF =", common_df, "· IDF =", round(common_idf, 4))
""",
                "A 문서의 `피자`는 1회, 전체 조각은 15개이므로 TF는 0.0667입니다. 세 문서 가운데 A에만 있으므로 DF는 1, IDF는 1.6931입니다. 두 값을 곱한 TF-IDF는 0.1129입니다. `파스`는 A와 B 두 문서에 있어 IDF가 1.2877로 더 작습니다.",
                """
1. `term_count_in_a`는 TF의 분자, `all_ngram_count_in_a`는 분모입니다.<br>
2. TF 계산을 `1 ÷ 15`처럼 실제 숫자로 출력해 공식과 결과를 연결합니다.<br>
3. `term_df`는 `피자`가 있는 문서만 세므로 1입니다.<br>
4. 세 문서라는 뜻의 3과 DF 1을 IDF 식에 넣어 1.6931을 얻습니다.<br>
5. `term_tf_in_a * rare_idf`는 0.0667과 1.6931을 곱해 0.1129를 만듭니다.<br>
6. 마지막 줄은 두 문서에 나타난 `파스`의 IDF가 더 작은지 비교합니다.
""",
            ),
            (
                "한국어 문서 세 편의 TF-IDF를 문서별로 계산하기",
                """
from collections import Counter
from hashlib import sha256
import json
import math
import re

corpus_folder = PROJECT_ROOT / "data" / "tfidf_korean_documents"
manifest = json.loads(
    (corpus_folder / "sources.json").read_text(encoding="utf-8")
)

documents = []
verification_results = []
for source in manifest["documents"]:
    document_path = corpus_folder / source["file"]
    document_text = document_path.read_text(encoding="utf-8")
    document_text = document_text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    document_bytes = document_text.encode("utf-8")
    actual_sha256 = sha256(document_bytes).hexdigest()
    is_verified = actual_sha256 == source["sha256"]
    verification_results.append(is_verified)
    if not is_verified:
        raise ValueError(f"원문 확인 실패: {source['file']}")
    documents.append((source["title"], document_text))
    print(
        source["title"],
        "→ 파일:", source["file"],
        "· 글자:", len(document_text),
        "· 원문 확인:", is_verified,
    )

stop_words = {
    "그리고", "그러나", "하지만", "또한", "대한", "대해", "위해",
    "통해", "있는", "있습니다", "있으며", "합니다", "됩니다",
    "입니다", "것을", "것이", "같은", "이러한", "다음", "가장",
    "사용", "사용할", "사용하여", "우리는", "여기서", "이를",
}

def count_korean_words(text):
    words = re.findall(r"[가-힣]{2,}", text)
    return Counter(word for word in words if word not in stop_words)

document_counts = [count_korean_words(text) for _, text in documents]
document_count = len(document_counts)
all_words = set().union(*(counts.keys() for counts in document_counts))
word_df = {
    word: sum(word in counts for counts in document_counts)
    for word in all_words
}

document_summaries = []
for (title, text), counts in zip(documents, document_counts):
    document_summaries.append(
        {
            "문서": title,
            "글자 수": len(text),
            "분석 단어 수": sum(counts.values()),
            "서로 다른 단어 수": len(counts),
        }
    )

print("\\n[문서별 기본 정보]")
display(pd.DataFrame(document_summaries))

document_top_terms = {}
document_term_details = {}
all_document_term_rows = []
for (title, _), counts in zip(documents, document_counts):
    total_words = sum(counts.values())
    calculated_rows = []
    for word, count in counts.items():
        tf = count / total_words
        idf = math.log((document_count + 1) / (word_df[word] + 1)) + 1
        calculated_rows.append(
            {
                "단어": word,
                "등장 횟수": count,
                "전체 단어 수": total_words,
                "TF": tf,
                "DF": word_df[word],
                "IDF": idf,
                "TF-IDF": tf * idf,
            }
        )

    ranked_rows = sorted(
        calculated_rows,
        key=lambda row: (-row["TF-IDF"], row["단어"]),
    )[:8]
    detail_rows = []
    for rank, row in enumerate(ranked_rows, 1):
        detail = {
            "순위": rank,
            "단어": row["단어"],
            "등장 횟수": row["등장 횟수"],
            "전체 단어 수": row["전체 단어 수"],
            "TF": round(row["TF"], 4),
            "DF": row["DF"],
            "IDF": round(row["IDF"], 4),
            "TF-IDF": round(row["TF-IDF"], 4),
            "계산": (
                f"{row['등장 횟수']} ÷ {row['전체 단어 수']} "
                f"× {row['IDF']:.4f} = {row['TF-IDF']:.4f}"
            ),
        }
        detail_rows.append(detail)
        all_document_term_rows.append({"문서": title, **detail})

    document_term_details[title] = detail_rows
    document_top_terms[title] = [row["단어"] for row in detail_rows]
    print(f"\\n[{title}] 상위 TF-IDF 계산표")
    display(pd.DataFrame(detail_rows))

comparison_words = ["뉴런", "이미지", "책임", "모델"]
cross_document_comparison = []
for compared_word in comparison_words:
    compared_df = word_df.get(compared_word, 0)
    compared_idf = math.log(
        (document_count + 1) / (compared_df + 1)
    ) + 1
    for (title, _), counts in zip(documents, document_counts):
        total_words = sum(counts.values())
        count = counts[compared_word]
        tf = count / total_words
        cross_document_comparison.append(
            {
                "단어": compared_word,
                "문서": title,
                "등장 횟수": count,
                "TF": round(tf, 4),
                "DF": compared_df,
                "IDF": round(compared_idf, 4),
                "TF-IDF": round(tf * compared_idf, 4),
            }
        )

print("\\n[같은 단어를 세 문서에서 비교]")
display(pd.DataFrame(cross_document_comparison))

document_sources_verified = all(verification_results)
""",
                """
#### 왜 이 단어의 점수가 높을까?

세 문서의 실제 1위는 다음과 같습니다.

| 문서 | 1위 단어 | 등장 횟수 | TF | DF | IDF | TF-IDF |
|---|---|---:|---:|---:|---:|---:|
| 신경망 소개 | 뉴런 | 6 | 0.0202 | 1 | 1.6931 | 0.0342 |
| 컴퓨터 비전 소개 | 이미지 | 18 | 0.0316 | 2 | 1.2877 | 0.0407 |
| 윤리적이고 책임 있는 AI | 책임 | 6 | 0.0153 | 1 | 1.6931 | 0.0260 |

`이미지`는 두 문서에 있어 IDF가 `뉴런`이나 `책임`보다 작습니다. 그래도 컴퓨터 비전 문서에서 18번 나타나 TF가 크기 때문에 세 1위 가운데 가장 높은 0.0407이 됩니다. 이처럼 최종 순위는 IDF 하나가 아니라 TF와 IDF의 곱으로 정해집니다.
""",
                """
1. `sources.json`에서 파일명·제목·원문 SHA-256을 읽습니다.<br>
2. 운영체제마다 다른 줄바꿈을 `LF`로 통일한 뒤 `sha256(document_bytes).hexdigest()`로 지문을 만들고 기록된 값과 비교합니다.<br>
3. `re.findall(r"[가-힣]{2,}", text)`는 한글 두 글자 이상인 덩어리만 찾습니다.<br>
4. `Counter`는 문서 안에서 각 단어가 몇 번 나왔는지 세어 TF의 재료를 만듭니다.<br>
5. `document_summaries`는 문서별 글자 수, 분석 단어 수, 서로 다른 단어 수를 먼저 보여 줍니다.<br>
6. `calculated_rows`에는 단어마다 등장 횟수부터 TF-IDF까지 계산한 값을 모두 저장합니다.<br>
7. `ranked_rows`는 TF-IDF가 큰 순서로 정렬하고 앞의 여덟 단어를 선택합니다.<br>
8. `계산` 열은 `6 ÷ 297 × 1.6931 = 0.0342`처럼 실제 숫자가 공식에 들어가는 과정을 남깁니다.<br>
9. `cross_document_comparison`은 같은 단어가 세 문서에서 몇 번 나타나고 얼마의 점수를 받는지 나란히 보여 줍니다.<br>
10. 이 결과는 이 세 문서 안에서의 상대적인 특징이며, 단어의 절대적인 중요도를 판정한 것이 아닙니다.
""",
            ),
            (
                "가상 취향과 다섯 메뉴를 점수 순서로 비교하기",
                """
query = "파스타 피자 면"
menu_texts = meal_df["menu_text"].tolist()
similarity_array = _tfidf_similarity(menu_texts, query)
similarities = [round(float(value), 3) for value in similarity_array]

query_terms = query.split()
query_ngram_set = set(_char_ngrams(query))
menu_similarity_ranking = []
for date, score, menu in zip(meal_df["date"], similarities, menu_texts):
    menu = str(menu)
    direct_hits = [term for term in query_terms if term in menu]
    common_ngrams = query_ngram_set.intersection(_char_ngrams(menu))
    menu_similarity_ranking.append(
        {
            "날짜": str(date),
            "유사도": score,
            "직접 포함 단어": ", ".join(direct_hits) if direct_hits else "없음",
            "공통 글자 조각 수": len(common_ngrams),
            "메뉴": menu,
        }
    )

menu_similarity_ranking.sort(
    key=lambda row: (-row["유사도"], row["날짜"])
)
for rank, row in enumerate(menu_similarity_ranking, 1):
    row["순위"] = rank

ranking_columns = [
    "순위", "날짜", "유사도", "직접 포함 단어", "공통 글자 조각 수", "메뉴"
]
print("[가상 취향과 메뉴의 TF-IDF 유사도 순위]")
display(pd.DataFrame(menu_similarity_ranking)[ranking_columns])

chapter_result = {
    "chapter": "03",
    "similarities": similarities,
    "query": query,
    "two_grams": two_grams,
    "common_idf": common_idf,
    "rare_idf": rare_idf,
    "document_count": document_count,
    "document_summaries": document_summaries,
    "document_term_details": document_term_details,
    "document_top_terms": document_top_terms,
    "document_sources_verified": document_sources_verified,
    "cross_document_comparison": cross_document_comparison,
    "menu_similarity_ranking": menu_similarity_ranking,
    "toy_tfidf_table": toy_tfidf_table,
}
""",
                """
2026-06-24 식단은 `미트볼로제파스타`와 `피자`를 직접 포함하고 공통 글자 조각도 많아 0.141로 1위입니다. 0.018처럼 작은 양수는 완전한 단어가 같지 않더라도 짧은 문자 조각 일부가 겹칠 때 나올 수 있습니다. 0.000은 이번 가상 취향과 비교할 공통 특징이 거의 없다는 뜻입니다.

유사도는 **텍스트 표현의 가까움**입니다. 0.141이 0.018보다 가상 취향과 더 비슷하다는 뜻이지, 14.1점짜리 음식이거나 더 건강하다는 뜻은 아닙니다.
""",
                """
1. `query`는 실제 학생 정보가 아닌 가상 취향 문장입니다.<br>
2. `tolist()`는 표의 메뉴 열을 문장 목록으로 바꿉니다.<br>
3. `_tfidf_similarity`는 취향과 각 메뉴의 TF-IDF 벡터 방향 가까움을 한 번에 계산합니다.<br>
4. `direct_hits`는 `파스타`, `피자`, `면`이 메뉴명에 그대로 들어 있는지 확인합니다.<br>
5. `common_ngrams`는 완전한 단어가 같지 않아도 함께 나타난 2~4글자 조각을 셉니다.<br>
6. `sort(...)`는 유사도가 큰 메뉴부터 정렬하고, `순위` 열에는 1부터 번호를 붙입니다.<br>
7. 마지막 표에서 어느 날짜가 얼마의 점수를 받았고 어떤 메뉴였는지 한 줄씩 확인합니다.
""",
            ),
        ],
        exercise_text="query의 단어를 ‘밥 국물’ 또는 자신이 고른 가상 취향으로 바꾸고 가장 높은 날짜를 찾으세요.",
        exercise_code="""
practice_query = "밥 국물"
practice_scores = _tfidf_similarity(meal_df["menu_text"].tolist(), practice_query)
best_index = int(practice_scores.argmax())
print("가상 취향:", practice_query)
print("가장 비슷한 메뉴:", meal_df.iloc[best_index]["menu_text"])
print("유사도:", round(float(practice_scores[best_index]), 3))
""",
        check_questions=[
            "`파스타`의 경계를 포함한 문자 2-gram 네 개를 적어 보세요.",
            "TF의 분자와 분모는 각각 무엇인가요?",
            "DF는 한 조각의 총 반복 횟수와 어떻게 다른가요?",
            "DF가 작아지면 IDF가 커지는 까닭을 식의 나눗셈과 연결해 설명해 보세요.",
            "TF-IDF는 어떤 두 값을 곱한 것인가요?",
            "실제 한국어 문서에서 불용어를 빼는 까닭은 무엇인가요?",
            "문서의 SHA-256을 확인하는 까닭은 무엇인가요?",
            "TF-IDF 유사도가 높으면 반드시 맛있거나 건강하다는 뜻인가요?",
        ],
        check_answer="""
1. `·파`, `파스`, `스타`, `타·`입니다.<br>
2. 분자는 그 문서 안의 해당 조각 횟수, 분모는 그 문서의 전체 n-gram 수입니다.<br>
3. DF는 같은 문서 안에서 몇 번 반복됐는지가 아니라 그 조각을 포함한 문서가 몇 개인지를 셉니다.<br>
4. DF가 작아지면 `(전체 문서 수+1)/(DF+1)`의 분모가 작아져 나눈 값과 IDF가 커집니다.<br>
5. 한 문서 안에서의 비율인 TF와 여러 문서에서의 희소성인 IDF를 곱합니다.<br>
6. `있습니다`, `그리고`처럼 여러 글에 흔하지만 주제를 구별하는 데 도움이 적은 말을 제외하기 위해서입니다.<br>
7. 내려받은 파일이 준비할 때 확인한 원문과 같은지, 빠지거나 바뀌지 않았는지 확인하기 위해서입니다.<br>
8. 아닙니다. 입력한 취향 글자와 메뉴 글자의 특징이 비슷하다는 뜻뿐입니다.
""",
        summary=[
            "문자 n-gram은 n칸짜리 창을 한 칸씩 옮겨 만드는 연속 글자 조각이다.",
            "TF는 한 문서 안의 비율, DF는 그 조각을 포함한 문서 수다.",
            "IDF는 여러 문서에서 드문 조각에 더 큰 값을 준다.",
            "TF-IDF는 한 메뉴에서 자주 나오면서 전체에서는 드문 특징을 크게 본다.",
            "출처가 다른 실제 문서도 같은 단어 분리 규칙과 TF-IDF 식으로 특징어를 비교할 수 있다.",
            "SHA-256은 내려받은 문서가 준비한 원문과 같은지 확인하는 파일 지문이다.",
            "텍스트 유사도는 취향 표현의 가까움이지 정답이 아니다.",
        ],
        next_text="04장에서는 벡터의 방향을 비교하고 숫자 특징이 비슷한 식단을 묶습니다.",
    )


def _chapter_04() -> dict:
    return _lesson(
        number="04",
        title="코사인 유사도와 식단 군집",
        question="비슷한 메뉴와 식단 묶음은 어떻게 찾을까?",
        session="3회차 후반",
        minutes=90,
        objectives=[
            "코사인 유사도를 벡터 방향의 가까움으로 설명할 수 있다.",
            "K-Means가 정답표 없이 중심을 옮기며 묶는 과정을 설명할 수 있다.",
            "군집 이름이 건강 등급이 아니라 상대적 설명임을 구분할 수 있다.",
        ],
        connection="두 메뉴에 비슷한 글자 조각이 많으면 숫자 화살표도 비슷한 방향을 가리킵니다. 방향이 가까운 메뉴를 찾고, 영양 수치의 모양이 비슷한 날짜끼리 묶으면 식단의 특징을 다른 관점에서 볼 수 있습니다.",
        keywords=[
            ("코사인 유사도", "두 벡터의 방향이 얼마나 비슷한지 나타내는 값"),
            ("군집", "특징이 비슷해 한 묶음으로 분류된 데이터"),
            ("K-Means", "가까운 중심을 찾아 반복해서 묶는 알고리즘"),
            ("표준화", "단위가 다른 숫자를 비교 가능한 크기로 바꾸는 일"),
        ],
        concept="""
두 학생이 같은 방향을 바라보면 키가 달라도 방향은 같습니다. 코사인 유사도는 벡터의 길이보다 방향을 비교합니다.

K-Means는 운동장에 몇 개의 깃발을 놓고 학생들이 가장 가까운 깃발로 모이는 모습을 떠올리면 됩니다. 모인 학생들의 가운데로 깃발을 옮기고 다시 모으기를 반복합니다.

이 프로젝트의 군집은 열량·탄수화물·단백질·지방·메뉴 수의 상대 패턴입니다. ‘가벼운 구성’과 ‘든든한 구성’은 데이터 안의 비교 이름일 뿐 좋고 나쁨이 아닙니다.
""",
        hand_example="""
점 A(1, 1), B(2, 2), C(1, 5)를 좌표에 찍어 보세요. A와 B는 같은 방향이지만 C는 다른 방향입니다. 코사인 유사도는 A와 B를 가깝게 봅니다.

K-Means 중심 이동도 숫자로 한 번 해 봅니다. A(1, 1), B(2, 2), C(8, 8), D(9, 9)가 있고 첫 중심을 A와 D에 놓습니다. 가까운 중심에 배정하면 A·B와 C·D로 나뉩니다. 첫 묶음의 새 중심은 `((1+2)/2, (1+2)/2)=(1.5, 1.5)`, 둘째 묶음은 `(8.5, 8.5)`로 이동합니다. 다시 배정해 묶음이 그대로면 반복을 멈춥니다.
""",
        prediction="- 파스타 가상 취향의 1위는 파스타·피자가 있는 날짜다.\n- 식단은 최소 두 종류의 상대 군집 이름으로 나뉜다.",
        code_sections=[
            (
                "유사도 1위 찾기",
                SETUP_CODE
                + """
from neis_meal_ai.recommender import _tfidf_similarity, cluster_meals

query = "파스타 피자 면"
scores = _tfidf_similarity(meal_df["menu_text"].tolist(), query)
best_position = int(scores.argmax())
top_similar_menu = meal_df.iloc[best_position]["menu_text"]
print("가상 취향:", query)
print("가장 비슷한 메뉴:", top_similar_menu)
print("유사도:", round(float(scores[best_position]), 3))
""",
                "1위는 취향 글자와 가장 비슷한 메뉴입니다. 다른 취향을 넣으면 1위가 달라질 수 있습니다.",
                """
1. `_tfidf_similarity`는 취향 문장과 각 메뉴의 유사도를 계산합니다.<br>
2. `scores.argmax()`는 가장 큰 유사도가 놓인 위치 번호를 찾습니다.<br>
3. `iloc[best_position]`은 그 위치의 메뉴 행을 꺼냅니다.<br>
4. `round(..., 3)`은 원래 값을 바꾸지 않고 출력할 때만 셋째 자리로 반올림합니다.
""",
            ),
            (
                "K-Means 군집 결과 읽기",
                """
clustered_df = cluster_meals(meal_df, max_clusters=3)
print(clustered_df[["date", "calories", "protein_g", "cluster_name"]].to_string(index=False))
cluster_names = sorted(set(clustered_df["cluster_name"]))
print("나타난 군집:", cluster_names)

chapter_result = {
    "chapter": "04",
    "top_similar_menu": top_similar_menu,
    "cluster_names": cluster_names,
}
""",
                "군집명이 같으면 이번 5행의 영양 수치 패턴이 상대적으로 비슷하다는 뜻입니다. 학생 개인의 건강 상태는 입력하지 않았습니다.",
                """
`cluster_meals` 안에서는 다음 순서가 반복됩니다.<br>
1. 열량·탄수화물·단백질·지방·메뉴 수를 비슷한 크기로 표준화합니다.<br>
2. 임시 중심을 놓고 각 날짜를 가장 가까운 중심에 배정합니다.<br>
3. 묶인 날짜들의 평균 위치로 중심을 옮깁니다.<br>
4. 더는 묶음이 바뀌지 않으면 상대적인 설명 이름을 붙입니다. `set`은 중복 이름을 한 번씩만 남깁니다.
""",
            ),
        ],
        exercise_text="max_clusters를 2와 3으로 각각 실행하고 군집 이름과 날짜 묶음이 어떻게 달라지는지 기록하세요.",
        exercise_code="""
practice_cluster_count = 2
practice_clustered = cluster_meals(meal_df, max_clusters=practice_cluster_count)
print(practice_clustered[["date", "cluster_name"]].to_string(index=False))
""",
        check_questions=[
            "코사인 유사도가 벡터의 무엇을 비교하나요?",
            "K-Means에서 중심을 반복해서 옮기는 이유는 무엇인가요?",
            "‘상대적 가벼운 구성’을 건강한 메뉴라고 바꿔 말하면 안 되는 이유는 무엇인가요?",
        ],
        check_answer="""
1. 벡터의 방향을 비교합니다.<br>
2. 각 데이터와 가까운 묶음의 중심을 더 잘 찾기 위해서입니다.<br>
3. 작은 공개 데이터 안의 숫자 비교일 뿐 개인 건강 적합성을 판단하지 않았기 때문입니다.
""",
        summary=[
            "코사인 유사도는 숫자 벡터의 방향을 비교한다.",
            "K-Means는 가까운 중심으로 데이터를 반복해서 묶는다.",
            "군집은 상대 패턴 설명이며 건강 등급이 아니다.",
        ],
        next_text="05장에서는 유사도에 명시적인 보너스와 감점을 합쳐 설명 가능한 추천 점수를 만듭니다.",
    )


def _chapter_05() -> dict:
    return _lesson(
        number="05",
        title="개인추천 점수 설계",
        question="개인마다 다른 추천 순위를 어떻게 만들까?",
        session="4회차 전반",
        minutes=90,
        objectives=[
            "추천 점수의 기준점·보너스·감점을 계산할 수 있다.",
            "같은 급식도 가상 취향에 따라 순위가 달라지는 이유를 설명할 수 있다.",
            "가상 알레르기 번호가 점수 계산 전에 제외되는 이유를 말할 수 있다.",
        ],
        connection="같은 급식표를 보고도 어떤 학생은 면을, 어떤 학생은 밥과 국을 먼저 고릅니다. 추천기는 이 차이를 점수 규칙으로 나타냅니다. 어떤 조건이 점수를 올리고 내리는지 누구나 확인할 수 있어야 합니다.",
        keywords=[
            ("기준점", "보너스와 감점을 적용하기 전의 출발 점수"),
            ("가중치", "어떤 조건을 얼마나 크게 반영할지 정한 수"),
            ("클리핑", "결과를 정한 최소·최대 범위 안으로 자르는 일"),
            ("설명 가능한 AI", "결과가 나온 근거를 사용자가 확인할 수 있는 AI"),
        ],
        concept="""
추천 점수는 시험 성적이 아니라 정렬을 위한 상대 숫자입니다.

**20 + 70×텍스트 유사도 + 8×좋아함 일치 + 5×유형 일치 - 18×기피 일치 - 3×매운맛 차이**

마지막에는 0~100 사이로 자릅니다. 20점 기준점은 감점이 0점 아래로 바로 사라지지 않게 합니다. 가상 알레르기 번호는 점수를 낮추는 것이 아니라 후보에서 먼저 제외합니다.
""",
        hand_example="""
유사도 0.2, 좋아함 1개, 면 유형 1개, 기피 0개, 매운맛 차이 1이라면<br>
20 + 14 + 8 + 5 - 0 - 3 = 44점입니다. 손으로 다시 계산해 보세요.
""",
        prediction="- 파스타·피자·면을 좋아하는 가상 프로필의 1위 이유에 좋아하는 키워드가 표시된다.\n- 추천 결과는 3행이다.",
        code_sections=[
            (
                "첫 가상 프로필 추천",
                SETUP_CODE
                + """
from neis_meal_ai.recommender import PreferenceProfile, recommend_menus

profile = PreferenceProfile(
    likes=("파스타", "피자"),
    avoids=("오이",),
    preferred_types=("면", "디저트"),
    spice_level=2,
    allergy_codes=(),
)
recommendation = recommend_menus(meal_df, profile, top_n=3)
print(recommendation[["date", "score", "menu_text", "reason"]].to_string(index=False))
""",
                "reason 열에서 텍스트 유사도, 좋아하는 키워드, 유형, 기피, 매운맛 차이를 확인할 수 있습니다.",
                """
1. `PreferenceProfile`은 좋아함, 기피, 선호 유형, 매운맛, 가상 번호를 묶습니다.<br>
2. `recommend_menus`는 `meal_df`와 `profile`을 비교해 상위 세 행을 만듭니다.<br>
3. `reason` 열에는 각 점수가 만들어진 근거가 기록됩니다.
""",
            ),
            (
                "서로 다른 두 취향 비교",
                """
profile_b = PreferenceProfile(
    likes=("밥", "국"),
    avoids=(),
    preferred_types=("밥", "국물"),
    spice_level=3,
    allergy_codes=(),
)
recommendation_b = recommend_menus(meal_df, profile_b, top_n=3)
print("A의 1위:", recommendation.iloc[0]["menu_text"])
print("B의 1위:", recommendation_b.iloc[0]["menu_text"])

chapter_result = {
    "chapter": "05",
    "recommendations": len(recommendation),
    "top_score": float(recommendation.iloc[0]["score"]),
    "top_reason": str(recommendation.iloc[0]["reason"]),
}
""",
                "데이터는 같아도 입력 취향이 달라지면 유사도와 보너스가 바뀌어 순위가 달라집니다.",
                """
1. `profile_b`에는 첫 프로필과 다른 밥·국 선호를 적습니다.<br>
2. `recommendation_b`는 같은 급식 표에 두 번째 취향을 적용한 결과입니다.<br>
3. 두 표의 `iloc[0]`을 비교하면 각 프로필의 1위가 보입니다.
""",
            ),
        ],
        exercise_text="활동 1의 조건은 그대로 두고 `practice_like`만 ‘파스타’에서 ‘치킨’으로 바꾸어 1위와 추천 이유를 비교하세요. 실제 의료 정보는 적지 않습니다.",
        exercise_code="""
practice_like = "치킨"
practice_profile = PreferenceProfile(
    likes=(practice_like, "피자"),
    avoids=("오이",),
    preferred_types=("면", "디저트"),
    spice_level=2,
    allergy_codes=(),
)
practice_result = recommend_menus(meal_df, practice_profile, top_n=1)
print("기준 프로필 1위:", recommendation.iloc[0]["menu_text"])
print("좋아함만 바꾼 뒤:")
print(practice_result[["date", "score", "reason"]].to_string(index=False))
""",
        check_questions=[
            "20점 기준점은 왜 있나요?",
            "기피 키워드는 점수에 어떻게 반영되나요?",
            "가상 알레르기 번호 일치 메뉴를 낮은 점수로 남기지 않고 제외하는 이유는 무엇인가요?",
        ],
        check_answer="""
1. 감점 효과가 0점 하한에서 바로 사라지지 않게 하기 위해서입니다.<br>
2. 한 번 일치할 때마다 18점을 뺍니다.<br>
3. 주의가 필요한 후보가 낮은 순위라도 추천 목록에 남지 않게 하기 위해서입니다. 그래도 실제 안전을 보장하지는 않습니다.
""",
        summary=[
            "추천 점수는 공개된 가감 규칙을 사용한다.",
            "같은 데이터도 입력 취향에 따라 순위와 이유가 달라진다.",
            "점수는 취향 비교용이며 건강·의학 점수가 아니다.",
        ],
        next_text="06장에서는 추천 함수를 Jupyter 안에서 누를 수 있는 화면으로 연결합니다.",
    )


def _chapter_06() -> dict:
    return _lesson(
        number="06",
        title="Jupyter 추천 화면",
        question="추천 기능을 누를 수 있는 화면으로 만들 수 있을까?",
        session="4회차 후반",
        minutes=90,
        objectives=[
            "입력 위젯과 출력 영역의 역할을 구분할 수 있다.",
            "버튼 클릭이 추천 함수로 이어지는 흐름을 설명할 수 있다.",
            "로컬 Jupyter 화면에서 가상 취향을 바꾸어 추천을 실행할 수 있다.",
        ],
        connection="친구에게 추천기를 체험하게 하면서 매번 코드 속 글자를 고쳐 달라고 할 수는 없습니다. 좋아하는 메뉴를 입력하고 버튼을 누르면 결과가 나타나는 화면을 만들면, 같은 추천 함수를 더 쉽게 사용할 수 있습니다.",
        keywords=[
            ("위젯", "클릭하거나 값을 입력할 수 있는 화면 부품"),
            ("콜백", "버튼 같은 사건이 생겼을 때 실행되는 함수"),
            ("상태", "화면에 현재 들어 있는 값"),
            ("프로토타입", "핵심 기능을 시험하는 초기 완성품"),
        ],
        concept="""
자동판매기의 버튼을 누르면 선택 정보가 내부 기능으로 전달되고 결과가 나옵니다. Jupyter 위젯도 같습니다.

이 판은 외부 공개 주소를 만들지 않고 현재 PC의 Jupyter 화면 안에서 작동합니다. 이름·학번 칸은 만들지 않으며 실제 알레르기·질병 정보도 입력하지 않습니다.
""",
        hand_example="""
종이에 입력 상자 두 개, 선택 상자, 슬라이더, 버튼, 결과 표를 그린 뒤 각 부품에서 추천 함수의 어느 매개변수로 값이 가는지 화살표를 그어 보세요.
""",
        prediction="- 좋아하는 메뉴, 기피 메뉴, 유형, 매운맛, 가상 번호 입력 부품이 보인다.\n- 기본 입력 콜백은 추천표 3행을 만든다.",
        code_sections=[
            (
                "추천 콜백 만들기",
                SETUP_CODE
                + """
from neis_meal_ai.service import run_recommendation

def recommend_from_inputs(likes, avoids, menu_types, spice, fake_allergies):
    return run_recommendation(
        meal_df,
        likes_text=likes,
        avoids_text=avoids,
        preferred_types=menu_types,
        spice_level=int(spice),
        allergy_codes=[int(code) for code in fake_allergies],
        top_n=3,
    )

preview_summary, preview_table = recommend_from_inputs(
    "파스타, 피자", "오이", ["면"], 2, []
)
print(preview_summary)
print(preview_table.to_string(index=False))
""",
                "콜백은 화면 값을 기존 추천 함수가 이해하는 형식으로 바꿉니다. 기본 호출에서 표 3행이 나오면 기능 연결이 준비된 것입니다.",
                """
1. `run_recommendation`은 05장에서 완성한 추천 규칙을 가져옵니다.<br>
2. `recommend_from_inputs`는 화면의 글자·선택값을 추천 함수의 이름표와 연결합니다.<br>
3. `top_n=3`은 결과를 세 행으로 제한합니다.<br>
4. `preview_...` 호출은 화면을 만들기 전에 추천 기능 자체부터 확인합니다.
""",
            ),
            (
                "입력 위젯 만들기",
                """
import os
import ipywidgets as widgets
from IPython.display import display

likes_widget = widgets.Text(value="파스타, 피자", description="좋아함")
avoids_widget = widgets.Text(value="오이", description="피함")
types_widget = widgets.SelectMultiple(
    options=["밥", "면", "국물", "튀김", "디저트"],
    value=("면",),
    description="유형",
)
spice_widget = widgets.IntSlider(value=2, min=1, max=5, description="매운맛")
allergy_widget = widgets.SelectMultiple(
    options=[str(code) for code in range(1, 20)],
    value=(),
    description="가상 번호",
)
run_button = widgets.Button(description="추천 실행", button_style="primary")
output_widget = widgets.Output()
callback_state = {"status": "not_run", "rows": 0, "message": ""}
print("입력 화면 준비 완료: 글자 입력 2개, 선택 상자 2개, 슬라이더 1개, 버튼 1개")
""",
                "Text는 글자 입력, SelectMultiple은 여러 항목 선택, IntSlider는 1~5 범위 선택을 맡습니다. `callback_state`는 버튼이 실제로 성공했는지 시험하기 위한 작은 기록장입니다.",
                """
1. `widgets.Text` 두 개는 좋아함과 피함을 쉼표로 입력받습니다.<br>
2. `SelectMultiple` 두 개는 메뉴 유형과 가상 알레르기 번호를 여러 개 선택합니다.<br>
3. `IntSlider`는 범위를 벗어난 매운맛 값이 들어오지 않게 합니다.<br>
4. Button은 사건을 만들고 Output은 안내문과 추천표가 나타날 자리를 만듭니다.
""",
            ),
            (
                "버튼 콜백 연결과 화면 조립",
                """

def on_recommend_clicked(_button):
    with output_widget:
        output_widget.clear_output()
        try:
            summary, table = recommend_from_inputs(
                likes_widget.value,
                avoids_widget.value,
                list(types_widget.value),
                spice_widget.value,
                list(allergy_widget.value),
            )
        except (TypeError, ValueError) as error:
            message = f"입력을 고쳐 주세요: {error}"
            callback_state.update(status="error", rows=0, message=message)
            print(message)
            return

        message = f"데이터 출처: {data_source}\\n{summary}"
        callback_state.update(status="success", rows=len(table), message=message)
        print(message)
        display(table)

run_button.on_click(on_recommend_clicked)
recommender_ui = widgets.VBox([
    widgets.HTML(
        "<h3>우리 학교 급식 AI 개인추천기</h3>"
        "<p>이름·학번·실제 의료 정보는 입력하지 않습니다.</p>"
    ),
    likes_widget,
    avoids_widget,
    types_widget,
    spice_widget,
    allergy_widget,
    run_button,
    output_widget,
])

if os.getenv("NEIS_JUPYTER_VERIFY") != "1":
    display(recommender_ui)
else:
    on_recommend_clicked(None)
    print("Jupyter 위젯 콜백 실행 완료")

chapter_result = {
    "chapter": "06",
    "widget_ready": isinstance(recommender_ui, widgets.Widget),
    "callback_rows": callback_state["rows"],
    "callback_status": callback_state["status"],
    "callback_source": data_source,
}
""",
                "화면은 로컬 Jupyter 안에 표시됩니다. 버튼은 콜백을 호출하고 콜백은 기존 추천 함수를 사용하므로 코드 셀 결과와 같은 규칙을 따릅니다.",
                """
1. `on_recommend_clicked`는 버튼을 눌렀을 때만 실행할 작업 묶음입니다.<br>
2. `try` 안에서는 입력값을 읽고, 잘못된 값이면 `except`가 한국어 수정 안내를 보여 줍니다.<br>
3. 성공하면 데이터 출처·안전 안내·추천표를 같은 출력 칸에 표시합니다.<br>
4. `on_click(...)`이 버튼과 함수를 연결하고, `VBox`가 부품을 위에서 아래로 배열합니다.<br>
5. 검증 모드에서는 콜백을 직접 한 번 실행해 행 수와 상태를 결과 계약에 기록합니다.
""",
            ),
        ],
        exercise_text="기본 화면의 다른 입력은 그대로 두고 `practice_spice`만 2에서 3으로 바꾸어 1위와 추천 이유를 비교하세요.",
        exercise_code="""
baseline_summary, baseline_table = recommend_from_inputs(
    "파스타, 피자", "오이", ["면"], 2, []
)
practice_spice = 3
practice_summary, practice_table = recommend_from_inputs(
    "파스타, 피자", "오이", ["면"], practice_spice, []
)
print("기준 매운맛 2의 1위:", baseline_table.iloc[0]["메뉴"])
print("매운맛만 3으로 바꾼 결과:")
print(practice_summary)
print(practice_table[["순위", "날짜", "추천 점수", "추천 이유"]].to_string(index=False))
""",
        check_questions=[
            "버튼을 눌렀을 때 실행되는 함수를 무엇이라고 하나요?",
            "화면 코드를 추천 알고리즘과 분리하면 어떤 장점이 있나요?",
            "이 화면에 실제 알레르기 정보를 입력하면 안 되는 이유는 무엇인가요?",
        ],
        check_answer="""
1. 콜백 함수라고 합니다.<br>
2. 추천 규칙을 한 곳에서 시험하고 화면만 따로 바꿀 수 있습니다.<br>
3. 수업용 프로토타입은 의료 안전을 보장하지 않으며 민감한 개인 정보를 수집하지 않기로 했기 때문입니다.
""",
        summary=[
            "위젯은 입력과 출력을 담당하고 콜백이 기능을 연결한다.",
            "Jupyter판 화면은 외부 공유 링크 없이 현재 PC에서 작동한다.",
            "실제 의료 정보가 아닌 가상 입력으로 동작만 시험한다.",
        ],
        next_text="07장에서는 여러 입력 사례를 자동으로 시험하고 모델 카드로 한계를 공개합니다.",
    )


def _chapter_07() -> dict:
    return _lesson(
        number="07",
        title="테스트와 모델 카드",
        question="추천기를 어떻게 시험하고 한계를 설명할까?",
        session="5회차",
        minutes=180,
        objectives=[
            "테스트의 예상·실행·판정 구조를 설명할 수 있다.",
            "입력을 바꾼 네 가지 사례로 추천 규칙을 확인할 수 있다.",
            "모델 카드에 목적·데이터·방법·한계·금지 사용을 기록할 수 있다.",
        ],
        connection="추천 버튼이 한 번 작동한 것만으로는 충분하지 않습니다. 취향이 비어 있거나 범위를 벗어난 값이 들어왔을 때도 정해 둔 방식으로 움직이는지 확인합니다. 모델 카드에는 시험 결과와 함께 이 추천기가 하지 못하는 일도 적습니다.",
        keywords=[
            ("테스트 사례", "특정 입력과 예상 결과를 짝지은 시험"),
            ("회귀", "고친 기능이 이후 변경으로 다시 망가지는 현상"),
            ("모델 카드", "AI의 목적, 데이터, 방법, 한계를 공개하는 설명서"),
            ("책임 있는 AI", "사람에게 미칠 영향과 한계를 고려하는 개발 태도"),
        ],
        concept="""
과학 실험은 예상하고, 실행하고, 결과를 비교합니다. 소프트웨어 테스트도 같습니다. ‘오이를 피하면 오이 메뉴 점수가 내려간다’처럼 구체적인 예상이 있어야 판정할 수 있습니다.

모델 카드는 제품 설명서와 비슷합니다. 잘하는 것만 쓰지 않고 데이터가 적다는 점, 실제 만족도를 모른다는 점, 의료 판단에 쓰면 안 된다는 점도 함께 공개합니다.
""",
        hand_example="""
‘좋아함을 파스타로 입력한다’라는 시험에 대해 예상 결과를 한 문장으로 적으세요. ‘잘 된다’보다 ‘파스타가 포함된 메뉴의 점수 또는 이유가 상승한다’가 더 좋은 예상입니다.
""",
        prediction="- 네 가지 시험 사례가 모두 통과한다.\n- 모델 카드의 다섯 필수 항목이 채워진다.",
        code_sections=[
            (
                "네 가지 추천 시험",
                SETUP_CODE
                + """
from neis_meal_ai.recommender import PreferenceProfile, recommend_menus, validate_profile

test_records = []

plain = recommend_menus(meal_df, PreferenceProfile((), (), (), 2, ()), top_n=3)
test_records.append(("빈 취향도 실행", len(plain) == 3))

liked = recommend_menus(
    meal_df, PreferenceProfile(("파스타",), (), ("면",), 2, ()), top_n=3
)
test_records.append(("파스타 좋아함 이유 표시", "파스타" in liked.iloc[0]["reason"]))

fake_allergy = recommend_menus(
    meal_df, PreferenceProfile((), (), (), 2, (1,)), top_n=5
)
test_records.append((
    "가상 번호 1 메뉴 제외",
    all(1 not in codes for codes in fake_allergy["allergy_codes"]),
))

try:
    validate_profile(PreferenceProfile((), (), (), 7, ()))
    invalid_spice_rejected = False
except ValueError:
    invalid_spice_rejected = True
test_records.append(("범위 밖 매운맛 차단", invalid_spice_rejected))

for name, passed in test_records:
    print("PASS" if passed else "FAIL", "-", name)
assert all(passed for _, passed in test_records)
""",
                "각 시험은 바뀐 입력이 어떤 결과를 만들어야 하는지 확인합니다. 실패가 나오면 발표 전에 원인을 찾아야 합니다.",
                """
1. `test_records`에는 시험 이름과 통과 여부를 한 쌍으로 저장합니다.<br>
2. `plain`, `liked`, `fake_allergy`는 서로 다른 입력 조건의 추천 결과입니다.<br>
3. `assert all(...)`은 네 시험 중 하나라도 실패하면 실행을 멈춥니다.
""",
            ),
            (
                "모델 카드 만들기",
                """
model_card = {
    "목적": "익명 가상 취향과 NEIS 메뉴 표현을 비교한 상대 추천",
    "데이터": "남악고 NEIS 공개 급식 예비 데이터 5행",
    "방법": "문자 n-gram TF-IDF, 코사인 유사도, 공개 가감점, 작은 K-Means",
    "한계": "실제 만족도, 숨은 재료, 개인 건강 적합도를 알 수 없음",
    "금지 사용": "의료 판단, 실제 알레르기 안전 결정, 학생 평가",
    "안전 문구": (
        "추천 결과는 취향 비교용입니다. 실제 식단과 알레르기 정보는 "
        "학교 급식표와 영양사 안내를 다시 확인하세요."
    ),
}
for key, value in model_card.items():
    print(f"- {key}: {value}")

chapter_result = {
    "chapter": "07",
    "tests_passed": sum(passed for _, passed in test_records),
    "model_card_complete": all(bool(value) for value in model_card.values()),
}
""",
                "모델 카드는 AI를 과장하지 않도록 목적과 금지 사용을 동시에 보여 줍니다. 발표 화면에도 안전 문구를 남깁니다.",
                """
1. `model_card`는 목적, 데이터, 방법, 한계, 금지 사용, 안전 문구를 한 사전에 모읍니다.<br>
2. `items()`로 각 항목을 빠짐없이 출력합니다.<br>
3. `all(bool(value) ...)`는 빈 항목이 있는지 확인합니다.
""",
            ),
        ],
        exercise_text="아래 팀 한계 문장을 메뉴 데이터의 부족한 점이 드러나도록 구체화하세요.",
        exercise_code="""
team_limit = "메뉴 이름에 적히지 않은 재료와 실제 학생 만족도를 알 수 없다."
print("우리 팀이 발표할 한계:", team_limit)
""",
        check_questions=[
            "좋은 테스트 예상은 왜 구체적이어야 하나요?",
            "모델 카드에 잘하는 것뿐 아니라 한계도 쓰는 이유는 무엇인가요?",
            "이 추천기를 학생의 건강 판단에 쓰면 안 되는 이유 두 가지를 말해 보세요.",
        ],
        check_answer="""
1. 실행 결과가 통과인지 실패인지 분명히 판정하기 위해서입니다.<br>
2. 사용자가 결과의 범위와 위험을 알고 과장해서 사용하지 않도록 하기 위해서입니다.<br>
3. 데이터가 5행으로 적고, 개인 건강 정보·숨은 재료·실제 만족도를 사용하지 않았기 때문입니다.
""",
        summary=[
            "테스트는 예상·실행·판정을 구체적으로 기록한다.",
            "입력 경계와 안전 제외 규칙도 시험한다.",
            "모델 카드는 AI의 목적과 한계, 금지 사용을 함께 공개한다.",
        ],
        next_text="08장에서는 완성된 프로젝트를 문제부터 한계까지 8개 구간으로 발표합니다.",
    )


def _chapter_08() -> dict:
    return _lesson(
        number="08",
        title="발표와 상호 체험",
        question="결과가 아니라 과정을 어떻게 설명할까?",
        session="6회차",
        minutes=180,
        objectives=[
            "프로젝트를 문제→데이터→AI→결과→한계 순서로 설명할 수 있다.",
            "시연 전에 필요한 항목을 확인할 수 있다.",
            "다른 가상 취향으로 체험한 결과를 근거와 함께 기록할 수 있다.",
        ],
        connection="발표를 듣는 사람은 완성 화면만 보고서는 추천이 어떻게 만들어졌는지 알기 어렵습니다. 문제, 데이터 출처, 계산 원리, 시험 결과, 한계를 차례로 보여 주어야 프로젝트를 정확하게 설명할 수 있습니다.",
        keywords=[
            ("시연", "실제 화면을 실행해 기능을 보여 주는 발표"),
            ("근거", "주장을 뒷받침하는 데이터·코드·실행 결과"),
            ("한계", "현재 데이터와 방법으로 할 수 없는 범위"),
            ("회고", "과정에서 배운 점과 다음 개선을 돌아보는 일"),
        ],
        concept="""
좋은 AI 발표는 ‘1위 메뉴가 이것입니다’에서 끝나지 않습니다. 어떤 문제를 골랐고, 데이터가 어디에서 왔고, 글자가 어떻게 숫자가 되었고, 추천 이유를 어떻게 시험했는지 보여 줍니다.

결과가 예상과 다르더라도 지우지 않습니다. 왜 그런 결과가 나왔는지 데이터와 규칙으로 설명하면 발표의 좋은 근거가 됩니다.
""",
        hand_example="""
친구에게 30초 안에 API, TF-IDF, 추천 점수 중 하나를 설명해 보세요. 전문 용어를 말한 뒤 바로 생활 속 비유를 붙이면 이해하기 쉬워집니다.
""",
        prediction="- 발표 흐름이 8개 구간으로 정리된다.\n- 시연 전 확인표와 관람자 체험 기록 틀이 준비된다.",
        code_sections=[
            (
                "8개 발표 구간 만들기",
                """
presentation_sections = [
    "1. 우리 학교에서 발견한 질문",
    "2. NEIS API와 공식 학교 코드",
    "3. JSON 기초와 원본 전처리",
    "4. 문자 n-gram TF-IDF와 코사인 유사도",
    "5. K-Means와 공개 추천 점수",
    "6. Jupyter 개인추천기 시연",
    "7. 테스트 결과",
    "8. 개인정보·한계·다음 버전",
]
for section in presentation_sections:
    print(section)
""",
                "8개 구간은 결과만 강조하지 않고 문제와 출처, 원리, 시험, 한계를 균형 있게 보여 줍니다.",
                """
1. `presentation_sections`는 발표 순서를 여덟 문장으로 정리한 목록입니다.<br>
2. `for section in presentation_sections`는 목록의 순서대로 한 줄씩 출력합니다.<br>
3. 앞 장의 결과를 어느 구간에서 보여 줄지 각 문장과 연결합니다.
""",
            ),
            (
                "시연 전 확인표",
                """
demo_checklist = {
    "00~07장과 부록 A 저장": True,
    "06장 새 커널에서 실행": True,
    "가상 입력만 사용": True,
    "예비 데이터 출처 표시": True,
    "안전 문구 표시": True,
    "실제 의료 정보 없음": True,
}
for item, ready in demo_checklist.items():
    print("확인" if ready else "점검 필요", "-", item)

chapter_result = {
    "chapter": "08",
    "presentation_sections": len(presentation_sections),
    "demo_checklist_ready": all(demo_checklist.values()),
}
""",
                "시연이 멈춰도 05장의 추천 표와 07장의 테스트·모델 카드로 원리를 설명할 수 있도록 준비합니다.",
                """
1. `demo_checklist`는 확인 항목과 준비 여부를 참·거짓으로 묶습니다.<br>
2. `ready`가 참이면 ‘확인’, 거짓이면 ‘점검 필요’를 출력합니다.<br>
3. `all(demo_checklist.values())`는 모든 항목이 준비되었는지 한 번에 판정합니다.
""",
            ),
        ],
        exercise_text="관람자가 넣은 가상 취향과 바뀐 1위, 추천 이유를 아래 세 변수에 기록하세요.",
        exercise_code="""
visitor_fake_preference = "밥, 국"
changed_top_menu = "실행 후 기록"
evidence_sentence = "추천 이유 열에서 좋아함 일치와 유형 일치를 확인했다."
print("가상 취향:", visitor_fake_preference)
print("바뀐 1위:", changed_top_menu)
print("근거:", evidence_sentence)
""",
        check_questions=[
            "AI 발표에서 데이터 출처를 말해야 하는 이유는 무엇인가요?",
            "예상과 다른 결과를 숨기지 않고 설명하면 어떤 장점이 있나요?",
            "발표 중 실제 알레르기 정보를 입력하지 않는 이유는 무엇인가요?",
        ],
        check_answer="""
1. 결과가 어떤 자료에 근거했는지 신뢰성과 한계를 판단할 수 있기 때문입니다.<br>
2. 알고리즘과 데이터의 한계를 발견하고 다음 개선 방향을 제시할 수 있습니다.<br>
3. 민감한 개인 정보이며 이 수업용 추천기는 의료 안전을 보장하지 않기 때문입니다.
""",
        summary=[
            "발표는 문제, 데이터, AI 원리, 결과, 테스트, 한계를 연결한다.",
            "시연 전에는 새 커널 실행과 가상 입력 원칙을 확인한다.",
            "체험 결과를 순위만이 아니라 추천 이유와 함께 기록한다.",
        ],
        next_text="여기까지 실행했다면 프로젝트가 끝났습니다. 내가 직접 바꾼 입력, 예상과 달랐던 결과, 이제 설명할 수 있는 개념을 회고에 남깁니다.",
    )


CHAPTER_BUILDERS: tuple[Callable[[], dict], ...] = (
    _chapter_00,
    _appendix_json,
    _chapter_01,
    _chapter_02,
    _chapter_03,
    _chapter_04,
    _chapter_05,
    _chapter_06,
    _chapter_07,
    _chapter_08,
)


def build_textbook(output_dir: str | Path = DEFAULT_OUTPUT) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, builder in zip(CHAPTER_FILES, CHAPTER_BUILDERS, strict=True):
        path = output / filename
        path.write_text(
            json.dumps(builder(), ensure_ascii=False, indent=1),
            encoding="utf-8",
            newline="\n",
        )
        paths.append(path)
    return paths


def main() -> int:
    paths = build_textbook()
    print(f"Jupyter 교과서 {len(paths)}개 장 생성 완료")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
