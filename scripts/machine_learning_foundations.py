"""머신러닝 기초 부록에 들어갈 교재 본문을 정의한다."""

# noqa: SIZE_OK - 실행 로직이 아니라 생성할 노트북의 수업 본문 데이터다.

from __future__ import annotations

from types import MappingProxyType
from typing import Final


MACHINE_LEARNING_LESSON: Final = MappingProxyType(
    {
        "number": "B",
        "title": "머신러닝 기초와 원-핫 벡터",
        "question": "데이터에서 규칙을 배운 컴퓨터는 처음 보는 값의 결과를 어떻게 예측할까?",
        "objectives": [
            "표의 한 행을 학습 예시로 보고 특성(X)과 정답(y)을 구분할 수 있다.",
            "음식 이름을 원-핫 벡터로 바꾸어 계산 가능한 특성으로 만들 수 있다.",
            "학습 데이터의 평균으로 만든 값을 모델의 가중치로 해석할 수 있다.",
            "원-핫 벡터와 가중치의 내적으로 예상 평점을 계산할 수 있다.",
            "실제 평점과 예측 평점의 절대오차를 구하고 모델의 한계를 설명할 수 있다.",
            "멀티-핫과 TF-IDF가 원-핫 표현을 어떻게 넓히는지 구분할 수 있다.",
        ],
        "connection": """
한 학생이 밥에는 4점과 5점, 국에는 3점과 4점을 주었다고 해 봅시다.
새 식단에 김치가 나왔을 때 아직 점수를 받지 못했더라도, 앞서 모은 평가에서
김치 점수의 규칙을 찾아 예상값을 만들 수 있습니다. 머신러닝은 이런 식으로
**예시에서 규칙을 배우고, 새 입력의 결과를 예측하는 과정**입니다.
""",
        "keywords": [
            ("데이터셋", "같은 기준으로 모은 여러 학습 예시의 표"),
            ("특성(X)", "예측할 때 컴퓨터에 알려 주는 입력 정보"),
            ("정답(y)", "입력과 함께 기록된 실제 결과 또는 목표값"),
            ("인코딩", "이름이나 문장을 계산할 수 있는 숫자로 바꾸는 일"),
            ("원-핫 벡터", "선택한 범주의 자리만 1이고 나머지는 0인 숫자 목록"),
            ("학습", "여러 예시에서 예측에 사용할 규칙이나 값을 정하는 과정"),
            ("예측", "학습된 규칙을 새 입력에 적용해 아직 모르는 결과를 계산하는 일"),
            ("오차", "실제값과 예측값이 얼마나 떨어져 있는지 나타낸 값"),
        ],
        "concept": """
### 1단계. 문제를 입력과 출력으로 나눈다

이번 문제는 **메뉴를 보고 예상 평점을 구하는 것**입니다. 입력은 메뉴 이름이고,
출력은 1점부터 5점까지의 평점입니다. 머신러닝에서는 입력을 보통 `X`, 정답을
`y`라고 씁니다. 알파벳 자체보다 역할이 중요합니다.

| 역할 | 이 예제의 값 | 뜻 |
|---|---|---|
| 특성 `X` | 메뉴 이름 | 예측할 때 알려 주는 정보 |
| 정답 `y` | 실제 평점 | 모델이 맞히려는 결과 |

표의 한 행은 `메뉴 하나 → 실제 평점 하나`로 이루어진 학습 예시입니다. 행이
여러 개 모이면 데이터셋이 됩니다.

### 2단계. 글자를 계산 가능한 숫자로 바꾼다

컴퓨터가 `김치`라는 글자를 평점과 바로 곱할 수는 없습니다. 먼저 음식 순서를
`["밥", "국", "김치", "돈까스"]`로 정하고, 선택한 음식의 칸만 1로 켭니다.

| 메뉴 | 밥 | 국 | 김치 | 돈까스 |
|---|---:|---:|---:|---:|
| 김치 | 0 | 0 | 1 | 0 |

이 숫자 목록이 **원-핫 벡터**입니다. 벡터의 각 자리가 무엇을 뜻하는지 정한
순서표를 함께 보관해야 숫자를 다시 음식 이름으로 읽을 수 있습니다.

### 3단계. 학습 데이터에서 값을 정한다

아주 단순한 모델부터 시작해 봅시다. 음식별로 받은 평점의 평균을 그 음식의
**학습된 값**으로 사용합니다.

| 메뉴 | 학습 평점 | 학습된 평균 |
|---|---|---:|
| 밥 | 4, 5 | 4.5 |
| 국 | 3, 4 | 3.5 |
| 김치 | 5, 5 | 5.0 |
| 돈까스 | 4, 4 | 4.0 |

어휘표 순서대로 평균을 나열하면 `[4.5, 3.5, 5.0, 4.0]`입니다. 이 예제에서는
이 네 값을 모델의 **가중치**로 봅니다. 복잡한 모델도 데이터에서 예측에 필요한
값을 정한다는 중심 생각은 같습니다.

### 4단계. 학습된 값으로 예측한다

김치의 원-핫 벡터와 학습된 가중치를 같은 자리끼리 곱하고 모두 더합니다.

`[0, 0, 1, 0] · [4.5, 3.5, 5.0, 4.0] = 5.0`

0이 있는 자리는 계산에서 사라지고 김치 자리의 5.0만 남습니다. 따라서 이
모델의 김치 예상 평점은 5.0점입니다.

### 5단계. 보지 않은 정답으로 오차를 확인한다

학습에 쓰지 않고 따로 둔 김치 평가가 실제로 4.0점이었다고 합시다. 예상은
5.0점이므로 절대오차는 다음과 같습니다.

`절대오차 = |실제값 - 예측값| = |4.0 - 5.0| = 1.0`

오차가 0이면 정확히 맞았고, 숫자가 커질수록 더 멀리 빗나갔다는 뜻입니다.
학습에 사용한 값을 다시 맞히는 것보다, 학습할 때 보지 않은 예시로 확인하는
편이 새 데이터에서도 잘 작동하는지 판단하는 데 도움이 됩니다.

### 6단계. 더 많은 정보를 표현한다

원-핫은 메뉴 한 종류를 나타내기에는 간단하지만 음식 사이의 비슷한 정도는
표현하지 못합니다. 한 식단의 여러 메뉴를 함께 나타낼 때는 1을 여러 개 켠
**멀티-핫**을 사용할 수 있습니다. 다음 장의 TF-IDF는 한 걸음 더 나아가,
문서나 메뉴 이름에서 어떤 글자 조각이 특징적인지 실수 가중치로 나타냅니다.
""",
        "hand_example": """
아래 표에서 `X`와 `y`를 먼저 찾아봅시다.

| 메뉴 | 평점 |
|---|---:|
| 밥 | 4 |
| 국 | 3 |
| 김치 | 5 |

메뉴 열은 예측할 때 주어지는 특성 `X`, 평점 열은 맞히려는 정답 `y`입니다.
김치를 `[0, 0, 1, 0]`으로 바꾸면 글자였던 `X`가 계산 가능한 숫자가 됩니다.
""",
        "prediction": """
- 여덟 행의 학습 데이터에서 음식별 평균을 계산하면 네 개의 학습된 값이 생길 것이다.
- 김치의 원-핫 벡터는 `[0, 0, 1, 0]`이므로 김치 가중치만 예측에 남을 것이다.
- 김치의 학습 평균이 5.0이면 예상 평점도 5.0이 될 것이다.
- 따로 둔 실제 평점이 4.0이면 절대오차는 1.0이 될 것이다.
""",
        "code_sections": [
            (
                "학습 데이터에서 특성 X와 정답 y 찾기",
                """
import pandas as pd
from IPython.display import display

feature_name = "menu"
target_name = "rating"
training_df = pd.DataFrame(
    {
        feature_name: ["밥", "밥", "국", "국", "김치", "김치", "돈까스", "돈까스"],
        target_name: [4, 5, 3, 4, 5, 5, 4, 4],
    }
)
training_rows = len(training_df)
X = training_df[[feature_name]]
y = training_df[target_name]

print("학습 예시 수:", training_rows)
print("특성(X):", feature_name)
print("정답(y):", target_name)
display(training_df)
""",
                "표의 각 행은 메뉴와 실제 평점이 짝을 이룬 학습 예시입니다. `X`에는 메뉴 열, `y`에는 평점 열이 들어갑니다. 지금은 여덟 행을 학습에 사용합니다.",
                """
1. `DataFrame`은 같은 길이의 열을 모아 표를 만듭니다.<br>
2. `feature_name`과 `target_name`은 열 이름을 한곳에서 관리합니다.<br>
3. `X`는 입력인 메뉴 열을, `y`는 정답인 평점 열을 선택합니다.<br>
4. 아직 학습하거나 예측하지 않고 데이터의 역할부터 확인합니다.
""",
            ),
            (
                "메뉴 이름을 원-핫 벡터로 바꾸기",
                """
labels = ["밥", "국", "김치", "돈까스"]


def make_one_hot(target_food: str) -> list[int]:
    return [1 if label == target_food else 0 for label in labels]


selected_food = "김치"
one_hot = make_one_hot(selected_food)
selected_index = one_hot.index(1)
decoded_food = labels[selected_index]
ones_count = one_hot.count(1)

encoded_df = training_df.copy()
encoded_df["one_hot"] = encoded_df[feature_name].map(make_one_hot)

print("선택한 음식:", selected_food)
print("원-핫 벡터:", one_hot)
print("다시 읽은 음식:", decoded_food)
print("1의 개수:", ones_count)
display(encoded_df)
""",
                "김치는 어휘표의 세 번째 항목이므로 `[0, 0, 1, 0]`이 됩니다. 모든 학습 행의 메뉴도 같은 규칙으로 변환되며, 벡터의 1은 항상 하나입니다.",
                """
1. `labels`의 순서가 벡터 각 자리의 뜻을 정합니다.<br>
2. `make_one_hot()`은 목표 음식과 같은 자리에는 1, 다른 자리에는 0을 넣습니다.<br>
3. `.map()`은 메뉴 열의 모든 값에 같은 변환 함수를 적용합니다.<br>
4. 1의 위치를 `labels`에서 찾으면 숫자를 다시 음식 이름으로 읽을 수 있습니다.
""",
            ),
            (
                "음식별 평균을 학습된 가중치로 만들기",
                """
mean_by_food = training_df.groupby(feature_name)[target_name].mean()
learned_weights = [
    round(float(mean_by_food[label]), 1)
    for label in labels
]

weight_table = pd.DataFrame(
    {
        "메뉴 자리": labels,
        "학습된 가중치": learned_weights,
    }
)

print("학습된 가중치:", learned_weights)
display(weight_table)
""",
                "밥 4.5, 국 3.5, 김치 5.0, 돈까스 4.0이 어휘표와 같은 순서로 저장됩니다. 이 모델의 학습은 음식별 평균 네 개를 정하는 과정입니다.",
                """
1. `groupby()`는 같은 음식 이름을 한 묶음으로 모읍니다.<br>
2. `.mean()`은 각 묶음의 평점 평균을 계산합니다.<br>
3. 평균을 `labels` 순서대로 꺼내 벡터 자리를 맞춥니다.<br>
4. 이렇게 데이터에서 정해진 값이 이후 예측에 사용됩니다.
""",
            ),
            (
                "원-핫 벡터와 가중치로 평점 예측하기",
                """
contributions = [
    feature_value * weight
    for feature_value, weight in zip(one_hot, learned_weights)
]
predicted_rating = round(sum(contributions), 1)

prediction_table = pd.DataFrame(
    {
        "메뉴 자리": labels,
        "특성(X)": one_hot,
        "학습된 값": learned_weights,
        "X × 학습된 값": contributions,
    }
)

print("예측 메뉴:", selected_food)
print("예상 평점:", predicted_rating)
display(prediction_table)
""",
                "김치 자리만 특성값이 1이므로 그 자리의 학습된 값 5.0만 남습니다. 이 계산은 원-핫 벡터와 가중치 벡터의 내적입니다.",
                """
1. `zip()`은 원-핫 벡터와 가중치의 같은 자리를 묶습니다.<br>
2. 같은 자리끼리 곱한 네 값을 `contributions`에 저장합니다.<br>
3. 네 값을 모두 더한 결과가 모델의 예상 평점입니다.<br>
4. 표의 마지막 열을 보면 어떤 자리가 예측에 기여했는지 확인할 수 있습니다.
""",
            ),
            (
                "처음 보는 실제 평점으로 절대오차 확인하기",
                """
test_row = {feature_name: "김치", target_name: 4.0}
actual_rating = float(test_row[target_name])
absolute_error = round(abs(actual_rating - predicted_rating), 1)

evaluation_table = pd.DataFrame(
    [
        {
            "메뉴": test_row[feature_name],
            "예상 평점": predicted_rating,
            "실제 평점": actual_rating,
            "절대오차": absolute_error,
        }
    ]
)

print("실제 평점:", actual_rating)
print("예상 평점:", predicted_rating)
print("절대오차:", absolute_error)
display(evaluation_table)
""",
                "학습에 넣지 않은 김치 평점은 4.0이고 모델의 예상은 5.0이므로 절대오차는 1.0입니다. 이 한 번의 결과만으로 모델 전체의 성능을 단정할 수는 없습니다.",
                """
1. `test_row`는 학습 표와 분리해 둔 새 평가 한 건입니다.<br>
2. `abs()`는 실제값과 예측값의 차이를 항상 0 이상으로 만듭니다.<br>
3. 오차는 예측이 얼마나 빗나갔는지 보여 주지만, 여러 평가를 모아야 일반적인 성능을 판단할 수 있습니다.<br>
4. 작은 데이터에서는 한 사람의 새 평가만으로 평균이 크게 달라질 수 있습니다.
""",
            ),
            (
                "한 식단을 멀티-핫으로 나타내고 TF-IDF와 연결하기",
                """
meal_foods = ["밥", "김치", "돈까스"]
multi_hot = [1 if label in meal_foods else 0 for label in labels]
multi_hot_ones = multi_hot.count(1)

representation_table = pd.DataFrame(
    [
        {"표현": "김치 한 가지", "벡터": one_hot, "1의 개수": ones_count},
        {"표현": "밥·김치·돈까스", "벡터": multi_hot, "1의 개수": multi_hot_ones},
    ]
)
display(representation_table)

chapter_result = {
    "chapter": "B",
    "training_rows": training_rows,
    "feature_name": feature_name,
    "target_name": target_name,
    "labels": labels,
    "selected_food": selected_food,
    "one_hot": one_hot,
    "decoded_food": decoded_food,
    "ones_count": ones_count,
    "learned_weights": learned_weights,
    "predicted_rating": predicted_rating,
    "actual_rating": actual_rating,
    "absolute_error": absolute_error,
    "multi_hot": multi_hot,
    "multi_hot_ones": multi_hot_ones,
    "tutorial_steps": 6,
}
""",
                "한 식단은 여러 메뉴를 포함하므로 멀티-핫 벡터에는 1이 세 개 있습니다. TF-IDF에서는 포함 여부를 넘어서 글자 조각마다 서로 다른 중요도 가중치를 사용합니다.",
                """
1. `meal_foods`에 들어 있는 항목은 1, 없는 항목은 0으로 표시합니다.<br>
2. 원-핫은 한 범주, 멀티-핫은 여러 범주를 함께 나타냅니다.<br>
3. `chapter_result`에는 학습부터 평가까지 확인한 결과를 저장합니다.<br>
4. 다음 장에서는 메뉴 글자를 n-gram으로 나누고 TF-IDF 가중치를 계산합니다.
""",
            ),
        ],
        "exercise_text": """
`practice_food`를 `"국"`으로 실행한 뒤 `"돈까스"`로 바꿔 다시 실행하세요.
원-핫 벡터에서 1의 위치가 어떻게 달라지는지, 그에 따라 어떤 학습된 값이
예상 평점으로 선택되는지 기록합니다.
""",
        "exercise_code": """
practice_food = "국"
practice_vector = make_one_hot(practice_food)
practice_contributions = [
    feature_value * weight
    for feature_value, weight in zip(practice_vector, learned_weights)
]
practice_prediction = round(sum(practice_contributions), 1)

print("바꾼 음식:", practice_food)
print("새 원-핫 벡터:", practice_vector)
print("예상 평점:", practice_prediction)
""",
        "check_questions": [
            "이 예제에서 특성 X와 정답 y는 각각 무엇인가요?",
            "메뉴 이름을 원-핫 벡터로 바꾸는 까닭은 무엇인가요?",
            "김치 벡터 `[0, 0, 1, 0]`과 가중치 `[4.5, 3.5, 5.0, 4.0]`을 곱해 더하면 왜 5.0이 되나요?",
            "실제 평점 4.0과 예상 평점 5.0의 절대오차는 얼마인가요?",
            "학습 데이터에 없던 메뉴는 이 모델이 바로 예측하기 어려운 까닭은 무엇인가요?",
            "멀티-핫과 TF-IDF는 원-핫 표현을 각각 어떻게 넓히나요?",
        ],
        "check_answer": """
1. 특성 `X`는 메뉴 이름이고 정답 `y`는 실제 평점입니다.
2. 글자인 메뉴 이름을 곱셈과 덧셈에 사용할 수 있는 숫자 특성으로 바꾸기 위해서입니다.
3. 김치 자리만 `1 × 5.0`이고 나머지 자리는 `0 × 가중치`이므로 합이 5.0입니다.
4. `|4.0 - 5.0|`이므로 1.0입니다.
5. 어휘표에 자리가 없고 그 메뉴의 학습된 평균도 없기 때문입니다. 새 메뉴를
다루려면 데이터를 더 모으거나 메뉴의 재료·글자 같은 다른 특성을 사용해야 합니다.
6. 멀티-핫은 한 행에서 여러 범주의 자리를 켜고, TF-IDF는 글자 조각마다
특징적인 정도에 따라 서로 다른 실수 가중치를 줍니다.
""",
        "summary": [
            "머신러닝은 학습 예시에서 규칙을 정하고 새 입력의 결과를 예측하는 과정이다.",
            "특성 X는 입력, 정답 y는 모델이 맞히려는 실제 결과다.",
            "원-핫 벡터는 범주 이름을 계산 가능한 숫자 특성으로 바꾼다.",
            "이 예제의 학습은 음식별 평균을 가중치로 정하는 과정이다.",
            "예측값과 실제값의 절대오차로 한 예측이 얼마나 빗나갔는지 확인할 수 있다.",
            "작은 데이터와 원-핫만으로는 새 메뉴나 음식 사이의 비슷한 정도를 충분히 표현하기 어렵다.",
        ],
        "next_text": "03장에서는 음식 이름과 한국어 문서를 n-gram으로 나누고, 각 글자 조각의 특징적인 정도를 TF-IDF 가중치로 계산합니다.",
    }
)
