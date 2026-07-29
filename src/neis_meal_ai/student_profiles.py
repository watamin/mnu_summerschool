"""교실 급식 평가 프로필과 평점을 SQLite에 보존한다."""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


_STUDENT_NAME_PATTERN = re.compile(r"[가-힣A-Za-z0-9 _-]{1,20}")
_QUESTION_COUNT = 30
_COMMON_COUNT = 15
_ROTATING_COUNT = 15
_POOL_SIZE = 45


@dataclass(frozen=True)
class SaveResult:
    """한 학생의 저장 직후 진행 상태."""

    saved_count: int
    total_questions: int
    complete: bool
    updated_at: str


def validate_student_name(name: object) -> str:
    """로그인 사용자 이름을 프로필 이름으로 쓸 수 있게 검증한다."""

    normalized = str(name or "").strip()
    if _STUDENT_NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "이름은 1~20자의 한글, 영문, 숫자, 공백, 밑줄, 붙임표만 사용할 수 있습니다."
        )
    return normalized


class StudentProfileStore:
    """학생별 질문 배정과 평점을 관리하는 작은 SQLite 저장소."""

    def __init__(self, db_path: str | Path, food_pool: Iterable[str]) -> None:
        foods = [str(food).strip() for food in food_pool]
        if len(foods) != _POOL_SIZE:
            raise ValueError("학생 평가용 음식 풀은 정확히 45개여야 합니다.")
        if any(not food for food in foods):
            raise ValueError("음식 이름은 비어 있을 수 없습니다.")
        if len(set(foods)) != len(foods):
            raise ValueError("학생 평가용 음식 풀에 중복 음식이 있습니다.")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.food_pool = tuple(foods)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS foods (
                    food_name TEXT PRIMARY KEY,
                    pool_order INTEGER NOT NULL UNIQUE,
                    is_core INTEGER NOT NULL CHECK (is_core IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    cohort_index INTEGER NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    profile_id INTEGER NOT NULL,
                    food_name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (profile_id, food_name),
                    UNIQUE (profile_id, position),
                    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id),
                    FOREIGN KEY (food_name) REFERENCES foods(food_name)
                );
                CREATE TABLE IF NOT EXISTS ratings (
                    profile_id INTEGER NOT NULL,
                    food_name TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (profile_id, food_name),
                    FOREIGN KEY (profile_id, food_name)
                        REFERENCES assignments(profile_id, food_name)
                );
                """
            )
            existing = connection.execute(
                "SELECT food_name FROM foods ORDER BY pool_order"
            ).fetchall()
            if not existing:
                connection.executemany(
                    "INSERT INTO foods(food_name, pool_order, is_core) VALUES (?, ?, ?)",
                    [
                        (food, index, int(index < _COMMON_COUNT))
                        for index, food in enumerate(self.food_pool)
                    ],
                )
            elif [row["food_name"] for row in existing] != list(self.food_pool):
                raise ValueError("기존 데이터베이스의 45개 음식 풀과 현재 데이터가 다릅니다.")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    def _ensure_profile(self, connection: sqlite3.Connection, name: str) -> int:
        row = connection.execute(
            "SELECT profile_id FROM profiles WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row is not None:
            return int(row["profile_id"])

        timestamp = self._now()
        cohort_row = connection.execute(
            "SELECT COALESCE(MAX(cohort_index), -1) + 1 AS next_index FROM profiles"
        ).fetchone()
        cohort_index = int(cohort_row["next_index"])
        cursor = connection.execute(
            """
            INSERT INTO profiles(name, cohort_index, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, cohort_index, timestamp, timestamp),
        )
        profile_id = int(cursor.lastrowid)

        common = list(self.food_pool[:_COMMON_COUNT])
        rotating_pool = list(self.food_pool[_COMMON_COUNT:])
        offset = (cohort_index * 5) % len(rotating_pool)
        rotating = [
            rotating_pool[(offset + index) % len(rotating_pool)]
            for index in range(_ROTATING_COUNT)
        ]
        connection.executemany(
            "INSERT INTO assignments(profile_id, food_name, position) VALUES (?, ?, ?)",
            [
                (profile_id, food, position)
                for position, food in enumerate(common + rotating, start=1)
            ],
        )
        return profile_id

    def load_survey(self, name: object) -> pd.DataFrame:
        """학생의 고정된 30개 질문과 이미 저장한 평점을 불러온다."""

        normalized = validate_student_name(name)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            profile_id = self._ensure_profile(connection, normalized)
            rows = connection.execute(
                """
                SELECT a.position, a.food_name, f.is_core, r.rating
                FROM assignments AS a
                JOIN foods AS f ON f.food_name = a.food_name
                LEFT JOIN ratings AS r
                  ON r.profile_id = a.profile_id AND r.food_name = a.food_name
                WHERE a.profile_id = ?
                ORDER BY a.position
                """,
                (profile_id,),
            ).fetchall()
            connection.commit()
        return pd.DataFrame.from_records(
            [
                {
                    "순서": int(row["position"]),
                    "음식": row["food_name"],
                    "구분": "공통" if row["is_core"] else "순환",
                    "평점": int(row["rating"]) if row["rating"] is not None else 0,
                }
                for row in rows
            ],
            columns=["순서", "음식", "구분", "평점"],
        )

    @staticmethod
    def _rating_value(value: object) -> int | None:
        if value is None or value == "":
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            raise ValueError("평점은 1부터 5까지의 정수여야 합니다.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("평점은 1부터 5까지의 정수여야 합니다.") from exc
        if math.isnan(numeric) or numeric == 0:
            return None
        if not numeric.is_integer() or not 1 <= numeric <= 5:
            raise ValueError("평점은 1부터 5까지의 정수여야 합니다.")
        return int(numeric)

    def save_ratings(
        self, name: object, rows: Iterable[Mapping[str, object]] | pd.DataFrame
    ) -> SaveResult:
        """답한 평점을 저장하고 빈 평점은 기존 응답에서도 지운다."""

        normalized = validate_student_name(name)
        records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
        parsed: list[tuple[str, int | None]] = []
        seen: set[str] = set()
        for row in records:
            food = str(row.get("음식", "")).strip()
            if not food or food in seen:
                raise ValueError("평가표의 음식 이름이 비었거나 중복되었습니다.")
            seen.add(food)
            parsed.append((food, self._rating_value(row.get("평점"))))

        timestamp = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            profile_id = self._ensure_profile(connection, normalized)
            assigned = {
                row["food_name"]
                for row in connection.execute(
                    "SELECT food_name FROM assignments WHERE profile_id = ?",
                    (profile_id,),
                )
            }
            unknown = [food for food, _ in parsed if food not in assigned]
            if unknown:
                raise ValueError(f"배정되지 않은 음식은 저장할 수 없습니다: {unknown[0]}")
            for food, rating in parsed:
                if rating is None:
                    connection.execute(
                        "DELETE FROM ratings WHERE profile_id = ? AND food_name = ?",
                        (profile_id, food),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO ratings(profile_id, food_name, rating, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(profile_id, food_name) DO UPDATE SET
                            rating = excluded.rating,
                            updated_at = excluded.updated_at
                        """,
                        (profile_id, food, rating, timestamp),
                    )
            connection.execute(
                "UPDATE profiles SET updated_at = ? WHERE profile_id = ?",
                (timestamp, profile_id),
            )
            count_row = connection.execute(
                "SELECT COUNT(*) AS count FROM ratings WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            connection.commit()
        saved_count = int(count_row["count"])
        return SaveResult(
            saved_count=saved_count,
            total_questions=_QUESTION_COUNT,
            complete=saved_count == _QUESTION_COUNT,
            updated_at=timestamp,
        )

    def status(self) -> pd.DataFrame:
        """프로필별 저장 진행률을 만든다."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.name, COUNT(r.rating) AS saved_count, p.updated_at
                FROM profiles AS p
                LEFT JOIN ratings AS r ON r.profile_id = p.profile_id
                GROUP BY p.profile_id
                ORDER BY p.cohort_index
                """
            ).fetchall()
        return pd.DataFrame.from_records(
            [
                {
                    "이름": row["name"],
                    "저장 문항": int(row["saved_count"]),
                    "전체 문항": _QUESTION_COUNT,
                    "완료": "완료" if row["saved_count"] == _QUESTION_COUNT else "진행 중",
                    "마지막 저장": row["updated_at"],
                }
                for row in rows
            ],
            columns=["이름", "저장 문항", "전체 문항", "완료", "마지막 저장"],
        )

    def rating_matrix(self) -> pd.DataFrame:
        """모든 프로필과 45개 음식으로 NaN 포함 평점 행렬을 만든다."""

        status = self.status()
        matrix = pd.DataFrame(
            index=pd.Index(status["이름"].tolist(), name="이름"),
            columns=list(self.food_pool),
            dtype=float,
        )
        if matrix.empty:
            return matrix
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.name, r.food_name, r.rating
                FROM ratings AS r
                JOIN profiles AS p ON p.profile_id = r.profile_id
                ORDER BY p.cohort_index, r.food_name
                """
            ).fetchall()
        for row in rows:
            matrix.loc[row["name"], row["food_name"]] = float(row["rating"])
        return matrix

    def export_ratings(self, path: str | Path) -> Path:
        """관측 평점을 긴 형식 UTF-8 BOM CSV로 내보낸다."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.name, r.food_name, r.rating, r.updated_at
                FROM ratings AS r
                JOIN profiles AS p ON p.profile_id = r.profile_id
                ORDER BY p.cohort_index, r.food_name
                """
            ).fetchall()
        frame = pd.DataFrame.from_records(
            [
                {
                    "이름": row["name"],
                    "음식": row["food_name"],
                    "평점": int(row["rating"]),
                    "저장 시각": row["updated_at"],
                }
                for row in rows
            ],
            columns=["이름", "음식", "평점", "저장 시각"],
        )
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        return destination
