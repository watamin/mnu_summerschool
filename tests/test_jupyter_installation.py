from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = {
    "ipykernel",
    "ipywidgets",
    "gradio",
    "matplotlib",
    "nbclient",
    "nbformat",
    "notebook",
    "numpy",
    "pandas",
    "requests",
}
INSTALL_COMMANDS = (
    "py -3 --version",
    "py -3 -m venv .venv",
    r".\.venv\Scripts\python.exe -m pip install --upgrade pip",
    r".\.venv\Scripts\python.exe -m pip install -r requirements-jupyter.txt",
    r".\.venv\Scripts\python.exe -m notebook",
)
WEB_COMMAND = r".\.venv\Scripts\python.exe web_app.py"


def _requirements() -> list[Requirement]:
    lines = (PROJECT_ROOT / "requirements-jupyter.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    return [
        Requirement(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_jupyter_requirements_match_the_verified_environment() -> None:
    requirements = _requirements()
    by_name = {requirement.name.casefold(): requirement for requirement in requirements}

    assert set(by_name) == REQUIRED_PACKAGES
    for name, requirement in by_name.items():
        assert requirement.specifier.contains(version(name)), (
            f"현재 검증 환경의 {name} {version(name)}이 {requirement.specifier} 범위 밖입니다."
        )


def test_plain_markdown_guide_orders_the_commands_needed_before_jupyter() -> None:
    guide = (PROJECT_ROOT / "jupyter_course" / "00_설치_준비.md").read_text(
        encoding="utf-8"
    )

    positions = [guide.index(command) for command in INSTALL_COMMANDS]
    assert positions == sorted(positions)
    assert "가상환경" in guide
    assert "설치 전에는 Jupyter Notebook을 열 수 없습니다" in guide
    assert "gradio" in guide.casefold()
    assert WEB_COMMAND in guide


def test_batch_launcher_files_are_removed() -> None:
    assert not (PROJECT_ROOT / "scripts" / "setup_jupyter.bat").exists()
    assert not (PROJECT_ROOT / "scripts" / "start_jupyter.bat").exists()


def test_students_reuse_one_working_folder_and_exclude_venv_from_backups() -> None:
    guides = [
        (PROJECT_ROOT / "jupyter_course" / filename).read_text(encoding="utf-8")
        for filename in ("README.md", "00_설치_준비.md")
    ]

    assert "처음 한 번만 작업 폴더" in guides[0]
    for guide in guides:
        assert "같은 작업 폴더를 계속 사용" in guide
        assert "백업할 때" in guide and "`.venv` 폴더" in guide and "제외" in guide


def test_student_surfaces_avoid_removed_batch_launcher_names() -> None:
    searchable_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "jupyter_course" / "README.md",
        PROJECT_ROOT / "jupyter_course" / "교사용_운영안.md",
        PROJECT_ROOT / "jupyter_course" / "notebook_support.py",
        PROJECT_ROOT / "scripts" / "build_jupyter_textbook.py",
        PROJECT_ROOT / "scripts" / "verify_jupyter_textbook.py",
    ]
    old_names = ("setup_jupyter.bat", "start_jupyter.bat")
    for path in searchable_paths:
        source = path.read_text(encoding="utf-8")
        assert not any(old_name in source for old_name in old_names), path
