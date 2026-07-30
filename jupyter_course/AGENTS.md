# Jupyter Textbook Guide

## Ownership

- `notebook_support.py` is handwritten runtime support shared by chapters and `web_app.py`.
- `00_설치_준비.md`, `README.md`, `교사용_운영안.md`, and `verification-report.md` are course/operator documentation.
- `chapters/*.ipynb` are generated, executed teaching artifacts owned by `scripts/build_jupyter_textbook.py` and `scripts/freeze_jupyter_outputs.py`.
- `verification/` outputs are disposable and are not textbook source.

`chapters/` has no separate guide because every notebook there is governed by this guide and `../scripts/AGENTS.md`.

## Chapter Set

The generated order is chapters `00` through `08` plus the JSON appendix `A` and one-hot-vector appendix `B`. Keep names, order, navigation, and learning progression synchronized with `CHAPTER_FILES` in the builder.

## Editing Workflow

For generated-cell or chapter-structure changes, edit the builder first and follow the ordered commands in `../scripts/AGENTS.md`. Do not hand-edit frozen outputs to make a check pass. The verifier executes every chapter in a fresh kernel and writes its own report under `verification/`.

## Course Constraints

- Examples must run from a fresh local setup using the documented project-local interpreter.
- Keep notebook support independent of the author's machine paths and current shell state.
- Preserve visible, pedagogically useful frozen outputs and their execution metadata.
- Use sample/fallback data for reproducible teaching. Live service access must remain optional.
- Do not place real names, student identifiers, health details, or credentials in cells or outputs.
- Keep explanations and UI strings clear for Korean introductory learners; retain safety notices around allergy and recommendation output.
- Keep each chapter independently executable from a fresh kernel; never rely on variables or imports left by a previous chapter.
- JSON, one-hot, and TF-IDF tutorials retain visible saved outputs, gradual examples, formula explanations, and source-to-vector interpretation.
- Never freeze a developer-specific absolute path or credential-file location into a notebook.

Tests under `tests/test_jupyter_*.py`, `tests/test_frozen_notebook_outputs.py`, `tests/test_jupyter_support.py`, and `tests/test_notebook.py` define the artifact contract.
