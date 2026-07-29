# Global Food Value Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a school-independent, one-row-per-food TF-IDF value ranking to the Mokpo meal service.

**Architecture:** Keep the aggregation and formula explanation as pure functions in `mokpo_analytics.py`. Precompute the top 50 rows for the Gradio tab, retain the existing school-specific analysis below it, and include the top 10 rows in the grounded NIM context.

**Tech Stack:** Python 3.12, pandas, NumPy, Gradio, pytest

## Global Constraints

- A food appears once in the global ranking even when it appears at multiple schools.
- Use `global TF = total food count / total corpus dish count`.
- Use `IDF = ln((1+school count)/(1+food school count))+1`.
- Rank by score descending, count descending, then food name ascending.
- Describe data value as corpus information, never nutrition, health, taste, or price.

---

### Task 1: Add global value calculation and explanation

**Files:**
- Modify: `src/neis_meal_ai/mokpo_analytics.py`
- Modify: `tests/test_mokpo_analytics.py`

**Interfaces:**
- Produces: `global_food_values(frame, *, top_n=None) -> pd.DataFrame`
- Produces: `global_food_value_explanation(values) -> str`

- [ ] Write a failing hand-calculated test for counts, global TF, IDF, score, and top-school attribution.
- [ ] Run the focused test and confirm the missing imports fail.
- [ ] Implement the two pure functions with input and `top_n` validation.
- [ ] Run all analytics tests and confirm they pass.

### Task 2: Put the global ranking in the service

**Files:**
- Modify: `src/neis_meal_ai/mokpo_ui.py`
- Modify: `tests/test_mokpo_ui.py`
- Modify: `README.md`
- Modify: `docs/교사용_목포_급식_AI_서비스.md`

**Interfaces:**
- The existing `school_value_analysis_callback` returns the global one-row-per-food table in its fifth output.
- `create_mokpo_app` shows a prefilled top-50 global table and formula before the school selector.
- `_school_chat_context` includes the global top 10 rows.

- [ ] Write failing callback and app-configuration tests for the unique global ranking and visible heading.
- [ ] Run the focused UI tests and confirm failure.
- [ ] Wire the pure functions into the callback, Gradio tab, and NIM context.
- [ ] Update the two operating documents with the global formula and interpretation.
- [ ] Run the full suite, local HTTP smoke, and secret scans.
- [ ] Commit and push the existing PR branch.
