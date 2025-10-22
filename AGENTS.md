# Repository Guidelines

## Project Structure & Module Organization
- `utils/`: Cryptographic primitives and helpers (e.g., `ImprovedPaillier.py`, `Threshold.py`, `SafeMul.py`).
- `TA/`: Trusted Authority logic (`TA.py`).
- `DO/`: Data Owner logic (`DO.py`).
- `CSP/`: Cloud/Computation Service Provider logic (`CSP.py`).
- `Test.py`: Example driver to run a local scenario.

Follow module naming as shown (lowercase with underscores for files; `CamelCase` for classes).

## Build, Test, and Development Commands
- Create venv (recommended): `python -m venv .venv` and `./.venv/Scripts/Activate.ps1` (Windows) or `source .venv/bin/activate` (Unix).
- Run example scenario: `python Test.py`.
- Lint (if installed): `flake8` or `ruff .`.
- Format (if installed): `black .`.

Pin dependencies per module import needs; no `requirements.txt` is tracked yet.

## Coding Style & Naming Conventions
- Python 3.9+ compatible; prefer standard library types and `int` for big integer math.
- Indentation: 4 spaces; line length target 100.
- Names: `snake_case` for functions/variables, `CamelCase` for classes, module files `lower_snake.py`.
- Avoid floating point in crypto logic; use integer arithmetic and explicit byte conversions.
- Add docstrings for public functions with argument/return descriptions.

## Testing Guidelines
- Place tests under `tests/` mirroring package paths: e.g., `tests/utils/test_improved_paillier.py`.
- Use `pytest` (preferred) or `unittest`. Run with `pytest -q`.
- Name tests `test_*.py`; aim for coverage of key math paths and edge cases (large moduli, zero/one, negative inputs where applicable).

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject, optional scope. Example: `feat(utils): add safe modular multiply`.
- Include a short body explaining rationale and constraints or security considerations.
- PRs: clear description, linked issues, minimal diff, include run/validation notes (e.g., `python Test.py` output) and any benchmarks.

## Security & Configuration Tips
- Never log secrets or private keys; scrub sensitive values in prints.
- Use deterministic seeds only in tests; document randomness sources.
- Validate inputs (key sizes, modulus properties) and raise explicit errors.

## Agent-Specific Instructions
- Keep changes minimal and scoped to the touched module.
- Preserve public APIs; update `Test.py` only to demonstrate behavior.
- Prefer pure functions, small units, and add tests alongside changes.
