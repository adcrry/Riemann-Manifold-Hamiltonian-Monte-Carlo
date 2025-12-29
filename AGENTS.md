# Agent Instructions

## Project Context
- **Goal**: Reproduce Riemannian Hamiltonian Monte Carlo (RMHMC) paper results.
- **Language**: Python (>=3.9).
- **Structure**: `src/` layout with `pyproject.toml`.

## Commands
- **Install**: `pip install -e .` (editable mode)
- **Test**: `pytest`
- **Run Single Test**: `pytest tests/path/to/test_file.py::test_name`
- **Lint/Format**: `ruff check .` (recommended) or `black .`

## Code Style & Conventions
- **Formatting**: Follow PEP 8. Use `black` or `ruff` defaults.
- **Imports**: 
  1. Standard library
  2. Third-party (numpy, jax, etc.)
  3. Local (`rmhmc`)
- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes.
- **Types**: Use Python type hints heavily (e.g., `def foo(x: float) -> float:`).
- **Libraries**: Use `jax` for autodiff/gradients where possible, `numpy` for general math.

## Rules
- No `.cursor/rules` or `.github/copilot-instructions.md` found.
