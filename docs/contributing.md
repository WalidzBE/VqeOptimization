# Contributing

Thanks for considering a contribution. The project focuses on clear, reproducible VQE workflows and modular
Hamiltonian construction.

## Development Setup
```bash
uv python install 3.12
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

## Guidelines
- Keep changes focused and aligned with the existing scientific scope.
- Add or update docstrings for new public functions and classes.
- Update the documentation pages for user-facing changes.
- Prefer deterministic examples by setting `--seed` where applicable.

## Documentation Build
```bash
uv run mkdocs build
```
