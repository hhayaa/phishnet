# Contributing to phishnet

## Setup
```bash
git clone https://github.com/hhayaa/phishnet.git
cd phishnet && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing
```bash
make test
```

## Code style
- PEP 8, 100-char lines, type hints, docstrings
- Conventional commits: feat:, fix:, docs:, test:
