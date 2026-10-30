# Python environment

Always use the virtual environment in `venv/` at the repo root (Python 3.13). Do not use the
system Python or install packages globally.

- Run scripts with `venv/bin/python` (e.g. `venv/bin/python experiments_all/online_mlp.py`), or
  `source venv/bin/activate` first in the same shell command.
- Install packages with `venv/bin/pip install ...`.
- Dependencies are listed in `requirements.txt`. When adding a new dependency, install it into the
  venv and add it to `requirements.txt` with a pinned version.
- `venv/` is gitignored; never commit it.
