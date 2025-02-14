# Code Preparation for Interviews

### Setup `poetry` for Python package management

Poetry sometimes bricks. To overcome this problem, use:

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

Then, install all necessary packages using:

```bash
poetry config virtualenvs.in-project true
poetry install --no-root
```

Finally, activate the built environment:

```bash
poetry shell
```
