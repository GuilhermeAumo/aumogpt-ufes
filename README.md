## AumoGPT

This repository manages the finetuning of AumoGPT

### How to initialize
Initialize poetry:
```bash
eval $(poetry env activate)

// or

poetry shell
```

Install dependencies
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install --no-root
```