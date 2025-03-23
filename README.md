## AumoGPT

This repository manages the finetuning of AumoGPT

### How to initialize

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) 

Test the command 
```bash
poetry shell
```

if you get 
```bash
The command "shell" does not exist. 
```

New version of poetry require you to install the shell apart, [here](https://github.com/python-poetry/poetry-plugin-shell) or : 
```bash
poetry self add poetry-plugin-shell
```

Install the virtual environment:
```
poetry install
```

After installing, start environment:
```
poetry shell
```

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
