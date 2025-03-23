## AumoGPT

This repository manages the finetuning of AumoGPT

### How to initialize

#### Install Poetry environment

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) 

Test the command 
```
poetry shell
```

if you get 
```
The command "shell" does not exist. 
```

New version of poetry require you to install the shell apart, [here](https://github.com/python-poetry/poetry-plugin-shell) or : 
```
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

#### Finetune using Oumi

Follow the [README.md](https://github.com/GuilhermeAumo/aumogpt-ufes/blob/main/oumi/readme.md) present on the Oumi directory

#### Try the model 

To try any model:

```
python gradio_evaluate base_model_id adapter_id
```

##### Try QLoRA model

```
python3 gradio_evaluate.py "meta-llama/Llama-3.2-3B-Instruct" "aumoai/llama3.2-3B-qlora-oumi-aumogpt-adapter"
```

##### Try LoRA model

```
python3 gradio_evaluate.py "meta-llama/Llama-3.2-3B-Instruct" "aumoai/llama3.2-3B-lora-oumi-aumogpt-adapter"
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
