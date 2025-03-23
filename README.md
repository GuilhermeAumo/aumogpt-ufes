## AumoGPT

This repository manages the finetuning of AumoGPT

### How to initialize

#### Install Poetry environment

Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) 

Test the command :
```bash
poetry shell
```

if you get :
```bash
The command "shell" does not exist. 
```

New version of poetry require you to install the shell apart, [here](https://github.com/python-poetry/poetry-plugin-shell) or : 
```bash
poetry self add poetry-plugin-shell
```

Install the virtual environment :
```bash
poetry install
```

After installing, start environment :
```bash
poetry shell
```

#### Finetune using Oumi

Follow the [README.md](https://github.com/GuilhermeAumo/aumogpt-ufes/blob/main/oumi/readme.md) present on the Oumi directory, the first step can be skipped since Oumi is already present on [pyproject.toml](https://github.com/GuilhermeAumo/aumogpt-ufes/blob/main/pyproject.toml), which is executed by `poetry install`.

#### Try the model 

To try any model :

```bash
python gradio_evaluate base_model_id adapter_id
```

##### Try QLoRA model

```bash
python3 gradio_evaluate.py "meta-llama/Llama-3.2-3B-Instruct" "aumoai/llama3.2-3B-qlora-oumi-aumogpt-adapter"
```

##### Try LoRA model

```bash
python3 gradio_evaluate.py "meta-llama/Llama-3.2-3B-Instruct" "aumoai/llama3.2-3B-lora-oumi-aumogpt-adapter"
```
