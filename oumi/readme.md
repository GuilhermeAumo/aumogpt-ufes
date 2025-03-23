## OUMI
Tool for finetuning, evaluation, etc.

### How to install OUMI
```bash
pip install oumi[gpu]
```

### How to fine tune a chatbot model

1. Create dataset in the format json:

```json
[
    {
        "messages": [
            {
                "content": "O AumoGPT consegue transformar dados de relat\u00f3rios em PDF em gr\u00e1ficos e tabelas?",
                "role": "user"
            },
            {
                "content": "Sim, ele pode transformar dados extra\u00eddos de PDFs em gr\u00e1ficos e tabelas, criando visualiza\u00e7\u00f5es claras e precisas para an\u00e1lise.",
                "role": "assistant"
            }
        ]
    },
    {
        "messages": [
            {
                "content": "O AumoGPT pode sugerir melhorias em relat\u00f3rios t\u00e9cnicos?",
                "role": "user"
            },
            {
                "content": "Sim, ele pode identificar \u00e1reas de melhoria e sugerir altera\u00e7\u00f5es para tornar os relat\u00f3rios t\u00e9cnicos mais claros e precisos.",
                "role": "assistant"
            }
        ]
    }
    ...
]
```

In our case, we created the dataset like [this](https://github.com/GuilhermeAumo/aumogpt-ufes/blob/main/datasets/AUMOGPT-dataset-final.json), and created the datasets used using the [python script](https://github.com/GuilhermeAumo/aumogpt-ufes/blob/main/generate_finetuning_dataset.py) `python3 generate_finetuning_dataset.py` to match the json format above.

2. Configure a .yml config file like the ones in this folder. Make sure that `dataset_path` variables in .yml are correct.

3. Run (For Single GPU)`oumi train -c <your-config.yml>`, comment `output_adapter_name: "aumoai/llama3.2-3B-qlora-oumi-aumogpt-adapter"` on the present YAML in this case

4. For Multiple GPU's (choose one option):
```
# Using DDP (DistributedDataParallel)
oumi distributed torchrun \
  -m oumi train \
  -c <your-config.yml>

# Using FSDP (Fully Sharded Data Parallel)
oumi distributed torchrun \
  -m oumi train \
  -c <your-config.yml> \
  --fsdp.enable_fsdp true \
  --fsdp.sharding_strategy FULL_SHARD
```

5. Save the model on HF (HuggingFace) `python3 upload_adapter_to_hf.py <your-config.yml>`, dont forget to uncomment `output_adapter_name: "aumoai/llama3.2-3B-qlora-oumi-aumogpt-adapter"` on the present YAML in this case. Be sure to have a `HF_TOKEN` system variable configured before using.
