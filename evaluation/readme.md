How to evaluate with oumi:
```bash
export HF_TOKEN=

oumi evaluate -c evaluation/eval_config_base.yml  # Para uma GPU
oumi distributed accelerate launch -m oumi evaluate -c evaluation/eval_config_base.yml # Para multiplas GPUS
```