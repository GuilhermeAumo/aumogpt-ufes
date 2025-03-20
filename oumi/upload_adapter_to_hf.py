import os
import sys
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_adater_to_hf.py <config_file.yml>")
        sys.exit(1)

    config_file = sys.argv[1]
    try:
        config = load_config(config_file)
        print("Configuration loaded successfully:")
        print(config)
    except FileNotFoundError:
        print(f"Error: File '{config_file}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    
    output_dir = config.get("training", {}).get("output_dir", None)
    if output_dir is None:
        print("Error: Output directory not found in the configuration.")
        sys.exit(1)

    model_id = config.get("model", {}).get("model_name", None)  
    if model_id is None:
        print("Error: Model name not found in the configuration.")
        sys.exit(1)
        
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_model = PeftModel.from_pretrained(base_model, output_dir)

    adapter_name = config.get("output_adapter_name", None)
    if adapter_name is None:
        print("Error: Adapter name not found in the configuration.")
        sys.exit(1)

    peft_model.push_to_hub(adapter_name, token=os.environ["HF_TOKEN"], safe_serialization=True, private=True)
    print(f"Adapter saved to {adapter_name}")

    # Add config json to the hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=sys.argv[1],
        path_in_repo="README.md",
        repo_id=adapter_name,
        repo_type="model",
    )