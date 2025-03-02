"""Read a JSON file, change it to the specified format, and send it to huggingface."""

import os
import datasets
import pandas as pd
from datasets import Dataset, DatasetDict
import json
from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

## CONFIGS
hf_dataset_name = "aumogpt-qa-disciplina-gpt"
hf_token = os.getenv("HF_TOKEN", None)
if hf_token is None:
    raise ValueError("Please set the HF_TOKEN environment variable in the .env or in the shell.")
# Load dados.json
with open('dados.json') as f:
    data = json.load(f)



# Convert to DataFrame
df_data = pd.DataFrame(data['qa'])

df_data["messages"] = df_data.apply(lambda row: [{"role": "user", "content": str(row["pergunta"])}, {"role": "assistant", "content": str(row["resposta"])}], axis=1)

print(">>> Dataframe com mensagens formatadas:")
print(df_data.head())

## Convert to Huggingface Dataset
dataset = Dataset.from_pandas(df_data)

#split dataset in train and test and validation
train_dataset = dataset.train_test_split(test_size=0.2)
test_dataset = train_dataset["test"].train_test_split(test_size=0.5)
validation_dataset = test_dataset["test"]

dict_dataset = DatasetDict({"train": train_dataset["train"], "test": test_dataset["train"], "validation": validation_dataset})
print(">>> Dataset após o split:")
print(dict_dataset)


## Save dataset to hf
# Authenticate (ensure you have a Hugging Face token set up)
api = HfApi()

# Create a new dataset repository on the Hugging Face Hub
try:
    api.create_repo(repo_id=f"aumoai/{hf_dataset_name}", private=True, token=hf_token, repo_type="dataset")
except Exception as e:
    print("Dataset already exists. Skipping repo creation.")

# Push the dataset
dict_dataset.push_to_hub(f"aumoai/{hf_dataset_name}", private=True, token=hf_token)

print(f"Dataset uploaded to Hugging Face under aumoai/{hf_dataset_name}")
