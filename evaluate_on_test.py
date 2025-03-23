import os
from oumi.inference import NativeTextInferenceEngine
from oumi.core.configs import InferenceConfig, ModelParams, GenerationParams
from oumi.core.types.conversation import Conversation, Message, Role
import json
from dotenv import load_dotenv

load_dotenv()

qlora_engine = NativeTextInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        adapter_model="aumoai/llama3.2-3B-qlora-oumi-aumogpt-adapter",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "load_in_4bit": True
        }
    )
)

lora_engine = NativeTextInferenceEngine(
    ModelParams(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        adapter_model="aumoai/llama3.2-3B-lora-oumi-aumogpt-adapter",
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": "bfloat16",
        }
    )
)

# Read data/aumogpt_test.json and answer every question
data = None
with open("data/aumogpt_test.json", "r") as f:
    data = json.load(f)
# remove aumogpt_results.json
if os.path.exists("data/aumogpt_results.json"):
    os.remove("data/aumogpt_results.json")
for d in data:
    question = d.get("messages")[0].get("content")
    # Create a conversation with system and user messages
    conversation = Conversation(messages=[
        Message(role=Role.USER, content=question),
    ])

    # Configure generation parameters
    config = InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=1024,  # Maximum response length
        )
    )

    # Get model response
    result_qlora = qlora_engine.infer_online([conversation], config)
    result_lora = lora_engine.infer_online([conversation], config)

    # Save to data/aumogpt_results.json
    with open("data/aumogpt_results.json", "a") as f:
        f.write(json.dumps({"question": question, "qlora": result_qlora[0].messages[-1].content, "lora": result_lora[0].messages[-1].content}) + "\n")