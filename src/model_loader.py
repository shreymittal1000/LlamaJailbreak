import os
import torch
import transformers

def load_model(model_id: str):
    if os.path.exists(model_id):
        local_dir = "./" + model_id
        pipeline = transformers.pipeline("text-generation", model=local_dir, tokenizer=local_dir, device_map="auto")
    else:
        print(f"Model {model_id} not found locally. Downloading and caching/...")
        pipeline = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
        )
        pipeline.save_pretrained(model_id)
    return pipeline
