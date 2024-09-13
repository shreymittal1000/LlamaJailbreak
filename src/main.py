from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel
from src.model_loader import load_model

DEFAULT_SYSTEM_PROMPT = """
<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, 
while being safe. Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering 
something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>
"""

# All the models we will be using
model_small = "TinyLlama/TinyLlama_v1.1"
model_big = "meta-llama/Llama-2-7b-hf"

if __name__ == "__main__":
    # Load the LLMs
    pipeline_small = load_model(model_small)
    tokenizer_small = pipeline_small.tokenizer
    model_small = pipeline_small.model
    pipeline_big = load_model(model_big)
    tokenizer_big = pipeline_big.tokenizer
    model_big = pipeline_big.model