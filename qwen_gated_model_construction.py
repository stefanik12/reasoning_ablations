import os

from transformers import AutoTokenizer
from transformers.models.qwen3 import Qwen3GatedForCausalLM

model_name = "Qwen/Qwen3-0.6B"  # or any dense Qwen3 checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)

OUT_PATH = "michal-stefanik/Qwen3-gated-0.6B"

model = Qwen3GatedForCausalLM.from_pretrained(model_name)

model.config.architectures = ['Qwen3GatedForCausalLM']

inputs = tokenizer("Hello, I am Gated Qwen. You can tell I'm conscious because ", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print("Model sainty check: ")
print(tokenizer.decode(out[0], skip_special_tokens=True))

print("Pushing model to HF")
model.push_to_hub(OUT_PATH, token=os.getenv("HF_TOKEN", None))
tokenizer.push_to_hub(OUT_PATH, token=os.getenv("HF_TOKEN", None))

print("Model after reload sanity check: ")
model = Qwen3GatedForCausalLM.from_pretrained(OUT_PATH)
tokenizer = AutoTokenizer.from_pretrained(OUT_PATH)

inputs = tokenizer("Hello, I am Gated Qwen. You can tell I'm conscious because ", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))
