from transformers import AutoTokenizer
from qwen3_dense_gated import Qwen3GatedForCausalLM

model_name = "Qwen/Qwen3-0.6B"  # or any dense Qwen3 checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = Qwen3GatedForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Hello, gated Qwen3!", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=True))

