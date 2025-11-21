# User input
prompt = "Lily picked up a flower."
model_name = "Maykeye/TinyLLama-v0"

# Tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    max_length=30,
    truncation=True,
)

# Generator
import torch

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

from tico.utils.record_input import RecordingInput

# past_key_values
# ---------------
# During prefill, "past_key_values" not None, but an empty Cache instance.
# Passing None makes torch.export happy.

input_to_remove = [
    "attention_mask",
    # For left pad,        [0, ⋯, 0, 1, ⋯, 1]
    # For right right pad, [1, ⋯, 1, 0, ⋯, 0]
    # ( 0 is pad-token )
    # This script uses right pad and pass all-1 attention mask (including pad).
    # Npu computes all positions whether it is pad or not.
]
condition_fn = lambda args_dict: args_dict["past_key_values"].get_seq_length() != 0

with torch.no_grad(), RecordingInput(model, condition_fn,
                                     input_to_remove=input_to_remove) as rec:
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    captured_input = rec.captured_input

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# Tico
import tico
from tico.serialize.operators.adapters.onert.llama_attention import (
    llama_attention_forward_adapter, )
from transformers.models.llama.modeling_llama import LlamaAttention

#LlamaAttention.forward = llama_attention_forward_adapter

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
circle_model = tico.convert(model, captured_input)
circle_model.save(f"decode_.circle")
