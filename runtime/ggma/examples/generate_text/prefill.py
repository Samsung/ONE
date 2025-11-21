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
    max_length=32,
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
    "past_key_values",
    # DynamicCache is flatten-able operator since 4.50.
    # See _pytree.py > tree_flatten
    # SUPPORTED_NODES has *transformers.DynamicCache*
    # After flattening, DynamicCache becomes { "key_cache": [] , "value_cache": [ ] }
    # dict.value is returne. dict.key is stored in treespec.
    #
    # On prefill, DynamicCache is empty, and dict is empty after flattening.
    # PyTorch removes empty dict!
    # If number of args is 4 (including cache), it becomes 3!
    # To avoid this error, don't pass empty cache, just pass None.
    "attention_mask",
    # For left pad,        [0, ⋯, 0, 1, ⋯, 1]
    # For right right pad, [1, ⋯, 1, 0, ⋯, 0]
    # ( 0 is pad-token )
    # This script uses right pad and pass all-1 attention mask (including pad).
    # Npu computes all positions whether it is pad or not.
    "cache_position"
    # It is the list of cache position like [0, 1, ..., 11].
    # For npu, we always store all values (including pad).
]

with torch.no_grad(), RecordingInput(model, input_to_remove=input_to_remove) as rec:
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

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
circle_model = tico.convert(model, captured_input)
circle_model.save(f"prefill.circle")
