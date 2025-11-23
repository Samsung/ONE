import argparse
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tico.utils.record_input import RecordingInput
import tico

# Constants
MODEL_ID = "Maykeye/TinyLLama-v0"
PROMPT = "Lily picked up a flower."


@dataclass
class ModeArg:
    max_length: int
    input_to_remove: List[str]
    condition: Optional[Callable]


MODE_ARGS = {
    "prefill":
    ModeArg(max_length=32,
            input_to_remove=["past_key_values", "attention_mask", "cache_position"],
            condition=None),
    "decode":
    ModeArg(
        max_length=30,
        input_to_remove=["attention_mask"],
        condition=lambda args_dict: args_dict["past_key_values"].get_seq_length() != 0)
}


def main():
    parser = argparse.ArgumentParser(
        description="Export TinyLlama model to Circle format.")
    parser.add_argument("--mode",
                        choices=["prefill", "decode"],
                        required=True,
                        help="Export mode: prefill or decode")
    args = parser.parse_args()

    # Get configuration for the selected mode
    config = MODE_ARGS[args.mode]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        padding="max_length",
        max_length=config.max_length,
        truncation=True,
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    rec_context = RecordingInput(model,
                                 config.condition,
                                 input_to_remove=config.input_to_remove)

    with torch.no_grad(), rec_context as rec:
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        captured_input = rec.captured_input

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

    # Tico Conversion
    # Reload model to ensure clean state for conversion if needed,
    # but prefill.py and decode.py re-instantiate model. Let's follow that pattern to be safe.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    if args.mode == "decode":
        # Monkey patch for decode mode
        from tico.serialize.operators.adapters.onert.llama_attention import (
            llama_attention_forward_adapter, )
        from transformers.models.llama.modeling_llama import LlamaAttention
        LlamaAttention.forward = llama_attention_forward_adapter

    circle_model = tico.convert(model, captured_input)
    output_file = f"{args.mode}.circle"
    circle_model.save(output_file)
    print(f"Model saved to {output_file}")


if __name__ == "__main__":
    main()
