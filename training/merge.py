import argparse
import json
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel


def _load_tokenizer(adapter_dir, base_model_path):
    last_error = None

    # Prefer the adapter tokenizer so saved special-token settings and chat template
    # follow the training run, but fall back to the base tokenizer when the adapter
    # directory does not contain enough backend files for this environment.
    load_attempts = [
        ("adapter", adapter_dir, True),
        ("adapter", adapter_dir, False),
        ("base model", base_model_path, True),
        ("base model", base_model_path, False),
    ]

    for source_name, source_path, use_fast in load_attempts:
        try:
            print(f"Loading tokenizer from {source_name}: {source_path} (use_fast={use_fast})")
            tokenizer = AutoTokenizer.from_pretrained(
                source_path,
                trust_remote_code=True,
                use_fast=use_fast,
            )
            return tokenizer
        except Exception as exc:
            last_error = exc
            print(f"Tokenizer load failed from {source_name} (use_fast={use_fast}): {exc}")

    raise RuntimeError(
        "Failed to load tokenizer from both adapter and base model."
    ) from last_error


def _maybe_apply_adapter_chat_template(tokenizer, adapter_dir):
    config_path = os.path.join(adapter_dir, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        print(f"Warning: could not read adapter tokenizer_config.json: {exc}")
        return

    chat_template = config.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
        print("Applied chat template from adapter tokenizer_config.json")


def _load_base_model(base_model_path, torch_dtype):
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    print(f"Base model config type: {type(config).__name__} (model_type={model_type})")

    common_kwargs = dict(
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Mistral 3 / Ministral 3 checkpoints are exposed through the image-text-to-text
    # auto class in current Transformers, even for text-only use.
    if model_type == "mistral3":
        print("Using AutoModelForImageTextToText for Mistral3/Ministral3 checkpoint")
        return AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            **common_kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **common_kwargs,
    )


def merge_adapter(base_model_path, adapter_dir, output_dir, dtype="float16"):

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    torch_dtype = dtype_map[dtype]

    print("=" * 60)
    print("Merging LoRA adapter")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------
    tokenizer = _load_tokenizer(adapter_dir, base_model_path)
    _maybe_apply_adapter_chat_template(tokenizer, adapter_dir)

    if tokenizer.pad_token is None:
        if "<|finetune_right_pad_id|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # ---------------------------------------------------------
    # Load base model
    # ---------------------------------------------------------
    print(f"Loading base model: {base_model_path}")
    base_model = _load_base_model(base_model_path, torch_dtype)

    base_vocab = base_model.get_input_embeddings().weight.shape[0]
    print(f"Base model vocab size: {base_vocab}")

    # ---------------------------------------------------------
    # Resize embeddings if tokenizer is larger
    # ---------------------------------------------------------
    if len(tokenizer) > base_vocab:
        print(f"Resizing embeddings: {base_vocab} → {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    # ---------------------------------------------------------
    # Load adapter
    # ---------------------------------------------------------
    print(f"Loading adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------
    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    print(f"Saving merged model to: {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Merge complete.")
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        required=True,
        help="HF model name OR local model directory",
    )

    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter directory",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged model",
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )

    args = parser.parse_args()

    merge_adapter(
        args.base_model,
        args.adapter,
        args.output,
        args.dtype,
    )
