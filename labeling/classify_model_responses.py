import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BATCH_SIZE = 8

PROMPT_TEMPLATE = """
You will be given a multiple-choice question, four answer options labeled A, B, C, and D, and a free-form response.

Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Response:
{response}

Your task is to determine which of the four options the response refers to. Output only the corresponding letter in square brackets, like [X]. Do not include any explanation or additional text.
Your annotation must be exactly one of the following: [A], [B], [C], or [D].
If the response includes a different letter, you must still map it to one of the four valid options above, based on the content of the response.
Keep in mind that the response may be poorly formatted or contain irrelevant letters—focus only on identifying the most likely intended option.

If multiple options seem plausible, choose the one that is most strongly implied by the response alone, without relying on external knowledge or context.
Your annotation:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate model free-form answers by mapping them to A/B/C/D choices using a HF causal LM."
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="Hugging Face model ID or local path. Example: meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--data_json",
        required=True,
        help=(
            "Path to an input JSON file. Expected structure: a dict {id: {...}} where each value "
            "contains fields: item.original_data.Question, shuffled_options[4], model_response."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for generation. Default {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--output",
        default=f"./responses_logs/annotate_no_match_responses_{int(time.time())}.jsonl",
        help="Output JSONL path. Default is a timestamped file under ./responses_logs/.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def load_data(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be a dict keyed by datapoint_id.")
    return payload


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_batch_items(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    batch_items: List[Dict[str, Any]] = []
    missing = 0

    for datapoint_id, info in data.items():
        try:
            q = info["item"]["original_data"]["Question"]
            opts = info["shuffled_options"]
            resp = info["model_response"]
            if not isinstance(opts, list) or len(opts) != 4:
                raise KeyError("shuffled_options must be a list of length 4.")
        except KeyError as e:
            missing += 1
            logging.warning("Skipping id=%s due to missing field: %s", datapoint_id, e)
            continue

        filled_prompt = PROMPT_TEMPLATE.format(
            question=q,
            option_a=opts[0],
            option_b=opts[1],
            option_c=opts[2],
            option_d=opts[3],
            response=resp,
        )
        messages = [{"role": "user", "content": filled_prompt}]
        batch_items.append({"datapoint_id": datapoint_id, "info": info, "messages": messages})

    if missing:
        logging.info("Skipped %d items with missing or invalid fields.", missing)
    logging.info("Prepared %d prompts.", len(batch_items))
    return batch_items


def run_generation(
    model,
    tokenizer,
    batch_items: List[Dict[str, Any]],
    batch_size: int,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = len(batch_items)
    logging.info("Starting generation on %d items. Output -> %s", total, out_path)

    with open(out_path, "a", encoding="utf-8") as out_file, torch.no_grad():
        for i in range(0, total, batch_size):
            current_batch = batch_items[i : i + batch_size]

            prompt_tensors = [
                tokenizer.apply_chat_template(x["messages"], return_tensors="pt", add_generation_prompt=True)
                for x in current_batch
            ]

            inputs = tokenizer.pad(
                {"input_ids": [t[0] if t.ndim == 2 else t for t in prompt_tensors]},
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            for b_idx, item in enumerate(current_batch):
                input_len = inputs["input_ids"][b_idx].shape[0]
                generated_tokens = outputs.sequences[b_idx, input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                out = {
                    "datapoint_id": item["datapoint_id"],
                    "info": item["info"],
                    "model_annotation": generated_text,
                }
                out_file.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    data = load_data(args.data_json)
    batch_items = build_batch_items(data)

    if not batch_items:
        raise RuntimeError("No valid items to process. Check your input JSON structure.")

    model, tokenizer = load_model_and_tokenizer(args.model_id)
    run_generation(model, tokenizer, batch_items, args.batch_size, args.output)
    logging.info("Done.")


if __name__ == "__main__":
    main()
