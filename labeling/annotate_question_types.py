import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_ANNOTATION = """
You are a medical expert assistant. Your task is to classify the type of each medical question according to the following categories:

1. Diagnosis – Identifying a disease or condition.
2. Treatment – Interventions like medication, surgery, or therapy.
3. Prognosis – The likely course or outcome of a disease.
4. Diagnostic Test – Lab tests, imaging, or diagnostic procedures.
5. Definition – Asking for an explanation of a medical concept.
6. Procedure/Operation – Questions about medical or surgical procedures.
7. Other

Assign the most appropriate type based on the question intent. Classify the following question. 
Provide only the corresponding label number (1–7) in square brackets as your response.
Q: {}
Question type:
"""

DEFAULT_BATCH_SIZE = 8

FILENAME_TO_SPECIALTY = {
    "biomedical_engineer_test.tsv": "Biomedical Engineering",
    "clinical_laboratory_scientist_test.tsv": "Clinical Laboratory Science",
    "clinical_psychologist_test.tsv": "Clinical Psychology",
    "occupational_therapist_test.tsv": "Occupational Therapy",
    "speech_pathologist_test.tsv": "Speech Language Pathology",
}


def read_medexqa_dir(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load MedExQA TSV files if a directory is provided and exists."""
    if not data_dir or not os.path.isdir(data_dir):
        logging.info("No MedExQA directory provided or not found. Skipping MedExQA loading.")
        return {}

    data_medexqa: Dict[str, List[Dict[str, Any]]] = {}
    for filename, specialty in FILENAME_TO_SPECIALTY.items():
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            logging.warning("Expected TSV not found: %s (specialty %s). Skipping.", path, specialty)
            continue

        df = pd.read_csv(path, sep="\t", header=None)
        data_medexqa[specialty] = []
        for _, row in df.iterrows():
            question = row[0]
            options = [row[1], row[2], row[3], row[4]]
            answer = row[7]  # A/B/C/D
            data_medexqa[specialty].append(
                {"original_data": {"Question": question, "Options": options, "Answer": answer}}
            )
    logging.info("Loaded MedExQA specialties: %s", list(data_medexqa.keys()))
    return data_medexqa


def read_custom_json(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a custom JSON dataset. See README section below for the expected schema."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Custom JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Accept two shapes:
    # 1) {"Specialty A": [{"original_data": {"Question": ...}}], "Specialty B": [...], ...}
    # 2) [{"Question": ...}, ...] -> mapped to a "Custom" specialty
    if isinstance(payload, dict):
        # Minimal validation
        for spec, items in payload.items():
            if not isinstance(items, list):
                raise ValueError(f'Specialty "{spec}" must map to a list.')
        return payload
    elif isinstance(payload, list):
        # Wrap into a single specialty
        return {"Custom": [{"original_data": {"Question": it.get("Question", str(it))}} for it in payload]}
    else:
        raise ValueError("Unsupported JSON format. Provide a dict of lists or a list of items.")


def build_batches(
    medexqa: Dict[str, List[Dict[str, Any]]],
    custom: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    batch_items: List[Dict[str, Any]] = []

    def _append_block(source: Dict[str, List[Dict[str, Any]]]):
        for specialty, datapoints_list in source.items():
            for idx, item in enumerate(datapoints_list):
                question_text = item.get("original_data", {}).get("Question")
                if not question_text:
                    logging.warning("Missing 'original_data.Question' at %s[%d]. Skipping.", specialty, idx)
                    continue
                prompt = PROMPT_ANNOTATION.format(question_text)
                messages = [{"role": "user", "content": prompt}]
                batch_items.append(
                    {"specialty": specialty, "index": idx, "item": item, "messages": messages}
                )

    _append_block(custom)
    _append_block(medexqa)
    logging.info("Prepared %d prompts.", len(batch_items))
    return batch_items


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


def run_inference(
    model, tokenizer, batch_items: List[Dict[str, Any]], batch_size: int, out_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = len(batch_items)
    logging.info("Starting generation on %d items. Output -> %s", total, out_path)

    with open(out_path, "a", encoding="utf-8") as out_file, torch.no_grad():
        for i in range(0, total, batch_size):
            current_batch = batch_items[i : i + batch_size]

            prompt_tensors: List[torch.Tensor] = []
            for x in current_batch:
                # apply_chat_template returns a tensor if return_tensors="pt"
                t = tokenizer.apply_chat_template(x["messages"], return_tensors="pt", add_generation_prompt=True)
                prompt_tensors.append(t[0] if t.ndim == 2 else t)

            inputs = tokenizer.pad({"input_ids": prompt_tensors}, return_tensors="pt", padding=True).to(
                model.device
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Compute per-sample response
            for b_idx, item in enumerate(current_batch):
                # Input length per sample
                input_len = inputs["input_ids"][b_idx].shape[0]
                generated_tokens = outputs.sequences[b_idx, input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                response_data = {
                    "specialty": item["specialty"],
                    "index": item["index"],
                    "model_response": generated_text,
                    "item": item["item"],
                }
                out_file.write(json.dumps(response_data, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy classification of question types with a HF causal LM."
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="Hugging Face model ID or local path. Example: meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--data_json",
        required=True,
        help="Path to a JSON file with questions to classify from S-MedQA. See 'Input JSON format' below.",
    )
    parser.add_argument(
        "--medexqa_dir",
        default="",
        help="Optional path to MedExQA test TSVs directory to include alongside your JSON.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--output",
        default=f"responses_logs/annotation_qtype_greedy_{int(time.time())}.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    # Load data
    custom_data = read_custom_json(args.data_json)
    medexqa_data = read_medexqa_dir(args.medexqa_dir) if args.medexqa_dir else {}

    batch_items = build_batches(medexqa=medexqa_data, custom=custom_data)

    if not batch_items:
        raise RuntimeError("No valid items to process. Check your JSON and optional MedExQA directory.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # Run
    run_inference(model, tokenizer, batch_items, args.batch_size, args.output)
    logging.info("Done.")


if __name__ == "__main__":
    main()
