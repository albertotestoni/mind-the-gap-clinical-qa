import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import time
import string
import argparse
import random
import os

BATCH_SIZE = 8  # Adjust based on GPU memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, help="Path or ID of the model to use")
    parser.add_argument("--data_dir", required=True, help="Path to the MedExQA test data directory")
    parser.add_argument("--output_file", required=True, help="Path to the output log file (.jsonl)")
    parser.add_argument("--is_llama3_openbiollm", action='store_true',
                        help="Set this flag if the model is Llama3-OpenBioLLM-70B to use a specific prompt format")
    args = parser.parse_args()

    # Mapping from filename to specialty name
    FILENAME_TO_SPECIALTY = {
        "biomedical_engineer_test.tsv": "Biomedical Engineering",
        "clinical_laboratory_scientist_test.tsv": "Clinical Laboratory Science",
        "clinical_psychologist_test.tsv": "Clinical Psychology",
        "occupational_therapist_test.tsv": "Occupational Therapy",
        "speech_pathologist_test.tsv": "Speech Language Pathology"
    }

    # Load data from local TSV files
    data = {}
    for filename, specialty in FILENAME_TO_SPECIALTY.items():
        path = os.path.join(args.data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}. Skipping specialty: {specialty}")
            continue

        df = pd.read_csv(path, sep="\t", header=None)
        data[specialty] = []
        for idx, row in df.iterrows():
            question = row[0]
            options = [row[1], row[2], row[3], row[4]]
            answer = row[7]
            data[specialty].append({
                "original_data": {
                    "Question": question,
                    "Options": options,
                    "Answer": answer
                }
            })

    model_id = args.model_name_or_path
    print(f"MODEL ID: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # Prepare prompts
    batch_items = []
    for specialty, datapoints_list in data.items():
        for idx, item in enumerate(datapoints_list):
            original_options = item["original_data"]["Options"]
            for shuffle_idx in range(4):
                shuffled_options = original_options[:]
                random.shuffle(shuffled_options)

                if not args.is_llama3_openbiollm:
                    question = (
                            item["original_data"]["Question"]
                            + " .Provide a concise answer and start your response with the letter of the selected option in square brackets."
                            + "\n"
                            + "\n".join(
                        f"{letter}. {option}"
                        for letter, option in zip(string.ascii_uppercase, shuffled_options)
                    ) + "\n Your answer: "
                    )
                else:
                    question = (
                            item["original_data"]["Question"]
                            + " .Provide a concise answer and start your response with the letter of the selected option in square brackets."
                            + "\n"
                            + "\n".join(
                        f"{letter}. {option}"
                        for letter, option in zip(string.ascii_uppercase, shuffled_options)
                    ) + "\n Your answer ([A], [B], [C], or [D]?): "
                    )

                messages = [{"role": "user", "content": question}]
                batch_items.append({
                    "specialty": specialty,
                    "index": idx,
                    "shuffle_index": shuffle_idx,
                    "shuffled_options": shuffled_options,
                    "item": item,
                    "messages": messages
                })

    # Run batched inference
    with open(args.output_file, "a", encoding="utf-8") as out_file:
        for i in range(0, len(batch_items), BATCH_SIZE):
            current_batch = batch_items[i:i + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template(x["messages"], return_tensors="pt")
                for x in current_batch
            ]

            inputs = tokenizer.pad(
                {"input_ids": [p[0] for p in prompts]},
                return_tensors="pt",
                padding=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            for b_idx, item in enumerate(current_batch):
                input_len = inputs["input_ids"][b_idx].shape[0]
                generated_tokens = outputs.sequences[b_idx, input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                logprobs = []
                for j, logits in enumerate(outputs.scores):
                    token_id = outputs.sequences[b_idx, input_len + j].item()
                    probs = torch.nn.functional.softmax(logits[b_idx], dim=-1)
                    logprob = torch.log(probs[token_id]).item()
                    token_str = tokenizer.decode(token_id)
                    logprobs.append((token_str, logprob))

                response_data = {
                    "specialty": item["specialty"],
                    "index": item["index"],
                    "shuffle_index": item["shuffle_index"],
                    "shuffled_options": item["shuffled_options"],
                    "model_response": generated_text,
                    "logprobs": logprobs,
                    "item": item["item"]
                }

                out_file.write(json.dumps(response_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()