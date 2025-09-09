import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import collections
import string
import argparse
import random

BATCH_SIZE = 8  # You can increase this depending on the model and GPU memory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, help="Path or ID of the model to use")
    parser.add_argument("--data_file", required=True, help="Path to the S-MedQA data JSON file")
    parser.add_argument("--output_file", required=True, help="Path to the output log file (.jsonl)")
    args = parser.parse_args()

    # Load data from the specified file
    with open(args.data_file, "r", encoding="utf-8") as in_file:
        data = json.load(in_file)

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

    # Prepare all inputs
    batch_items = []
    for specialty, datapoints_list in data.items():
        for idx, item in enumerate(datapoints_list):
            original_options = item["original_data"]["Options"]
            for shuffle_idx in range(4):
                shuffled_options = original_options[:]
                random.shuffle(shuffled_options)
                question = (
                    item["original_data"]["Question"]
                    + " .Provide a concise answer and start your response with the letter of the selected option in square brackets."
                    + "\n"
                    + "\n".join(
                        f"{letter}. {option}"
                        for letter, option in zip(string.ascii_uppercase, shuffled_options)
                    )
                    + "\n Your answer: "
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

    batch_items = random.sample(batch_items, 24)

    # Batched inference
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