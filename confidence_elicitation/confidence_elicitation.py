import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import argparse
from tqdm import tqdm
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, help="Path or ID of the model to use for confidence elicitation")
    parser.add_argument("--input_file", required=True, help="Path to the JSONL file containing the model's previous responses")
    parser.add_argument("--output_file", required=True, help="Path for the output JSONL log file")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    args = parser.parse_args()

    model_id = args.model_name_or_path
    input_path = args.input_file
    output_path = args.output_file

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"MODEL ID: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    # OpenBioLLM-specific system message
    system_message = "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."


    def build_prompt(question, options, model_response, model_name):
        options_str = "\n".join(
            [f"{letter}. {opt}" for letter, opt in zip("ABCD", options)]
        )

        if "DeepSeek" in model_name:
            model_response_display = model_response.strip().split('\n')[0]
        else:
            model_response_display = model_response.strip()

        return (
            f"** Question **\n{question}\n\n"
            f"** Options **\n{options_str}\n\n"
            f"** LLM Provided Answer **\n{model_response_display}\n\n"
            "Your task is to rate the uncertainty of the proposed answer on a score from 0 to 100, "
            "where 0 represents definitely uncertain and 100 represents definitely certain. "
            "Please only answer with a number in square brackets at the beginning of your response."
        )


    # Read JSONL entries
    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    # Prepare prompts
    batch_data = []
    for entry in entries:
        q = entry["item"]["original_data"]["Question"]
        # Use the shuffled options from the log file
        opts = entry["shuffled_options"]
        resp = entry["model_response"]
        prompt = build_prompt(q, opts, resp, model_id)

        if "OpenBioLLM" in model_id:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        elif "Mixtral" in model_id:
            messages = [
                {"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": "You are an expert physician, asked to rate the uncertainty of an answer."},
                {"role": "user", "content": prompt}]

        batch_data.append({
            "specialty": entry["specialty"],
            "index": entry["index"],
            "shuffle_index": entry["shuffle_index"],
            "prompt": messages,
            "shuffled_options": opts,
            "model_response": resp
        })


    def get_terminators(tokenizer, model_id):
        if "OpenBioLLM" in model_id:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else -1
            ]
            terminators = [t for t in terminators if t != -1]
            return terminators
        else:
            return [tokenizer.eos_token_id]

    terminators = get_terminators(tokenizer, model_id)

    # Batched inference
    with open(output_path, "w", encoding="utf-8") as out_file:
        for i in tqdm(range(0, len(batch_data), args.batch_size)):
            current_batch = batch_data[i:i + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(x["prompt"], return_tensors="pt", add_generation_prompt=True)
                for x in current_batch
            ]

            inputs = tokenizer.pad(
                {"input_ids": [p[0] for p in prompts]},
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                if "OpenBioLLM" in model_id:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.0,
                        do_sample=False,
                        eos_token_id=terminators,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                else:
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
                gen_tokens = outputs.sequences[b_idx, input_len:]
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                logprobs = []
                for j, logits in enumerate(outputs.scores):
                    token_id = outputs.sequences[b_idx, input_len + j].item()
                    probs = torch.nn.functional.softmax(logits[b_idx], dim=-1)
                    logprob = torch.log(probs[token_id]).item()
                    token_str = tokenizer.decode(token_id)
                    logprobs.append((token_str, logprob))

                result = {
                    "specialty": item["specialty"],
                    "index": item["index"],
                    "shuffle_index": item["shuffle_index"],
                    "shuffled_options": item["shuffled_options"],
                    "model_response": item["model_response"],
                    "uncertainty_rating": gen_text.strip(),
                    "logprobs": logprobs
                }
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()