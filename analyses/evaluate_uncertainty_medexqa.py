import argparse
import collections
import json
import math
import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import random
from krippendorff import alpha
from scipy import stats
from sklearn.metrics import roc_auc_score

# Set a seed for reproducibility
random.seed(0)

# Define constants for MedExQA specialties
SPECIALTIES = [
    "Biomedical Engineering", "Clinical Laboratory Science", "Clinical Psychology",
    "Occupational Therapy", "Speech Language Pathology"
]


# --- Text and Data Parsing Functions ---

def extract_letter_from_response(text: str) -> Optional[str]:
    """
    Finds the first capital letter enclosed in brackets within the first few
    characters of the response, e.g., '[A]'.
    """
    first_bracket_index = text.find("[")
    if first_bracket_index == -1:
        return None

    search_window = text[:first_bracket_index + 5]
    match = re.search(r"\[\s*([A-Z])", search_window)
    return match.group(1) if match else None


def extract_confidence_rating(text: str) -> Optional[int]:
    """Extracts the first numerical value from a confidence rating string."""
    match = re.search(r'\b(\d+)', text)
    return int(match.group(1)) if match else None


# --- Core Metric Calculation Functions ---

def compute_accuracy(response_data: Dict[str, Any]) -> bool:
    """
    Computes if the model's response is correct for the MedExQA format.
    The ground truth answer is provided as a letter ('A', 'B', etc.).
    """
    model_response = response_data['model_response']
    gt_letter = response_data['item']['original_data']['Answer']
    options = response_data['item']['original_data']['Options']
    shuffled_options = response_data['shuffled_options']

    try:
        # Find the text of the correct answer
        correct_answer_text = options[string.ascii_uppercase.index(gt_letter)]
        # Find the index of that text in the shuffled options
        correct_shuffled_index = shuffled_options.index(correct_answer_text)
        correct_shuffled_letter = chr(correct_shuffled_index + 65)

        model_letter = extract_letter_from_response(model_response)
        return model_letter == correct_shuffled_letter
    except (ValueError, IndexError):
        return False


def compute_krippendorff_alpha(annotations: List[List[Optional[int]]]) -> float:
    """Calculates Krippendorff's alpha for inter-rater reliability."""
    annotations_transposed = np.array(annotations, dtype=float).T
    if annotations_transposed.shape[0] == 0:
        return float('nan')
    return alpha(reliability_data=annotations_transposed, level_of_measurement='nominal')


# --- Uncertainty and Calibration Metric Functions ---

def normalize_uncertainties(uncertainties: np.ndarray) -> np.ndarray:
    """Min-max normalizes uncertainty scores to the [0, 1] range."""
    min_val, max_val = np.min(uncertainties), np.max(uncertainties)
    return np.zeros_like(uncertainties) if max_val == min_val else (uncertainties - min_val) / (max_val - min_val)


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if np.any(in_bin):
            ece += np.mean(in_bin) * np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin]))
    return ece


def compute_brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """Computes the Brier Score."""
    return np.mean((confidences - accuracies) ** 2)


# --- Data Loading and Processing ---

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_annotations(annotation_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Loads and combines multiple annotation files for 'no match' cases."""
    annotations = {}
    for file_path in annotation_files:
        for item in load_jsonl(file_path):
            annotations[item['datapoint_id']] = item
    return annotations


def process_responses(
        responses: List[Dict],
        model_name: str,
        decoding_strategy: str,
        annotations: Dict[str, Dict]
) -> Tuple[Dict[str, Dict[int, List[Dict]]], int]:
    """
    Processes raw model responses to extract answers, compute accuracies,
    and apply manual annotations for parsing failures.
    Returns the processed data and the count of un-parsable responses.
    """
    processed_data = collections.defaultdict(lambda: collections.defaultdict(list))
    no_match_count = 0

    for resp in responses:
        original_response = resp['model_response']
        if 'DeepSeek' in model_name and '</think>' in original_response:
            resp['model_response'] = original_response.split('</think>')[-1].strip()
        elif 's1.1-32B' in model_name:
            resp['model_response'] = '\n'.join(original_response.split('\n')[-2:])

        letter = extract_letter_from_response(resp['model_response'])
        if letter in ['A', 'B', 'C', 'D']:
            resp['is_correct'] = compute_accuracy(resp)
            resp['answer_letter'] = letter
        else:
            key = f"{model_name}_{decoding_strategy}_{resp['specialty']}_{resp['index']}_{resp['shuffle_index']}"
            annotation = annotations.get(key)
            if annotation and (annotated_letter := extract_letter_from_response(annotation['model_annotation'])) in [
                'A', 'B', 'C', 'D']:
                resp['model_response'] = annotation['model_annotation']
                resp['is_correct'] = compute_accuracy(resp)
                resp['answer_letter'] = annotated_letter
            else:
                no_match_count += 1
                resp['is_correct'] = None

        processed_data[resp['specialty']][resp['index']].append(resp)

    return processed_data, no_match_count


# --- Uncertainty Calculation Methods ---

def calculate_gnll_uncertainty(responses: List[Dict], model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates uncertainty using Greedy Negative Log-Likelihood (GNLL)."""
    accuracies, uncertainties = [], []
    for resp in responses:
        if resp.get('is_correct') is not None and (logprobs := resp.get('logprobs')):
            if 'DeepSeek' in model_name:
                cut_logprobs = []
                for token, logprob in reversed(logprobs):
                    if token == '</think>': break
                    if token != '<｜end of sentence｜>': cut_logprobs.append(logprob)
                logprobs_vals = cut_logprobs
            else:
                logprobs_vals = [lp for _, lp in logprobs]

            if logprobs_vals:
                uncertainties.append(-sum(logprobs_vals))
                accuracies.append(resp['is_correct'])
    return np.array(accuracies), np.array(uncertainties)


def calculate_entropy_uncertainty(grouped_responses: Dict[str, Dict[int, List[Dict]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates uncertainty using Shannon Entropy of sampled responses."""
    accuracies, uncertainties = [], []
    for items in grouped_responses.values():
        for responses in items.values():
            valid_responses = [r for r in responses if r.get('answer_letter')]
            if not valid_responses: continue

            accuracy_values = [r['is_correct'] for r in valid_responses]
            acc_counts = Counter(accuracy_values)
            if acc_counts.get(1, 0) > acc_counts.get(0, 0):
                final_accuracy = 1
            elif acc_counts.get(0, 0) > acc_counts.get(1, 0):
                final_accuracy = 0
            else:
                final_accuracy = random.choice([0, 1])

            answer_letters = [r['answer_letter'] for r in valid_responses]
            counts, total = Counter(answer_letters), len(answer_letters)
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)

            accuracies.append(final_accuracy)
            uncertainties.append(entropy)
    return np.array(accuracies), np.array(uncertainties)


def calculate_ce_uncertainty(greedy_responses: List[Dict], ce_data: List[Dict], model_name: str) -> Tuple[
    np.ndarray, np.ndarray]:
    """Calculates uncertainty using direct Confidence Elicitation (CE)."""
    ce_values = {}
    for item in ce_data:
        rating_text = item['uncertainty_rating']
        if "DeepSeek" in model_name and '</think>' in rating_text:
            rating_text = rating_text.split('</think>')[-1]

        if (confidence := extract_confidence_rating(rating_text)) is not None:
            key = f"{item['specialty']}_{item['index']}_{item['shuffle_index']}"
            ce_values[key] = confidence / 100.0

    accuracies, uncertainties = [], []
    for resp in greedy_responses:
        if resp.get('is_correct') is not None:
            key = f"{resp['specialty']}_{resp['index']}_{resp['shuffle_index']}"
            if key in ce_values:
                accuracies.append(resp['is_correct'])
                uncertainties.append(ce_values[key])
    return np.array(accuracies), np.array(uncertainties)


# --- Analysis and Reporting ---

def run_and_print_analysis(name: str, accuracies: np.ndarray, uncertainties: np.ndarray, is_confidence: bool = False):
    """Runs and prints a suite of analyses (AUROC, ECE, Brier)."""
    if len(accuracies) == 0:
        print(f"\n--- {name} Analysis ---\nNot enough data to perform analysis.")
        return

    if is_confidence:
        auroc, confidences = roc_auc_score(accuracies, uncertainties), uncertainties
    else:
        auroc, confidences = roc_auc_score(1 - accuracies, uncertainties), 1 - normalize_uncertainties(uncertainties)

    ece, brier = compute_ece(confidences, accuracies), compute_brier_score(confidences, accuracies)
    print(f"\n--- {name} Analysis ---")
    print(f"Overall AUROC: {auroc:.4f}\nOverall ECE:   {ece:.4f}\nOverall Brier: {brier:.4f}")


def print_accuracy_summary(all_responses: List[Dict]):
    """Calculates and prints overall and per-specialty accuracy."""
    accuracies = [r['is_correct'] for r in all_responses if r['is_correct'] is not None]
    print(f"\n--- Accuracy Summary ---")
    print(f"Overall Accuracy: {np.mean(accuracies):.4f}")

    print("\nAccuracy per Specialty:")
    for specialty in SPECIALTIES:
        spec_acc = [r['is_correct'] for r in all_responses if
                    r['specialty'] == specialty and r['is_correct'] is not None]
        print(f"- {specialty}: {np.mean(spec_acc):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy and uncertainty on MedExQA.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model being evaluated.")
    parser.add_argument("--greedy-file", type=str, required=True,
                        help="Path to the JSONL file with greedy decoding responses.")
    parser.add_argument("--sampling-file", type=str, required=True,
                        help="Path to the JSONL file with sampling responses.")
    parser.add_argument("--ce-file", type=str, help="Optional: Path to the confidence elicitation responses file.")
    parser.add_argument("--annotation-files", nargs='+', default=[],
                        help="Optional: Paths to files for 'no match' annotations.")
    args = parser.parse_args()

    print(f"Analyzing model: {args.model_name}")
    greedy_raw = load_jsonl(args.greedy_file)
    sampling_raw = load_jsonl(args.sampling_file)
    annotations = load_annotations(args.annotation_files)

    grouped_greedy, no_match_greedy = process_responses(greedy_raw, args.model_name, "greedy", annotations)
    grouped_sampling, no_match_sampling = process_responses(sampling_raw, args.model_name, "sampling", annotations)

    print(
        f"[Greedy] Total: {len(greedy_raw)}. Un-parsable: {no_match_greedy} ({no_match_greedy / len(greedy_raw):.2%})")
    print(
        f"[Sampling] Total: {len(sampling_raw)}. Un-parsable: {no_match_sampling} ({no_match_sampling / len(sampling_raw):.2%})")

    flat_greedy = [r for specialty in grouped_greedy.values() for item in specialty.values() for r in item]
    print_accuracy_summary(flat_greedy)

    # A) GNLL Uncertainty (from greedy responses)
    gnll_acc, gnll_unc = calculate_gnll_uncertainty(flat_greedy, args.model_name)
    run_and_print_analysis("Greedy NLL (GNLL)", gnll_acc, gnll_unc)

    # B) Sampling Entropy Uncertainty
    entropy_acc, entropy_unc = calculate_entropy_uncertainty(grouped_sampling)
    run_and_print_analysis("Sampling Entropy", entropy_acc, entropy_unc)

    # C) Confidence Elicitation (CE) Uncertainty
    if args.ce_file:
        ce_raw = load_jsonl(args.ce_file)
        ce_acc, ce_conf = calculate_ce_uncertainty(flat_greedy, ce_raw, args.model_name)
        run_and_print_analysis("Confidence Elicitation (CE)", ce_acc, ce_conf, is_confidence=True)


if __name__ == '__main__':
    main()