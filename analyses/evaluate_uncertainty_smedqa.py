import argparse
import collections
import json
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import random
from krippendorff import alpha
from scipy import stats
from sklearn.metrics import roc_auc_score

# Set a seed for reproducibility
random.seed(0)

# Define constants for medical specialties
SPECIALTIES = [
    "Gastroenterology", "Cardiology", "Obstetrics and gynecology",
    "Neurology", "Infectious diseases", "Pediatrics"
]


# --- Text and Data Parsing Functions ---

def extract_letter_from_response(text: str) -> Optional[str]:
    """
    Finds the first capital letter enclosed in brackets within the first few
    characters of the response, e.g., '[A]'.
    """
    # Search only in the initial part of the text to avoid parsing long rationales
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
    """Computes if the model's response is correct."""
    model_response = response_data['model_response']
    correct_answer = response_data['item']['original_data']['Answer']
    shuffled_options = response_data['shuffled_options']

    try:
        correct_answer_index = shuffled_options.index(correct_answer)
        correct_letter = chr(correct_answer_index + 65)
        model_letter = extract_letter_from_response(model_response)
        return model_letter == correct_letter
    except (ValueError, IndexError):
        return False


def compute_krippendorff_alpha(annotations: List[List[Optional[int]]]) -> float:
    """Calculates Krippendorff's alpha for inter-rater reliability."""
    # The library expects rows to be raters and columns to be items.
    annotations_transposed = np.array(annotations, dtype=float).T
    return alpha(reliability_data=annotations_transposed, level_of_measurement='nominal')


# --- Uncertainty and Calibration Metric Functions ---

def normalize_uncertainties(uncertainties: np.ndarray) -> np.ndarray:
    """Min-max normalizes uncertainty scores to the [0, 1] range."""
    min_val = np.min(uncertainties)
    max_val = np.max(uncertainties)
    if max_val == min_val:
        return np.zeros_like(uncertainties)
    return (uncertainties - min_val) / (max_val - min_val)


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if np.any(in_bin):
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = np.mean(in_bin)
            ece += bin_weight * np.abs(bin_confidence - bin_accuracy)
    return ece


def compute_brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """Computes the Brier Score."""
    return np.mean((confidences - accuracies) ** 2)


# --- Data Loading and Processing ---

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Processes raw model responses to extract answers, compute accuracies,
    and apply manual annotations for parsing failures.
    """
    processed_data = collections.defaultdict(lambda: collections.defaultdict(list))
    no_match_count = 0

    for resp in responses:
        # --- Pre-process model-specific output formats ---
        original_response = resp['model_response']
        if 'DeepSeek' in model_name and '</think>' in original_response:
            resp['model_response'] = original_response.split('</think>')[-1].strip()
        elif 's1.1-32B' in model_name:
            resp['model_response'] = '\n'.join(original_response.split('\n')[-2:])

        # --- Extract answer and compute accuracy ---
        letter = extract_letter_from_response(resp['model_response'])
        if letter in ['A', 'B', 'C', 'D']:
            resp['is_correct'] = compute_accuracy(resp)
            resp['answer_letter'] = letter
        else:
            # Fallback to manual annotations if parsing fails
            key = f"{model_name}_{decoding_strategy}_{resp['specialty']}_{resp['index']}_{resp['shuffle_index']}"
            annotation = annotations.get(key)
            if annotation:
                annotated_letter = extract_letter_from_response(annotation['model_annotation'])
                if annotated_letter in ['A', 'B', 'C', 'D']:
                    resp['model_response'] = annotation['model_annotation']
                    resp['is_correct'] = compute_accuracy(resp)
                    resp['answer_letter'] = annotated_letter
                else:
                    no_match_count += 1
                    resp['is_correct'] = None
            else:
                no_match_count += 1
                resp['is_correct'] = None

        processed_data[resp['specialty']][resp['index']].append(resp)

    print(
        f"[{decoding_strategy.capitalize()}] Total responses: {len(responses)}. Un-parsable responses: {no_match_count} ({no_match_count / len(responses):.2%})")
    return processed_data


# --- Uncertainty Calculation Methods ---

def calculate_gnll_uncertainty(responses: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates uncertainty using Greedy Negative Log-Likelihood (GNLL)."""
    accuracies, uncertainties = [], []
    for resp in responses:
        if resp.get('is_correct') is not None:
            # Handle model-specific logprob slicing
            logprobs = resp.get('logprobs', [])
            if 'DeepSeek' in resp.get('model_name', ''):
                cut_logprobs = []
                for token, logprob in reversed(logprobs):
                    if token == '</think>': break
                    if token != '<｜end of sentence｜>': cut_logprobs.append((token, logprob))
                logprobs = cut_logprobs

            if logprobs:
                gnll = -sum(lp for _, lp in logprobs)
                uncertainties.append(gnll)
                accuracies.append(resp['is_correct'])

    return np.array(accuracies), np.array(uncertainties)


def calculate_entropy_uncertainty(grouped_responses: Dict[str, Dict[int, List[Dict]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates uncertainty using Shannon Entropy of sampled responses."""
    accuracies, uncertainties = [], []
    for specialty, items in grouped_responses.items():
        for idx, responses in items.items():
            valid_responses = [r for r in responses if r.get('answer_letter')]
            if not valid_responses:
                continue

            # Majority vote for accuracy
            accuracy_values = [r['is_correct'] for r in valid_responses]
            if not accuracy_values: continue

            acc_counts = Counter(accuracy_values)
            if acc_counts.get(1, 0) > acc_counts.get(0, 0):
                final_accuracy = 1
            elif acc_counts.get(0, 0) > acc_counts.get(1, 0):
                final_accuracy = 0
            else:
                final_accuracy = random.choice([0, 1])  # Tie-break

            # Calculate entropy
            answer_letters = [r['answer_letter'] for r in valid_responses]
            counts = Counter(answer_letters)
            total = len(answer_letters)
            probs = [count / total for count in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)

            accuracies.append(final_accuracy)
            uncertainties.append(entropy)

    return np.array(accuracies), np.array(uncertainties)


def calculate_ce_uncertainty(greedy_responses: List[Dict], ce_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates uncertainty using direct Confidence Elicitation (CE)."""
    ce_values = {}
    for item in ce_data:
        # Handle model-specific response formats
        rating_text = item['uncertainty_rating']
        if "DeepSeek" in item.get("model_name", ""):
            if '</think>' in rating_text:
                rating_text = rating_text.split('</think>')[-1]
            elif '</n>' in rating_text:
                rating_text = rating_text.split('</n>')[-1]

        confidence = extract_confidence_rating(rating_text)
        if confidence is not None:
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

def run_and_print_analysis(
        name: str,
        accuracies: np.ndarray,
        uncertainties: np.ndarray,
        is_confidence: bool = False
):
    """
    Runs a suite of analyses (AUROC, ECE, Brier) and prints the results.

    Args:
        name (str): The name of the uncertainty method.
        accuracies (np.ndarray): Array of 1s (correct) and 0s (incorrect).
        uncertainties (np.ndarray): Array of corresponding uncertainty scores.
        is_confidence (bool): If True, treats scores as confidence (higher is better).
                              If False, treats as uncertainty (lower is better).
    """
    if len(accuracies) == 0 or len(uncertainties) == 0:
        print(f"\n--- {name} Analysis ---")
        print("Not enough data to perform analysis.")
        return

    # For AUROC, uncertainty should correlate with incorrectness (label 1)
    # If using confidence, flip both confidence and accuracy for roc_auc_score.
    if is_confidence:
        auroc = roc_auc_score(accuracies, uncertainties)
        confidences = uncertainties
    else:
        auroc = roc_auc_score(1 - accuracies, uncertainties)
        confidences = 1 - normalize_uncertainties(uncertainties)

    ece = compute_ece(confidences, accuracies)
    brier = compute_brier_score(confidences, accuracies)

    print(f"\n--- {name} Analysis ---")
    print(f"Overall AUROC: {auroc:.4f}")
    print(f"Overall ECE:   {ece:.4f}")
    print(f"Overall Brier: {brier:.4f}")


def main():
    """Main function to run the model evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate model accuracy and uncertainty.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model being evaluated.")
    parser.add_argument("--greedy-file", type=str, required=True,
                        help="Path to the JSONL file with greedy decoding responses.")
    parser.add_argument("--sampling-file", type=str, required=True,
                        help="Path to the JSONL file with sampling responses.")
    parser.add_argument("--ce-file", type=str, help="Optional: Path to the confidence elicitation responses file.")
    parser.add_argument("--annotation-files", nargs='+', default=[],
                        help="Optional: Paths to files for 'no match' annotations.")
    args = parser.parse_args()

    # --- 1. Load Data ---
    print(f"Analyzing model: {args.model_name}")
    greedy_responses_raw = load_jsonl(args.greedy_file)
    sampling_responses_raw = load_jsonl(args.sampling_file)
    annotations = load_annotations(args.annotation_files) if args.annotation_files else {}

    # --- 2. Process and Calculate Base Metrics ---
    grouped_greedy = process_responses(greedy_responses_raw, args.model_name, "greedy", annotations)
    grouped_sampling = process_responses(sampling_responses_raw, args.model_name, "sampling", annotations)

    # --- 3. Run Uncertainty Analyses ---

    # A) GNLL Uncertainty (from greedy responses)
    flat_greedy_responses = [resp for specialty in grouped_greedy.values() for item in specialty.values() for resp in
                             item]
    gnll_acc, gnll_unc = calculate_gnll_uncertainty(flat_greedy_responses)
    run_and_print_analysis("Greedy NLL (GNLL)", gnll_acc, gnll_unc)

    # B) Sampling Entropy Uncertainty
    entropy_acc, entropy_unc = calculate_entropy_uncertainty(grouped_sampling)
    run_and_print_analysis("Sampling Entropy", entropy_acc, entropy_unc)

    # C) Confidence Elicitation (CE) Uncertainty
    if args.ce_file:
        ce_responses_raw = load_jsonl(args.ce_file)
        ce_acc, ce_conf = calculate_ce_uncertainty(flat_greedy_responses, ce_responses_raw)
        run_and_print_analysis("Confidence Elicitation (CE)", ce_acc, ce_conf, is_confidence=True)


if __name__ == '__main__':
    main()