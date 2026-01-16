import json
import re
import sys

def normalize_text(text):
    """Simple normalization to remove trailing punctuation and whitespace."""
    if not isinstance(text, str):
        return str(text)
    return text.strip().lower().rstrip('.')

def extract_boxed_answer(response):
    """Extracts content inside \boxed{...} typically used in Math/GSM8K."""
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()
    return None

def extract_multiple_choice(response):
    """Extracts 'A', 'B', 'C', 'D', 'E' from phrases like 'The best answer is C'."""
    # Look for "The best answer is [X]" or "Answer: [X]"
    match = re.search(r'(?:best answer|answer) is\s*\(?([A-E])\)?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_boolean(response):
    """Extracts Yes/No or True/False from the end of the response."""
    # Look for Yes/No/True/False at the very end of the text
    response = response.strip()
    match = re.search(r'\b(Yes|No|True|False)\W*$', response, re.IGNORECASE)
    if match:
        word = match.group(1).lower()
        if word in ['yes', 'true']:
            return True
        if word in ['no', 'false']:
            return False
    return None

def is_correct(entry):
    """
    Determines if the 'response' matches the ground truth based on available keys.
    """
    response = entry.get('response', '')
    
    # --- Case 1: Multiple Choice (ARC, CommonsenseQA, Date) ---
    # These usually have an 'answerKey' field (A, B, C, D).
    if 'answerKey' in entry:
        prediction = extract_multiple_choice(response)
        ground_truth = entry['answerKey']
        return prediction == ground_truth

    # --- Case 2: NLI (Anli) ---
    # These have a 'label' ('entailment', 'contradiction') and response ends in True/False.
    if 'label' in entry and 'premise' in entry:
        ground_truth_label = entry['label'] # 'entailment' or 'contradiction'
        
        # Extract "Final Answer: False/True"
        match = re.search(r'Final Answer:\s*(True|False)', response, re.IGNORECASE)
        if match:
            pred_bool_str = match.group(1).lower()
            is_pred_true = (pred_bool_str == 'true')
            
            # Logic: 
            # If label is 'entailment', we expect True.
            # If label is 'contradiction', we expect False.
            if ground_truth_label == 'entailment':
                return is_pred_true
            elif ground_truth_label == 'contradiction':
                return not is_pred_true
        return False

    # --- Case 3: Boolean QA (StrategyQA) ---
    # The 'answer' is a boolean True/False. Response usually ends in "Yes" or "No".
    if 'answer' in entry and isinstance(entry['answer'], bool):
        prediction_bool = extract_boolean(response)
        return prediction_bool == entry['answer']

    # --- Case 4: Math / Free Text (GSM8K, MATH, Tables) ---
    # The 'answer' is a string or number. Response usually contains \boxed{answer}.
    if 'answer' in entry:
        ground_truth = str(entry['answer']).strip()
        prediction = extract_boxed_answer(response)
        
        if prediction:
            # Try comparing as numbers first (to handle 55 vs 55.0)
            try:
                return abs(float(prediction) - float(ground_truth)) < 1e-9
            except ValueError:
                # Fallback to string comparison
                return normalize_text(prediction) == normalize_text(ground_truth)
        
    return False

def process_file(input_file, output_file):
    total = 0
    kept = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            total += 1
            try:
                data = json.loads(line)
                if is_correct(data):
                    fout.write(json.dumps(data) + '\n')
                    kept += 1
            except json.JSONDecodeError:
                continue
                
    print(f"Processed {total} lines.")
    print(f"Kept {kept} correct entries.")
    print(f"Filtered out {total - kept} incorrect entries.")

# Example usage
if __name__ == "__main__":
    # Replace these filenames with your actual file paths
    input_path = "dataset.jsonl" 
    output_path = "dataset_clean.jsonl"
    
    # Or use command line args: python filter.py input.jsonl output.jsonl
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]

    process_file(input_path, output_path)