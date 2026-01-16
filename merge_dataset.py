import json
import jsonlines

def merge_datasets(original_jsonl_path, generated_json_path, output_jsonl_path):
    """
    Merge original dataset with generated responses.
    
    Args:
        original_jsonl_path: Path to original JSONL file
        generated_json_path: Path to generated JSON file with responses
        output_jsonl_path: Path to save merged JSONL file
    """
    # Load original dataset
    original_data = []
    with jsonlines.open(original_jsonl_path) as reader:
        for obj in reader:
            original_data.append(obj)
    
    # Load generated responses
    with open(generated_json_path, 'r') as f:
        generated_data = json.load(f)
    
    # Create mapping from question to generated response
    # Using question as key since pid might not be reliable
    question_to_generated = {}
    for detail in generated_data['details']:
        # Extract question from prompt (remove the instruction part)
        prompt = detail['prompt']
        question = prompt.replace('Question: ', '').replace('\n\nPlease reason step by step, and put your final answer within \\boxed{}.', '').strip()
        question_to_generated[question] = detail['generated']
    
    # Merge datasets
    merged_data = []
    matched_count = 0
    unmatched_count = 0
    
    for original in original_data:
        merged_entry = original.copy()
        
        # Try to find matching generated response
        if original['question'] in question_to_generated:
            merged_entry['response'] = question_to_generated[original['question']]
            matched_count += 1
        else:
            # Keep original response if no match found
            unmatched_count += 1
            print(f"Warning: No generated response found for question: {original['question'][:50]}...")
        
        merged_data.append(merged_entry)
    
    # Write merged data to output file
    with jsonlines.open(output_jsonl_path, mode='w') as writer:
        writer.write_all(merged_data)
    
    # Print statistics
    print(f"\nMerge Statistics:")
    print(f"Total original entries: {len(original_data)}")
    print(f"Total generated responses: {len(generated_data['details'])}")
    print(f"Matched entries: {matched_count}")
    print(f"Unmatched entries: {unmatched_count}")
    print(f"Output saved to: {output_jsonl_path}")

# Usage
if __name__ == "__main__":
    original_file = "/scratch/vjd5zr/project/icml_kd/data/math/test.jsonl"
    generated_file = "/scratch/vjd5zr/project/icml_kd/results/math/Qwen2.5-3B-Instruct/eval_results.json"
    output_file = "/scratch/vjd5zr/project/icml_kd/data/math/test_3B.jsonl"
    
    merge_datasets(original_file, generated_file, output_file)

# # Main execution
# if __name__ == "__main__":
#     # File paths
#     original_file = "/scratch/vjd5zr/project/icml_kd/data/math/test.jsonl"
#     generated_file = "/scratch/vjd5zr/project/icml_kd/results/math/Qwen2.5-3B-Instruct/eval_results.json"
#     output_file = "/scratch/vjd5zr/project/icml_kd/data/math/test_3B.jsonl"
    
#     # Load datasets
#     print("Loading datasets...")
#     original_data = load_original_dataset(original_file)
#     generated_data = load_generated_responses(generated_file)
    
#     # Merge datasets
#     print("\nMerging datasets...")
#     merged_data = merge_datasets(original_data, generated_data)
    
#     # Save merged dataset
#     print(f"\nSaving merged dataset to {output_file}...")
#     save_merged_dataset(merged_data, output_file)
#     print("Done!")
