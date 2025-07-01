# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GitHub patches dataset for search-replace reward format
"""

import argparse
import os
import json
from typing import Dict, Any

import datasets


def extract_file_context_from_diff(diff_text: str) -> Dict[str, Any]:
    """
    Extract file context from a unified diff.
    This is a simplified version - in production you'd want more robust parsing.
    
    Returns:
        Dict with 'code_context' and 'oracle_new_content' keys
    """
    # This is a placeholder - in reality, you'd parse the diff to extract:
    # 1. Original file contents (code_context)
    # 2. Modified file contents (oracle_new_content)
    
    # For now, just return the diff as-is
    return {
        "code_context": {},
        "oracle_new_content": {},
        "diff": diff_text
    }


def convert_github_patches_to_search_replace_format(example, idx, split):
    """Convert a GitHub patches example to VERL format with search-replace reward support."""
    
    # Extract context from the golden diff if possible
    context_info = extract_file_context_from_diff(example["golden_diff"])
    
    # Create VERL format
    data = {
        "data_source": "github_patches_search_replace",
        "prompt": [
            {
                "role": "user",
                "content": example["prompt"]
            }
        ],
        "ability": "github_patches_search_replace",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["golden_diff"]
        },
        "extra_info": {
            "split": split,
            "index": idx,
            # Include file context for search-replace reward calculation
            "code_context": context_info.get("code_context", {}),
            "oracle_new_content": context_info.get("oracle_new_content", {}),
            # Keep the original diff for fallback
            "golden_diff": example["golden_diff"]
        }
    }
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/root/persistent/data/github_patches_search_replace")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--test_size", type=int, default=64, help="Number of samples for test set")
    parser.add_argument("--dataset_name", default="rasdani/SWE-smith-oracle-4k-context-1k-diff", 
                        help="HuggingFace dataset to use")

    args = parser.parse_args()

    print(f"Loading the {args.dataset_name} dataset from huggingface...")
    dataset = datasets.load_dataset(args.dataset_name)

    # The dataset only has a train split, so we'll create our own train/test split
    full_dataset = dataset["train"]
    print(f"Original dataset size: {len(full_dataset)}")
    
    # Limit samples if specified (useful for testing)
    if args.max_samples:
        full_dataset = full_dataset.select(range(min(args.max_samples, len(full_dataset))))
        print(f"Limited dataset to {len(full_dataset)} samples for testing")
    
    # Create train/test split with fixed test size
    dataset_size = len(full_dataset)
    test_size = min(args.test_size, dataset_size)
    train_size = dataset_size - test_size
    
    print(f"Creating train/test split: {train_size} train, {test_size} test samples")
    
    train_dataset = full_dataset.select(range(train_size))
    test_dataset = full_dataset.select(range(train_size, dataset_size))
    
    print(f"Processing {len(train_dataset)} train samples and {len(test_dataset)} test samples...")

    # Process the dataset
    def make_map_fn(split):
        def process_fn(example, idx):
            try:
                return convert_github_patches_to_search_replace_format(example, idx, split)
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                # Return a minimal valid example to avoid breaking the pipeline
                return {
                    "data_source": "github_patches_search_replace",
                    "prompt": [{"role": "user", "content": "Error processing this example"}],
                    "ability": "github_patches_search_replace", 
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {"split": split, "index": idx, "error": str(e)}
                }

        return process_fn

    processed_train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    processed_test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_output_file = os.path.join(local_dir, "train.parquet")
    test_output_file = os.path.join(local_dir, "test.parquet")
    
    processed_train_dataset.to_parquet(train_output_file)
    processed_test_dataset.to_parquet(test_output_file)
    
    print(f"Saved processed train dataset to {train_output_file}")
    print(f"Saved processed test dataset to {test_output_file}")

    if hdfs_dir is not None:
        raise NotImplementedError("HDFS is not supported")

    print("Dataset preprocessing completed!")
    print(f"Total train samples processed: {len(processed_train_dataset)}")
    print(f"Total test samples processed: {len(processed_test_dataset)}")
    
    # Show a sample of the processed data
    if len(processed_train_dataset) > 0:
        sample = processed_train_dataset[0]
        print("\nSample processed data:")
        print(f"Data source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"Prompt length: {len(sample['prompt'][0]['content'])}")
        print(f"Extra info keys: {list(sample['extra_info'].keys())}")