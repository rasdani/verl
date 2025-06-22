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
Preprocess the SWE-Fixer dataset to parquet format
"""

import argparse
import os
import json
import ast
from typing import Dict, Any

import datasets

# from verl.utils.hdfs_io import copy, makedirs



def convert_github_patches_to_verl_format(example, idx, split):
    """Convert a single SWE-Fixer example to VERL format."""
    verification_info = json.loads(example["verification_info"])
        
    # Create VERL format
    data = {
        "data_source": "github_patches",
        "prompt": [
            {
                "role": "user",
                "content": example["prompt"]
            }
        ],
        "ability": "github_patches",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["golden_diff"]
        },
        "extra_info": {
            "split": split,
            "index": idx,
            # "problem_id": example["problem_id"],
            # "in_source_id": example["in_source_id"],
            # "task_type": example["task_type"],
            "verification_info": verification_info
        }
    }
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/root/persistent/data/github_patches")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--test_size", type=int, default=32, help="Number of samples for test set (default: 1024 - one typical batch)")

    args = parser.parse_args()

    data_source = "rasdani/github-patches-genesys-2k-context-1k-diff"

    print(f"Loading the {data_source} dataset from huggingface...")
    dataset = datasets.load_dataset(data_source)

    # The dataset only has a train split, so we'll create our own train/test split
    full_dataset = dataset["train"]
    print(f"Original dataset size: {len(full_dataset)}")
    
    # Limit samples if specified (useful for testing)
    if args.max_samples:
        full_dataset = full_dataset.select(range(min(args.max_samples, len(full_dataset))))
        print(f"Limited dataset to {len(full_dataset)} samples for testing")

    # Shuffle the dataset to avoid repository clustering
    print("Shuffling dataset to avoid repository clustering...")
    # full_dataset = full_dataset.shuffle(seed=42)
    
    # Create train/test split with fixed test size (one batch)
    dataset_size = len(full_dataset)
    test_size = min(args.test_size, dataset_size)  # Don't exceed dataset size
    train_size = dataset_size - test_size
    
    print(f"Creating train/test split: {train_size} train, {test_size} test samples")
    print(f"Test set size matches typical batch size: {test_size}")
    
    train_dataset = full_dataset.select(range(train_size))
    test_dataset = full_dataset.select(range(train_size, dataset_size))
    
    print(f"Processing {len(train_dataset)} train samples and {len(test_dataset)} test samples...")

    # Process the dataset
    def make_map_fn(split):
        def process_fn(example, idx):
            try:
                return convert_github_patches_to_verl_format(example, idx, split)
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                # Return a minimal valid example to avoid breaking the pipeline
                return {
                    "data_source": "github_patches",
                    "prompt": [{"role": "user", "content": "Error processing this example"}],
                    "ability": "github_patches", 
                    "reward_model": {"style": "rule", "ground_truth": "{}"},
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
        raise NotImplementedError("HDFS is not supported for SWE-Fixer")
        # makedirs(hdfs_dir)
        # copy(src=local_dir, dst=hdfs_dir)
        # print(f"Copied to HDFS: {hdfs_dir}")

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