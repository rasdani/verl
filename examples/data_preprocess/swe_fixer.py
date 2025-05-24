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


def parse_verification_info(verification_info_str: str) -> Dict[str, Any]:
    """Parse the verification_info string into a dictionary."""
    try:
        return ast.literal_eval(verification_info_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing verification_info: {e}")
        return {}


def parse_golden_solution(golden_solution_str: str) -> Dict[str, Any]:
    """Parse the golden_standard_solution string into a dictionary."""
    try:
        return ast.literal_eval(golden_solution_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing golden_standard_solution: {e}")
        return {}


def convert_swe_fixer_to_verl_format(example, idx, split):
    """Convert a single SWE-Fixer example to VERL format."""
    # Parse verification info and golden solution
    verification_info = parse_verification_info(example["verification_info"])
    golden_solution = parse_golden_solution(example["golden_standard_solution"])
    
    # Extract original files and expected patches from verification info
    original_files = verification_info.get("input", {}).get("files to be modified", [])
    expected_patches = verification_info.get("output", {}).get("edited code", [])
    
    # Create VERL format
    data = {
        "data_source": "swe_fixer",
        "prompt": [
            {
                "role": "user",
                "content": example["prompt"]
            }
        ],
        "ability": "swe_fixer",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(golden_solution)
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "problem_id": example["problem_id"],
            "in_source_id": example["in_source_id"],
            "task_type": example["task_type"],
            "original_files": original_files,
            "expected_patches": expected_patches,
            "verification_info": verification_info
        }
    }
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/swe_fixer")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")

    args = parser.parse_args()

    data_source = "rasdani/swe-fixer-70k"

    print(f"Loading the {data_source} dataset from huggingface...")
    dataset = datasets.load_dataset(data_source)

    # The dataset should have a train split
    train_dataset = dataset["train"]
    
    # Limit samples if specified (useful for testing)
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        print(f"Limited dataset to {len(train_dataset)} samples for testing")

    print(f"Processing {len(train_dataset)} samples...")

    # Process the dataset
    def make_map_fn(split):
        def process_fn(example, idx):
            try:
                return convert_swe_fixer_to_verl_format(example, idx, split)
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                # Return a minimal valid example to avoid breaking the pipeline
                return {
                    "data_source": "swe_fixer",
                    "prompt": [{"role": "user", "content": "Error processing this example"}],
                    "ability": "swe_fixer", 
                    "reward_model": {"style": "rule", "ground_truth": "{}"},
                    "extra_info": {"split": split, "index": idx, "error": str(e)}
                }

        return process_fn

    processed_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    output_file = os.path.join(local_dir, "train.parquet")
    processed_dataset.to_parquet(output_file)
    print(f"Saved processed dataset to {output_file}")

    if hdfs_dir is not None:
        raise NotImplementedError("HDFS is not supported for SWE-Fixer")
        # makedirs(hdfs_dir)
        # copy(src=local_dir, dst=hdfs_dir)
        # print(f"Copied to HDFS: {hdfs_dir}")

    print("Dataset preprocessing completed!")
    print(f"Total samples processed: {len(processed_dataset)}")
    
    # Show a sample of the processed data
    if len(processed_dataset) > 0:
        sample = processed_dataset[0]
        print("\nSample processed data:")
        print(f"Data source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"Prompt length: {len(sample['prompt'][0]['content'])}")
        print(f"Extra info keys: {list(sample['extra_info'].keys())}")