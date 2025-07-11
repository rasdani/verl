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
import argparse
import os

import datasets


def convert_swe_rl_to_verl_format(example, idx, split):
    """Convert a single SWE-RL example to VERL format."""
    data = {
        "data_source": "swe_rl",
        "prompt": example["messages"],
        "ability": "swe_rl",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["patch"]
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "parsed_commit_content": example["parsed_commit_content"],
            "file_context": example["file_context"]
        }
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/swe_rl_8k")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--test_size", type=int, default=64, help="Number of samples for test set (default: 1024 - one typical batch)")

    args = parser.parse_args()

    data_source = "rasdani/SkyRL-v0-293-data-oracle-8k-context"

    print(f"Loading the {data_source} dataset from huggingface...")
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]
    
    print(f"Processing {len(train_dataset)} train samples and {len(test_dataset)} test samples...")

    processed_train_dataset = train_dataset.map(function=lambda example, idx: convert_swe_rl_to_verl_format(example, idx, "train"), with_indices=True)
    processed_test_dataset = test_dataset.map(function=lambda example, idx: convert_swe_rl_to_verl_format(example, idx, "test"), with_indices=True)
    # processed_test_dataset = processed_test_dataset.select(range(2))

    local_dir = os.path.expanduser(args.local_dir)

    train_output_file = os.path.join(local_dir, "train.parquet")
    test_output_file = os.path.join(local_dir, "test.parquet")
    
    processed_train_dataset.to_parquet(train_output_file)
    processed_test_dataset.to_parquet(test_output_file)
    
    print(f"Saved processed train dataset to {train_output_file}")
    print(f"Saved processed test dataset to {test_output_file}")
    print(f"Total train samples processed: {len(processed_train_dataset)}")
    print(f"Total test samples processed: {len(processed_test_dataset)}")
    
    if len(processed_train_dataset) > 0:
        sample = processed_train_dataset[0]
        print("\nSample processed data:")
        print(f"Data source: {sample['data_source']}")
        print(f"Ability: {sample['ability']}")
        print(f"Prompt length: {len(sample['prompt'][0]['content'])}")
        print(f"Extra info keys: {list(sample['extra_info'].keys())}")
