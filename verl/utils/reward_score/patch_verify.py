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

import re
import cydifflib


def parse_last_diff_codeblock(markdown_str):
    matches = re.finditer(r"```diff\s*(.*?)\s*```", markdown_str, re.DOTALL)
    if matches:
        last_match = matches[-1]
        return last_match.group(1).strip()
    else:
        return None

def normalize_diff(diff_text: str) -> str:
    diff_text = re.sub(r'(?m)^index [^\n]*\n', '', diff_text)
    diff_text = re.sub(r'(?m)^(@@[^@]*@@).*', r'\1', diff_text)
    diff_text = diff_text.strip() + "\n"
    return diff_text

def score_patching(model_diff, ground_truth, debug=False):
    """
    Score how well the model's patches match the expected patches.

    Args:
        verification_info (dict): Contains the golden diff
        debug (bool): Whether to print debug information

    Returns:
        float: Score between 0.0 and 1.0
    """
    try:
        score = cydifflib.SequenceMatcher(
            None,
            a=model_diff,
            b=ground_truth,
            autojunk=False,
        ).ratio()
        if debug:
            print(f"DEBUG: Model diff: \n{model_diff}")
            print(f"DEBUG: Ground truth: \n{ground_truth}")
            print(f"\nDEBUG: Diff similarity score: {score}")
        return score

    except Exception as e:
        if debug:
            print(f"DEBUG: Exception in score_patching: {repr(e)}")
        return 0.0


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute score for SWE fixer based on solution and ground truth.
    
    Args:
        solution_str (str): The model's output containing patches
        ground_truth (str): Expected output or reference
        extra_info (dict, optional): Additional verification information including:
            - original_files: List of files to be modified
            - expected_patches: Expected code patches
            
    Returns:
        float: Score between 0.0 and 1.0
    """
    
    try:
        after_thinking = solution_str.split("</think>")[-1].strip()
        if "```diff" in after_thinking:
            breakpoint()
        model_diff = parse_last_diff_codeblock(solution_str)
        if not model_diff:
            return 0.0
        
        verification_info = {
            "golden_diff": ground_truth,
        }
        # verification_info = json.loads(extra_info)

        return score_patching(model_diff, ground_truth=verification_info["golden_diff"], debug=False)
        
    except Exception:
        return 0.0


if __name__ == "__main__":
    """
    Test the SWE-Fixer reward function with actual processed data.
    """
    import pandas as pd
    import os
    
    # Path to the processed parquet file
    data_path = os.path.expanduser("~/data/github_patches/train.parquet")
    
    # Load the processed data
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")
    
    
    first_row = df.iloc[0]
    ground_truth = first_row['reward_model']['ground_truth']
    # extra_info = first_row['extra_info']
   