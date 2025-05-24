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

import json
import re
import ast
import cydifflib


LINE_NUMBER_REGEX = re.compile(r"^\d+\s", re.MULTILINE)


def parse_json_codeblock_from_model_output(markdown_str):
    """Extract JSON content from markdown codeblock or think tags."""
    # Get everything after </think>, if it exists
    match = re.search(r"</think>(.*?)$", markdown_str, re.DOTALL)
    answer_str = match.group(1).strip() if match else markdown_str.strip()
    # Extract everything between ```json and ``` markers
    match = re.search(r"```json\s*(.*?)\s*```", answer_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return answer_str.strip()


def remove_line_numbers(content):
    """Remove line numbers from code content."""
    return LINE_NUMBER_REGEX.sub("", content)


def remove_empty_lines(code):
    """Remove empty lines from code."""
    lines = code.splitlines()
    filtered_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(filtered_lines)


def check_syntax(code):
    """Check if Python code has valid syntax."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def apply_patches(files_to_modify, patches):
    """
    Apply a list of code-edit patches to an iterable of files and return the
    fully-patched workspace.

    Args:
        files_to_modify (list[dict]): items from verification_info["input"]["files to be modified"]
        patches (list[dict]): items structured like verification_info["output"]["edited code"]
                            or the model's JSON output.

    Returns:
        dict[str, str]: file-path -> patched file content
    """
    workspace = {f["file"]: remove_line_numbers(f["file content"])
                for f in files_to_modify}
    failed_file_paths = []

    for patch in patches:
        file_path = patch["file"]
        snippet_old = remove_line_numbers(
            patch["code snippet to be modified"]
        ).strip()
        snippet_new = patch["edited code snippet"].strip()

        current = workspace.get(file_path, "")
        if snippet_old:
            if snippet_old not in current:
                # Model failed to localize the code snippet to be modified
                failed_file_paths.append(file_path)
                continue
            current = current.replace(snippet_old, snippet_new)
        elif current == "":           # brand-new file
            current = snippet_new
        workspace[file_path] = current

    # Set the file content to None for files that failed to be patched
    workspace = {k: None if k in failed_file_paths else v for k, v in workspace.items()}
    return workspace


def get_diff(before, after):
    """Get unified diff between two text contents."""
    diff = cydifflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="")
    lines = list(diff)[2:]  # Keep relevant parts of the diff
    return "\n".join(lines)


def score_patching(verification_info, json_output, debug=False):
    """
    Score how well the model's patches match the expected patches.

    Args:
        verification_info (dict): Contains the original files to modify and golden patches
                                in verification_info["input"]["files to be modified"] and
                                verification_info["output"]["edited code"] respectively
        json_output (str): The model's output as a JSON string containing patches to apply
        debug (bool): Whether to print debug information

    Returns:
        float: Score between 0.0 and 1.0
    """
    try:
        model_patches = json.loads(json_output)
        if debug:
            print(f"DEBUG: Parsed model patches: {model_patches}")
    except Exception as e:
        if debug:
            print(f"DEBUG: Failed to parse JSON: {e}")
        return 0.0

    try:
        original_files = verification_info["input"]["files to be modified"]
        golden_patches = verification_info["output"]["edited code"]
        
        if debug:
            print(f"DEBUG: Original files count: {len(original_files)}")
            print(f"DEBUG: Golden patches count: {len(golden_patches)}")
            print(f"DEBUG: Model patches count: {len(model_patches)}")

        original_workspace = apply_patches(original_files, [])
        golden_workspace = apply_patches(original_files, golden_patches)
        predicted_workspace = apply_patches(original_files, model_patches)
        
        if debug:
            print(f"DEBUG: Original workspace files: {list(original_workspace.keys())}")
            print(f"DEBUG: Golden workspace files: {list(golden_workspace.keys())}")
            print(f"DEBUG: Predicted workspace files: {list(predicted_workspace.keys())}")

        scores = []
        for file_path in golden_workspace:
            if debug:
                print(f"DEBUG: Processing file: {file_path}")
                
            if predicted_workspace[file_path] is None:
                if debug:
                    print(f"DEBUG: File {file_path} failed to patch")
                scores.append(0.0)  # model failed to localize edit location
                continue
                
            golden_file_content = golden_workspace[file_path]
            predicted_file_content = predicted_workspace.get(file_path, "")
            
            if debug:
                print(f"DEBUG: Golden content length: {len(golden_file_content)}")
                print(f"DEBUG: Predicted content length: {len(predicted_file_content)}")
                print(f"DEBUG: Contents match exactly: {predicted_file_content == golden_file_content}")

            if predicted_file_content == golden_file_content:
                scores.append(1.0)
                continue

            syntax_ok = check_syntax(predicted_file_content)
            if not syntax_ok:
                if debug:
                    print(f"DEBUG: Syntax check failed for file {file_path}")
                return 0.0

            golden_diff = get_diff(before=original_workspace[file_path], after=golden_file_content)
            model_diff = get_diff(before=original_workspace[file_path], after=predicted_file_content)

            score = cydifflib.SequenceMatcher(
                None,
                a=model_diff,
                b=golden_diff,
                autojunk=False,
            ).ratio()
            if debug:
                print(f"DEBUG: Diff similarity score for {file_path}: {score}")
            scores.append(score)

        final_score = sum(scores) / len(scores) if scores else 0.0
        if debug:
            print(f"DEBUG: Individual scores: {scores}")
            print(f"DEBUG: Final average score: {final_score}")
        return final_score

    except Exception as e:
        if debug:
            print(f"DEBUG: Exception in score_patching: {e}")
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
    if extra_info is None:
        # Fallback: simple string comparison if no extra verification info
        return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
    
    try:
        # Extract JSON from solution string (may contain markdown or think tags)
        json_output = parse_json_codeblock_from_model_output(solution_str)
        
        # Parse the JSON to get the actual patches
        parsed_output = json.loads(json_output)
        
        # Extract the actual patches from the "edited code" key if it exists
        if "edited code" in parsed_output:
            model_patches = parsed_output["edited code"]
        else:
            model_patches = parsed_output  # Assume it's already the patch list
        
        # Create verification info structure
        verification_info = {
            "input": {"files to be modified": extra_info.get("original_files", [])},
            "output": {"edited code": extra_info.get("expected_patches", [])}
        }
        
        # Convert model patches back to JSON for score_patching
        model_patches_json = json.dumps(model_patches)
        
        # Score the patches
        return score_patching(verification_info, model_patches_json, debug=False)
        
    except Exception:
        return 0.0


if __name__ == "__main__":
    """
    Test the SWE-Fixer reward function with actual processed data.
    """
    import pandas as pd
    import os
    
    # Path to the processed parquet file
    data_path = os.path.expanduser("~/data/swe_fixer/train.parquet")
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please run the preprocessing script first:")
        print("python examples/data_preprocess/swe_fixer.py --max_samples 100")
        exit(1)
    
    # Load the processed data
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Get the first row for testing
    if len(df) == 0:
        print("No data found in the parquet file")
        exit(1)
    
    first_row = df.iloc[0]
    print(f"\nTesting with first row (index {first_row['extra_info']['index']}):")
    print(f"Problem ID: {first_row['extra_info']['problem_id']}")
    print(f"Prompt length: {len(first_row['prompt'][0]['content'])} characters")
    
    # Extract data for testing
    ground_truth = first_row['reward_model']['ground_truth']
    extra_info = first_row['extra_info']
    
    print(f"\nGround truth: {ground_truth[:200]}..." if len(ground_truth) > 200 else f"\nGround truth: {ground_truth}")
    print(f"Original files: {len(extra_info['original_files'])}")
    print(f"Expected patches: {len(extra_info['expected_patches'])}")
    
    # Test 1: Use ground truth as prediction (should get perfect score)
    print("\n" + "="*50)
    print("TEST 1: Using ground truth as prediction (should score 1.0)")
    print("="*50)
    
    print("DEBUG: Checking ground truth format...")
    print(f"Ground truth type: {type(ground_truth)}")
    print(f"Ground truth content (first 500 chars): {ground_truth[:500]}")
    
    # Parse the ground truth to see what format it's in
    try:
        parsed_gt = json.loads(ground_truth)
        print(f"Parsed ground truth keys: {list(parsed_gt.keys())}")
        if 'edited code' in parsed_gt:
            print(f"Number of edited code entries: {len(parsed_gt['edited code'])}")
    except Exception as e:
        print(f"Error parsing ground truth: {e}")
    
    # Create a debug version of compute_score for testing
    def compute_score_debug(solution_str, ground_truth, extra_info=None):
        if extra_info is None:
            return 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
        
        try:
            json_output = parse_json_codeblock_from_model_output(solution_str)
            print(f"DEBUG: Parsed JSON output: {json_output[:200]}...")
            
            # Parse the JSON to get the actual patches
            parsed_output = json.loads(json_output)
            print(f"DEBUG: Parsed output keys: {list(parsed_output.keys())}")
            
            # Extract the actual patches from the "edited code" key
            if "edited code" in parsed_output:
                model_patches = parsed_output["edited code"]
                print(f"DEBUG: Extracted {len(model_patches)} model patches")
            else:
                print("DEBUG: No 'edited code' key found in model output")
                model_patches = parsed_output  # Assume it's already the patch list
            
            verification_info = {
                "input": {"files to be modified": extra_info.get("original_files", [])},
                "output": {"edited code": extra_info.get("expected_patches", [])}
            }
            
            # Convert model patches back to JSON for score_patching
            model_patches_json = json.dumps(model_patches)
            print(f"DEBUG: Model patches JSON: {model_patches_json[:200]}...")
            
            return score_patching(verification_info, model_patches_json, debug=True)
        except Exception as e:
            print(f"DEBUG: Exception in compute_score_debug: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    score_perfect = compute_score_debug(
        solution_str=ground_truth,  # Use ground truth as prediction
        ground_truth=ground_truth,
        extra_info=extra_info
    )
    print(f"Score with perfect prediction: {score_perfect}")
    
    # Test 2: Use empty JSON as prediction (should get 0.0)
    print("\n" + "="*50)
    print("TEST 2: Using empty JSON as prediction (should score 0.0)")
    print("="*50)
    
    score_empty = compute_score(
        solution_str="[]",  # Empty patches
        ground_truth=ground_truth,
        extra_info=extra_info
    )
    print(f"Score with empty prediction: {score_empty}")
    
    # Test 3: Use malformed JSON as prediction (should get 0.0)
    print("\n" + "="*50)
    print("TEST 3: Using malformed JSON as prediction (should score 0.0)")
    print("="*50)
    
    score_malformed = compute_score(
        solution_str="not valid json",
        ground_truth=ground_truth,
        extra_info=extra_info
    )
    print(f"Score with malformed prediction: {score_malformed}")
    
    # Test 4: Test without extra_info (fallback mode)
    print("\n" + "="*50)
    print("TEST 4: Testing fallback mode without extra_info")
    print("="*50)
    
    score_fallback_perfect = compute_score(
        solution_str=ground_truth,
        ground_truth=ground_truth,
        extra_info=None
    )
    print(f"Score with fallback mode (perfect): {score_fallback_perfect}")
    
    score_fallback_wrong = compute_score(
        solution_str="wrong answer",
        ground_truth=ground_truth,
        extra_info=None
    )
    print(f"Score with fallback mode (wrong): {score_fallback_wrong}")
    
    # Test 5: Test with slightly modified patch (should get partial score)
    print("\n" + "="*50)
    print("TEST 5: Testing with slightly modified patch (should get partial score)")
    print("="*50)
    
    # Parse the ground truth and modify one of the patches slightly
    try:
        parsed_gt = json.loads(ground_truth)
        modified_patches = parsed_gt.copy()
        
        if len(modified_patches["edited code"]) > 0:
            # Take the first patch and modify it slightly
            first_patch = modified_patches["edited code"][0].copy()
            original_snippet = first_patch["edited code snippet"]
            
            # Make a small change - add a comment or change spacing
            modified_snippet = original_snippet.replace("    def all(self, **kwargs):", "    def all(self, **kwargs):  # Modified comment")
            first_patch["edited code snippet"] = modified_snippet
            modified_patches["edited code"][0] = first_patch
            
            print(f"DEBUG: Original snippet length: {len(original_snippet)}")
            print(f"DEBUG: Modified snippet length: {len(modified_snippet)}")
            print(f"DEBUG: Change made: Added comment to function definition")
            
            modified_ground_truth = json.dumps(modified_patches)
            
            score_partial = compute_score(
                solution_str=modified_ground_truth,
                ground_truth=ground_truth,
                extra_info=extra_info
            )
            print(f"Score with slightly modified prediction: {score_partial}")
            
        else:
            print("No patches available to modify")
            score_partial = 0.0
            
    except Exception as e:
        print(f"Error creating modified patch: {e}")
        score_partial = 0.0
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Perfect prediction score: {score_perfect} (expected: 1.0)")
    print(f"Empty prediction score: {score_empty} (expected: 0.0)")
    print(f"Malformed prediction score: {score_malformed} (expected: 0.0)")
    print(f"Fallback perfect score: {score_fallback_perfect} (expected: 1.0)")
    print(f"Fallback wrong score: {score_fallback_wrong} (expected: 0.0)")
    print(f"Partial modification score: {score_partial} (expected: 0.0 < score < 1.0)")
    
    # Verify the results
    success = True
    if score_perfect != 1.0:
        print(f"❌ Perfect prediction should score 1.0, got {score_perfect}")
        success = False
    if score_empty != 0.0:
        print(f"❌ Empty prediction should score 0.0, got {score_empty}")
        success = False
    if score_malformed != 0.0:
        print(f"❌ Malformed prediction should score 0.0, got {score_malformed}")
        success = False
    if score_fallback_perfect != 1.0:
        print(f"❌ Fallback perfect should score 1.0, got {score_fallback_perfect}")
        success = False
    if score_fallback_wrong != 0.0:
        print(f"❌ Fallback wrong should score 0.0, got {score_fallback_wrong}")
        success = False
    if not (0.0 < score_partial < 1.0):
        print(f"❌ Partial modification should score between 0.0 and 1.0, got {score_partial}")
        success = False
    
    if success:
        print("✅ All tests passed! SWE-Fixer reward function is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        exit(1)