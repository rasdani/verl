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
Search-Replace reward function for code changes.
Ported from Meta's implementation for VERL compatibility.
"""

import re
import warnings
from typing import TypedDict, Dict, List, Tuple, Optional

# Try to import cydifflib first (faster), fall back to standard difflib
try:
    import cydifflib as difflib
except ImportError:
    import difflib

# Tag constants for thought-solution format
THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

# Regex for parsing search-replace blocks
SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"


class FormatError(Exception):
    """Raised when the output format is invalid."""
    pass


class ChangeSimilarity(TypedDict):
    """Type definition for change similarity results."""
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def extract_thought_solution(output: str) -> Tuple[str, str]:
    """
    Extract the thought and solution from the output. It is expected to have the following format:
    <think>
    ...
    </think>
    <solution>
    ...
    </solution>
    """
    for tag in [THINK_START, THINK_END, ANSWER_START, ANSWER_END]:
        if output.count(tag) != 1:
            raise FormatError(f"count of {tag} is not 1")

    thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
    answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
    if len(thought) == 0:
        raise FormatError("Thought is empty")
    return thought, answer


def parse_search_replace(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A dictionary where the key is the file path and the value is a list of search/replace pairs.
    """
    path_search_replaces: List[Tuple[str, str, str]] = re.findall(
        SEARCH_REPLACE_REGEX, text
    )
    path_search_replace_dict = {}
    for path, search, replace in path_search_replaces:
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


def generate_unified_diff(
    old_code: str,
    new_code: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code.

    Args:
        old_code: The original code.
        new_code: The modified code.
        n_context: The number of context lines to show.

    Returns:
        A string representing the unified diff."""

    original_lines = old_code.splitlines(keepends=True)
    modified_lines = new_code.splitlines(keepends=True)

    # Use difflib for consistency with other reward functions
    diff = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    ))
    
    try:
        # Skip the first two lines (file headers)
        if len(diff) >= 2:
            diff_code = "".join(diff[2:])
            return diff_code
        else:
            return ""
    except Exception:
        return ""


def apply_code_change(
    code_context: Dict[str, str],
    search_replace_dict: Dict[str, List[Tuple[str, str]]],
    silent: bool = False,
) -> Dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.
        silent: Whether to suppress the error messages.

    Returns:
        A dictionary containing the file path and the new content of the code.
    """
    new_content_dict = {}
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            # Ensure search block can be matched
            # "\n" + search to ensure the indentations are correct
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if not silent and search not in new_content:
                raise FormatError(f"Search block not found in the code: {search}")
            new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        new_content_dict[path] = new_content[1:]
    return new_content_dict


def get_normalized_patch(
    code_context: Dict[str, str],
    new_content_dict: Dict[str, str],
) -> Dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = {}
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        # but it can occur in general-purpose usages
        if patch:
            patch_dict[path] = patch
    return patch_dict


def compute_change_similarities(
    pred_patch: Dict[str, str],
    oracle_patch: Dict[str, str],
) -> List[ChangeSimilarity]:
    """Compute similarities between predicted and oracle patches."""
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = []
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # Both are empty changes, meaning search = replace. We should penalize this to avoid
            # the model predicting empty changes to hack the reward.
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            )
        )
    return similarities


def calculate_reward(
    code_context: Dict[str, str],
    oracle_new_content: Dict[str, str],
    pred_new_content: Dict[str, str],
) -> Tuple[float, dict]:
    """
    Compute the SWE-RL reward given the code context, oracle patch, and the model output.
    Note that this function is a general version of the reward calculation, which can be used
    for code changes in any form, not just search/replace edits. For search/replace edits, use
    `calculate_search_replace_reward`.

    The return value is always within the range of [0, 1].

    Args:
        code_context: path -> original content of the file. It doesn't need to
            contain the entire codebase, only the files that are affected by the oracle patch.
        oracle_new_content: path -> oracle new content of the file after change.
        pred_new_content: path -> predicted new content of the file after change.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    # Obtain a unified diff for each file, for both the predicted and the oracle patch
    oracle_patch = get_normalized_patch(code_context, oracle_new_content)
    pred_patch = get_normalized_patch(code_context, pred_new_content)
    # Calculate the reward based on the similarity between the predicted and the oracle patch
    similarities = compute_change_similarities(pred_patch, oracle_patch)
    # assert len(similarities) > 0
    # This means oracle_patch and pred_patch are both empty, then they are identical and we reward 1.0
    if len(similarities) == 0:
        assert len(oracle_patch) == 0 and len(pred_patch) == 0
        return 1.0, dict(similarities=[])
    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


def calculate_search_replace_reward(
    code_context: Dict[str, str],
    oracle_new_content: Dict[str, str],
    output: str,
) -> Tuple[float, dict]:
    """
    The search/replace version of the reward calculation. It expects the output to contain
    the thought and solution in the following format:
    <think>
    ...
    </think>
    <solution>
    ...
    </solution>

    Args:
        code_context: path -> original content of the file.
        oracle_new_content: path -> oracle new content of the file after change.
        output: The output from the model containing the thought and solution.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    try:
        # Extract the thought and solution from the output
        thought, answer = extract_thought_solution(output)
        # Parse the search/replace edits from the solution
        pred_search_replaces = parse_search_replace(answer)
        if len(pred_search_replaces) == 0:
            raise FormatError("No valid search blocks found")
        # Get the new content of each file after applying the search/replace edits
        pred_new_content = apply_code_change(code_context, pred_search_replaces)
        reward, metadata = calculate_reward(
            code_context, oracle_new_content, pred_new_content
        )
        metadata["thought"] = thought
        metadata["answer"] = answer
        return reward, metadata
    except FormatError as e:
        return -1.0, dict(error=str(e))


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """
    Compute score for search-replace based code changes (VERL-compatible interface).
    
    For compatibility with existing VERL reward functions, this function compares
    the model's output containing search-replace blocks against a ground truth diff.
    
    Args:
        solution_str: The model's output containing search-replace edits in <think>/<solution> format
        ground_truth: Expected unified diff 
        extra_info: Additional information (optional, may contain file context)
        
    Returns:
        float: Score between -1.0 and 1.0
    """
    try:
        # First, try to extract thought and solution with search-replace blocks
        try:
            thought, answer = extract_thought_solution(solution_str)
            pred_search_replaces = parse_search_replace(answer)
            
            # If we have file context in extra_info, use the full search-replace reward
            if extra_info and isinstance(extra_info, dict):
                if "code_context" in extra_info and "oracle_new_content" in extra_info:
                    reward, _ = calculate_search_replace_reward(
                        extra_info["code_context"],
                        extra_info["oracle_new_content"], 
                        solution_str
                    )
                    return reward
        except FormatError:
            # If format is wrong, fall back to diff comparison
            pass
            
        # Fall back to diff-based comparison (similar to other reward functions)
        # Extract diff from solution (after </think> if present)
        think_splits = solution_str.split("</think>")
        after_think = think_splits[1].strip() if len(think_splits) == 2 else solution_str
        
        # Try to extract a diff block
        import re
        diff_patterns = [
            r"```diff\s*(.*?)\s*```",  # Standard diff block
            r"<diff>\s*(.*?)\s*</diff>",  # XML-style diff
            r"<patch>\s*(.*?)\s*</patch>",  # Patch tag
        ]
        
        model_diff = ""
        for pattern in diff_patterns:
            matches = re.findall(pattern, after_think, re.DOTALL)
            if matches:
                model_diff = matches[-1].strip() + "\n"  # Take the last match
                break
                
        if not model_diff:
            # If we found search-replace blocks but no diff, it's still valid format
            # Just can't compute similarity without file context
            try:
                thought, answer = extract_thought_solution(solution_str)
                pred_search_replaces = parse_search_replace(answer)
                if len(pred_search_replaces) > 0:
                    # Valid search-replace format but no context to compute diff
                    return 0.0  # Return 0 instead of -1 to indicate valid format
            except:
                pass
            return -1.0
            
        # Compare diffs using sequence matcher
        score = difflib.SequenceMatcher(
            None,
            a=model_diff,
            b=ground_truth,
            autojunk=False,
        ).ratio()
        
        return score
        
    except Exception:
        return -1.0


# Additional functions for unified diff handling
try:
    from unidiff import PatchedFile, PatchSet
    from unidiff.errors import UnidiffParseError
    UNIDIFF_AVAILABLE = True
except ImportError:
    UNIDIFF_AVAILABLE = False
    warnings.warn("unidiff package not available. Some features will be limited.")


def get_filelevel_diff(patch_text: str) -> Dict[str, str]:
    """
    Convert a unified diff text into a dictionary of file patches.
    """
    if not UNIDIFF_AVAILABLE:
        return {}
        
    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as e:
        # NOTE: sometimes unidiff throws other exceptions (e.g. UnboundLocalError) than
        # UnidiffParseError, which is unexpected, but we should still handle it.
        warnings.warn(f"Unexpected unidiff parsing error: {str(e)}")
        return {}
    
    result = {}
    for patchfile in patch:
        patchfile: PatchedFile = patchfile
        if patchfile.is_binary_file:
            # We don't consider binary files
            continue
        if patchfile.is_rename:
            # Add a special header for renamed files
            source_file = patchfile.source_file
            target_file = patchfile.target_file
            if source_file.startswith("a/"):
                source_file = source_file[2:]
            if target_file.startswith("b/"):
                target_file = target_file[2:]
            header = f"rename from {source_file} to {target_file}"
            path = source_file
        else:
            header = ""
            path = patchfile.path
        body = "\n".join(str(hunk).strip() for hunk in patchfile)
        content = header + "\n" + body
        content = content.strip()
        result[path] = content
    return result


def calculate_reward_unidiff(
    oracle_patches: List[str], pred_patches: List[str]
) -> Tuple[float, dict]:
    """
    Compute the SWE-RL reward given two sets of unified diffs.

    The return value is always within the range of [0, 1].

    Args:
        oracle_patches: A list of oracle diffs.
        pred_patches: A list of predicted diffs.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    # Calculate the reward based on the similarity between the predicted and the oracle patch
    pred_patch_dict = {}
    oracle_patch_dict = {}

    for patch_text in oracle_patches:
        oracle_patch_dict.update(get_filelevel_diff(patch_text))

    for patch_text in pred_patches:
        pred_patch_dict.update(get_filelevel_diff(patch_text))

    similarities = compute_change_similarities(pred_patch_dict, oracle_patch_dict)
    if len(similarities) == 0:
        assert len(pred_patch_dict) == 0 and len(oracle_patch_dict) == 0
        return 1.0, dict(similarities=[])
    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


if __name__ == "__main__":
    """
    Test the search-replace reward function.
    """
    # Example 1: Test with search-replace format
    example_output = """<think>
I need to fix the bug in the calculate function.
</think>
<solution>
```python
### utils/math.py
<<<<<<< SEARCH
def calculate(a, b):
    return a + b
=======
def calculate(a, b):
    if b == 0:
        return None
    return a / b
>>>>>>> REPLACE
```
</solution>"""

    example_context = {
        "utils/math.py": "def calculate(a, b):\n    return a + b"
    }
    
    example_oracle = {
        "utils/math.py": "def calculate(a, b):\n    if b == 0:\n        return None\n    return a / b"
    }
    
    # Test the full search-replace reward calculation
    reward, metadata = calculate_search_replace_reward(
        example_context,
        example_oracle,
        example_output
    )
    print(f"Search-Replace Reward: {reward}")
    print(f"Metadata: {metadata}")
    
    # Example 2: Test VERL-compatible interface with diff
    example_ground_truth_diff = """@@ -1,2 +1,5 @@
 def calculate(a, b):
-    return a + b
+    if b == 0:
+        return None
+    return a / b"""
    
    score = compute_score(example_output, example_ground_truth_diff)
    print(f"\nVERL-compatible score (with search-replace format): {score}")
    
    # Example 3: Test with regular diff format
    example_diff_output = """<think>
Need to handle division by zero.
</think>
```diff
@@ -1,2 +1,5 @@
 def calculate(a, b):
-    return a + b  
+    if b == 0:
+        return None
+    return a / b
```"""
    
    score2 = compute_score(example_diff_output, example_ground_truth_diff)
    print(f"VERL-compatible score (with diff format): {score2}")