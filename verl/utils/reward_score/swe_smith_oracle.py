# adapted from SWE-bench and SWE-smith
import re
import cydifflib

DIFF_PATTERN = re.compile(r"^diff(?:.*)")
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)

# from swe-bench
def get_first_idx(charlist):
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)

def get_last_idx(charlist):
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1

def strip_content(hunk):
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1

def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta

def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]

def extract_minimal_patch(model_patch):
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        diff_header = DIFF_PATTERN.findall(patch)
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch

def score_patching(model_diff, ground_truth, debug=False):
    """
    Score how well the model's patches match the expected patches.

    Args:
        verification_info (dict): Contains the golden diff
        debug (bool): Whether to print debug information

    Returns:
        float: Score between -1.0 and 1.0
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
        return -1.0


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
        float: Score between -1.0 and 1.0
    """
    ground_truth = extract_minimal_patch(ground_truth)
    
    try:
        think_splits = solution_str.split("</think>")
        after_think = think_splits[1].strip() if len(think_splits) == 2 else ""
        if not after_think:
            return -1.0
        model_diff = extract_diff(after_think)
        model_diff = extract_minimal_patch(model_diff)
        if not model_diff:
            return -1.0
        
        verification_info = {
            "golden_diff": ground_truth,
        }
        # verification_info = json.loads(extra_info)

        return score_patching(model_diff, ground_truth=verification_info["golden_diff"], debug=False)
        
    except Exception:
        return -1.0


if __name__ == "__main__":
    import pandas as pd
    import os
    
    # Path to the processed parquet file
    data_path = os.path.expanduser("~/persistent/data/github_patches_4k/test.parquet")
    
    # Load the processed data
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")
    
    first_row = df.iloc[1]
    ground_truth = first_row['reward_model']['ground_truth']
    # print(f"Ground truth: {ground_truth}")
    # extra_info = first_row['extra_info']

    solution_str = """\
diff --git a/cupyx/scipy/ndimage/filters.py b/cupyx/scipy/ndimage/filters.py
--- a/cupyx/scipy/ndimage/filters.py
+++ b/cupyx/scipy/ndimage/filters.py
@@ -1,5 +1,3 @@
-import numpy
-
 import cupy
 from cupy import util
 
@@ -80,7 +78,7 @@
 
 def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                            convolution):
-    if input.dtype in (numpy.complex64, numpy.complex128, numpy.complex256):
+    if input.dtype.kind == 'c':
         raise TypeError('Complex type not supported.')
     if not hasattr(origin, '__getitem__'):
         origin = [origin, ] * input.ndim
"""

    score = compute_score(solution_str, ground_truth)
    print(f"Score: {score}")
