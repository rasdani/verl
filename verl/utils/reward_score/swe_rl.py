# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import difflib
import re
import warnings
from typing import TypedDict
import os
import subprocess
import uuid
import difflib
import re
import warnings
from typing import TypedDict
import ast

from unidiff import PatchedFile, PatchSet
from unidiff.errors import UnidiffParseError

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"

PLAYGROUND_DIR = "./tmp_swe_rl_playground"

class FormatError(Exception):
    pass

def extract_python_blocks(text):
    # Regular expression pattern to match ```python\n{text}\n```
    pattern = r"```python\n(.*?)\n```"

    # Use re.findall to find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    return matches

def split_edit_multifile_commands(commands: list[str]) -> dict[str, list[str]]:
    """Split commands based on edited files."""
    file_to_commands = OrderedDict()  # type: ignore
    for command in commands:
        file_name = None
        for subcommand in command.split(">>>>>>> REPLACE")[:-1]:
            subcommand = subcommand.strip()
            if "<<<<<<< SEARCH" in subcommand:
                fn = subcommand.split("<<<<<<< SEARCH")[0].lstrip("#").strip()
                if fn:
                    file_name = fn

            if len(subcommand.split("<<<<<<< SEARCH")) != 2:
                continue
            converted_command = (
                "<<<<<<< SEARCH"
                + subcommand.split("<<<<<<< SEARCH")[1]
                + "\n"
                + ">>>>>>> REPLACE"
            )
            # deduplicate
            if file_name is not None and (
                file_name not in file_to_commands
                or converted_command not in file_to_commands[file_name]
            ):
                file_to_commands.setdefault(file_name, []).append(converted_command)
    return file_to_commands

def parse_diff_edit_commands(commands: list[str], content: str) -> str:
    replaced = False
    # apply the edits from the end of file to the beginning of file
    # this is to make sure context is correct
    # since we want to replace the original context, let's first check for all edits.
    can_apply = []
    for subcommand in commands:
        if not subcommand.startswith("<<<<<<< SEARCH") and subcommand.endswith(
            ">>>>>>> REPLACE"
        ):
            continue

        subcommand = "\n".join(subcommand.splitlines()[1:-1])
        if len(subcommand.split("\n=======\n")) != 2:
            continue

        original, replace = subcommand.split("\n=======\n")

        if original in content:
            can_apply.append(subcommand)

    # apply edits backwards
    # for subcommand in can_apply[::-1]:
    # NOTE(yuxiang): 02/16, not needed; just apply them forwards
    for subcommand in can_apply:
        original, replace = subcommand.split("\n=======\n")
        content = content.replace(original, replace)
        replaced = True

    if not replaced:
        print("not replaced")

    return content

def _post_process_multifile_repair(
    raw_output: str, file_contents: dict[str, str]
) -> tuple[list[str], list[str]]:
    edit_multifile_commands = extract_python_blocks(raw_output)
    edited_files = list[str]()
    new_contents = list[str]()
    file_to_commands = split_edit_multifile_commands(edit_multifile_commands)

    for edited_file_key in file_to_commands:
        edited_file = ""
        new_content = ""
        edit_commands = file_to_commands[edited_file_key]
        edited_file = edited_file_key
        if edited_file not in file_contents:
            continue

        content = file_contents[edited_file]
        new_content = parse_diff_edit_commands(edit_commands, content)

        if edited_file == "" or new_content == "":
            continue
        edited_files.append(edited_file)
        new_contents.append(new_content)

    return edited_files, new_contents

def extract_thought_solution(output: str) -> tuple[str, str]:
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

def fake_git_repo(repo_playground, file_pathes, old_contents, new_contents) -> str:
    """create a fake git repo to obtain git diff format"""

    if not isinstance(file_pathes, list):
        # for backwards compatibility
        file_pathes = [file_pathes]
        old_contents = [old_contents]
        new_contents = [new_contents]

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    # create a fake git repo
    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    for file_path, old_content, new_content in zip(
        file_pathes, old_contents, new_contents
    ):
        # create a file
        subprocess.run(
            f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
        )

        with open(f"{repo_playground}/{file_path}", "w") as f:
            f.write(old_content)

        # add file to git
        # same message is okay
        subprocess.run(
            f"cd {repo_playground} && git add {file_path} && git commit -m 'initial commit'",
            shell=True,
        )

    for file_path, old_content, new_content in zip(
        file_pathes, old_contents, new_contents
    ):
        # edit file
        with open(f"{repo_playground}/{file_path}", "w") as f:
            f.write(new_content)

    # get git diff
    o = subprocess.run(
        f"cd {repo_playground} && git diff .", shell=True, capture_output=True
    )

    s = o.stdout.decode("utf-8")

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return s

def check_syntax(code):
    if not isinstance(code, list):
        code = [code]

    for c in code:
        if (
            not c.strip()
        ):  # Check for cases where the model didn't return a python block
            return False
        try:
            ast.parse(c)
        except SyntaxError as e:
            return False
    return True

def remove_empty_lines(code: str) -> str:
    # Split the code into lines
    lines = code.splitlines()
    # Remove empty lines
    filtered_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(filtered_lines)

def check_code_differ_by_just_empty_lines(codes, prev_codes) -> bool:

    if not isinstance(codes, list):
        codes = [codes]
        prev_codes = [prev_codes]

    normalized_code1 = ""
    normalized_code2 = ""

    for code, prev_code in zip(codes, prev_codes):
        # Normalize both code snippets
        normalized_code1 += remove_empty_lines(code)
        normalized_code2 += remove_empty_lines(prev_code)

    return normalized_code1 == normalized_code2

def post_process_raw_output(raw_output_text: str, file_contents: dict[str, str]):
    git_diffs = ""
    raw_git_diffs = ""
    edited_files = list[str]()
    new_contents = list[str]()
    contents = list[str]()
    edited_files, new_contents = _post_process_multifile_repair(
        raw_output_text, file_contents
    )

    contents = [file_contents[edited_file] for edited_file in edited_files]

    git_diff = fake_git_repo(
        PLAYGROUND_DIR, edited_files, contents, new_contents
    )

    raw_git_diffs += "\n" + git_diff.replace("\\ No newline at end of file\n", "")

    syntax_success = check_syntax(new_contents)

    differ_by_empty_lines = check_code_differ_by_just_empty_lines(
        new_contents, contents
    )

    if syntax_success and not differ_by_empty_lines:
        git_diffs = raw_git_diffs
    else:
        git_diffs = ""  # no need to evaluate

    return git_diffs, raw_git_diffs, contents, edited_files, new_contents


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A dictionary where the key is the file path and the value is a list of search/replace pairs.
    """
    path_search_replaces: list[tuple[str, str, str]] = re.findall(
        SEARCH_REPLACE_REGEX, text
    )
    path_search_replace_dict = dict[str, list[tuple[str, str]]]()
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

    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""


def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
    silent: bool = False,
) -> dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.
        silent: Whether to suppress the error messages.

    Returns:
        A dictionary containing the file path and the new content of the code.
    """
    new_content_dict = dict[str, str]()
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
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
) -> dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = dict[str, str]()
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        # but it can occur in general-purpose usages
        if patch:
            patch_dict[path] = patch
    return patch_dict


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> list[ChangeSimilarity]:
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = list[ChangeSimilarity]()
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # Both are empty changes, meaning search = replace. We should penalize this to avoid
            # the model predicting empty changes to hack the reward.
            # NOTE: this should not happen due to (1) the search == replace check in `apply_code_change`
            # and (2) the `if patch` check in `get_normalized_patch`.
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
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    pred_new_content: dict[str, str],
) -> tuple[float, dict]:
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
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    output: str,
) -> tuple[float, dict]:
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

def calculate_reward_against_oracle_patch(
    code_context: dict[str, str],
    oracle_patch: dict[str, str],
    pred_new_content: dict[str, str],
) -> tuple[float, dict]:
    """
    Compute the SWE-RL reward given the code context, oracle patch, and the model output.
    Note that this function is a general version of the reward calculation, which can be used
    for code changes in any form, not just search/replace edits. For search/replace edits, use
    `calculate_search_replace_reward`.

    The return value is always within the range of [0, 1].

    Args:
        code_context: path -> original content of the file. It doesn't need to
            contain the entire codebase, only the files that are affected by the oracle patch.
        oracle_patch: path -> oracle patch.
        pred_new_content: path -> predicted new content of the file after change.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    # Obtain a unified diff for each file, for both the predicted and the oracle patch
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

def calculate_search_replace_reward_against_oracle_patch(
    code_context: dict[str, str],
    oracle_patch: dict[str, str],
    output: str,
) -> tuple[float, dict]:
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
        oracle_patch: path -> oracle patch.
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
        reward, metadata = calculate_reward_against_oracle_patch(
            code_context, oracle_patch, pred_new_content
        )
        metadata["thought"] = thought
        metadata["answer"] = answer
        return reward, metadata
    except FormatError as e:
        return -1.0, dict(error=str(e))

def get_filelevel_diff(patch_text: str) -> dict[str, str]:
    """
    Convert a unified diff text into a dictionary of file patches.
    """
    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as e:
        # NOTE: sometimes unidiff throws other exceptions (e.g. UnboundLocalError) than
        # UnidiffParseError, which is unexpected, but we should still handle it.
        warnings.warn(f"Unexpected unidiff parsing error: {str(e)}")
        return {}
    result = dict[str, str]()
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
    oracle_patches: list[str], pred_patches: list[str]
) -> tuple[float, dict]:
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
    pred_patch_dict = dict[str, str]()
    oracle_patch_dict = dict[str, str]()

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
    oracle_patch = get_filelevel_diff(ground_truth)
    # verification_info = extra_info["verification_info"]
    code_context = json.loads(extra_info["code_context"])
    
    try:
        reward, metadata = calculate_search_replace_reward_against_oracle_patch(
            code_context=code_context,
            oracle_patch=oracle_patch,
            output=solution_str
        )
        return reward
        
    except Exception:
        return -1.0

if __name__ == "__main__":
    import datasets
    import json
    from collections import OrderedDict
    
    # Read data from parquet file
    parquet_file = "data/train.parquet"
    if not os.path.exists(parquet_file):
        print(f"Error: {parquet_file} does not exist")
        exit(1)
    
    print(f"Loading dataset from {parquet_file}...")
    dataset = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    
    if len(dataset) == 0:
        print("Error: No examples found in the dataset")
        exit(1)
    
    # Get the first example
    example = dataset[0]
    
    print("=== Example Data ===")
    print(f"Data source: {example['data_source']}")
    print(f"Ability: {example['ability']}")
    print(f"Reward model: {example['reward_model']}")
    
    # Extract ground truth and extra info
    ground_truth = example['reward_model']['ground_truth']
    extra_info = example['extra_info']
    
    print(f"\nGround truth type: {type(ground_truth)}")
    print(f"Extra info keys: {list(extra_info.keys())}")
    
    # Parse verification_info if it's a string
    # verification_info = extra_info['verification_info']
    # if isinstance(verification_info, str):
    #     verification_info = json.loads(verification_info)
    code_context = extra_info['code_context']
    
    
    # Hardcoded test solution
    solution_str = """<think>
This is a test solution for the SWE-RL reward computation.
I need to analyze the code and make the necessary changes.
</think>
<solution>
```
### test_file.py
<<<<<<< SEARCH
print("hello")
=======
print("hello world")
>>>>>>> REPLACE
```
</solution>"""
    
    # Compute the score
    try:
        score = compute_score(solution_str, ground_truth, extra_info)
        print(f"\n=== Result ===")
        print(f"Computed score: {score}")
        
        if score == -1.0:
            print("Score is -1.0, indicating an error in processing")
        elif score >= 0.0:
            print(f"Score is {score:.4f}, indicating similarity ratio")
        else:
            print("Unexpected score value")
            
    except Exception as e:
        print(f"Error computing score: {e}")
        import traceback
        traceback.print_exc()
