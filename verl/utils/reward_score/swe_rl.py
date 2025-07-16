# @article{wei2025swerl,
#   title={SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution},
#   author={Yuxiang Wei and Olivier Duchenne and Jade Copet and Quentin Carbonneaux and Lingming Zhang and Daniel Fried and Gabriel Synnaeve and Rishabh Singh and Sida I. Wang},
#   year={2025},
#   journal={arXiv preprint arXiv:2502.18449}
# }

import json
import re
from collections import defaultdict

import cydifflib
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from verl.utils.reward_score.swe_rl_utils.utils import FileDiff, extract_minimal_patch

EDITS_PATTERN = re.compile(
    r"```.*?\n"
    r"### (.*)\n"
    r"<<<<<<< SEARCH\n"
    r"([\s\S]*?)\n"
    r"=======\n"
    r"([\s\S]*?)\n"
    r">>>>>>> REPLACE\n"
    r"```"
)


def parse_thinking(completion: str) -> str:
    think_splits = completion.split("</think>")
    after_think = think_splits[1].strip() if len(think_splits) == 2 else ""
    return after_think


def parse_edits(input_text: str) -> dict[str, list[tuple[str, str]]]:
    edits = defaultdict(list)
    matches = EDITS_PATTERN.finditer(input_text)
    for match in matches:
        file_path = match.group(1)
        search_content = match.group(2)
        replace_content = match.group(3)
        edits[file_path].append((search_content, replace_content))
    return edits


def create_patched_file_context(
    edited_file_context: dict[str, str],
    file_diffs: list[FileDiff],
) -> dict[str, str]:
    patch_dict = dict[str, str]()
    for file_diff in file_diffs:
        file_path = file_diff.header.file.path
        # update the file content with the edited file context
        file_diff.new_file_content = edited_file_context.get(file_path, "")
        file_diff.generate_hunks_from_content()
        predicted_patch = file_diff.get_patch()
        if predicted_patch.strip():
            patch_dict[file_path] = predicted_patch
    return patch_dict


def get_unidiff_from_patched_file_context(patched_file_context: dict[str, str]) -> str:
    try:
        patches = list(patched_file_context.values())
        first_patch = patches.pop(0)
        patch_set = PatchSet(first_patch)
        for patch in patches:
            patch_set.extend(PatchSet(patch))
        return str(patch_set)
    except UnidiffParseError:
        return ""


def apply_edits(file_context: dict[str, str], edits: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    edited_file_context = {}
    for file_path, file_edits in edits.items():
        edited_file_content = f"\n{file_context.get(file_path, '')}"
        for search_str, replace_str in file_edits:
            if search_str not in edited_file_content:
                return None
            edited_file_content = edited_file_content.replace(f"\n{search_str}", f"\n{replace_str}")
        edited_file_context[file_path] = edited_file_content.lstrip("\n")
    return edited_file_context


def score_patch(pred_patch: str, oracle_patch: str) -> float:
    try:
        score = cydifflib.SequenceMatcher(
            None,
            a=pred_patch,
            b=oracle_patch,
            autojunk=False,
        ).ratio()
        return score
    except Exception:
        return -1.0


def score(solution_str: str, file_context: dict[str, str], file_diffs: list[FileDiff], oracle_patch: str) -> float:
    after_think = parse_thinking(solution_str)
    edits = parse_edits(after_think)
    if len(edits) == 0:
        return -1.0
    edited_file_context = apply_edits(file_context, edits)
    if edited_file_context is None:
        return -1.0
    patched_file_context = create_patched_file_context(edited_file_context, file_diffs)
    pred_patch = get_unidiff_from_patched_file_context(patched_file_context)
    min_pred_patch = extract_minimal_patch(pred_patch)
    min_oracle_patch = extract_minimal_patch(oracle_patch)
    return score_patch(min_pred_patch, min_oracle_patch)


def compute_score(solution_str: str, ground_truth: str, extra_info: dict[str, str] = None) -> float:
    parsed_commit_content = json.loads(extra_info["parsed_commit_content"])
    file_diffs = parsed_commit_content.get("file_diffs")
    file_diffs = [FileDiff(**file_diff) for file_diff in file_diffs]

    # verification_info = extra_info["verification_info"]
    file_context = json.loads(extra_info["file_context"])

    try:
        return score(
            solution_str=solution_str, file_context=file_context, file_diffs=file_diffs, oracle_patch=ground_truth
        )
    except Exception:
        return -1.0
