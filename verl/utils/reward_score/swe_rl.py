# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Ported to VERL by open-source contributors.
"""SWE-RL reward implementation.

This module provides a `compute_score` function with the same signature as the other
reward helpers in :pymod:`verl.utils.reward_score`.  Internally, it computes a
diff-level similarity between the model-predicted patch(es) and the oracle patch(es)
using the algorithm described in the SWE-RL paper.

It is **fully compatible** with the dataset produced by
`examples/data_preprocess/github_patches.py` – simply call

>>> from verl.utils.reward_score import swe_rl
>>> score = swe_rl.compute_score(model_output, golden_diff)

Just like the other helpers, the returned value lies in ``[0, 1]`` when the input
is well-formed and ``-1.0`` when the model output cannot be parsed.
"""

from __future__ import annotations

import difflib
import re
import warnings
from typing import List, Tuple, TypedDict

# The `unidiff` package is used to parse unified-diff strings.  In case it is
# missing from the runtime environment we fall back to a *very* naive parser
# which treats the whole diff as a single change under an artificial file path
# so that evaluation can still proceed (albeit with slightly degraded
# granularity).  This makes the reward function robust to environments where
# installing extra dependencies is cumbersome.

try:
    from unidiff import PatchedFile, PatchSet  # type: ignore
    from unidiff.errors import UnidiffParseError  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback implementation

    class _DummyPatchedFile:  # minimal stand-in so type checkers are satisfied
        def __init__(self, diff_text: str):
            self.is_binary_file = False
            self.is_rename = False
            self.path = "__unknown__"
            self._text = diff_text

        def __iter__(self):
            yield self._text

    class _DummyPatchSet(list):
        def __init__(self, diff_text: str):
            super().__init__([_DummyPatchedFile(diff_text)])

    PatchedFile = _DummyPatchedFile  # type: ignore
    PatchSet = _DummyPatchSet  # type: ignore

    class UnidiffParseError(Exception):
        pass

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

# Matches blocks of the form:
# ```
# ### <path>
# <<<<<<< SEARCH
# <search>
# =======
# <replace>
# >>>>>>> REPLACE
# ```
SEARCH_REPLACE_REGEX = (
    r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"
)


class FormatError(Exception):
    """Raised when the model output does not follow the expected format."""


# ---------------------------------------------------------------------------
# Extract <think> / <solution> blocks
# ---------------------------------------------------------------------------

def _extract_thought_solution(output: str) -> Tuple[str, str]:
    """Return the stripped text inside the <think> and <solution> tags."""
    for tag in (THINK_START, THINK_END, ANSWER_START, ANSWER_END):
        if output.count(tag) != 1:
            raise FormatError(f"count of {tag} is not 1")

    thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
    answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()

    if not thought:
        raise FormatError("Thought is empty")
    return thought, answer


# ---------------------------------------------------------------------------
# Search / replace utilities (unused for unidiff reward today, but kept for
#                 future compatibility with the full SWE-RL benchmark)
# ---------------------------------------------------------------------------

def _parse_search_replace(text: str) -> dict[str, List[Tuple[str, str]]]:
    """Parse all search-replace directives from *text*.

    Returns a mapping ``path -> [(search, replace), ...]``.
    """
    path_search_replaces = re.findall(SEARCH_REPLACE_REGEX, text)
    path_search_replace_dict: dict[str, List[Tuple[str, str]]] = {}
    for path, search, replace in path_search_replaces:
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


def _apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, List[Tuple[str, str]]],
    *,
    silent: bool = False,
) -> dict[str, str]:
    """Apply *search_replace_dict* to *code_context* and return the new content."""
    new_content_dict: dict[str, str] = {}
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if not silent and search not in new_content:
                raise FormatError(f"Search block not found in the code: {search}")
            new_content = new_content.replace(search, replace)
        new_content_dict[path] = new_content[1:]  # strip leading \n
    return new_content_dict


# ---------------------------------------------------------------------------
# Diff utilities
# ---------------------------------------------------------------------------

def _generate_unified_diff(old_code: str, new_code: str, n_context: int = 3) -> str:
    """Return a unified diff between *old_code* and *new_code*."""
    diff_iter = difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    # Skip the header lines produced by difflib
    try:
        next(diff_iter)
        next(diff_iter)
    except StopIteration:
        return ""
    return "\n".join(diff_iter)


def _get_normalized_patch(code_ctx: dict[str, str], new_content: dict[str, str]) -> dict[str, str]:
    """Return ``path -> unified diff`` for all files that changed."""
    patches: dict[str, str] = {}
    for path, new_code in new_content.items():
        old_code = code_ctx.get(path, "")
        diff_text = _generate_unified_diff(old_code, new_code)
        if diff_text:  # skip empty diffs
            patches[path] = diff_text
    return patches


class _ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def _compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> List[_ChangeSimilarity]:
    """Compute per-file diff similarity using :pyclass:`difflib.SequenceMatcher`."""
    similarities: List[_ChangeSimilarity] = []
    for path in set(oracle_patch) | set(pred_patch):
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if not oracle_change or not pred_change:
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None, pred_change, oracle_change, autojunk=False
            ).ratio()
        similarities.append(
            {
                "path": path,
                "pred_change": pred_change,
                "oracle_change": oracle_change,
                "similarity": change_similarity,
            }
        )
    return similarities


# ---------------------------------------------------------------------------
# File-level helpers for unified diff similarity (used in VERL dataset)
# ---------------------------------------------------------------------------

def _get_filelevel_diff(patch_text: str) -> dict[str, str]:
    """Convert a unified diff *patch_text* to ``path -> diff`` mapping."""
    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as exc:  # pragma: no cover – defensive coding
        warnings.warn(f"Unexpected unidiff parsing error: {exc}")
        return {}

    result: dict[str, str] = {}
    for patched_file in patch:
        if patched_file.is_binary_file:
            continue  # ignore binaries

        if patched_file.is_rename:
            src = patched_file.source_file[2:] if patched_file.source_file.startswith("a/") else patched_file.source_file
            tgt = patched_file.target_file[2:] if patched_file.target_file.startswith("b/") else patched_file.target_file
            header = f"rename from {src} to {tgt}\n"
            path = src  # diff similarity keyed by original path
        else:
            header = ""
            path = patched_file.path
        body = "\n".join(str(hunk).strip() for hunk in patched_file)
        result[path] = (header + body).strip()
    return result


def _calculate_reward_unidiff(
    oracle_patches: List[str],
    pred_patches: List[str],
) -> Tuple[float, dict]:
    """Return *(reward, metadata)* using unified-diff similarity."""
    oracle_patch_dict: dict[str, str] = {}
    for txt in oracle_patches:
        oracle_patch_dict.update(_get_filelevel_diff(txt))

    pred_patch_dict: dict[str, str] = {}
    for txt in pred_patches:
        pred_patch_dict.update(_get_filelevel_diff(txt))

    similarities = _compute_change_similarities(pred_patch_dict, oracle_patch_dict)
    if not similarities:
        assert not pred_patch_dict and not oracle_patch_dict
        return 1.0, {"similarities": []}

    reward_val = sum(s["similarity"] for s in similarities) / len(similarities)
    return reward_val, {"similarities": similarities}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_score(solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Compute similarity score between *solution_str* and *ground_truth*.

    The implementation is **robust** to various formatting styles found in LLM
    outputs – it takes everything after the closing ``</think>`` tag and tries
    to extract a ``diff``/``patch`` code block.  If no such block is found, the
    entire remainder is treated as a patch text.
    """
    # Extract part after </think>
    think_split = solution_str.split("</think>")
    after_think = think_split[1].strip() if len(think_split) == 2 else solution_str.strip()
    if not after_think:
        return -1.0

    # We reuse the robust diff extractor from swe_smith_oracle to keep behaviour
    try:
        from . import swe_smith_oracle  # local import to avoid cycles

        model_patch = swe_smith_oracle.extract_diff(after_think)
    except Exception:  # fallback to simple regex if anything goes wrong
        code_block_match = re.search(r"```diff\s*(.*?)```", after_think, re.DOTALL)
        model_patch = code_block_match.group(1).strip() if code_block_match else after_think

    if not model_patch:
        return -1.0

    try:
        reward, _ = _calculate_reward_unidiff([ground_truth], [model_patch])
        return float(reward)
    except Exception:
        # Defensive: never crash the learner because of the reward
        return -1.0


__all__ = [
    "compute_score",
]