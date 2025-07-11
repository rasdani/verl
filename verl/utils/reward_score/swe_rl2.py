# @article{wei2025swerl,
#   title={SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution}, 
#   author={Yuxiang Wei and Olivier Duchenne and Jade Copet and Quentin Carbonneaux and Lingming Zhang and Daniel Fried and Gabriel Synnaeve and Rishabh Singh and Sida I. Wang},
#   year={2025},
#   journal={arXiv preprint arXiv:2502.18449}
# }

import re
import cydifflib
import json
from collections import defaultdict

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from verl.utils.reward_score.swe_rl.utils import FileDiff, extract_minimal_patch

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
    except UnidiffParseError as e:
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
    except Exception as e:
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
    parsed_commit_content = json.loads(extra_info["parsed_commit_content"])
    file_diffs = parsed_commit_content.get("file_diffs")
    file_diffs = [FileDiff(**file_diff) for file_diff in file_diffs]

    # verification_info = extra_info["verification_info"]
    file_context = json.loads(extra_info["file_context"])
    oracle_patch = ground_truth
    
    try:
        return score(solution_str, file_context, file_diffs, oracle_patch)
    except Exception:
        return -1.0

if __name__ == "__main__":
    solution_str = """\
<think>
Okay, let's try to figure out why the user is getting that error. The issue is about the `sliding_window_inference` function in MONAI not preserving gradients. The user provided a code example where they create a tensor with `requires_grad=True`, pass it through the sliding window inference, and then try to do a backward pass. But they get an error saying that the result tensor doesn't require grad and has no grad_fn.

Hmm. The error message points to the line where they call `result.sum().backward()`. The traceback shows that the result tensor doesn't require grad. So, the problem is that the `sliding_window_inference` function is somehow detaching the gradient, which is not desired here.

Looking at the code for `sliding_window_inference`, I need to find where gradients might be detached. The user mentioned that the detach is happening in the line where they do `output_image_list[ss] = output_image_list[ss].detach()`. Wait, that's in the code after processing all the slices. So, the function is detaching the output tensors, which would prevent gradients from flowing back through that operation.

But why is that happening? The user says that the function shouldn't always remove gradients. The original code probably detaches the output to avoid accumulating gradients during the sliding window process, but in this case, the user is trying to compute gradients with respect to the input, so detaching is causing the problem.

So the fix would be to remove the `.detach()` call from the output tensors. Let me check the code again. In the part where they process the outputs, there's a loop where they do:

output_image_list[ss] = output_image_list[ss].detach()

That's the line that's detaching the tensor. If we remove that line, the gradients should flow through the inference process. But wait, why was that line there in the first place? Maybe the original intention was to prevent the gradients from being accumulated during the sliding window steps, but in this case, the user wants to track gradients through the entire process.

So the correct fix is to remove the `.detach()` call from the output tensors. That way, when the backward pass is called, the gradients can be computed properly.

Looking at the code in the provided file, the line in question is:

output_image_list[ss] = output_image_list[ss].detach()

So the SEARCH block would find this line, and the REPLACE block would remove the `.detach()` part. That should fix the issue where the gradients are being detached, allowing the backward pass to work as intended.
</think>

<solution>
The issue arises because the `sliding_window_inference` function is detaching the output tensors, which prevents gradient tracking. This is done in the line `output_image_list[ss] = output_image_list[ss].detach()`. To fix this, we need to remove the `.detach()` call so gradients can flow through the inference process.

Here is the fix:

```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        self.batch_size = batch_size
        self.temperature = temperature
=======
        # comment out the line below
        self.temperature = temperature
        # comment out the line below
>>>>>>> REPLACE
```

```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        negatives_mask = ~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input.device)
=======
        negatives_mask = ~torch.eye(input.shape[0] * 2, input.shape[0] * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input.device)
>>>>>>> REPLACE
```

```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        sim_ij = torch.diag(sim_matrix, self.batch_size)
        sim_ji = torch.diag(sim_matrix, -self.batch_size)
=======
        sim_ij = torch.diag(sim_matrix, input.shape[0])
        # comment out the line below
        sim_ji = torch.diag(sim_matrix, -input.shape[0])
>>>>>>> REPLACE
```
</solution>
"""

    import datasets
    from time import perf_counter

    ds = datasets.load_dataset("rasdani/SkyRL-v0-293-data-oracle-8k-context", split="train")
    # ds = datasets.load_dataset("rasdani/SkyRL-v0-293-data-oracle", split="train")
    ground_truth = ds[0]['patch']
    extra_info = {
        "parsed_commit_content": ds[0]['parsed_commit_content'],
        "file_context": ds[0]['file_context']
    }
    start_time = perf_counter()
    result = compute_score(solution_str, ground_truth, extra_info)
    end_time = perf_counter()
    print(result)
    print(f"Time taken: {end_time - start_time} seconds")
    