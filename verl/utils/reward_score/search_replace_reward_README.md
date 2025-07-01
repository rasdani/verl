# Search-Replace Reward Function

This module implements a sophisticated reward function for evaluating code changes expressed in a search-replace format, ported from Meta's implementation for VERL compatibility.

## Overview

The search-replace reward function evaluates how well a model's code changes match expected changes. It supports two formats:

1. **Search-Replace Format**: Model outputs contain explicit search-replace blocks
2. **Unified Diff Format**: Traditional unified diff format (fallback)

## Key Features

- **Precise Change Evaluation**: Compares actual code changes rather than just diff text
- **Multi-file Support**: Handles changes across multiple files
- **Format Flexibility**: Works with both search-replace and diff formats
- **Detailed Metadata**: Returns similarity scores and change details

## Usage

### Basic Usage with Search-Replace Format

```python
from verl.utils.reward_score.search_replace_reward import calculate_search_replace_reward

# Original file contents
code_context = {
    "src/main.py": "def hello():\n    return 'Hello'"
}

# Expected changes
oracle_new_content = {
    "src/main.py": "def hello():\n    return 'Hello, World!'"
}

# Model output with search-replace blocks
model_output = """<think>
I need to update the return value.
</think>
<solution>
```python
### src/main.py
<<<<<<< SEARCH
def hello():
    return 'Hello'
=======
def hello():
    return 'Hello, World!'
>>>>>>> REPLACE
```
</solution>"""

# Calculate reward
reward, metadata = calculate_search_replace_reward(
    code_context, oracle_new_content, model_output
)
print(f"Reward: {reward}")  # 1.0 for perfect match
```

### VERL-Compatible Interface

```python
from verl.utils.reward_score import default_compute_score

# For use with the VERL training pipeline
score = default_compute_score(
    "github_patches_search_replace",
    model_output,
    ground_truth_diff,
    extra_info={
        "code_context": code_context,
        "oracle_new_content": oracle_new_content
    }
)
```

## Output Format

The model output should follow this format:

```
<think>
[Reasoning about the changes]
</think>
<solution>
```python
### path/to/file.py
<<<<<<< SEARCH
[Original code to be replaced]
=======
[New code to replace with]
>>>>>>> REPLACE
```
</solution>
```

## Reward Calculation

The reward is calculated as:
1. Parse search-replace blocks from the model output
2. Apply changes to create new file contents
3. Generate unified diffs for both predicted and oracle changes
4. Compare diffs using sequence matching
5. Average similarity scores across all files

Returns:
- **1.0**: Perfect match
- **0.0-1.0**: Partial match based on similarity
- **-1.0**: Invalid format or error

## Dataset Preprocessing

Use the provided preprocessing script for GitHub patches datasets:

```bash
python examples/data_preprocess/github_patches_search_replace.py \
    --dataset_name "rasdani/SWE-smith-oracle-4k-context-1k-diff" \
    --local_dir ~/data/github_patches_sr \
    --max_samples 1000
```

## Integration with VERL

The reward function is registered as `"github_patches_search_replace"` in VERL's reward registry and can be used directly in training configurations.