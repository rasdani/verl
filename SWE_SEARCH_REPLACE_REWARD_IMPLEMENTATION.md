# SWE Search-Replace Reward Function Implementation

## Overview

This document describes the implementation of a new reward function for SWE (Software Engineering) search-replace tasks, ported from Meta's implementation and integrated into the VERL framework.

## Implementation Details

### New File: `verl/utils/reward_score/swe_search_replace.py`

This file contains the complete implementation of the search-replace reward function, including:

#### Key Features

1. **Search-Replace Format Support**: Handles solutions in search-replace format with `<think></think>` and `<solution></solution>` tags
2. **Unified Diff Comparison**: Compares patches using difflib similarity matching
3. **File-level Change Analysis**: Analyzes changes at the file level for accurate scoring
4. **Flexible Input Handling**: Supports both search-replace blocks and direct unified diff inputs
5. **Error Handling**: Graceful fallback for malformed inputs (returns -1.0)

#### Core Functions

- `extract_thought_solution()`: Parses `<think>` and `<solution>` tags from model output
- `parse_search_replace()`: Extracts search-replace blocks from solution text
- `apply_code_change()`: Applies search-replace edits to code context
- `generate_unified_diff()`: Creates unified diffs between old and new code
- `calculate_reward()`: Computes similarity-based rewards for code changes
- `calculate_reward_unidiff()`: Computes rewards directly from unified diff patches
- `compute_score()`: Main entry point compatible with VERL framework

#### Dependencies

- **Required**: `difflib`, `re`, `warnings`, `typing`
- **Optional**: `unidiff` (for enhanced unified diff parsing)
  - Install with: `pip install unidiff`
  - Function works without it but with reduced functionality

### Integration with VERL Framework

#### Updated File: `verl/utils/reward_score/__init__.py`

Added support for the new reward function by adding:

```python
elif data_source == "swe_search_replace":
    from . import swe_search_replace
    res = swe_search_replace.compute_score(solution_str, ground_truth, extra_info)
```

## Usage

### Basic Usage

```python
from verl.utils.reward_score import default_compute_score

# For search-replace tasks
score = default_compute_score(
    data_source="swe_search_replace",
    solution_str=model_output,  # Contains <think> and <solution> tags
    ground_truth=expected_diff,  # Unified diff format
    extra_info=None  # Optional additional info
)
```

### Direct Function Usage

```python
from verl.utils.reward_score.swe_search_replace import compute_score

score = compute_score(solution_str, ground_truth, extra_info)
```

### Expected Input Formats

#### Model Output Format
```
<think>
I need to analyze the issue and make the appropriate changes.
</think>
<solution>
```diff
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello World")
+    print("Hello, World!")
```
</solution>
```

#### Ground Truth Format
```
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello World")
+    print("Hello, World!")
```

## Scoring Methodology

The reward function uses the following scoring approach:

1. **Patch Extraction**: Extracts unified diff patches from both predicted and ground truth
2. **File-level Analysis**: Breaks down patches by file for granular comparison
3. **Similarity Calculation**: Uses `difflib.SequenceMatcher` to compute similarity ratios
4. **Aggregation**: Averages similarity scores across all modified files
5. **Range**: Returns scores in range [0.0, 1.0], with -1.0 for format errors

### Score Interpretation

- **1.0**: Perfect match between predicted and ground truth patches
- **0.5-0.9**: Partial match with some differences
- **0.0-0.4**: Poor match with significant differences
- **-1.0**: Format error or unparseable input

## Compatibility

### Dataset Compatibility

The implementation is compatible with datasets created by `examples/data_preprocess/github_patches.py`, which expect:

- Data source: "github_patches" (currently) or "swe_search_replace" (new)
- Reward model format with ground truth in unified diff format
- Standard VERL data structure

### Backward Compatibility

The existing "github_patches" data source continues to use `swe_smith_oracle.py`. The new reward function is available as "swe_search_replace" to avoid breaking existing configurations.

## Testing

The implementation includes built-in tests that verify:

1. **Perfect Match**: Identical patches receive score ≈ 1.0
2. **Format Errors**: Invalid formats receive score = -1.0
3. **Partial Matches**: Different patches receive intermediate scores
4. **Graceful Degradation**: Works without optional dependencies

## Error Handling

The function handles various error conditions:

- **Missing unidiff package**: Falls back to basic string comparison
- **Malformed input**: Returns -1.0 for unparseable solutions
- **Missing tags**: Attempts direct diff extraction
- **Empty patches**: Handles empty/no-change scenarios

## Performance Considerations

- **Efficient**: Uses compiled regex patterns for parsing
- **Memory-conscious**: Processes patches incrementally
- **Scalable**: File-level analysis allows handling large codebases
- **Robust**: Multiple fallback mechanisms for edge cases

## Future Enhancements

Potential improvements for future versions:

1. **Enhanced Search-Replace Support**: Full support for search-replace block parsing with code context
2. **Semantic Analysis**: Consider semantic similarity in addition to textual similarity
3. **Multi-file Coordination**: Better handling of related changes across multiple files
4. **Performance Optimization**: Caching and optimization for large-scale evaluation

## Migration Guide

To migrate from existing reward functions:

1. Update `data_source` from "github_patches" to "swe_search_replace"
2. Ensure model outputs include `<think>` and `<solution>` tags
3. Verify unified diff format in ground truth data
4. Install `unidiff` package for full functionality: `pip install unidiff`

## Conclusion

The new SWE search-replace reward function provides a robust, flexible solution for evaluating code modification tasks in the VERL framework. It maintains compatibility with existing datasets while offering enhanced functionality for search-replace based approaches.