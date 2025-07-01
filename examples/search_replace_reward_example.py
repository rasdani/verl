#!/usr/bin/env python3
"""
Example of using the search-replace reward function with VERL.

This demonstrates how to:
1. Format model outputs with search-replace blocks
2. Calculate rewards using the search-replace reward function
3. Compare with traditional diff-based rewards
"""

from verl.utils.reward_score.search_replace_reward import (
    calculate_search_replace_reward,
    compute_score,
)
from verl.utils.reward_score import default_compute_score


def example_search_replace_reward():
    """Demonstrate the search-replace reward calculation."""
    
    # Example 1: Perfect match case
    print("=== Example 1: Perfect Match ===")
    
    # Original file content
    code_context = {
        "src/calculator.py": """def calculate(a, b, operation):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    else:
        return None"""
    }
    
    # Expected changes (oracle)
    oracle_new_content = {
        "src/calculator.py": """def calculate(a, b, operation):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b != 0:
            return a / b
        else:
            raise ValueError("Division by zero")
    else:
        return None"""
    }
    
    # Model output with search-replace format
    model_output = """<think>
I need to add multiply and divide operations to the calculate function. 
I should also handle division by zero.
</think>
<solution>
```python
### src/calculator.py
<<<<<<< SEARCH
    elif operation == 'subtract':
        return a - b
    else:
        return None
=======
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b != 0:
            return a / b
        else:
            raise ValueError("Division by zero")
    else:
        return None
>>>>>>> REPLACE
```
</solution>"""

    # Calculate reward
    reward, metadata = calculate_search_replace_reward(
        code_context, oracle_new_content, model_output
    )
    
    print(f"Reward: {reward}")
    print(f"Thought: {metadata.get('thought', 'N/A')[:100]}...")
    print(f"Number of file changes: {len(metadata.get('similarities', []))}")
    
    # Example 2: Partial match
    print("\n=== Example 2: Partial Match ===")
    
    partial_output = """<think>
I'll add the multiply operation.
</think>
<solution>
```python
### src/calculator.py
<<<<<<< SEARCH
    elif operation == 'subtract':
        return a - b
    else:
        return None
=======
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    else:
        return None
>>>>>>> REPLACE
```
</solution>"""

    reward2, metadata2 = calculate_search_replace_reward(
        code_context, oracle_new_content, partial_output
    )
    
    print(f"Reward: {reward2}")
    print(f"Explanation: Partial implementation - only added multiply, not divide")
    
    # Example 3: Using VERL's default_compute_score interface
    print("\n=== Example 3: VERL Interface ===")
    
    # Simulate ground truth as a diff
    ground_truth_diff = """@@ -3,6 +3,14 @@ def calculate(a, b, operation):
         return a + b
     elif operation == 'subtract':
         return a - b
+    elif operation == 'multiply':
+        return a * b
+    elif operation == 'divide':
+        if b != 0:
+            return a / b
+        else:
+            raise ValueError("Division by zero")
     else:
         return None"""
    
    # Test with search-replace format
    score1 = compute_score(model_output, ground_truth_diff)
    print(f"Score with search-replace format: {score1}")
    
    # Test with diff format
    diff_output = """<think>
Need to add multiply and divide operations.
</think>
```diff
@@ -3,6 +3,14 @@ def calculate(a, b, operation):
         return a + b
     elif operation == 'subtract':
         return a - b
+    elif operation == 'multiply':
+        return a * b
+    elif operation == 'divide':
+        if b != 0:
+            return a / b
+        else:
+            raise ValueError("Division by zero")
     else:
         return None
```"""
    
    score2 = compute_score(diff_output, ground_truth_diff)
    print(f"Score with diff format: {score2}")
    
    # Example 4: Using VERL's registry
    print("\n=== Example 4: VERL Registry ===")
    
    # With extra_info containing file context
    extra_info = {
        "code_context": code_context,
        "oracle_new_content": oracle_new_content
    }
    
    score3 = default_compute_score(
        "github_patches_search_replace",
        model_output,
        ground_truth_diff,
        extra_info
    )
    print(f"Score via VERL registry with context: {score3}")


if __name__ == "__main__":
    example_search_replace_reward()