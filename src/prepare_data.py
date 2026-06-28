"""
Data preparation for StraTA training.
Generates training tasks from public datasets (HumanEval, MBPP) and creates
structured task files for the CodeGym sandbox.
"""
import os
import json
import random
from typing import List, Dict


def create_humaneval_tasks(output_path: str, max_tasks: int = 164) -> List[Dict]:
    """Convert HumanEval problems to multi-step interactive tasks"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")
    except Exception:
        print("HumanEval dataset not available, generating synthetic tasks...")
        return create_synthetic_tasks(output_path, max_tasks)

    tasks = []
    for i, item in enumerate(dataset):
        if i >= max_tasks:
            break

        task = {
            "id": f"humaneval_{i}",
            "description": f"Implement the following Python function:\n\n{item['prompt']}\n\n"
                          f"Your solution must pass all test cases.",
            "context": "Empty project. Create a Python file with your implementation.",
            "files": {},
            "test_command": f"python3 -c \"\n{item['prompt']}\n{item['entry_point']}()\n\" 2>&1",
            "test_files": [],
            "difficulty": "medium",
            "max_steps": 10,
            "expected_solution": item.get("canonical_solution", ""),
            "test_cases": item.get("test", ""),
        }
        tasks.append(task)

    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"Created {len(tasks)} HumanEval tasks -> {output_path}")
    return tasks


def create_mbpp_tasks(output_path: str, max_tasks: int = 200) -> List[Dict]:
    """Convert MBPP problems to multi-step interactive tasks"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("mbpp", "full", split="test")
    except Exception:
        print("MBPP dataset not available")
        return []

    tasks = []
    for i, item in enumerate(dataset):
        if i >= max_tasks:
            break

        test_code = "\n".join(item.get("test_list", []))
        task = {
            "id": f"mbpp_{i}",
            "description": f"Write a Python function that: {item['text']}\n\n"
                          f"The function should handle the test cases correctly.",
            "context": "Empty project. Create a Python file with your implementation.",
            "files": {},
            "test_command": f"python3 -c \"\nimport solution\n{test_code}\n\" 2>&1",
            "test_files": [],
            "difficulty": "medium",
            "max_steps": 10,
        }
        tasks.append(task)

    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"Created {len(tasks)} MBPP tasks -> {output_path}")
    return tasks


def create_synthetic_tasks(output_path: str, max_tasks: int = 200) -> List[Dict]:
    """Create synthetic coding tasks covering common scenarios"""
    templates = [
        # Simple tasks
        {
            "desc": "Create a Python function `add(a, b)` that returns the sum of two numbers. "
                   "Handle both integers and floats. Include type checking.",
            "files": {},
            "test": "python3 -c \"from solution import add; assert add(1,2)==3; assert add(1.5,2.5)==4.0; print('PASS')\"",
            "difficulty": "easy",
            "steps": 5,
        },
        {
            "desc": "Create a Python function `is_palindrome(s)` that checks if a string is a palindrome. "
                   "Ignore case and spaces.",
            "files": {},
            "test": "python3 -c \"from solution import is_palindrome; assert is_palindrome('Race car'); assert not is_palindrome('hello'); print('PASS')\"",
            "difficulty": "easy",
            "steps": 5,
        },
        {
            "desc": "Create a Python class `Stack` with methods: push(item), pop(), peek(), is_empty(), size(). "
                   "Raise IndexError on pop/peek of empty stack.",
            "files": {},
            "test": "python3 -c \"from solution import Stack; s=Stack(); s.push(1); s.push(2); assert s.pop()==2; assert s.peek()==1; assert s.size()==1; print('PASS')\"",
            "difficulty": "easy",
            "steps": 8,
        },
        {
            "desc": "Create a function `fibonacci(n)` that returns the nth Fibonacci number. "
                   "Use memoization for efficiency. Handle n=0 returning 0, n=1 returning 1.",
            "files": {},
            "test": "python3 -c \"from solution import fibonacci; assert fibonacci(0)==0; assert fibonacci(1)==1; assert fibonacci(10)==55; assert fibonacci(50)==12586269025; print('PASS')\"",
            "difficulty": "medium",
            "steps": 8,
        },
        {
            "desc": "Create a function `binary_search(arr, target)` that performs binary search on a sorted array. "
                   "Return the index of target, or -1 if not found.",
            "files": {},
            "test": "python3 -c \"from solution import binary_search; assert binary_search([1,3,5,7,9],5)==2; assert binary_search([1,3,5,7,9],4)==-1; assert binary_search([],1)==-1; print('PASS')\"",
            "difficulty": "medium",
            "steps": 8,
        },
        {
            "desc": "Create a function `merge_sort(arr)` that sorts a list using merge sort algorithm. "
                   "Return a new sorted list without modifying the input.",
            "files": {},
            "test": "python3 -c \"from solution import merge_sort; assert merge_sort([3,1,4,1,5,9])==[1,1,3,4,5,9]; assert merge_sort([])==[]; assert merge_sort([1])==[1]; print('PASS')\"",
            "difficulty": "medium",
            "steps": 10,
        },
        {
            "desc": "Create a function `flatten(nested_list)` that flattens a nested list of arbitrary depth. "
                   "Example: flatten([1,[2,[3,4],5]]) -> [1,2,3,4,5]",
            "files": {},
            "test": "python3 -c \"from solution import flatten; assert flatten([1,[2,[3,4],5]])==[1,2,3,4,5]; assert flatten([])==[]; assert flatten([[[1]]])==[1]; print('PASS')\"",
            "difficulty": "medium",
            "steps": 8,
        },
        {
            "desc": "Create a function `lru_cache_decorator(max_size)` that implements an LRU cache decorator. "
                   "When cache is full, evict the least recently used item.",
            "files": {},
            "test": "python3 -c \"from solution import lru_cache_decorator; @lru_cache_decorator(2)\ndef f(x): return x*2\nf(1); f(2); f(3); print('PASS')\"",
            "difficulty": "hard",
            "steps": 15,
        },
        {
            "desc": "Create a Python class `LRUCache` with get(key) and put(key, value) methods. "
                   "Both operations must be O(1) time complexity. Evict LRU item when capacity exceeded.",
            "files": {},
            "test": "python3 -c \"from solution import LRUCache; c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; print('PASS')\"",
            "difficulty": "hard",
            "steps": 15,
        },
        {
            "desc": "Create a function `regex_match(pattern, text)` that implements basic regex matching. "
                   "Support: '.' (any char), '*' (zero or more of previous), '+' (one or more of previous).",
            "files": {},
            "test": "python3 -c \"from solution import regex_match; assert regex_match('a*b', 'b'); assert regex_match('a.*b', 'axxb'); assert not regex_match('a+b', 'b'); print('PASS')\"",
            "difficulty": "hard",
            "steps": 15,
        },
        # Multi-file tasks
        {
            "desc": "Create a simple REST API calculator module. Create `calculator.py` with add, subtract, "
                   "multiply, divide functions (divide should handle ZeroDivisionError). "
                   "Create `test_calculator.py` with pytest tests.",
            "files": {},
            "test": "python3 -m pytest test_calculator.py -v --tb=short 2>&1",
            "difficulty": "medium",
            "steps": 12,
        },
        {
            "desc": "Create a `config_parser.py` module that reads INI-style config files. "
                   "Support sections [section], key=value pairs, and comments (#). "
                   "Include proper error handling for malformed configs.",
            "files": {},
            "test": "python3 -c \"from config_parser import ConfigParser; cp=ConfigParser(); print('PASS')\"",
            "difficulty": "medium",
            "steps": 12,
        },
        {
            "desc": "Create a `linked_list.py` with a LinkedList class supporting: append, prepend, "
                   "delete(value), find(value), to_list(). Also create `test_linked_list.py` with tests.",
            "files": {},
            "test": "python3 -m pytest test_linked_list.py -v --tb=short 2>&1",
            "difficulty": "medium",
            "steps": 12,
        },
        {
            "desc": "Create a `rate_limiter.py` implementing a token bucket rate limiter. "
                   "Class with: __init__(rate, capacity), allow() -> bool. "
                   "The limiter should allow `rate` requests per second with burst up to `capacity`.",
            "files": {},
            "test": "python3 -c \"from rate_limiter import RateLimiter; rl=RateLimiter(2,2); assert rl.allow(); assert rl.allow(); print('PASS')\"",
            "difficulty": "hard",
            "steps": 15,
        },
        {
            "desc": "Create a `task_queue.py` implementing a priority queue with: push(item, priority), "
                   "pop() -> (item, priority), peek(), is_empty(). Lower priority number = higher priority.",
            "files": {},
            "test": "python3 -c \"from task_queue import PriorityQueue; pq=PriorityQueue(); pq.push('a',2); pq.push('b',1); assert pq.pop()==('b',1); print('PASS')\"",
            "difficulty": "medium",
            "steps": 10,
        },
    ]

    tasks = []
    for i, tmpl in enumerate(templates):
        task = {
            "id": f"synthetic_{i}",
            "description": tmpl["desc"],
            "context": "Empty project directory.",
            "files": {},
            "test_command": tmpl["test"],
            "test_files": [],
            "difficulty": tmpl["difficulty"],
            "max_steps": tmpl["steps"],
        }
        tasks.append(task)

    # Duplicate with variations to reach max_tasks
    while len(tasks) < max_tasks:
        base = random.choice(templates)
        task = {
            "id": f"synthetic_{len(tasks)}",
            "description": base["desc"] + f" (variant {len(tasks)})",
            "context": "Empty project directory.",
            "files": {},
            "test_command": base["test"],
            "test_files": [],
            "difficulty": base["difficulty"],
            "max_steps": base["steps"],
        }
        tasks.append(task)

    with open(output_path, "w") as f:
        json.dump(tasks[:max_tasks], f, indent=2, ensure_ascii=False)
    print(f"Created {min(len(tasks), max_tasks)} synthetic tasks -> {output_path}")
    return tasks[:max_tasks]


def create_sft_data(output_path: str, n_samples: int = 300):
    """Create SFT training data for format alignment"""
    templates = [
        {
            "task": "Add input validation to a function that processes user emails.",
            "strategy": "1. Read the current function implementation.\n"
                       "2. Add email format validation using regex.\n"
                       "3. Add type checking for the input parameter.\n"
                       "4. Write tests for valid and invalid inputs.\n"
                       "5. Run tests to verify.",
            "actions": [
                "read:main.py",
                "write:main.py\nimport re\n\ndef validate_email(email: str) -> bool:\n    if not isinstance(email, str):\n        raise TypeError('Email must be a string')\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n\ndef process_email(email: str) -> dict:\n    if not validate_email(email):\n        raise ValueError(f'Invalid email: {email}')\n    return {'email': email, 'domain': email.split('@')[1]}",
                "write:test_main.py\nimport pytest\nfrom main import validate_email, process_email\n\ndef test_valid_email():\n    assert validate_email('user@example.com')\n    assert validate_email('test.user@domain.org')\n\ndef test_invalid_email():\n    assert not validate_email('not-an-email')\n    assert not validate_email('@domain.com')\n\ndef test_process_email():\n    result = process_email('user@example.com')\n    assert result['domain'] == 'example.com'\n\ndef test_invalid_input():\n    with pytest.raises(TypeError):\n        validate_email(123)",
                "test:",
            ]
        },
        {
            "task": "Fix a bug where the function crashes on empty input.",
            "strategy": "1. Read the buggy function to understand the crash.\n"
                       "2. Identify the root cause (missing empty check).\n"
                       "3. Add guard clause for empty input.\n"
                       "4. Write a regression test.\n"
                       "5. Run all tests.",
            "actions": [
                "read:solution.py",
                "write:solution.py\ndef process_items(items: list) -> int:\n    if not items:\n        return 0\n    total = 0\n    for item in items:\n        total += item\n    return total",
                "write:test_solution.py\nfrom solution import process_items\n\ndef test_normal():\n    assert process_items([1, 2, 3]) == 6\n\ndef test_empty():\n    assert process_items([]) == 0\n\ndef test_single():\n    assert process_items([42]) == 42",
                "test:",
            ]
        },
    ]

    samples = []
    for i in range(n_samples):
        tmpl = templates[i % len(templates)]
        # Build conversation format
        messages = [
            {"role": "user", "content": tmpl["task"]},
            {"role": "assistant", "content": f"<strategy>{tmpl['strategy']}</strategy>"},
        ]
        for j, action in enumerate(tmpl["actions"]):
            messages.append({
                "role": "assistant",
                "content": f"<action>{action}</action>"
            })

        samples.append({
            "id": f"sft_{i}",
            "messages": messages,
            "task": tmpl["task"],
        })

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Created {n_samples} SFT samples -> {output_path}")
    return samples


def main():
    data_dir = "/root/strata-project/data"
    os.makedirs(f"{data_dir}/train", exist_ok=True)
    os.makedirs(f"{data_dir}/eval", exist_ok=True)

    # Create training data
    print("=== Creating training data ===")
    all_tasks = []

    # Synthetic tasks (always available)
    synth = create_synthetic_tasks(f"{data_dir}/train/synthetic.json", 200)
    all_tasks.extend(synth)

    # HumanEval (if available)
    he = create_humaneval_tasks(f"{data_dir}/train/humaneval.json", 164)
    all_tasks.extend(he)

    # MBPP (if available)
    mbpp = create_mbpp_tasks(f"{data_dir}/train/mbpp.json", 200)
    all_tasks.extend(mbpp)

    # Combined training data
    with open(f"{data_dir}/train/all_tasks.json", "w") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)
    print(f"\nTotal training tasks: {len(all_tasks)}")

    # Create eval data (subset)
    eval_tasks = random.sample(all_tasks, min(100, len(all_tasks)))
    with open(f"{data_dir}/eval/eval_tasks.json", "w") as f:
        json.dump(eval_tasks, f, indent=2, ensure_ascii=False)
    print(f"Eval tasks: {len(eval_tasks)}")

    # Create SFT data
    print("\n=== Creating SFT data ===")
    create_sft_data(f"{data_dir}/train/sft_data.json", 300)


if __name__ == "__main__":
    main()
