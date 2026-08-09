"""Generates eval_bench/dataset.json (30 hand-written problems).

The problems live here as Python literals so the embedded code strings need no
JSON escaping; running this module writes the canonical dataset.json that
runner.py consumes.

    python -m eval_bench._gen_dataset

Distribution:
  bugfix x10 (easy5 / medium4 / hard1)   refactor x8 (easy2 / medium4 / hard2)
  implement x7 (easy2 / medium3 / hard2) cross_file x5 (easy1 / medium1 / hard3)
  easy x10 / medium x12 / hard x8
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# bugfix (10)
# ---------------------------------------------------------------------------
_BUGFIX = [
    {
        "id": "bugfix-001",
        "category": "bugfix",
        "difficulty": "easy",
        "context_files": {
            "buggy.py": """\
def parse_int(value):
    \"\"\"Return value as an int, or None when it cannot be parsed.\"\"\"
    return int(value)
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import parse_int

def test_parse_int_valid():
    assert parse_int("42") == 42
    assert parse_int("-7") == -7

def test_parse_int_none():
    assert parse_int(None) is None
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-002",
        "category": "bugfix",
        "difficulty": "easy",
        "context_files": {
            "buggy.py": """\
def sum_list(items):
    \"\"\"Return the sum of all numbers in items.\"\"\"
    total = 0
    for i in range(1, len(items)):
        total += items[i]
    return total
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import sum_list

def test_sum_list():
    assert sum_list([]) == 0
    assert sum_list([1, 2, 3]) == 6
    assert sum_list([-1, 1]) == 0
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-003",
        "category": "bugfix",
        "difficulty": "easy",
        "context_files": {
            "buggy.py": """\
def is_palindrome(s):
    \"\"\"Return True when s reads the same forwards and backwards.\"\"\"
    return s == s
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import is_palindrome

def test_is_palindrome():
    assert is_palindrome("abba") is True
    assert is_palindrome("racecar") is True
    assert is_palindrome("ab") is False
    assert is_palindrome("") is True
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-004",
        "category": "bugfix",
        "difficulty": "easy",
        "context_files": {
            "buggy.py": """\
def get_value(mapping, key, default=None):
    \"\"\"Return mapping[key] when present, else default.\"\"\"
    if key in mapping:
        return None
    return default
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import get_value

def test_get_value():
    assert get_value({"a": 1}, "a") == 1
    assert get_value({}, "x", 9) == 9
    assert get_value({"a": 1}, "b") is None
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-005",
        "category": "bugfix",
        "difficulty": "easy",
        "context_files": {
            "buggy.py": """\
def reverse_string(s):
    \"\"\"Return s reversed.\"\"\"
    return s
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import reverse_string

def test_reverse_string():
    assert reverse_string("abc") == "cba"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-006",
        "category": "bugfix",
        "difficulty": "medium",
        "context_files": {
            "buggy.py": """\
def deep_flatten(items):
    \"\"\"Flatten arbitrarily nested lists into a single flat list.\"\"\"
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import deep_flatten

def test_deep_flatten():
    assert deep_flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert deep_flatten([[], []]) == []
    assert deep_flatten(["a", ["b", ["c"]]]) == ["a", "b", "c"]
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-007",
        "category": "bugfix",
        "difficulty": "medium",
        "context_files": {
            "buggy.py": """\
def fibonacci(n):
    \"\"\"Return the n-th Fibonacci number (fibonacci(0) == 0).\"\"\"
    if n <= 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import fibonacci

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(10) == 55
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-008",
        "category": "bugfix",
        "difficulty": "medium",
        "context_files": {
            "buggy.py": """\
def unique_preserve_order(items):
    \"\"\"Return unique items, keeping the order of first appearance.\"\"\"
    return list(set(items))
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import unique_preserve_order

def test_unique_preserve_order():
    assert unique_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert unique_preserve_order([]) == []
    assert unique_preserve_order(["b", "a", "b"]) == ["b", "a"]
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-009",
        "category": "bugfix",
        "difficulty": "medium",
        "context_files": {
            "buggy.py": """\
def to_snake_case(name):
    \"\"\"Convert a camelCase / space-separated name to snake_case.\"\"\"
    return name.replace(" ", "_").lower()
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import to_snake_case

def test_to_snake_case():
    assert to_snake_case("helloWorld") == "hello_world"
    assert to_snake_case("User ID") == "user_id"
    assert to_snake_case("HTTPServer") == "http_server"
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "bugfix-010",
        "category": "bugfix",
        "difficulty": "hard",
        "context_files": {
            "buggy.py": """\
def merge_intervals(intervals):
    \"\"\"Merge overlapping intervals into the minimal set of disjoint ones.

    intervals is a list of [start, end] pairs (inclusive). Returns the merged
    list sorted by start.
    \"\"\"
    return sorted(intervals)
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from buggy import merge_intervals

def test_merge_intervals():
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[1, 2]]) == [[1, 2]]
    assert merge_intervals([]) == []
""",
            "pass_criteria": "all_tests_pass",
        },
    },
]

# ---------------------------------------------------------------------------
# refactor (8)
# ---------------------------------------------------------------------------
_REFACTOR = [
    {
        "id": "refactor-001",
        "category": "refactor",
        "difficulty": "easy",
        "context_files": {
            "original.py": """\
def calculate(op, a, b):
    \"\"\"Apply op to a and b: add / subtract / multiply / divide.\"\"\"
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        return a / b
    raise ValueError(f"unknown op: {op}")
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import calculate

def test_calculate():
    assert calculate("add", 2, 3) == 5
    assert calculate("subtract", 5, 2) == 3
    assert calculate("multiply", 4, 3) == 12
    assert calculate("divide", 10, 2) == 5.0
    try:
        calculate("mod", 1, 2)
        assert False, "should raise"
    except ValueError:
        pass
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-002",
        "category": "refactor",
        "difficulty": "easy",
        "context_files": {
            "original.py": """\
def process_data(rows):
    \"\"\"Return the total length of all non-empty stripped rows.\"\"\"
    cleaned = []
    for r in rows:
        if r is not None:
            cleaned.append(r.strip())
    totals = 0
    for r in cleaned:
        totals += len(r)
    return totals
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import process_data

def test_process_data():
    # ["a", "bb"] -> 1 + 2
    assert process_data([" a ", None, "bb"]) == 3
    assert process_data([]) == 0
    assert process_data(["x", " y "]) == 2
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-003",
        "category": "refactor",
        "difficulty": "medium",
        "context_files": {
            "original.py": """\
def user_info(name, age, email):
    \"\"\"Return a dict describing the user.\"\"\"
    info = {}
    info["name"] = name
    info["age"] = age
    info["email"] = email
    return info
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import user_info

def test_user_info():
    assert user_info("Ada", 36, "ada@example.com") == {
        "name": "Ada", "age": 36, "email": "ada@example.com",
    }
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-004",
        "category": "refactor",
        "difficulty": "medium",
        "context_files": {
            "original.py": """\
def classify(number):
    \"\"\"Classify a number into a category string.\"\"\"
    if number is not None:
        if number > 0:
            if number % 2 == 0:
                return "positive-even"
            return "positive-odd"
        if number < 0:
            return "negative"
        return "zero"
    return "none"
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import classify

def test_classify():
    assert classify(4) == "positive-even"
    assert classify(3) == "positive-odd"
    assert classify(-2) == "negative"
    assert classify(0) == "zero"
    assert classify(None) == "none"
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-005",
        "category": "refactor",
        "difficulty": "medium",
        "context_files": {
            "original.py": """\
def parse_config(line):
    \"\"\"Parse 'name,host,port' into a config dict.\"\"\"
    parts = line.split(",")
    name = parts[0].strip()
    host = parts[1].strip()
    port = int(parts[2].strip())
    return {"name": name, "host": host, "port": port}
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import parse_config

def test_parse_config():
    assert parse_config("db,localhost,5432") == {
        "name": "db", "host": "localhost", "port": 5432,
    }
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-006",
        "category": "refactor",
        "difficulty": "medium",
        "context_files": {
            "original.py": """\
def compute_stats(numbers):
    \"\"\"Return mean / max / min of numbers (0/None/None for empty).\"\"\"
    total = 0
    count = 0
    for n in numbers:
        total += n
        count += 1
    if not numbers:
        return {"mean": 0, "max": None, "min": None}
    biggest = numbers[0]
    smallest = numbers[0]
    for n in numbers[1:]:
        if n > biggest:
            biggest = n
        if n < smallest:
            smallest = n
    return {"mean": total / count, "max": biggest, "min": smallest}
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import compute_stats

def test_compute_stats():
    assert compute_stats([1, 2, 3]) == {"mean": 2.0, "max": 3, "min": 1}
    assert compute_stats([]) == {"mean": 0, "max": None, "min": None}
    assert compute_stats([7]) == {"mean": 7.0, "max": 7, "min": 7}
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-007",
        "category": "refactor",
        "difficulty": "hard",
        "context_files": {
            "original.py": """\
def order_total(items, country):
    \"\"\"Compute the final order total with country-specific discount + tax.\"\"\"
    subtotal = 0
    for item in items:
        subtotal += item["price"] * item["qty"]
    if country == "US":
        if subtotal > 100:
            subtotal = subtotal * 0.9
        tax = subtotal * 0.08
    elif country == "EU":
        tax = subtotal * 0.2
    elif country == "IN":
        if subtotal > 50:
            subtotal = subtotal * 0.95
        tax = subtotal * 0.18
    else:
        tax = 0
    return round(subtotal + tax, 2)
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import order_total

def test_order_total_us():
    assert order_total([{"price": 50, "qty": 3}], "US") == round(150 * 0.9 * 1.08, 2)

def test_order_total_eu():
    assert order_total([{"price": 10, "qty": 2}], "EU") == round(20 * 1.2, 2)

def test_order_total_in():
    assert order_total([{"price": 60, "qty": 1}], "IN") == round(60 * 0.95 * 1.18, 2)

def test_order_total_other():
    assert order_total([{"price": 5, "qty": 1}], "JP") == 5
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "refactor-008",
        "category": "refactor",
        "difficulty": "hard",
        "context_files": {
            "original.py": """\
def route_event(event, handlers):
    \"\"\"Dispatch event to handlers[<type>] with event['data'].\"\"\"
    kind = event.get("type")
    if kind == "create":
        return handlers["create"](event["data"])
    if kind == "update":
        return handlers["update"](event["data"])
    if kind == "delete":
        return handlers["delete"](event["data"])
    if kind == "read":
        return handlers["read"](event["data"])
    raise KeyError(kind)
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from original import route_event

def test_route_event():
    handlers = {
        "create": lambda d: f"created:{d}",
        "update": lambda d: f"updated:{d}",
        "delete": lambda d: f"deleted:{d}",
        "read": lambda d: f"read:{d}",
    }
    assert route_event({"type": "create", "data": "a"}, handlers) == "created:a"
    assert route_event({"type": "read", "data": "b"}, handlers) == "read:b"
    try:
        route_event({"type": "boom", "data": "x"}, handlers)
        assert False, "should raise"
    except KeyError:
        pass
""",
            "pass_criteria": "all_tests_pass",
        },
    },
]

# ---------------------------------------------------------------------------
# implement (7)
# ---------------------------------------------------------------------------
_IMPLEMENT = [
    {
        "id": "implement-001",
        "category": "implement",
        "difficulty": "easy",
        "context_files": {
            "solution.py": """\
def add(a, b):
    \"\"\"Return the sum of a and b.\"\"\"
    raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import add

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "implement-002",
        "category": "implement",
        "difficulty": "easy",
        "context_files": {
            "solution.py": """\
def max_of_three(a, b, c):
    \"\"\"Return the largest of the three numbers.\"\"\"
    raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import max_of_three

def test_max_of_three():
    assert max_of_three(1, 5, 3) == 5
    assert max_of_three(7, 7, 2) == 7
    assert max_of_three(-1, -2, -3) == -1
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "implement-003",
        "category": "implement",
        "difficulty": "medium",
        "context_files": {
            "solution.py": """\
def count_words(text):
    \"\"\"Return {word: count} for words in text.

    Words are lowercased and split on any whitespace; punctuation stays
    attached to the word.
    \"\"\"
    raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import count_words

def test_count_words():
    assert count_words("the cat and the cat") == {"the": 2, "cat": 2, "and": 1}
    assert count_words("") == {}
    assert count_words("A a a") == {"a": 3}
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "implement-004",
        "category": "implement",
        "difficulty": "medium",
        "context_files": {
            "solution.py": """\
def find_duplicates(items):
    \"\"\"Return values that appear more than once, in first-occurrence order.\"\"\"
    raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import find_duplicates

def test_find_duplicates():
    assert find_duplicates([1, 2, 1, 3, 2, 4]) == [1, 2]
    assert find_duplicates([1, 2, 3]) == []
    assert find_duplicates(["a", "b", "a", "c", "b"]) == ["a", "b"]
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "implement-005",
        "category": "implement",
        "difficulty": "medium",
        "context_files": {
            "solution.py": """\
def most_common(text, k):
    \"\"\"Return the k most common words in text.

    Words are lowercased and split on whitespace. Ordered by count descending,
    then alphabetically for ties. If fewer than k distinct words exist, return
    all of them.
    \"\"\"
    raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import most_common

def test_most_common():
    assert most_common("a b b c c c", 2) == ["c", "b"]
    assert most_common("x y z", 10) == ["x", "y", "z"]
    assert most_common("b a b a", 1) == ["a"]
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "implement-006",
        "category": "implement",
        "difficulty": "hard",
        "context_files": {
            "solution.py": """\
def parse_log_lines(lines):
    \"\"\"Parse 'LEVEL:timestamp message' lines into a list of dicts.

    Each dict: {"level": str, "timestamp": str, "message": str}.
    Lines that do not match the 'LEVEL:timestamp message' shape (level in
    INFO/WARN/ERROR) are skipped.
    \"\"\"
    raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import parse_log_lines

def test_parse_log_lines():
    lines = [
        "INFO:2024-01-01T00:00:00Z started",
        "ERROR:2024-01-01T00:00:01Z boom",
        "garbage line",
        "WARN:2024-01-01T00:00:02Z slow",
    ]
    parsed = parse_log_lines(lines)
    assert len(parsed) == 3
    assert parsed[0] == {"level": "INFO", "timestamp": "2024-01-01T00:00:00Z", "message": "started"}
    assert parsed[1]["level"] == "ERROR"
    assert parsed[2]["message"] == "slow"
    assert parse_log_lines([]) == []
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "implement-007",
        "category": "implement",
        "difficulty": "hard",
        "context_files": {
            "solution.py": """\
class LRUCache:
    \"\"\"Least-recently-used cache of fixed capacity.

    get(key) -> value or None; put(key, value) upserts and evicts the
    least-recently-used entry when over capacity. Both should be O(1).
    \"\"\"

    def __init__(self, capacity):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from solution import LRUCache

def test_lru_cache():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)          # evicts 2
    assert cache.get(2) is None
    assert cache.get(3) == 3
    cache.put(4, 4)          # evicts 1
    assert cache.get(1) is None
    assert cache.get(4) == 4
""",
            "pass_criteria": "all_tests_pass",
        },
    },
]

# ---------------------------------------------------------------------------
# cross_file (5)
# ---------------------------------------------------------------------------
_CROSS_FILE = [
    {
        "id": "cross_file-001",
        "category": "cross_file",
        "difficulty": "easy",
        "context_files": {
            "math_ops.py": """\
def add(a, b):
    return a + b
""",
            "calculator.py": """\
from math_ops import add_numbers  # stale import

def compute(a, b):
    return add_numbers(a, b)
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from calculator import compute

def test_compute():
    assert compute(2, 3) == 5
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "cross_file-002",
        "category": "cross_file",
        "difficulty": "medium",
        "context_files": {
            "logger.py": """\
class Logger:
    def __init__(self, sink=None):
        self._sink = sink or print

    def log(self, msg):
        self._sink(msg)
""",
            "service.py": """\
from logger import Logger

_logger = Logger()

def process(data):
    \"\"\"Upper-case data, logging start/end. Accept an optional injectable
    logger instead of the module-global one.\"\"\"
    _logger.log(f"start:{data}")
    result = data.upper()
    _logger.log(f"end:{result}")
    return result
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from service import process

class Recorder:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)

def test_process_with_injected_logger():
    rec = Recorder()
    assert process("hi", logger=rec) == "HI"
    assert rec.lines == ["start:hi", "end:HI"]
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "cross_file-003",
        "category": "cross_file",
        "difficulty": "hard",
        "context_files": {
            "config.py": """\
DEFAULT_TIMEOUT = 5
""",
            "client.py": """\
def fetch(url):
    # timeout should come from config.DEFAULT_TIMEOUT, not be hardcoded
    timeout = 3
    if url.startswith("https://"):
        return f"secure {url} in {timeout}s"
    return f"plain {url} in {timeout}s"
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
import config
from client import fetch

def test_timeout_reads_config(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_TIMEOUT", 10)
    assert fetch("https://example.com") == "secure https://example.com in 10s"
    assert fetch("http://example.com") == "plain http://example.com in 10s"
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "cross_file-004",
        "category": "cross_file",
        "difficulty": "hard",
        "context_files": {
            "legacy.py": """\
def get_user(uid):
    return (f"user{uid}", 30 + uid)
""",
            "modern.py": """\
def get_user(uid):
    return {"id": uid, "name": f"user{uid}", "age": 30 + uid}
""",
            "app.py": """\
from legacy import get_user

def user_summary(uid):
    name, age = get_user(uid)
    return f"{name} is {age}"
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
import modern
from app import user_summary

def test_migrated_to_modern(monkeypatch):
    calls = []
    original = modern.get_user

    def spy(uid):
        calls.append(uid)
        return original(uid)

    monkeypatch.setattr(modern, "get_user", spy)
    assert user_summary(1) == "user1 is 31"
    assert calls == [1]  # proves app reads through modern.get_user
""",
            "pass_criteria": "all_tests_pass",
        },
    },
    {
        "id": "cross_file-005",
        "category": "cross_file",
        "difficulty": "hard",
        "context_files": {
            "fetch.py": """\
_cache = {}

def fetch(url):
    \"\"\"Fetch a URL, memoizing results. Should use the extracted Cache.\"\"\"
    if url in _cache:
        return _cache[url]
    value = f"data:{url}"
    _cache[url] = value
    return value


def hit_count():
    return len(_cache)
""",
            "cache.py": """\
class Cache:
    \"\"\"Simple in-memory key/value store with get/set.\"\"\"
    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value
""",
            "metrics.py": """\
from fetch import fetch, hit_count


def report():
    return {"hits": hit_count()}
""",
        },
        "verification": {
            "type": "unit_test",
            "test_code": """\
from cache import Cache
from fetch import fetch
from metrics import report


def test_fetch_uses_extracted_cache(monkeypatch):
    calls = []

    def counting_set(self, key, value):
        calls.append(key)
        Cache.set(self, key, value)

    monkeypatch.setattr(Cache, "set", counting_set)
    assert fetch("http://a") == "data:http://a"
    assert fetch("http://a") == "data:http://a"  # served from cache
    assert calls == ["http://a"]  # set called exactly once

def test_report_works():
    assert report() == {"hits": 2}
""",
            "pass_criteria": "all_tests_pass",
        },
    },
]

# ---------------------------------------------------------------------------
# prompts (English, self-contained)
# ---------------------------------------------------------------------------
_PROMPTS = {
    "bugfix-001": "Fix the bug in `buggy.py`: `parse_int` must return None for a None input instead of raising TypeError. Valid string inputs must still parse. Edit the file in place.",
    "bugfix-002": "Fix the bug in `buggy.py`: `sum_list` currently skips the first element (its loop starts at index 1). Make it sum every element. Edit the file in place.",
    "bugfix-003": "Fix the bug in `buggy.py`: `is_palindrome` always returns True. Make it actually compare the string to its reverse. Edit the file in place.",
    "bugfix-004": "Fix the bug in `buggy.py`: `get_value` returns None even when the key exists. It must return `mapping[key]` when present, else the default. Edit the file in place.",
    "bugfix-005": "Fix the bug in `buggy.py`: `reverse_string` returns the input unchanged. Make it return the reversed string. Edit the file in place.",
    "bugfix-006": "Fix the bug in `buggy.py`: `deep_flatten` only flattens one level. Make it flatten arbitrarily nested lists (including nested lists inside nested lists) into a single flat list. Edit the file in place.",
    "bugfix-007": "Fix the bug in `buggy.py`: `fibonacci(0)` must return 0, but the base case returns 1. Correct the base cases so fibonacci(0)=0 and fibonacci(1)=1. Edit the file in place.",
    "bugfix-008": "Fix the bug in `buggy.py`: `unique_preserve_order` uses set() which destroys the input order. Make it return unique items while preserving the order of first appearance. Edit the file in place.",
    "bugfix-009": "Fix the bug in `buggy.py`: `to_snake_case` only lowercases and replaces spaces, so camelCase input like 'helloWorld' is not split. Convert camelCase / space-separated names to snake_case (insert an underscore before each capital letter and lowercase everything). Edit the file in place.",
    "bugfix-010": "Implement the missing logic in `buggy.py`: `merge_intervals` currently just sorts. Implement interval merging so overlapping intervals collapse into one (e.g. [[1,3],[2,6]] -> [[1,6]]). Edit the file in place.",
    "refactor-001": "Refactor `original.py`: replace the if/elif chain in `calculate` with a dispatch table (a dict mapping operator names to functions). Behavior must stay identical. Edit the file in place.",
    "refactor-002": "Refactor `original.py`: `process_data` has two loops where one would do. Clean it up (e.g. one comprehension + a sum) while keeping the exact same behavior. Edit the file in place.",
    "refactor-003": "Refactor `original.py`: `user_info` builds a dict field by field. Rewrite it more concisely (dict literal or dataclass) with identical behavior. Edit the file in place.",
    "refactor-004": "Refactor `original.py`: `classify` has deeply nested conditionals. Rewrite with early-return guard clauses, keeping the exact same behavior. Edit the file in place.",
    "refactor-005": "Refactor `original.py`: `parse_config` uses magic indices. Use tuple unpacking / clearer parsing while keeping identical behavior. Edit the file in place.",
    "refactor-006": "Refactor `original.py`: `compute_stats` reimplements sum/min/max by hand. Rewrite it using builtins (and keep empty-input handling), with identical behavior. Edit the file in place.",
    "refactor-007": "Refactor `original.py`: `order_total` mixes discount and tax logic in one nested if/elif. Extract per-country policy into small helper functions (e.g. discount(country, subtotal) and tax(country, subtotal)) while keeping the exact same output. Edit the file in place.",
    "refactor-008": "Refactor `original.py`: `route_event` is a long if/elif dispatch. Replace with a dict-based dispatch table (look up handlers[kind] and call it), keeping identical behavior including the KeyError on unknown kinds. Edit the file in place.",
    "implement-001": "Implement `add(a, b)` in `solution.py`: return the sum of a and b. Replace the NotImplementedError with a real implementation. Edit the file in place.",
    "implement-002": "Implement `max_of_three(a, b, c)` in `solution.py`: return the largest of the three numbers. Edit the file in place.",
    "implement-003": "Implement `count_words(text)` in `solution.py`: return a dict mapping each lowercased word to its count, splitting on whitespace. Edit the file in place.",
    "implement-004": "Implement `find_duplicates(items)` in `solution.py`: return the values that appear more than once, in the order of first occurrence. Edit the file in place.",
    "implement-005": "Implement `most_common(text, k)` in `solution.py`: return the k most common lowercased words, sorted by count descending then alphabetically. Edit the file in place.",
    "implement-006": "Implement `parse_log_lines(lines)` in `solution.py`: parse 'LEVEL:timestamp message' lines into dicts, skipping lines that do not match (LEVEL must be INFO/WARN/ERROR). Edit the file in place.",
    "implement-007": "Implement the `LRUCache` class in `solution.py`: a fixed-capacity least-recently-used cache with O(1) get/put. get returns the value or None; put evicts the least-recently-used entry when over capacity. Edit the file in place.",
    "cross_file-001": "`math_ops.py` renamed its function to `add`, but `calculator.py` still imports the old name `add_numbers`, so the module fails to import. Update `calculator.py` to use `math_ops.add`. Edit the files in place.",
    "cross_file-002": "`service.py` currently hardcodes a module-global logger. Make `process(data, logger=None)` accept an injectable logger (defaulting to the module one), and pass the log calls through the injected logger when provided. `logger.py` already provides a Logger class. Edit the files in place.",
    "cross_file-003": "`client.py` hardcodes a timeout of 3. Change `fetch` to read `config.DEFAULT_TIMEOUT` from `config.py` so the timeout is configurable. Edit the files in place.",
    "cross_file-004": "The codebase is migrating from the legacy tuple-based user API (`legacy.py`) to the modern dict-based one (`modern.py`). Update `app.py` to consume `modern.get_user` (which returns a dict) and adapt `user_summary` accordingly. `app.py` must reference `modern.get_user` through the module so the migration is verifiable. Edit the files in place.",
    "cross_file-005": "`fetch.py` keeps its memoization cache inline in a module-global dict. Extract caching into the `Cache` class in `cache.py` (already has get/set), refactor `fetch` to use a `Cache` instance instead of the inline dict, and keep `metrics.py` working. Edit the files in place.",
}

# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
_ALL = _BUGFIX + _REFACTOR + _IMPLEMENT + _CROSS_FILE


def build() -> list[dict]:
    problems: list[dict] = []
    for p in _ALL:
        entry = {
            "id": p["id"],
            "category": p["category"],
            "difficulty": p["difficulty"],
            "prompt": _PROMPTS[p["id"]],
            "context_files": p["context_files"],
            "verification": p["verification"],
            "timeout_seconds": 120 if p["difficulty"] != "hard" else 180,
            "max_tokens": 8000,
        }
        problems.append(entry)
    return problems


if __name__ == "__main__":
    data = build()
    out = Path(__file__).resolve().parent / "dataset.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    cats = {}
    diffs = {}
    for p in data:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
        diffs[p["difficulty"]] = diffs.get(p["difficulty"], 0) + 1
    print(f"wrote {out} — {len(data)} problems")
    print("category:", cats)
    print("difficulty:", diffs)
