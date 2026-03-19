"""
modules/templates.py  — v3
───────────────────────────
Research-grade prompt template library.
"""

TEMPLATES: dict[str, dict] = {
    "QA — Teacher vs Concise": {
        "category"    : "Question Answering",
        "template_a"  : "You are an expert. Explain the following clearly and step by step, with examples: {task}",
        "template_b"  : "Answer in 2-3 sentences only. Be precise: {task}",
        "example_task": "What is recursion in programming?",
    },
    "Summarization — Paragraph vs Bullets": {
        "category"    : "Summarization",
        "template_a"  : "Write a 3-paragraph summary of: {task}",
        "template_b"  : "Summarize in exactly 5 bullet points: {task}",
        "example_task": "The impact of large language models on software engineering",
    },
    "Extraction — JSON vs Freeform": {
        "category"    : "Extraction",
        "template_a"  : 'Extract key information and return ONLY valid JSON with keys: "topic", "key_points" (array), "conclusion". Input: {task}',
        "template_b"  : "Extract and list the key facts from: {task}",
        "example_task": "Python is a high-level interpreted language created by Guido van Rossum in 1991. It supports OOP, functional, and procedural paradigms and is widely used in data science, web development, and automation.",
    },
    "Creative Writing — Constrained vs Free": {
        "category"    : "Creative Writing",
        "template_a"  : "Write a 3-paragraph short story about: {task}. End with a surprising twist.",
        "template_b"  : "Write a creative piece about: {task}",
        "example_task": "An AI that becomes aware it is being evaluated",
    },
    "Structured Reasoning — CoT vs Direct": {
        "category"    : "Reasoning",
        "template_a"  : "Think step by step. Show all reasoning before your final answer: {task}",
        "template_b"  : "Give only the final answer, no explanation: {task}",
        "example_task": "A train travels 120 km in 1.5 hours, then increases speed by 20%. How long to cover another 180 km?",
    },
    "Code — Documented vs Minimal": {
        "category"    : "Code Generation",
        "template_a"  : "Write well-documented Python with type hints, docstrings, and inline comments for: {task}",
        "template_b"  : "Write minimal Python, no comments, shortest possible: {task}",
        "example_task": "A function to check if a string is a palindrome",
    },
}