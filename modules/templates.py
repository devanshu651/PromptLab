"""
modules/templates.py
─────────────────────
Built-in prompt template library.
Each template has: name, category, template_a, template_b, example_task
"""

TEMPLATES = {
    "QA — Teacher vs Direct": {
        "category": "Question Answering",
        "template_a": "You are an expert teacher. Explain {task} step by step, with examples, suitable for a beginner.",
        "template_b": "Answer this question concisely in 2-3 sentences: {task}",
        "example_task": "What is recursion in programming?",
    },
    "Summarization — Detailed vs Bullet": {
        "category": "Summarization",
        "template_a": "Write a comprehensive summary of the following topic in 3 paragraphs: {task}",
        "template_b": "Summarize the following in exactly 5 bullet points. Be concise: {task}",
        "example_task": "The impact of artificial intelligence on the job market",
    },
    "Extraction — Structured vs Freeform": {
        "category": "Extraction",
        "template_a": "Extract all key information from the following and return it as valid JSON with keys: topic, main_points, conclusion. Input: {task}",
        "template_b": "Read the following and list the key facts: {task}",
        "example_task": "Python is a high-level, interpreted programming language known for its clear syntax and readability. Created by Guido van Rossum and first released in 1991, Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    },
    "Creative Writing — Constrained vs Free": {
        "category": "Creative Writing",
        "template_a": "Write a creative short story (exactly 3 paragraphs) about: {task}. Include a twist ending.",
        "template_b": "Write a short creative piece about: {task}",
        "example_task": "A robot who discovers it can dream",
    },
    "Structured Reasoning — Chain of Thought vs Direct": {
        "category": "Reasoning",
        "template_a": "Think step by step. Show your reasoning process clearly before giving a final answer. Problem: {task}",
        "template_b": "Give the final answer only, no explanation: {task}",
        "example_task": "If a train travels 120km in 1.5 hours, then increases speed by 20%, how long will it take to travel another 180km?",
    },
    "Code Generation — Documented vs Minimal": {
        "category": "Code",
        "template_a": "Write well-documented Python code with docstrings, type hints, and inline comments for: {task}",
        "template_b": "Write minimal Python code (no comments) for: {task}",
        "example_task": "A function that checks if a string is a palindrome",
    },
}