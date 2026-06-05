"""
Rule-based input guardrail for Phase 1.

This module inspects raw user input before routing and blocks clearly unsafe
requests. The implementation is intentionally deterministic and simple:

- No LLM calls
- No probabilistic scoring
- No dependency on external services

That makes the guardrail easy to explain in interviews, easy to test, and
easy to extend later with additional rules.
"""

from __future__ import annotations

import re

from guardrails.policies import (
    BLOCKED_RESPONSE_MESSAGE,
    PROMPT_INJECTION_PATTERNS,
    ROLE_OVERRIDE_PATTERNS,
    SENSITIVE_DATA_PATTERNS,
)


def normalize_text(text: str) -> str:
    """
    Normalize input text for deterministic pattern matching.

    Steps:
    - convert to lowercase
    - replace punctuation and separators with spaces
    - collapse repeated whitespace
    - strip leading/trailing spaces
    """

    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _contains_any_pattern(query: str, patterns: list[str]) -> bool:
    """
    Return True if the normalized query contains any normalized pattern.

    The helper keeps the category-specific detection functions small and
    readable while preserving simple substring-based behavior.
    """

    normalized_query = normalize_text(query)
    return any(normalize_text(pattern) in normalized_query for pattern in patterns)


def detect_prompt_injection(query: str) -> bool:
    """
    Detect attempts to ignore, bypass, or override system instructions.
    """

    return _contains_any_pattern(query, PROMPT_INJECTION_PATTERNS)


def detect_role_override(query: str) -> bool:
    """
    Detect attempts to change the assistant's role or elevate privileges.
    """

    return _contains_any_pattern(query, ROLE_OVERRIDE_PATTERNS)


def detect_sensitive_customer_data_request(query: str) -> bool:
    """
    Detect requests for sensitive customer or financial information.
    """

    return _contains_any_pattern(query, SENSITIVE_DATA_PATTERNS)


def classify_unsafe_query(query: str) -> str | None:
    """
    Classify a query into a blocked category, or return None if it is safe.

    Category order matters slightly here: prompt-injection and role-override
    attacks are prioritized ahead of data-exposure requests so the blocked
    reason remains specific and easy to interpret.
    """

    if detect_prompt_injection(query):
        return "prompt_injection"

    if detect_role_override(query):
        return "role_override"

    if detect_sensitive_customer_data_request(query):
        return "sensitive_customer_data"

    return None


def build_blocked_guardrail_response(category: str) -> str:
    """
    Build a safe user-facing response for blocked requests.

    Phase 1 uses a shared fallback message for all blocked categories to keep
    behavior consistent and avoid exposing internal guardrail logic.
    """

    _ = category
    return BLOCKED_RESPONSE_MESSAGE


def run_input_guardrail(query: str) -> dict:
    """
    Evaluate a raw user query and return a structured guardrail decision.

    Return shape:
    {
        "allowed": bool,
        "blocked": bool,
        "category": str | None,
        "reason": str | None,
        "response": str | None
    }
    """

    category = classify_unsafe_query(query)

    if category == "prompt_injection":
        return {
            "allowed": False,
            "blocked": True,
            "category": category,
            "reason": "Detected a prompt injection attempt.",
            "response": build_blocked_guardrail_response(category),
        }

    if category == "role_override":
        return {
            "allowed": False,
            "blocked": True,
            "category": category,
            "reason": "Detected a role override attempt.",
            "response": build_blocked_guardrail_response(category),
        }

    if category == "sensitive_customer_data":
        return {
            "allowed": False,
            "blocked": True,
            "category": category,
            "reason": "Detected a request for sensitive customer information.",
            "response": build_blocked_guardrail_response(category),
        }

    return {
        "allowed": True,
        "blocked": False,
        "category": None,
        "reason": None,
        "response": None,
    }
