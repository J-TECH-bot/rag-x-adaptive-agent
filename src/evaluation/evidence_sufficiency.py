import re
from collections import Counter


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "as", "from", "that", "this",
    "what", "why", "how", "does", "do", "used",
    "use", "can", "be", "it", "its", "their",
    "they", "than", "into", "during", "which"
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized tokens.
    """

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def token_overlap(
    reference_answer: str,
    evidence_text: str
) -> float:
    """
    Calculate token overlap between the reference answer
    and retrieved evidence.

    Returns a value between 0 and 1.
    """

    reference_tokens = set(
        tokenize(reference_answer)
    )

    evidence_tokens = set(
        tokenize(evidence_text)
    )

    if not reference_tokens:
        return 0.0

    overlap = (
        reference_tokens & evidence_tokens
    )

    return len(overlap) / len(reference_tokens)


def combine_evidence(results: list[dict]) -> str:
    """
    Combine retrieved chunks into one evidence context.
    """

    return "\n".join(
        result["text"]
        for result in results
    )


def check_evidence_sufficiency(
    reference_answer: str,
    results: list[dict],
    threshold: float = 0.40
) -> dict:
    """
    Determine whether retrieved evidence contains enough
    lexical information to support the reference answer.

    This is a baseline heuristic, not a semantic proof.
    """

    evidence_text = combine_evidence(results)

    score = token_overlap(
        reference_answer,
        evidence_text
    )

    sufficient = score >= threshold

    return {
        "sufficient": sufficient,
        "score": float(score),
        "threshold": float(threshold),
        "evidence_text": evidence_text
    }