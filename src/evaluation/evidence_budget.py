from dataclasses import dataclass


@dataclass
class EvidenceBudgetResult:
    selected_results: list[dict]
    token_count: int
    budget: int
    candidates_considered: int
    candidates_selected: int


class EvidenceBudget:
    """
    Select reranked evidence under a fixed token budget.

    The selector keeps evidence in reranker order and adds a result
    only when its text fits within the remaining budget.

    This module does not perform retrieval, reranking, or judging.
    It only controls how much evidence is passed downstream.
    """

    def __init__(self, tokenizer, max_tokens: int = 512):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None
        )

        return len(encoded["input_ids"])

    def select(
        self,
        results: list[dict]
    ) -> EvidenceBudgetResult:

        selected = []
        total_tokens = 0

        for result in results:
            text = result["text"]

            token_count = self.count_tokens(text)

            if total_tokens + token_count > self.max_tokens:
                continue

            selected.append(result)
            total_tokens += token_count

        return EvidenceBudgetResult(
            selected_results=selected,
            token_count=total_tokens,
            budget=self.max_tokens,
            candidates_considered=len(results),
            candidates_selected=len(selected)
        )
