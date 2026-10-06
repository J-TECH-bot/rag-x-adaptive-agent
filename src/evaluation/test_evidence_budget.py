from transformers import AutoTokenizer

from evaluation.evidence_budget import EvidenceBudget


MODEL_NAME = "google/flan-t5-base"


def main():
    print("=" * 70)
    print("EVIDENCE BUDGET UNIT TEST")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    budget = EvidenceBudget(
        tokenizer=tokenizer,
        max_tokens=512
    )

    results = [
        {
            "chunk_index": 1,
            "score": 10.0,
            "text": "This is a short evidence passage."
        },
        {
            "chunk_index": 2,
            "score": 9.0,
            "text": "This is another evidence passage containing additional information."
        },
        {
            "chunk_index": 3,
            "score": 8.0,
            "text": "This is a third evidence passage that should also fit within the budget."
        }
    ]

    result = budget.select(results)

    print("\nBudget:", result.budget)
    print("Tokens selected:", result.token_count)
    print("Candidates considered:", result.candidates_considered)
    print("Candidates selected:", result.candidates_selected)

    print("\nSelected chunks:")

    for item in result.selected_results:
        print(
            f"  Chunk {item['chunk_index']} "
            f"(score={item['score']})"
        )

    assert result.token_count <= 512
    assert result.candidates_selected > 0

    print("\n✓ Evidence budget test passed.")


if __name__ == "__main__":
    main()
