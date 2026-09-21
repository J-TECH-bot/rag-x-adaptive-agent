from evaluation.evidence_support import EvidenceSupportJudge


def main():

    judge = EvidenceSupportJudge()

    examples = [
        {
            "question": "What are the two forms of memory used by RAG?",
            "evidence": """
            RAG combines parametric memory stored in the
            neural network with non-parametric memory stored
            in an external dense vector index.
            """
        },
        {
            "question": "What GPU model was used to train the RAG models?",
            "evidence": """
            RAG is evaluated on several knowledge-intensive
            NLP tasks including question answering and fact
            verification. The experiments compare different
            retrieval configurations.
            """
        }
    ]

    for i, example in enumerate(examples, start=1):

        result = judge.judge(
            question=example["question"],
            evidence=example["evidence"]
        )

        print("=" * 80)
        print(f"Example {i}")
        print("=" * 80)

        print(f"Question: {example['question']}")
        print(f"Prediction: {result['label']}")
        print(f"Raw output: {result['raw_output']}")
        print()


if __name__ == "__main__":
    main()