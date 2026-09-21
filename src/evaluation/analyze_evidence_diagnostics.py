import json
import statistics


DIAGNOSTIC_PATH = "data/evaluation/evidence_diagnostics.json"


def summarize(values):
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values)
    }


def main():

    with open(DIAGNOSTIC_PATH, "r") as f:
        diagnostics = json.load(f)

    answerable = [
        item for item in diagnostics
        if item["answerability"] == "answerable"
    ]

    unanswerable = [
        item for item in diagnostics
        if item["answerability"] == "unanswerable"
    ]

    # ---------------------------------------------------------
    # Extract scores
    # ---------------------------------------------------------

    answerable_lexical = [
        item["lexical_score"]
        for item in answerable
    ]

    unanswerable_lexical = [
        item["lexical_score"]
        for item in unanswerable
    ]

    answerable_semantic = [
        item["semantic_score"]
        for item in answerable
    ]

    unanswerable_semantic = [
        item["semantic_score"]
        for item in unanswerable
    ]


    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVIDENCE DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    print("\nQuestion distribution:")
    print(f"Answerable:   {len(answerable)}")
    print(f"Unanswerable: {len(unanswerable)}")


    # ---------------------------------------------------------
    # Lexical statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("LEXICAL SCORE")
    print("-" * 80)

    print("\nAnswerable:")
    print(summarize(answerable_lexical))

    print("\nUnanswerable:")
    print(summarize(unanswerable_lexical))


    # ---------------------------------------------------------
    # Semantic statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("SEMANTIC SCORE")
    print("-" * 80)

    print("\nAnswerable:")
    print(summarize(answerable_semantic))

    print("\nUnanswerable:")
    print(summarize(unanswerable_semantic))


    # ---------------------------------------------------------
    # Find overlap
    # ---------------------------------------------------------

    answerable_min_semantic = min(answerable_semantic)
    unanswerable_max_semantic = max(unanswerable_semantic)

    print("\n" + "-" * 80)
    print("SEMANTIC DISTRIBUTION OVERLAP")
    print("-" * 80)

    print(
        f"Lowest answerable semantic score: "
        f"{answerable_min_semantic:.4f}"
    )

    print(
        f"Highest unanswerable semantic score: "
        f"{unanswerable_max_semantic:.4f}"
    )


    if unanswerable_max_semantic >= answerable_min_semantic:
        print(
            "\nThere is substantial overlap between "
            "answerable and unanswerable semantic scores."
        )
    else:
        print(
            "\nThere is no overlap between the two "
            "semantic score ranges."
        )


    # ---------------------------------------------------------
    # Sort semantic scores
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("UNANSWERABLE QUESTIONS WITH HIGHEST SEMANTIC SCORES")
    print("-" * 80)

    sorted_unanswerable = sorted(
        unanswerable,
        key=lambda x: x["semantic_score"],
        reverse=True
    )

    for item in sorted_unanswerable:

        print(
            f"Q{item['id']:02d} | "
            f"Semantic: {item['semantic_score']:.4f} | "
            f"Lexical: {item['lexical_score']:.4f} | "
            f"{item['question']}"
        )


    # ---------------------------------------------------------
    # Answerable questions with lowest semantic scores
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("ANSWERABLE QUESTIONS WITH LOWEST SEMANTIC SCORES")
    print("-" * 80)

    sorted_answerable = sorted(
        answerable,
        key=lambda x: x["semantic_score"]
    )

    for item in sorted_answerable[:10]:

        print(
            f"Q{item['id']:02d} | "
            f"Semantic: {item['semantic_score']:.4f} | "
            f"Lexical: {item['lexical_score']:.4f} | "
            f"{item['question']}"
        )


    # ---------------------------------------------------------
    # Lexical false-positive analysis
    # ---------------------------------------------------------

    lexical_false_positive = [
        item for item in unanswerable
        if item["lexical_sufficient"]
    ]

    print("\n" + "-" * 80)
    print("LEXICAL FALSE POSITIVES")
    print("-" * 80)

    print(
        f"Unanswerable questions incorrectly considered "
        f"lexically sufficient: "
        f"{len(lexical_false_positive)}/{len(unanswerable)}"
    )


    # ---------------------------------------------------------
    # Important diagnostic examples
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("IMPORTANT CASES")
    print("-" * 80)

    important_ids = [56, 57, 58, 59, 60]

    for item in diagnostics:

        if item["id"] in important_ids:

            print(
                f"\nQ{item['id']}: "
                f"{item['question']}"
            )

            print(
                f"  Lexical score:  "
                f"{item['lexical_score']:.4f}"
            )

            print(
                f"  Semantic score: "
                f"{item['semantic_score']:.4f}"
            )

            print(
                f"  Expected:       "
                f"{item['expected_behavior']}"
            )


if __name__ == "__main__":
    main()