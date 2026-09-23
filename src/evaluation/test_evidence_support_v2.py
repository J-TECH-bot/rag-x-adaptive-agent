from evaluation.evidence_support import EvidenceSupportJudge


judge = EvidenceSupportJudge()


tests = [
    {
        "name": "Supported",
        "question": "What are the two forms of memory used by RAG?",
        "evidence": """
RAG combines pre-trained parametric memory with
non-parametric memory in the form of a dense vector index
of Wikipedia.
"""
    },
    {
        "name": "Insufficient",
        "question": "What GPU model was used to train the RAG models?",
        "evidence": """
RAG models are trained and evaluated on several
knowledge-intensive NLP tasks.
"""
    },
    {
        "name": "Contradicted",
        "question": "Does the RAG paper show that a larger retrieval depth always produces better results?",
        "evidence": """
Models are trained with either 5 or 10 retrieved latent
documents. We do not observe significant differences in
performance between them. Performance peaks for RAG-Token
at 10 retrieved documents. Retrieving more documents can
also improve one metric at the expense of another.
"""
    }
]


for test in tests:

    result = judge.judge(
        question=test["question"],
        evidence=test["evidence"]
    )

    print("\n" + "=" * 70)
    print(test["name"])
    print("=" * 70)
    print("Expected:", test["name"].upper())
    print("Predicted:", result["label"])
    print("Raw output:", result["raw_output"])