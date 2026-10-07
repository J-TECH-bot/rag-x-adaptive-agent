from adaptive.evidence_selector import (
    AdaptiveEvidenceSelector,
)


class MockEvidenceJudge:
    """
    Deterministic evidence judge used only for unit testing.
    """

    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = []

    def judge(self, query, evidence_text):
        label = next(self.labels)

        self.calls.append({
            "query": query,
            "evidence_text": evidence_text
        })

        return {
            "label": label,
            "raw_output": label
        }


def make_candidates(count=10):

    return [
        {
            "chunk_index": i,
            "score": float(10 - i),
            "text": f"Evidence chunk {i}"
        }
        for i in range(1, count + 1)
    ]


def test_supported_at_top_1():

    judge = MockEvidenceJudge([
        "SUPPORTED"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "SUPPORTED"
    assert result.selected_depth == 1
    assert len(result.selected_evidence) == 1
    assert len(result.rounds) == 1


def test_supported_at_top_3():

    judge = MockEvidenceJudge([
        "INSUFFICIENT",
        "SUPPORTED"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "SUPPORTED"
    assert result.selected_depth == 3
    assert len(result.selected_evidence) == 3
    assert len(result.rounds) == 2


def test_supported_at_top_5():

    judge = MockEvidenceJudge([
        "INSUFFICIENT",
        "INSUFFICIENT",
        "SUPPORTED"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "SUPPORTED"
    assert result.selected_depth == 5
    assert len(result.selected_evidence) == 5
    assert len(result.rounds) == 3


def test_supported_at_top_10():

    judge = MockEvidenceJudge([
        "INSUFFICIENT",
        "INSUFFICIENT",
        "INSUFFICIENT",
        "SUPPORTED"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "SUPPORTED"
    assert result.selected_depth == 10
    assert len(result.selected_evidence) == 10
    assert len(result.rounds) == 4


def test_abstain_when_all_depths_are_insufficient():

    judge = MockEvidenceJudge([
        "INSUFFICIENT",
        "INSUFFICIENT",
        "INSUFFICIENT",
        "INSUFFICIENT"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "ABSTAIN"
    assert result.selected_depth == 10
    assert len(result.selected_evidence) == 10
    assert len(result.rounds) == 4


def test_contradicted_causes_abstention():

    judge = MockEvidenceJudge([
        "CONTRADICTED"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "ABSTAIN"
    assert result.selected_depth == 1
    assert len(result.rounds) == 1


def test_custom_depths():

    judge = MockEvidenceJudge([
        "INSUFFICIENT",
        "SUPPORTED"
    ])

    selector = AdaptiveEvidenceSelector(
        evidence_judge=judge,
        evidence_depths=(2, 4)
    )

    result = selector.select(
        query="test query",
        ranked_candidates=make_candidates()
    )

    assert result.status == "SUPPORTED"
    assert result.selected_depth == 4
    assert len(result.selected_evidence) == 4


if __name__ == "__main__":

    test_supported_at_top_1()
    test_supported_at_top_3()
    test_supported_at_top_5()
    test_supported_at_top_10()
    test_abstain_when_all_depths_are_insufficient()
    test_contradicted_causes_abstention()
    test_custom_depths()

    print("\n✓ All adaptive evidence selector tests passed.")
