from dataclasses import dataclass
from typing import Literal


Decision = Literal[
    "ANSWER",
    "EXPAND_RETRIEVAL",
    "ABSTAIN"
]


@dataclass
class DecisionResult:
    """
    Decision made by the RAG-X adaptive controller.

    The controller decides whether the current evidence is
    sufficient, whether retrieval should be expanded, or
    whether the system should abstain.
    """

    decision: Decision
    reason: str
    evidence_label: str
    candidate_count: int
    evidence_count: int


class AdaptiveDecisionEngine:
    """
    Controls the retrieval -> reranking -> evidence decision loop.

    The engine intentionally does NOT generate an answer.

    Responsibilities:
        1. Accept evidence judgment.
        2. Decide whether more retrieval is needed.
        3. Answer only when evidence is supported.
        4. Abstain when retrieval expansion is exhausted.

    This separation allows the project to evaluate retrieval
    and answer generation independently.
    """

    def __init__(
        self,
        max_retrieval_rounds: int = 2
    ):
        if max_retrieval_rounds < 1:
            raise ValueError(
                "max_retrieval_rounds must be >= 1"
            )

        self.max_retrieval_rounds = max_retrieval_rounds

    def decide(
        self,
        evidence_label: str,
        candidate_count: int,
        evidence_count: int,
        retrieval_round: int
    ) -> DecisionResult:
        """
        Decide what RAG-X should do next.

        Parameters
        ----------
        evidence_label:
            Output from EvidenceSupportJudge.

            Expected values:
                SUPPORTED
                INSUFFICIENT
                CONTRADICTED

        candidate_count:
            Number of retrieved candidates available
            before final evidence selection.

        evidence_count:
            Number of chunks passed to the evidence judge.

        retrieval_round:
            Current adaptive retrieval round.
            Starts at 1.
        """

        normalized_label = (
            evidence_label
            .strip()
            .upper()
        )

        if retrieval_round < 1:
            raise ValueError(
                "retrieval_round must be >= 1"
            )

        # ----------------------------------------------------
        # Supported evidence
        # ----------------------------------------------------

        if normalized_label == "SUPPORTED":

            return DecisionResult(
                decision="ANSWER",
                reason=(
                    "The evidence judge determined that "
                    "the retrieved evidence supports the query."
                ),
                evidence_label=normalized_label,
                candidate_count=candidate_count,
                evidence_count=evidence_count
            )

        # ----------------------------------------------------
        # Insufficient evidence
        # ----------------------------------------------------

        if normalized_label == "INSUFFICIENT":

            if retrieval_round < self.max_retrieval_rounds:

                return DecisionResult(
                    decision="EXPAND_RETRIEVAL",
                    reason=(
                        "Current evidence is insufficient. "
                        "Additional retrieval should be attempted "
                        "before abstaining."
                    ),
                    evidence_label=normalized_label,
                    candidate_count=candidate_count,
                    evidence_count=evidence_count
                )

            return DecisionResult(
                decision="ABSTAIN",
                reason=(
                    "Evidence remained insufficient after "
                    "the maximum retrieval depth was reached."
                ),
                evidence_label=normalized_label,
                candidate_count=candidate_count,
                evidence_count=evidence_count
            )

        # ----------------------------------------------------
        # Contradicted evidence
        # ----------------------------------------------------

        if normalized_label == "CONTRADICTED":

            return DecisionResult(
                decision="ABSTAIN",
                reason=(
                    "Retrieved evidence contradicts the "
                    "claim or does not support it as stated."
                ),
                evidence_label=normalized_label,
                candidate_count=candidate_count,
                evidence_count=evidence_count
            )

        # ----------------------------------------------------
        # Unknown label
        # ----------------------------------------------------

        # Fail closed.
        return DecisionResult(
            decision="ABSTAIN",
            reason=(
                "The evidence judge returned an unknown label. "
                "The system therefore fails closed instead of "
                "answering without verified evidence."
            ),
            evidence_label=normalized_label,
            candidate_count=candidate_count,
            evidence_count=evidence_count
        )
