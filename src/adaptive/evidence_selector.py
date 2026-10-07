from dataclasses import dataclass


@dataclass
class EvidenceSelectionResult:
    """
    Result of adaptive evidence-depth selection.

    The selector progressively increases the amount of evidence
    passed to the evidence judge.
    """

    status: str
    selected_depth: int
    selected_evidence: list[dict]
    rounds: list[dict]


class AdaptiveEvidenceSelector:
    """
    Adaptively selects evidence using progressive depths.

    Default strategy:

        Top-1
          ↓ insufficient
        Top-3
          ↓ insufficient
        Top-5
          ↓ insufficient
        Top-10
          ↓ insufficient
        ABSTAIN

    This class does not retrieve or rerank documents.

    It receives already-ranked candidates and delegates evidence
    sufficiency judgment to the supplied judge.
    """

    def __init__(
        self,
        evidence_judge,
        evidence_depths: tuple[int, ...] = (1, 3, 5, 10)
    ):
        if not evidence_depths:
            raise ValueError(
                "evidence_depths must contain at least one depth"
            )

        if any(depth < 1 for depth in evidence_depths):
            raise ValueError(
                "All evidence depths must be >= 1"
            )

        if tuple(sorted(evidence_depths)) != evidence_depths:
            raise ValueError(
                "evidence_depths must be in ascending order"
            )

        self.evidence_judge = evidence_judge
        self.evidence_depths = evidence_depths

    @staticmethod
    def combine_evidence(
        results: list[dict]
    ) -> str:
        """
        Combine selected evidence into the context passed
        to the evidence judge.
        """

        return "\n\n".join(
            result["text"]
            for result in results
        )

    def select(
        self,
        query: str,
        ranked_candidates: list[dict]
    ) -> EvidenceSelectionResult:
        """
        Progressively increase evidence depth until the evidence
        is supported or all configured depths are exhausted.
        """

        rounds = []

        for depth in self.evidence_depths:

            evidence = ranked_candidates[:depth]

            evidence_text = self.combine_evidence(
                evidence
            )

            judgment = self.evidence_judge.judge(
                query,
                evidence_text
            )

            label = (
                judgment["label"]
                .strip()
                .upper()
            )

            round_result = {
                "depth": depth,
                "evidence_count": len(evidence),
                "evidence_label": label,
                "raw_judge_output": judgment["raw_output"]
            }

            rounds.append(round_result)

            # ------------------------------------------------
            # Supported evidence -> stop immediately.
            # ------------------------------------------------

            if label == "SUPPORTED":

                return EvidenceSelectionResult(
                    status="SUPPORTED",
                    selected_depth=depth,
                    selected_evidence=evidence,
                    rounds=rounds
                )

            # ------------------------------------------------
            # Contradicted evidence -> fail closed.
            # ------------------------------------------------

            if label == "CONTRADICTED":

                return EvidenceSelectionResult(
                    status="ABSTAIN",
                    selected_depth=depth,
                    selected_evidence=evidence,
                    rounds=rounds
                )

            # ------------------------------------------------
            # INSUFFICIENT -> continue to next depth.
            # ------------------------------------------------

        # ----------------------------------------------------
        # No evidence depth was sufficient.
        # ----------------------------------------------------

        final_evidence = (
            ranked_candidates[:self.evidence_depths[-1]]
        )

        return EvidenceSelectionResult(
            status="ABSTAIN",
            selected_depth=min(
                self.evidence_depths[-1],
                len(ranked_candidates)
            ),
            selected_evidence=final_evidence,
            rounds=rounds
        )
