from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from evaluation.evidence_support_v2 import EvidenceSupportJudgeV2

from adaptive.decision_engine import (
    AdaptiveDecisionEngine,
)


class AdaptiveRetriever:
    """
    RAG-X adaptive retrieval controller.

    Retrieval strategy:

        Round 1:
            Hybrid retrieval
            -> reranking
            -> evidence judgment

        Round 2:
            If evidence is insufficient:
            expand candidate depth
            -> reranking
            -> evidence judgment

        Final:
            SUPPORTED     -> ANSWER
            INSUFFICIENT  -> ABSTAIN
            CONTRADICTED  -> ABSTAIN

    The class does not generate the final natural-language
    answer. It only determines whether enough evidence exists.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        evidence_judge: EvidenceSupportJudgeV2,
        decision_engine: AdaptiveDecisionEngine | None = None,
        evidence_top_k: int = 10,
        initial_candidate_k: int = 20,
        expanded_candidate_k: int = 50
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.evidence_judge = evidence_judge

        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else AdaptiveDecisionEngine()
        )

        self.evidence_top_k = evidence_top_k
        self.initial_candidate_k = initial_candidate_k
        self.expanded_candidate_k = expanded_candidate_k

    @staticmethod
    def combine_evidence(
        results: list[dict]
    ) -> str:
        """
        Combine retrieved chunks into evidence context.
        """

        return "\n\n".join(
            result["text"]
            for result in results
        )

    def retrieve(
        self,
        query: str
    ) -> dict:
        """
        Execute adaptive retrieval for one query.

        Returns a complete trace of the adaptive process.
        """

        rounds = []

        candidate_depths = [
            self.initial_candidate_k,
            self.expanded_candidate_k
        ]

        for round_number, candidate_k in enumerate(
            candidate_depths,
            start=1
        ):

            # ------------------------------------------------
            # Hybrid retrieval
            # ------------------------------------------------

            candidates = self.hybrid_retriever.retrieve(
                query,
                top_k=candidate_k,
                candidate_k=candidate_k
            )

            # ------------------------------------------------
            # Cross-encoder reranking
            # ------------------------------------------------

            reranked = self.reranker.rerank(
                query,
                candidates
            )

            # ------------------------------------------------
            # Final evidence for this round
            # ------------------------------------------------

            evidence = reranked[
                :self.evidence_top_k
            ]

            evidence_text = self.combine_evidence(
                evidence
            )

            # ------------------------------------------------
            # Evidence judgment
            # ------------------------------------------------

            judgment = self.evidence_judge.judge(
                query,
                evidence_text
            )

            # ------------------------------------------------
            # Adaptive decision
            # ------------------------------------------------

            decision = self.decision_engine.decide(
                evidence_label=judgment["label"],
                candidate_count=len(candidates),
                evidence_count=len(evidence),
                retrieval_round=round_number
            )

            round_result = {
                "round": round_number,
                "candidate_k": candidate_k,
                "candidate_count": len(candidates),
                "evidence_count": len(evidence),
                "evidence_label": judgment["label"],
                "raw_judge_output": judgment["raw_output"],
                "decision": decision.decision,
                "reason": decision.reason,
                "evidence": evidence
            }

            rounds.append(round_result)

            # ------------------------------------------------
            # Stop when the controller has made a final
            # decision.
            # ------------------------------------------------

            if decision.decision in {
                "ANSWER",
                "ABSTAIN"
            }:

                return {
                    "query": query,
                    "final_decision": decision.decision,
                    "final_reason": decision.reason,
                    "final_evidence_label": judgment["label"],
                    "rounds_used": round_number,
                    "evidence": evidence,
                    "trace": rounds
                }

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        return {
            "query": query,
            "final_decision": "ABSTAIN",
            "final_reason": (
                "Adaptive retrieval completed without "
                "sufficient verified evidence."
            ),
            "final_evidence_label": "INSUFFICIENT",
            "rounds_used": len(rounds),
            "evidence": (
                rounds[-1]["evidence"]
                if rounds
                else []
            ),
            "trace": rounds
        }
