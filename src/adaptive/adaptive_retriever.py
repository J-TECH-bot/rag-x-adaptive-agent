from __future__ import annotations

from adaptive.decision_engine import AdaptiveDecisionEngine
from adaptive.evidence_selector import AdaptiveEvidenceSelector
from evaluation.evidence_support_v2 import EvidenceSupportJudgeV2
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker


class AdaptiveRetriever:
    """
    RAG-X adaptive retrieval controller.

    The system adapts at two levels:

    1. Candidate retrieval depth:
       initial_candidate_k -> expanded_candidate_k

    2. Evidence depth:
       1 -> 3 -> 5 -> 10

    The evidence selector determines whether the current ranked
    candidates contain sufficient evidence.

    The decision engine determines whether the system should:
        - ANSWER
        - EXPAND_RETRIEVAL
        - ABSTAIN
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        evidence_judge: EvidenceSupportJudgeV2,
        decision_engine: AdaptiveDecisionEngine | None = None,
        evidence_selector: AdaptiveEvidenceSelector | None = None,
        initial_candidate_k: int = 20,
        expanded_candidate_k: int = 50,
    ):
        if initial_candidate_k < 1:
            raise ValueError(
                "initial_candidate_k must be >= 1"
            )

        if expanded_candidate_k < initial_candidate_k:
            raise ValueError(
                "expanded_candidate_k must be >= initial_candidate_k"
            )

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.evidence_judge = evidence_judge

        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else AdaptiveDecisionEngine(
                max_retrieval_rounds=2
            )
        )

        self.evidence_selector = (
            evidence_selector
            if evidence_selector is not None
            else AdaptiveEvidenceSelector(
                evidence_judge=evidence_judge,
                evidence_depths=(1, 3, 5, 10),
            )
        )

        self.candidate_depths = [
            initial_candidate_k,
            expanded_candidate_k,
        ]

    def retrieve(self, query: str) -> dict:
        """
        Run adaptive retrieval for a single query.

        Returns a structured result containing:
            - final decision
            - evidence label
            - selected evidence
            - selected evidence depth
            - retrieval/evidence trace
        """

        trace = []

        for retrieval_round, candidate_k in enumerate(
            self.candidate_depths,
            start=1,
        ):
            # ---------------------------------------------------------
            # 1. Hybrid retrieval
            # ---------------------------------------------------------
            retrieved = self.hybrid_retriever.retrieve(
                query,
                top_k=candidate_k,
            )

            candidate_count = len(retrieved)

            # ---------------------------------------------------------
            # 2. Cross-encoder reranking
            # ---------------------------------------------------------
            reranked = self.reranker.rerank(
                query,
                retrieved,
            )

            # ---------------------------------------------------------
            # 3. Adaptive evidence selection
            #
            # Evidence depths:
            # 1 -> 3 -> 5 -> 10
            # ---------------------------------------------------------
            evidence_result = self.evidence_selector.select(
                query=query,
                ranked_candidates=reranked,
            )

            evidence_rounds = evidence_result.rounds

            # The final evidence round tells us why the selector
            # stopped.
            final_evidence_label = (
                evidence_rounds[-1]["evidence_label"]
                if evidence_rounds
                else "UNKNOWN"
            )

            # ---------------------------------------------------------
            # 4. Adaptive decision engine
            # ---------------------------------------------------------
            if evidence_result.status == "SUPPORTED":
                decision_result = self.decision_engine.decide(
                    evidence_label="SUPPORTED",
                    candidate_count=candidate_count,
                    evidence_count=len(
                        evidence_result.selected_evidence
                    ),
                    retrieval_round=retrieval_round,
                )

            elif final_evidence_label == "CONTRADICTED":
                decision_result = self.decision_engine.decide(
                    evidence_label="CONTRADICTED",
                    candidate_count=candidate_count,
                    evidence_count=len(
                        evidence_result.selected_evidence
                    ),
                    retrieval_round=retrieval_round,
                )

            else:
                decision_result = self.decision_engine.decide(
                    evidence_label="INSUFFICIENT",
                    candidate_count=candidate_count,
                    evidence_count=len(
                        evidence_result.selected_evidence
                    ),
                    retrieval_round=retrieval_round,
                )

            # ---------------------------------------------------------
            # 5. Store complete adaptive trace
            #
            # IMPORTANT:
            # Store reranked candidates so evaluation can distinguish:
            #
            #   retrieval failure
            #       vs
            #   ranking failure
            #
            # This is diagnostic information only and does not affect
            # the adaptive decision itself.
            # ---------------------------------------------------------
            trace_entry = {
                "retrieval_round": retrieval_round,
                "candidate_k": candidate_k,
                "candidate_count": candidate_count,

                "retrieved_candidates": retrieved,
                "reranked_candidates": reranked,

                "evidence_status": evidence_result.status,
                "selected_evidence_depth": (
                    evidence_result.selected_depth
                ),
                "evidence_rounds": evidence_rounds,

                "decision": decision_result.decision,
                "decision_reason": decision_result.reason,
                "evidence_label": decision_result.evidence_label,
                "evidence_count": decision_result.evidence_count,
            }

            trace.append(trace_entry)

            # ---------------------------------------------------------
            # 6. Final answer
            # ---------------------------------------------------------
            if decision_result.decision == "ANSWER":
                return {
                    "query": query,
                    "decision": "ANSWER",
                    "reason": decision_result.reason,
                    "evidence_label": decision_result.evidence_label,
                    "selected_evidence_depth": (
                        evidence_result.selected_depth
                    ),
                    "evidence": evidence_result.selected_evidence,
                    "trace": trace,
                }

            # ---------------------------------------------------------
            # 7. Explicit abstention
            # ---------------------------------------------------------
            if decision_result.decision == "ABSTAIN":
                return {
                    "query": query,
                    "decision": "ABSTAIN",
                    "reason": decision_result.reason,
                    "evidence_label": decision_result.evidence_label,
                    "selected_evidence_depth": (
                        evidence_result.selected_depth
                    ),
                    "evidence": evidence_result.selected_evidence,
                    "trace": trace,
                }

            # ---------------------------------------------------------
            # 8. EXPAND_RETRIEVAL
            #
            # Continue to the next candidate depth.
            # ---------------------------------------------------------
            if decision_result.decision == "EXPAND_RETRIEVAL":
                continue

        # -------------------------------------------------------------
        # Safety fallback
        # -------------------------------------------------------------
        if trace:
            final_trace = trace[-1]

            return {
                "query": query,
                "decision": "ABSTAIN",
                "reason": (
                    "All adaptive retrieval depths were exhausted "
                    "without sufficient supporting evidence."
                ),
                "evidence_label": final_trace["evidence_label"],
                "selected_evidence_depth": (
                    final_trace["selected_evidence_depth"]
                ),
                "evidence": [],
                "trace": trace,
            }

        return {
            "query": query,
            "decision": "ABSTAIN",
            "reason": "No retrieval results were produced.",
            "evidence_label": "UNKNOWN",
            "selected_evidence_depth": 0,
            "evidence": [],
            "trace": [],
        }
