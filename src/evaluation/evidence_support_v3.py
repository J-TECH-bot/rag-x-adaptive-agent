from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EvidenceSupportJudgeV3:
    """
    Claim-level evidence verification judge.

    Labels:
        SUPPORTED
        INSUFFICIENT
        CONTRADICTED

    The judge verifies whether the evidence directly supports
    the claim required by the question.

    It does not use gold answers or outside knowledge.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base"
    ):
        self.model_name = model_name

        print(
            f"Loading evidence judge V3: {model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        print("Evidence judge V3 loaded.")

    def judge(
        self,
        question: str,
        evidence: str
    ) -> dict:

        prompt = f"""
You are a strict claim-level evidence verification system.

Your task is to determine whether the EVIDENCE directly
supports the exact claim required by the QUESTION.

Return exactly ONE label:

SUPPORTED
INSUFFICIENT
CONTRADICTED

Definitions:

SUPPORTED:
The evidence directly and explicitly supports the claim
required by the question.

INSUFFICIENT:
The evidence is related to the question, but does not
provide enough information to establish the exact claim.

CONTRADICTED:
The evidence explicitly conflicts with the claim.

IMPORTANT RULES:

1. Topic relevance is NOT the same as support.

2. Do not use outside knowledge.

3. Do not assume facts that are not written in the evidence.

4. Quantitative claims must have direct quantitative support.

5. Claims containing words such as:
   always, never, every, all, none, exactly, only,
   guaranteed, must, best, optimal, proves, completely
   require explicit evidence for that strong claim.

6. If the evidence supports only a weaker statement,
   return INSUFFICIENT.

7. If the evidence gives a counterexample, limitation,
   exception, tradeoff, or conflicting statement that
   disproves a universal or absolute claim, return
   CONTRADICTED.

8. A related experiment does not prove a universal claim.

9. If the exact claim cannot be established from the
   evidence, prefer INSUFFICIENT.

10. Be conservative. When uncertain, return INSUFFICIENT.

QUESTION:
{question}

EVIDENCE:
{evidence}

LABEL:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

        raw_output = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        normalized = raw_output.upper()

        if normalized.startswith("SUPPORTED"):
            label = "SUPPORTED"

        elif normalized.startswith("CONTRADICTED"):
            label = "CONTRADICTED"

        else:
            # Fail closed.
            label = "INSUFFICIENT"

        return {
            "label": label,
            "raw_output": raw_output
        }
