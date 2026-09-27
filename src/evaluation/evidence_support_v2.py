from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EvidenceSupportJudgeV2:
    """
    Binary evidence sufficiency judge.

    The judge answers one question:

        Does the provided evidence contain enough
        information to answer the question?

    Output:

        SUPPORTED
        INSUFFICIENT

    The judge does not use gold answers.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base"
    ):
        self.model_name = model_name

        print(
            f"Loading evidence judge V2: {model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        print("Evidence judge V2 loaded.")

    def judge(
        self,
        question: str,
        evidence: str
    ) -> dict:

        prompt = f"""
You are an evidence verification system.

Your task is to determine whether the EVIDENCE
contains enough information to answer the QUESTION.

Return exactly ONE label:

SUPPORTED
INSUFFICIENT

SUPPORTED means:
The evidence explicitly contains the information
needed to answer the question.

INSUFFICIENT means:
The evidence does not contain enough information
to answer the question.

Important rules:

1. Related information is NOT enough.
2. Do not use your own outside knowledge.
3. Do not assume information that is not written
   in the evidence.
4. If the answer cannot be determined directly
   from the evidence, return INSUFFICIENT.
5. A question about a fact that is absent from
   the evidence must be INSUFFICIENT.

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

        else:
            label = "INSUFFICIENT"

        return {
            "label": label,
            "raw_output": raw_output
        }
