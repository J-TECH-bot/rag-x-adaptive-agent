from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EvidenceSupportJudge:
    """
    Determines the relationship between a question and
    the provided evidence.

    Possible labels:

        SUPPORTED
        INSUFFICIENT
        CONTRADICTED

    The judge only receives:
        - question
        - retrieved evidence

    It does not use evaluation-only gold answers.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base"
    ):
        self.model_name = model_name

        print(f"Loading evidence judge: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print("Evidence judge loaded.")

    def judge(
        self,
        question: str,
        evidence: str
    ) -> dict:

        prompt = f"""
You are an evidence verification system.

Determine the relationship between the QUESTION and the EVIDENCE.

Return exactly ONE label:

SUPPORTED
INSUFFICIENT
CONTRADICTED

Definitions:

SUPPORTED:
The evidence directly provides enough information to support
the claim or answer the question.

INSUFFICIENT:
The evidence is related to the question, but it does not
contain enough information to establish the answer or claim.

CONTRADICTED:
The evidence contains information that conflicts with the
claim or shows that the claim is false, too strong, or not
supported as stated.

Important:
Pay special attention to words such as:
- always
- never
- all
- none
- proves
- guarantees
- exactly
- must

Do not treat evidence about a related topic as support for
a stronger claim.

Question:
{question}

Evidence:
{evidence}

Label:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
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

        elif normalized.startswith("INSUFFICIENT"):
            label = "INSUFFICIENT"

        elif normalized.startswith("SUFFICIENT"):
            # FLAN-T5 sometimes uses SUFFICIENT instead of SUPPORTED.
            label = "SUPPORTED"

        else:
            # Fail closed.
            label = "INSUFFICIENT"

        return {
            "label": label,
            "raw_output": raw_output
        }