from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EvidenceSupportJudge:
    """
    Determines whether retrieved evidence is sufficient
    to answer a question.

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

Determine whether the provided evidence contains enough
information to answer the question.

Return exactly one label:

SUPPORTED
INSUFFICIENT

SUPPORTED means:
The evidence contains enough information to answer the question.

INSUFFICIENT means:
The evidence does not contain enough information to answer
the question, even if the evidence is related to the topic.

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

        elif normalized.startswith("SUFFICIENT"):
            label = "SUPPORTED"

        elif normalized.startswith("INSUFFICIENT"):
            label = "INSUFFICIENT"
        else:
            label = "INSUFFICIENT"

        return {
            "label": label,
            "raw_output": raw_output
        }