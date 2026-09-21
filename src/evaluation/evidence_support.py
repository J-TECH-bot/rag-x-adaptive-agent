from transformers import pipeline


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

        self.model = pipeline(
            "text2text-generation",
            model=model_name
        )

    def judge(
        self,
        question: str,
        evidence: str
    ) -> dict:

        prompt = f"""
You are an evidence verification system.

Your task is to determine whether the provided evidence
contains enough information to answer the question.

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

        output = self.model(
            prompt,
            max_new_tokens=10,
            do_sample=False
        )

        raw_output = output[0]["generated_text"].strip()

        if "SUPPORTED" in raw_output.upper():
            label = "SUPPORTED"
        else:
            label = "INSUFFICIENT"

        return {
            "label": label,
            "raw_output": raw_output
        }