from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EvidenceSupportJudgeV3_1:
    """
    Strict claim-level evidence verification judge.

    Labels:
        SUPPORTED
        INSUFFICIENT
        CONTRADICTED

    V3.1 improves on V3 by explicitly separating:

        1. The exact claim being tested
        2. The evidence that supports or contradicts that claim
        3. The scope of the claim

    In particular, evidence supporting some cases must not be
    treated as support for universal claims such as:

        always
        never
        every
        all
        none
        exactly
        only
        optimal
        guaranteed
        proves
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base"
    ):
        self.model_name = model_name

        print(
            f"Loading evidence judge V3.1: {model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        print("Evidence judge V3.1 loaded.")

    def judge(
        self,
        question: str,
        evidence: str
    ) -> dict:

        prompt = f"""
You are a strict claim verification system.

Your task is to verify the EXACT CLAIM expressed by the
QUESTION using ONLY the provided EVIDENCE.

You must evaluate whether the evidence logically supports
the entire claim, not merely whether the evidence is related
to the topic.

Return exactly ONE label:

SUPPORTED
INSUFFICIENT
CONTRADICTED

--------------------------------------------------
STEP 1: IDENTIFY THE EXACT CLAIM
--------------------------------------------------

Interpret the question as a claim that needs verification.

Pay attention to:

- subject
- action
- condition
- quantity
- comparison
- scope
- certainty
- universal or absolute wording

Do NOT weaken the claim.

For example:

"Does retrieval always improve generation quality?"

means the strong claim:

"Retrieval improves generation quality in every relevant
case."

It does NOT mean:

"Retrieval improves generation quality in some experiments."

Similarly:

"Does retrieving exactly 10 documents always produce
the best results?"

means:

"10 documents are always the optimal retrieval depth."

--------------------------------------------------
STEP 2: VERIFY THE COMPLETE CLAIM
--------------------------------------------------

SUPPORTED means:

The evidence directly establishes the complete claim,
including its scope, quantity, conditions, and certainty.

Evidence supporting only some examples or some experiments
is NOT enough to support a universal claim.

INSUFFICIENT means:

The evidence is related to the claim but does not establish
the complete claim.

If the evidence shows improvement in some cases but does not
establish that the improvement occurs in every case, return
INSUFFICIENT.

If the evidence reports an experiment but does not prove a
universal conclusion, return INSUFFICIENT.

If the evidence gives only partial support, return
INSUFFICIENT.

--------------------------------------------------
STEP 3: CHECK FOR CONTRADICTION
--------------------------------------------------

CONTRADICTED means:

The evidence explicitly conflicts with the claim.

Examples:

Claim:
"Retrieving more documents always improves performance."

Evidence:
"Performance peaks at 10 documents."

Result:
CONTRADICTED

Claim:
"The retriever never makes mistakes."

Evidence:
"The top retrieved document matches gold evidence in 71%
of cases."

Result:
CONTRADICTED

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

1. Topic relevance is NOT evidence of entailment.

2. Do not use outside knowledge.

3. Do not assume information that is not explicitly present.

4. Quantitative claims require direct quantitative evidence.

5. Universal or absolute words such as:

   always
   never
   every
   all
   none
   exactly
   only
   optimal
   guaranteed
   proves
   completely

   require evidence covering the full scope of the claim.

6. Evidence supporting "some" cases cannot support "all"
   cases.

7. Evidence showing an average, percentage, experiment,
   or benchmark result cannot by itself establish a universal
   statement.

8. If the evidence supports a weaker version of the claim,
   return INSUFFICIENT.

9. If the evidence contains a clear exception, limitation,
   tradeoff, or counterexample to a universal claim, return
   CONTRADICTED.

10. If uncertain between SUPPORTED and INSUFFICIENT,
    choose INSUFFICIENT.

11. If uncertain between INSUFFICIENT and CONTRADICTED,
    choose INSUFFICIENT unless the evidence clearly conflicts
    with the claim.

12. Never treat semantic similarity as proof.

--------------------------------------------------
QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
EVIDENCE
--------------------------------------------------

{evidence}

--------------------------------------------------
FINAL LABEL
--------------------------------------------------

Return exactly one of:

SUPPORTED
INSUFFICIENT
CONTRADICTED

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
