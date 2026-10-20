"""Published BEAM answer/judge protocols, reproduced for an apples-to-apples Cortext run.

exabase  -- the protocol past.dev ran (pastdotdev/benchmarks beam/, harness v2):
            ExaBase's published BEAM adapter prompts
            (https://fabric.so/p/beam-3VjBcqEVRofZyA5CeazeX, sha256 584b672d...fc43),
            copied from pastdotdev/benchmarks beam/exabase_prompts.py. These prompt texts are
            ExaBase's work and are NOT covered by this repository's license; they are kept
            here, attributed, for local evaluation only. Note: the answer prompt feeds the
            answerer dataset annotations (ordering topics, rubric items, time points,
            preference/instruction under test, why a question may be unanswerable).
            Scoring: per-rubric-item judge 0/0.5/1 (rounded at .25/.75); event ordering by
            LLM item equivalence + Kendall tau-b mapped to 0..1; question score = mean.
amb      -- the protocol Hindsight ran (vectorize-io/agent-memory-benchmark, single-query):
            mem0's BEAM answer prompt (one prompt for all categories, no annotations) and
            mem0's nugget judge per rubric item; event ordering scored as "should mention"
            nuggets (no Kendall tau).
"""

from __future__ import annotations

import math
import re


# --- annotations (past.dev dataset.annotations) ----------------------------------------------

def annotations(category: str, item: dict) -> dict:
    result: dict = {}
    rubric = item.get("rubric")
    if rubric:
        result["rubric"] = rubric if isinstance(rubric, list) else [str(rubric)]
    if category == "abstention":
        result["why_unanswerable"] = item.get("why_unanswerable", "")
    elif category == "instruction_following":
        result["instruction_being_tested"] = item.get("instruction_being_tested", "")
        result["compliance_indicators"] = item.get("compliance_indicators", [])
    elif category == "preference_following":
        result["preference_being_tested"] = item.get("preference_being_tested", "")
        result["compliance_indicators"] = item.get("compliance_indicators", [])
    elif category == "temporal_reasoning":
        result["time_points"] = item.get("time_points", [])
        result["calculation_required"] = item.get("calculation_required", "")
    elif category == "event_ordering":
        result["ordering_tested"] = item.get("ordering_tested", [])
    return result


# --- ExaBase prompts (verbatim bodies) -------------------------------------------------------

def exabase_answer_prompt(query: str, context: str, cat: str, meta: dict) -> str:
    category_guidance = ""
    if cat == "abstention":
        why = meta.get("why_unanswerable", "")
        category_guidance = (
            "IMPORTANT — ABSTENTION TASK: This question may be about something NOT "
            "mentioned in the conversation. If the context does not explicitly contain "
            "the requested information, you MUST respond with: "
            "\"Based on the provided chat, there is no information related to [topic].\" "
            "Do NOT fabricate, infer, or guess. Only answer if the information is "
            "directly present in the retrieved context."
            + (f"\n\nHint about why this may be unanswerable: {why}" if why else "")
        )
    elif cat == "event_ordering":
        topics = meta.get("ordering_tested", [])
        topics_str = "\n".join(f"  - {t}" for t in topics) if topics else ""
        category_guidance = (
            "IMPORTANT — EVENT ORDERING TASK: The following specific topics need to be "
            "listed in the order they were FIRST mentioned in the conversation.\n"
            + (f"\nTopics to order:\n{topics_str}\n" if topics_str else "")
            + "\nFor each topic, find the FIRST time it was mentioned (use Turn IDs and "
            "timestamps as ordering cues). List ONLY these topics in chronological order. "
            "Number each item. Do NOT add extra topics not listed above."
        )
    elif cat == "contradiction_resolution":
        category_guidance = (
            "IMPORTANT — CONTRADICTION TASK: The conversation contains contradictory "
            "statements. Identify both contradictory statements and explicitly note the "
            "contradiction. Do not give a definitive answer — instead say what was "
            "claimed at different points."
        )
    elif cat == "knowledge_update":
        category_guidance = (
            "IMPORTANT — KNOWLEDGE UPDATE TASK: Information was updated during the "
            "conversation. Report ONLY the most recent value. If you see older "
            "conflicting info, note the update."
        )
    elif cat == "temporal_reasoning":
        time_pts = meta.get("time_points", [])
        calc = meta.get("calculation_required", "")
        guidance = (
            "IMPORTANT — TEMPORAL REASONING TASK: Calculate the exact time duration. "
            "Show your arithmetic step by step."
        )
        if time_pts:
            guidance += f"\n\nKey time points from context: {'; '.join(time_pts)}"
        if calc:
            guidance += f"\n\nCalculation hint: {calc}"
        category_guidance = guidance
    elif cat == "preference_following":
        pref = meta.get("preference_being_tested", "")
        category_guidance = (
            "IMPORTANT — PREFERENCE FOLLOWING TASK: The user has stated a specific "
            "preference earlier in the conversation. Your answer MUST respect and comply "
            "with that stated preference."
            + (f"\n\nUser's preference (from conversation): {pref}" if pref else "")
        )
    elif cat == "instruction_following":
        instr = meta.get("instruction_being_tested", "")
        indicators = meta.get("compliance_indicators", [])
        guidance = (
            "IMPORTANT — INSTRUCTION FOLLOWING TASK: The user gave a specific formatting "
            "or style instruction earlier in the conversation. Follow it exactly."
        )
        if instr:
            guidance += f"\n\nInstruction to follow: {instr}"
        if indicators:
            guidance += f"\n\nCompliance indicators: {', '.join(indicators)}"
        category_guidance = guidance
    elif cat == "summarization":
        rubric_items = meta.get("rubric", [])
        hints = []
        for r in rubric_items:
            hint = r.split(": ", 1)[-1] if ": " in r else r
            hints.append(hint)
        hints_str = "\n".join(f"  - {h}" for h in hints) if hints else ""
        category_guidance = (
            "IMPORTANT — SUMMARIZATION TASK: Provide a comprehensive chronological "
            "summary. Be specific about dates, versions, and key technical decisions.\n"
            + (f"\nMake sure your summary covers these key aspects:\n{hints_str}" if hints_str else "")
        )
    elif cat == "multi_session_reasoning":
        category_guidance = (
            "IMPORTANT — MULTI-SESSION REASONING TASK: The answer requires "
            "combining facts from multiple parts of the conversation.\n\n"
            "Steps:\n"
            "1. Before giving any count or total, FIRST list each distinct item "
            "you found, with its Turn number or source reference.\n"
            "2. Review your list — merge items that are the same thing described "
            "differently. Do NOT split a single item into sub-items or count "
            "variations of the same thing separately.\n"
            "3. Only then compute your count or total from the deduplicated list.\n"
            "4. If the question asks for a total amount, show the arithmetic "
            "(e.g. $X + $Y + $Z = $Total).\n\n"
            "Count only items the user explicitly mentioned — do not infer or "
            "generalize beyond what is stated."
        )
    elif cat == "information_extraction":
        category_guidance = (
            "IMPORTANT — INFORMATION EXTRACTION TASK: Answer with the specific "
            "fact, value, or detail as it was originally stated at the point in "
            "the conversation the question refers to. Use Turn numbers to locate "
            "the relevant statement.\n\n"
            "If a value was later updated or changed, report what was stated at "
            "the time the question asks about (typically the FIRST mention), NOT "
            "the updated value. If the question says 'did I say' or 'did I "
            "mention', report the original statement."
        )

    return f"""You are a helpful assistant answering questions based on a long conversation history.
Answer the question using ONLY information found in the retrieved context below.

{category_guidance}

Question: {query}

Retrieved Context:
{context}

Answer:"""


def exabase_rubric_prompt(query: str, answer: str, rubric_item: str) -> str:
    prompt = f"""You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## QUESTION:
{query}

## LLM RESPONSE:
{answer}

## RUBRIC CRITERION:
{rubric_item}

## SCORING:
- **1.0** = Fully satisfied: The response clearly and completely addresses this rubric criterion.
- **0.5** = Partially satisfied: The response addresses this criterion but is incomplete, vague, or only partially correct.
- **0.0** = Not satisfied: The response does not address this criterion at all, or is incorrect.

Evaluate the response against ONLY this specific rubric criterion. Provide your score and a brief reason."""
    return prompt


def equivalence_prompt(ref_item: str, sys_item: str) -> str:
    prompt = f"""Do these two items refer to the same event, topic, or concept? Answer only YES or NO.

Item A: {ref_item}
Item B: {sys_item}

Answer (YES or NO):"""
    return prompt


RUBRIC_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": {"score": {"type": "number", "description": "Score: 0.0, 0.5, or 1.0"},
                                "reason": {"type": "string"}},
                 "required": ["score", "reason"]}
EQUIVALENCE_SCHEMA = {"type": "object", "additionalProperties": False,
                      "properties": {"answer": {"type": "string", "description": "YES or NO"}},
                      "required": ["answer"]}


# --- AMB / mem0 prompts (verbatim bodies) ----------------------------------------------------

def amb_answer_prompt(query: str, context: str) -> str:
    return (
        "You are an AI assistant with access to stored memories from prior conversations with a user.\n"
        "Use these memories to answer the following question as accurately and completely as possible.\n\n"
        "IMPORTANT RULES:\n"
        "1. Scan ALL provided memories before answering — do not stop after the first relevant one.\n"
        "2. If multiple memories contain relevant information, combine and cross-reference them.\n"
        "3. The question may refer to something indirectly (e.g. 'the person I met at the festival') — resolve such references by searching the memories before concluding. Only if the specific information is genuinely absent after a careful search, say exactly: \"I don't have enough information to answer this question.\"\n"
        "4. Be specific and direct — include exact names, dates, numbers, and details from the memories.\n"
        "5. Do NOT invent or assume information that isn't in the memories.\n"
        f"\nQUESTION: {query}\n\n"
        f"RETRIEVED MEMORIES:\n{context}\n\n"
        "ANSWER:"
    )


def amb_rubric_prompt(query: str, answer: str, rubric_item: str) -> str:
    return f"""Evaluate whether the following LLM response demonstrates compliance with the specified RUBRIC CRITERION.

QUESTION:
{query}

LLM RESPONSE:
{answer}

RUBRIC CRITERION:
{rubric_item}

SCORING GUIDELINES:

First, determine whether the rubric criterion is a POSITIVE requirement (the response SHOULD include something) or a NEGATIVE constraint (the response SHOULD NOT include something).

**For POSITIVE requirements** (response should contain, mention, or demonstrate something):
- **1.0 (Complete Compliance)**: The required element is present, accurate, and complete.
- **0.5 (Partial Compliance)**: The required element is partially present, has minor inaccuracies, or is incomplete.
- **0.0 (No Compliance)**: The required element is missing, incorrect, or the response is entirely off-topic / non-responsive.

**For NEGATIVE constraints** (response should NOT contain or should avoid something):
- **1.0**: The response is responsive AND the prohibited element is absent.
- **0.5**: The response is responsive but contains a borderline or ambiguous reference to the prohibited element.
- **0.0**: The prohibited element is present, OR the response is non-responsive.

**Compound statements** ("and"/commas joining multiple required elements): all present=1.0, some=0.5, none=0.0.

EVALUATION RULES:
1. Semantic tolerance: paraphrases and synonyms are acceptable.
2. Numeric/date equivalence: "$68,000"="68k"="sixty-eight thousand dollars"; "2 years"="24 months".
3. Case/punctuation/whitespace tolerance.
4. Hedging tolerance: do not penalize "I think"/"probably"/passive/verbosity if content satisfies the criterion.
5. Style neutrality: do not penalize tone/format/length unless the criterion requires it.
6. Responsiveness: completely off-topic or refusal = 0.0.
7. Independence: evaluate this criterion in isolation.
8. Specificity: vague generic answers score lower than specific, detailed ones.

Return JSON: {{"score": <0.0 or 0.5 or 1.0>, "reason": "<one concise sentence>"}}"""


# --- scoring helpers -------------------------------------------------------------------------

def round_rubric_score(raw: float) -> float:
    return 1.0 if raw >= 0.75 else 0.5 if raw >= 0.25 else 0.0


def list_items(text: str) -> list[str]:
    items = []
    for line in text.strip().splitlines():
        item = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", line).strip()
        if item:
            items.append(item)
    return items


def kendall_tau_b(x: list[int], y: list[int]) -> float:
    """scipy.stats.kendalltau(variant='b') for small lists (no scipy dependency)."""
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float("nan") if denom == 0 else (conc - disc) / denom


def ordering_score(reference: list[str], answer_order: list[str]) -> float:
    topics = list(dict.fromkeys(reference + answer_order))
    missing = len(topics) + 1

    def ranks(order):
        position = {t: i + 1 for i, t in enumerate(order)}
        return [position.get(t, missing) for t in topics]

    tau = kendall_tau_b(ranks(reference), ranks(answer_order))
    return 0.0 if math.isnan(tau) else (tau + 1) / 2
