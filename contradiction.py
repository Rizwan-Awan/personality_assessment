"""Detect incompatible answers across paired personality items."""

RESPONSE_SCORE = {
    "Strongly Agree": 5,
    "Agree": 4,
    "Neutral": 3,
    "Disagree": 2,
    "Strongly Disagree": 1,
}


def detect_contradictions(questions, answers, threshold=3):
    """Return contradictions where paired items imply materially different positions.

    Reverse-scored questions are transformed so a high normalized value consistently
    means the trait is endorsed. Two linked questions are contradictory when their
    normalized scores differ by at least `threshold`.
    """
    by_id = {q["id"]: q for q in questions}
    found = []
    seen = set()
    for q in questions:
        pid = q.get("pair_id")
        other_id = q.get("paired_question_id")
        if not pid or not other_id or q["id"] in seen or other_id not in answers:
            continue
        if q["id"] not in answers:
            continue
        other = by_id.get(other_id)
        if not other:
            continue

        def normalized(question):
            raw = RESPONSE_SCORE[answers[question["id"]]]
            return 6 - raw if question.get("reverse") else raw

        a = normalized(q)
        b = normalized(other)
        if abs(a - b) >= threshold:
            found.append({
                "pair_id": pid,
                "q1_id": q["id"],
                "q1_text": q["text"],
                "q1_answer": answers[q["id"]],
                "q2_id": other["id"],
                "q2_text": other["text"],
                "q2_answer": answers[other["id"]],
                "trait": q["trait"],
                "difference": abs(a - b),
            })
        seen.add(q["id"])
        seen.add(other_id)
    return found
