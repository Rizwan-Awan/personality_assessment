"""Detect incompatible answers across paired personality items."""

RESPONSE_SCORE = {
    "Strongly Agree": 5,
    "Agree": 4,
    "Slightly Agree": 4,
    "Neutral": 3,
    "Slightly Disagree": 2,
    "Disagree": 2,
    "Strongly Disagree": 1,
}


def detect_contradictions(questions, answers, threshold=3):
    """Return contradictions where paired items imply materially different positions."""
    by_id = {q["id"]: q for q in questions}
    found = []
    seen = set()

    for q in questions:
        q_id = q["id"]
        pid = q.get("pair_id")
        other_id = q.get("paired_question_id")

        if q_id in seen or not pid or not other_id:
            continue
        if q_id not in answers or other_id not in answers:
            continue

        other = by_id.get(other_id)
        if not other:
            continue

        def normalized(question):
            ans = answers.get(question["id"])
            raw = RESPONSE_SCORE.get(ans, 3)
            return 6 - raw if question.get("reverse") else raw

        a = normalized(q)
        b = normalized(other)

        if abs(a - b) >= threshold:
            found.append({
                "pair_id": pid,
                "q1_id": q_id,
                "q1_text": q["text"],
                "q1_answer": answers[q_id],
                "q2_id": other["id"],
                "q2_text": other["text"],
                "q2_answer": answers[other_id],
                "trait": q["trait"],
                "difference": abs(a - b),
            })

        seen.add(q_id)
        seen.add(other_id)

    return found
