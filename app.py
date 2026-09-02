import random
import time
from collections import defaultdict

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from assessor import combine_scores, deterministic_scores, ml_scores, overall_summary
from contradiction import detect_contradictions
from questions import OPTIONS, QUESTIONS, TRAITS

st.set_page_config(page_title="ISSB Personality Practice Simulator", page_icon="🧠", layout="wide")

# -----------------------------
# Attempt + timer helpers
# -----------------------------
DEFAULT_MINUTES = 20
QUESTION_COUNT = 200


def new_attempt_questions(attempt_number, previous_pair_ids=None):
    """Select 10 complete question-pairs per trait, giving 200 questions.

    A later attempt uses a new random seed and avoids the previous pair set when
    possible, so the candidate gets genuinely different MCQs instead of only a new order.
    """
    pairs = defaultdict(list)
    for q in QUESTIONS:
        if q.get("pair_id"):
            pairs[q["pair_id"]].append(q)

    by_trait = defaultdict(list)
    for pid, items in pairs.items():
        if len(items) == 2:
            by_trait[items[0]["trait"]].append((pid, items))

    previous_pair_ids = set(previous_pair_ids or [])
    rng = random.Random(100000 + attempt_number * 7919)
    selected = []

    for trait in TRAITS:
        candidates = by_trait[trait][:]
        rng.shuffle(candidates)
        preferred = [x for x in candidates if x[0] not in previous_pair_ids]
        chosen = preferred[:10]
        if len(chosen) < 10:
            chosen += candidates[:10 - len(chosen)]
        selected.extend(chosen)

    rng.shuffle(selected)
    result = [q for _, pair_items in selected for q in pair_items]
    rng.shuffle(result)
    return result


def start_new_attempt():
    attempt = st.session_state.get("attempt", 0) + 1
    previous_ids = [q.get("pair_id") for q in st.session_state.get("questions", [])]
    st.session_state.attempt = attempt
    st.session_state.questions = new_attempt_questions(attempt, previous_ids)
    st.session_state.answers = {}
    st.session_state.current = 0
    st.session_state.started_at = time.time()
    st.session_state.deadline = st.session_state.started_at + st.session_state.duration_minutes * 60
    st.session_state.timed_out = False


if "duration_minutes" not in st.session_state:
    st.session_state.duration_minutes = DEFAULT_MINUTES
if "attempt" not in st.session_state:
    st.session_state.attempt = 0
if "questions" not in st.session_state or len(st.session_state.questions) != QUESTION_COUNT:
    start_new_attempt()
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "current" not in st.session_state:
    st.session_state.current = 0
if "deadline" not in st.session_state:
    st.session_state.deadline = time.time() + st.session_state.duration_minutes * 60
if "timed_out" not in st.session_state:
    st.session_state.timed_out = False

questions = st.session_state.questions
answers = st.session_state.answers

# Refresh once per second while test is active to keep the timer live and to auto-submit.
if st.session_state.current < len(questions):
    st_autorefresh(interval=1000, key=f"timer_{st.session_state.attempt}")
    remaining = max(0, int(st.session_state.deadline - time.time()))
    if remaining <= 0:
        st.session_state.timed_out = True
        st.session_state.current = len(questions)
        st.rerun()


def timer_text(seconds):
    minutes, secs = divmod(max(0, seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


st.title("🧠 ISSB-Style Personality Practice Simulator")
st.caption(
    "200-question timed practice • different question set on later attempts • trait analysis • contradiction detection"
)
st.warning(
    "Practice simulator only — not an official ISSB test and not a validated psychological diagnosis. "
    "ISSB states that its psychologist dimension uses carefully designed psychological tests and that candidates should be honest and straightforward."
)

with st.sidebar:
    st.header("Test Controls")
    st.write(f"Attempt: **{st.session_state.attempt}**")
    st.write(f"Question set: **{len(questions)} MCQs**")

    if st.session_state.current < len(questions):
        remaining = max(0, int(st.session_state.deadline - time.time()))
        st.metric("Time remaining", timer_text(remaining))
    else:
        st.metric("Time limit", f"{st.session_state.duration_minutes} min")

    answered = len(answers)
    st.progress(min(1.0, answered / len(questions)))
    st.write(f"Answered: **{answered}/{len(questions)}**")

    if st.session_state.current == 0 and not answers and st.session_state.attempt == 1:
        st.session_state.duration_minutes = st.select_slider(
            "Practice time limit",
            options=[10, 15, 20, 25, 30, 40],
            value=st.session_state.duration_minutes,
            format_func=lambda x: f"{x} minutes",
        )
        if st.button("Apply time limit"):
            st.session_state.started_at = time.time()
            st.session_state.deadline = st.session_state.started_at + st.session_state.duration_minutes * 60
            st.rerun()

    st.divider()
    if st.button("Start New Attempt", type="secondary"):
        start_new_attempt()
        st.rerun()

# -----------------------------
# Test UI
# -----------------------------
if st.session_state.current < len(questions):
    q = questions[st.session_state.current]
    number = st.session_state.current + 1
    remaining = max(0, int(st.session_state.deadline - time.time()))

    st.subheader(f"Question {number} of {len(questions)}")
    st.progress(number / len(questions))
    if remaining <= 120:
        st.error(f"⏱️ Time remaining: **{timer_text(remaining)}**")
    else:
        st.info(f"⏱️ Time remaining: **{timer_text(remaining)}**")

    st.write(f"**{q['text']}**")

    current_answer = answers.get(q["id"])
    choice = st.radio(
        "Choose one:",
        OPTIONS,
        index=OPTIONS.index(current_answer) if current_answer in OPTIONS else None,
        key=f"answer_{st.session_state.attempt}_{q['id']}",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Previous", disabled=number == 1):
            if choice is not None:
                answers[q["id"]] = choice
            st.session_state.current -= 1
            st.rerun()
    with c2:
        if st.button("Save & Next", type="primary", disabled=choice is None):
            answers[q["id"]] = choice
            st.session_state.current += 1
            st.rerun()
    with c3:
        if st.button("Finish Now"):
            if choice is not None:
                answers[q["id"]] = choice
            st.session_state.current = len(questions)
            st.rerun()
else:
    if st.session_state.timed_out:
        st.error("⏰ Time is up. Your current answers have been submitted for analysis.")
    else:
        st.success("Assessment completed.")

    rule = deterministic_scores(questions, answers)
    model = ml_scores(questions, answers)
    final = combine_scores(rule, model)
    contradictions = detect_contradictions(questions, answers)
    strengths, growth = overall_summary(final)

    st.header("Assessment Report")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall practice score", f"{sum(final.values()) / len(final):.1f}%")
    with col2:
        st.metric("Answered", f"{len(answers)}/{len(questions)}")
    with col3:
        st.metric("Contradicting pairs", len(contradictions))

    st.subheader("Personality dimensions")
    df = pd.DataFrame({"Dimension": [TRAITS[k] for k in final], "Score": list(final.values())})
    st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Dimension"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Relative strengths")
        for trait, score in strengths:
            st.write(f"**{TRAITS[trait]} — {score}%**")
    with c2:
        st.subheader("Areas to reflect on")
        for trait, score in reversed(growth):
            st.write(f"**{TRAITS[trait]} — {score}%**")

    st.subheader("Contradiction analysis")
    if not contradictions:
        st.success("No strong contradictions were detected among the answered linked questions.")
    else:
        st.error(f"Contradicting answers detected: {len(contradictions)} pair(s)")
        for c in contradictions:
            with st.expander(f"{c['pair_id']} • {TRAITS[c['trait']]} • difference {c['difference']}"):
                st.markdown(f"**{c['q1_id']}:** {c['q1_text']}  ")
                st.write(f"Answer: **{c['q1_answer']}**")
                st.markdown(f"**{c['q2_id']}:** {c['q2_text']}  ")
                st.write(f"Answer: **{c['q2_answer']}**")
                st.caption("These answers point in materially different directions for the same underlying trait.")

    st.subheader("AI / ML assessment")
    st.write(
        "The report combines a transparent questionnaire rubric with a Random Forest model. "
        "The model is trained on synthetic response patterns for this demo, so it should not be treated as a validated psychological assessment."
    )
    ml_df = pd.DataFrame({
        "Dimension": [TRAITS[k] for k in final],
        "Rubric": [rule[k] for k in final],
        "ML estimate": [model[k] for k in final],
        "Final": [final[k] for k in final],
    })
    st.dataframe(ml_df.sort_values("Final", ascending=False), use_container_width=True, hide_index=True)

    result_df = pd.DataFrame([
        {
            "Attempt": st.session_state.attempt,
            "Question": q["id"],
            "Trait": TRAITS[q["trait"]],
            "Question Text": q["text"],
            "Answer": answers.get(q["id"], "Not answered"),
            "Pair ID": q.get("pair_id", ""),
        }
        for q in questions
    ])
    st.download_button(
        "Download Answers as CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name=f"assessment_attempt_{st.session_state.attempt}.csv",
        mime="text/csv",
    )

    st.divider()
    st.info(
        "A new attempt creates a fresh 200-question set from the larger question bank and changes the order. "
        "This helps reduce memorization of the previous attempt."
    )
    if st.button("Take Another Attempt", type="primary"):
        start_new_attempt()
        st.rerun()
