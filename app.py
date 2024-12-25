import streamlit as st
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease, syllable_count
from nltk.tokenize import word_tokenize
import nltk
import json

from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Functions
def preprocess_text(text):
    """Preprocess text by removing extra spaces."""
    return text.strip()

def compute_relevance(reference_answer, student_answer):
    """Calculate semantic relevance between the reference and student's answers."""
    ref_embedding = model.encode(reference_answer, convert_to_tensor=True)
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(ref_embedding, student_embedding).item()
    return similarity_score

def compute_detailedness(student_answer, max_marks):
    """Assess detailedness of the student's answer based on word count."""
    word_count = len(word_tokenize(student_answer))
    expected_words = max_marks * 15
    word_score = min(word_count / expected_words, 1)
    return word_score

def compute_language_quality(student_answer):
    """Assess language quality based on readability and syllable count."""
    readability_score = flesch_reading_ease(student_answer)
    syllables = syllable_count(student_answer)
    readability_score = max(min(readability_score / 100, 1), 0)  # Scale between 0 and 1
    syllable_score = min(syllables / 50, 1)  # Assume 50 syllables for a balanced answer
    return (readability_score + syllable_score) / 2

def calculate_final_score(relevance, detailedness, language_quality, max_marks):
    """Combine all feature scores into a final score."""
    weighted_score = (0.5 * relevance) + (0.3 * detailedness) + (0.2 * language_quality)
    final_score = round(weighted_score * max_marks, 2)
    return final_score

def evaluate_answer(reference_answer, student_answer, max_marks):
    """Evaluate a single answer."""
    reference_answer = preprocess_text(reference_answer)
    student_answer = preprocess_text(student_answer)

    relevance = compute_relevance(reference_answer, student_answer)
    detailedness = compute_detailedness(student_answer, max_marks)
    language_quality = compute_language_quality(student_answer)
    
    # If relevance is very low, set score to 0
    if relevance < 0.1:
        return {"relevance": 0, "detailedness": 0, "language_quality": 0, "final_score": 0}

    final_score = calculate_final_score(relevance, detailedness, language_quality, max_marks)
    return {
        "relevance": relevance,
        "detailedness": detailedness,
        "language_quality": language_quality,
        "final_score": final_score
    }

# Streamlit App
st.title("Automated Exam Evaluator")

# Section 1: Input Questions
st.sidebar.header("Input Questions")
num_questions = st.sidebar.number_input("Number of Questions", min_value=1, value=3)
questions = []
for i in range(num_questions):
    question = st.sidebar.text_area(f"Question {i+1}", key=f"q_{i}")
    reference_answer = st.sidebar.text_area(f"Reference Answer {i+1}", key=f"ref_{i}")
    marks = st.sidebar.number_input(f"Marks for Question {i+1}", min_value=1, value=2, key=f"marks_{i}")
    if question and reference_answer:
        questions.append({"question": question, "reference": reference_answer, "marks": marks})

# Section 2: Input Student Answers
st.sidebar.header("Input Student Answers")
num_students = st.sidebar.number_input("Number of Students", min_value=1, value=2)
students = {}
for i in range(num_students):
    student_name = st.sidebar.text_input(f"Student {i+1} Name", key=f"student_{i}")
    if student_name:
        student_answers = []
        for j, q in enumerate(questions):
            answer = st.sidebar.text_area(f"{student_name} Answer for Q{j+1}", key=f"{student_name}_q{j}")
            student_answers.append({"answer": answer})
        students[student_name] = student_answers

# Evaluate and Display Results
if st.button("Evaluate"):
    if not questions or not students:
        st.error("Please input questions and student answers.")
    else:
        st.subheader("Evaluation Results")
        results = {}
        for student, answers in students.items():
            student_scores = []
            total_score = 0
            for i, question in enumerate(questions):
                reference_answer = question["reference"]
                max_marks = question["marks"]
                student_answer = answers[i]["answer"]
                
                evaluation = evaluate_answer(reference_answer, student_answer, max_marks)
                total_score += evaluation["final_score"]
                
                student_scores.append({
                    "question": question["question"],
                    "max_marks": max_marks,
                    "final_score": evaluation["final_score"],
                    "relevance": evaluation["relevance"],
                    "detailedness": evaluation["detailedness"],
                    "language_quality": evaluation["language_quality"],
                })
            
            results[student] = {"scores": student_scores, "total_score": total_score}

        # Display results
        for student, data in results.items():
            st.write(f"### {student} - Total Score: {data['total_score']}")
            for score in data["scores"]:
                st.write(f"- **Question**: {score['question']}")
                st.write(f"  - Max Marks: {score['max_marks']}")
                st.write(f"  - Final Score: {score['final_score']}")
                st.write(f"  - Relevance: {score['relevance']:.2f}")
                st.write(f"  - Detailedness: {score['detailedness']:.2f}")
                st.write(f"  - Language Quality: {score['language_quality']:.2f}")

       
