import streamlit as st
from PyPDF2 import PdfReader
import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from textblob import TextBlob

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

# ===== SCORING CONFIGURATION =====
SCORING_WEIGHTS = {
    "keywords": 0.4,      # Emphasize domain-specific terms
    "semantic": 0.3,      # Answer relevance to ideal response
    "sentiment": 0.1,     # Positivity (especially for Behavioral/HR)
    "length": 0.2         # Encourage concise but complete answers
}

# ===== SCORING FUNCTIONS =====
def semantic_similarity(answer, ideal_answer):
    """Compare answer to ideal response using cosine similarity (0-1)"""
    emb1 = model.encode(answer, convert_to_tensor=True)
    emb2 = model.encode(ideal_answer, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def extract_keywords(jd, n=10):
    """Extract top keywords from Job Description using TF-IDF"""
    if not jd.strip():  # Handle empty JD
        return []
    tfidf = TfidfVectorizer(stop_words='english', max_features=n)
    tfidf.fit([jd])
    return tfidf.get_feature_names_out().tolist()

def sentiment_analysis(answer):
    """Analyze positivity/negativity (-1 to 1 scale)"""
    analysis = TextBlob(answer)
    return analysis.sentiment.polarity

def unified_scorer(answer, persona, jd="", ideal_answer=""):
    """
    Consolidated scoring with all components.
    Returns: (total_score, score_breakdown)
    """
    # --- 1. Keyword Scoring ---
    static_weights = {
        "Technical": {"algorithm": 2, "python": 3, "architecture": 2},
        "Behavioral": {"example": 2, "situation": 2, "result": 2},
        "HR": {"culture": 3, "growth": 2, "mission": 2},
        "General": {"strength": 2, "experience": 3, "describe": 2}
    }
    jd_keywords = extract_keywords(jd) if jd else []
    combined_weights = {**static_weights[persona], **{kw: 2 for kw in jd_keywords}}
    
    keyword_score = sum(
        weight * answer.lower().count(keyword)
        for keyword, weight in combined_weights.items()
    )

    # --- 2. Semantic Similarity ---
    if ideal_answer:
        semantic_score = semantic_similarity(answer, ideal_answer) * 10
    else:
        semantic_score = 5  # Neutral baseline

    # --- 3. Sentiment Analysis ---
    sentiment_score = max(0, sentiment_analysis(answer) * 2)  # Only penalize negativity (0-2 scale)

    # --- 4. Length Scoring ---
    word_count = len(answer.split())
    length_score = np.exp(-0.5 * ((word_count - 500) / 100) ** 2) * 3  # 0-3 scale

    # --- Combine All Scores ---
    weighted_scores = {
        "keywords": keyword_score * SCORING_WEIGHTS["keywords"],
        "semantic": semantic_score * SCORING_WEIGHTS["semantic"],
        "sentiment": sentiment_score * SCORING_WEIGHTS["sentiment"],
        "length": length_score * SCORING_WEIGHTS["length"]
    }
    
    total_score = min(10, max(1, sum(weighted_scores.values())))
    
    return total_score, weighted_scores

# ===== SESSION STATE INITIALIZATION =====
required_states = {
    "messages": [],
    "interview_started": False,
    "follow_up_count": 0,
    "scores": [],
    "current_question": "",
    "selected_persona": "General",
    "current_ideal_answer": ""
}

for key, default_val in required_states.items():
    st.session_state.setdefault(key, default_val)

# ===== GROQ API CONFIG =====
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # Replace with your actual key
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

# ===== PERSONA CONFIGURATION =====
PERSONAS = {
    "General": {"emoji": "🌐", "prompt": "Ask common interview questions about:"},
    "Technical": {"emoji": "👩💻", "prompt": "Ask technical interview questions about:"},
    "Behavioral": {"emoji": "🗣️", "prompt": "Ask behavioral interview questions about:"},
    "HR": {"emoji": "💼", "prompt": "Ask HR screening questions about:"}
}

def ask_groq(prompt):
    """Get AI response from Groq's API"""
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        if 'response' in locals():
            st.json(response.json())
        return None

# ===== STREAMLIT UI =====
st.title("🤖 Interview Prep Buddy (Groq Powered)")

# Persona Selector
with st.container(border=True):
    cols = st.columns([3, 1])
    with cols[0]:
        new_persona = st.radio(
            "Interview Type:",
            options=list(PERSONAS.keys()),
            index=list(PERSONAS.keys()).index(st.session_state.selected_persona),
            format_func=lambda x: f"{x} {PERSONAS[x]['emoji']}",
            horizontal=True,
            key="persona_selector"
        )
    with cols[1]:
        if st.button("🔄 New Interview"):
            for key in required_states:
                st.session_state[key] = required_states[key]
            st.rerun()

    if new_persona != st.session_state.selected_persona:
        st.session_state.selected_persona = new_persona
        st.rerun()

# Inputs
jd = st.text_area("Job Description:", height=150)
resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Progress display
if st.session_state.scores:
    avg_score = sum(st.session_state.scores) / len(st.session_state.scores)
    st.progress(min(1.0, avg_score / 10))
    st.caption(f"📊 Average Score: {avg_score:.1f}/10")
else:
    st.progress(0.0)
    st.caption("🎯 No answers scored yet")

# Chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Interview Logic ---
if st.button("New Question" if st.session_state.messages else "Start Interview"):
    if not jd:
        st.warning("⚠️ Please enter a Job Description first!")
        st.stop()
    
    resume_text = ""
    if resume:
        try:
            resume_text = PdfReader(resume).pages[0].extract_text()
        except Exception as e:
            st.warning(f"Couldn't read resume: {str(e)}")
        
    prompt = f"""
    Act as a {st.session_state.selected_persona.lower()} interviewer. 
    {PERSONAS[st.session_state.selected_persona]["prompt"]}
    Job: {jd}
    Resume: {resume_text if resume_text else "Not provided"}
    """
    
    question = ask_groq(prompt)
    if question:
        st.session_state.current_ideal_answer = ask_groq(f"Generate a model answer for: {question}")
        st.session_state.messages.append({"role": "assistant", "content": question})
        st.session_state.current_question = question
        st.session_state.follow_up_count = 0
        st.rerun()

# Handle user answers
if user_input := st.chat_input("Your answer:"):
    score, breakdown = unified_scorer(
        answer=user_input,
        persona=st.session_state.selected_persona,
        jd=jd,
        ideal_answer=st.session_state.current_ideal_answer
    )
    
    # Store score and message
    st.session_state.scores.append(score)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show score breakdown
    with st.expander("🔍 Score Breakdown"):
        st.write(f"**Keywords**: {breakdown['keywords']:.1f}")
        st.write(f"**Relevance**: {breakdown['semantic']:.1f}")
        st.write(f"**Sentiment**: {breakdown['sentiment']:.1f}")
        st.write(f"**Length**: {breakdown['length']:.1f}")
        st.progress(score / 10)

        # Add breakdown as a system message
    breakdown_text = (
        f"🔍 **Score Breakdown**\n"
        f"- Keywords: {breakdown['keywords']:.1f}\n"
        f"- Relevance: {breakdown['semantic']:.1f}\n"
        f"- Sentiment: {breakdown['sentiment']:.1f}\n"
        f"- Length: {breakdown['length']:.1f}"
    )
    st.session_state.messages.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": breakdown_text}
    ])
    
    # Generate response based on score
    if score < 8:
        model_answer = ask_groq(f"Suggest improvements for: {user_input}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
            f"📝 Answer score is (Score {score:.1f}/10)\n\n"  # First line with score
            f"**Suggested Answer**\n\n"                  # Bold heading with extra line break
            f"{model_answer}" ) 
    })
        st.session_state.follow_up_count = 0
    elif st.session_state.follow_up_count < 1:
        follow_up = ask_groq(f"Ask 1 short follow-up about: {user_input}")
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        st.session_state.follow_up_count += 1
        st.session_state.current_question = follow_up
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ Great job! (Score {score}/10)"
        })
        st.session_state.follow_up_count = 0
    
    st.rerun()

if GROQ_API_KEY.startswith("gsk_"):
    st.caption("Using Groq's free tier (500 requests/day)")













