import streamlit as st
from PyPDF2 import PdfReader
import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from textblob import TextBlob
from collections import Counter
import nltk
import uuid
nltk.download('stopwords')  # First-time download
from nltk.corpus import stopwords
import string


# ===== NEW SESSION MANAGEMENT =====
def init_session():
    """Initialize/reset all session state variables"""
    st.session_state.update({
        "messages": [],
        "scores": [],
        "current_question": "",
        "selected_persona": "General",
        "current_ideal_answer": "",
        "jd_text": "",
        "asked_questions": [],  # Track asked questions
        "question_embeddings": np.array([]),  # For semantic checks
        "follow_up_count": 0,
        "combined_weights": {},
        "session_id": str(uuid.uuid4()),
        "debug_info": []  # Track errors and warnings
    })

# Initialize if needed
if 'session_id' not in st.session_state:
    init_session()    

if 'asked_questions' not in st.session_state:  # INITIALIZE IF MISSING
    init_session()

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

# ===== SCORING CONFIGURATION =====
SCORING_WEIGHTS = {
    "keywords": 0.3,      # Emphasize domain-specific terms
    "semantic": 0.5,      # Answer relevance to ideal response
    "sentiment": 0.1,     # Positivity (especially for Behavioral/HR)
    "length": 0.1         # Encourage concise but complete answers
}

# ===== NEW ANTI-REPETITION FUNCTIONS =====
def is_duplicate_question(new_question, threshold=0.75):
    """Check semantic similarity with previous questions"""
    if not st.session_state.asked_questions:
        return False
    
    new_embed = model.encode([new_question])
    similarities = util.cos_sim(new_embed, st.session_state.question_embeddings)
    return np.max(similarities.numpy()) > threshold

# ===== MODIFIED FEEDBACK FORMATTING =====
def format_feedback_content(score, breakdown_content, feedback, guidelines, example):
    """Format feedback with auto-generation fallbacks"""
    # Generate fallback content if missing
    if not feedback.strip():
        feedback = ask_groq(f"Provide brief constructive feedback for this answer: '{user_input}' to question: '{st.session_state.current_question}'") or "Could not generate feedback"
    
    if not example.strip():
        example = ask_groq(f"Create a concise example answer for: '{st.session_state.current_question}'") or "Could not generate example"
    
    # Format guidelines nicely
    formatted_guidelines = guidelines
    if guidelines and not guidelines.startswith("-"):
        formatted_guidelines = "\n- " + "\n- ".join(guidelines.split("\n"))
    
    return f"""
📊 Score: {score:.1f}/10

{breakdown_content}

➤ Constructive Feedback  
{feedback or 'Could not generate feedback'}



➤ Example Response  
*"{example}"*
"""

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

def extract_top_keywords(answer, n=3):
    """Extract most frequent meaningful keywords from answer"""
    words = [word.lower() for word in answer.split() 
             if word.lower() not in stopwords.words('english') 
             and word not in string.punctuation]
    return ", ".join([word for word, _ in Counter(words).most_common(n)])

def format_score_breakdown(score, breakdown, answer):
    """Format score breakdown in specified structure"""
    word_count = len(answer.split())

     # Get both extracted and relevant keywords
    extracted_keywords = extract_top_keywords(answer)
    persona_jd_keywords = list(st.session_state.combined_weights.keys())  # Add this to unified_scorer
    
    # Find actual matches that contributed to scoring
    relevant_keywords = [
        kw for kw in extracted_keywords.split(", ")
        if kw in persona_jd_keywords
    ]
    
    keyword_display = (
        ", ".join(relevant_keywords) 
        if relevant_keywords 
        else "No exact matches found"
    )

    return f"""
📊 Score: {score:.1f}/10
- Keywords: {breakdown['keywords']:.1f} ({keyword_display})
- Relevance: {breakdown['semantic']:.1f} ({int((breakdown['semantic']/3)*100)}% semantic match)
- Sentiment: {breakdown['sentiment']:.1f} ({"Positive" if breakdown['sentiment'] > 0 else "Neutral" if breakdown['sentiment'] == 0 else "Negative"} tone)
- Length: {breakdown['length']:.1f} ({word_count} words)
"""    

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
        "General": {"strength": 2, "experience": 3, "mission": 3,"describe": 2}
    }
    jd_keywords = extract_keywords(jd) if jd else []
    combined_weights = {**static_weights[persona], **{kw: 2 for kw in jd_keywords}}

    # NEW LINE HERE 👇
    st.session_state.combined_weights = combined_weights  # Track scoring keywords
    
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
    length_score = np.exp(-0.5 * ((word_count - 150) / 100) ** 2) * 3  # 0-3 scale

    # --- Combine All Scores ---
    weighted_scores = {
        "keywords": keyword_score * SCORING_WEIGHTS["keywords"],
        "semantic": semantic_score * SCORING_WEIGHTS["semantic"],
        "sentiment": sentiment_score * SCORING_WEIGHTS["sentiment"],
        "length": length_score * SCORING_WEIGHTS["length"]
    }
    
    total_score = min(10, max(1, sum(weighted_scores.values())))
    
    return total_score, weighted_scores

# ===== ROBUST PARSING FUNCTION =====
def parse_feedback(raw: str) -> tuple:
    """Enhanced parser with multiple fallback strategies"""
    try:
        # Strategy 1: Exact header matching
        feedback_match = re.search(r"Feedback:(.*?)(?=Guidelines:|$)", raw, re.DOTALL | re.IGNORECASE)
        guidelines_match = re.search(r"Guidelines:(.*?)(?=Example Answer:|$)", raw, re.DOTALL | re.IGNORECASE)
        example_match = re.search(r"Example Answer:(.*?)(?=== END FORMAT ===|$)", raw, re.DOTALL | re.IGNORECASE)
        
        # Strategy 2: Fallback to flexible matching
        if not feedback_match:
            feedback_match = re.search(r"(Feedback|Analysis):(.*?)(?=Guidelines|Recommendations|$)", raw, re.DOTALL | re.IGNORECASE)
        if not guidelines_match:
            guidelines_match = re.search(r"Guidelines:(.*?)(?=Example|Model|$)", raw, re.DOTALL | re.IGNORECASE)
        if not example_match:
            example_match = re.search(r"(Example Answer|Model Response):(.*)", raw, re.DOTALL | re.IGNORECASE)
        
        # Extract content
        feedback = feedback_match.group(1).strip() if feedback_match else ""
        guidelines = guidelines_match.group(1).strip() if guidelines_match else ""
        example = example_match.group(1).strip() if example_match else ""
        
        # Try to extract bullet points if guidelines are messy
        if guidelines and "-" not in guidelines:
            bullet_points = re.findall(r"\d+\.\s+(.*)|-\s+(.*)", guidelines)
            if bullet_points:
                guidelines = "\n- " + "\n- ".join([bp[0] or bp[1] for bp in bullet_points])
        
        return feedback, guidelines, example
        
    except Exception as e:
        st.session_state.debug_info.append(f"Parse error: {str(e)}")
        st.session_state.debug_info.append(f"Raw response: {raw[:500]}")
        return "", "", ""

# ===== NEW IDEAL ANSWER GENERATION FUNCTION =====
def generate_ideal_answer(question):
    """Robust model answer generation with fallback"""
    prompt = f"""
    As an expert in interview coaching, generate a comprehensive model answer for this question:
    "{question}"
    
    Structure your response:
    1. Key principles to demonstrate
    2. Ideal structure for the answer
    3. Concrete examples to include
    4. Common mistakes to avoid
    
    Return only the model answer text without any additional labels.
    """
    answer = ask_groq(prompt)
    return answer if answer else "Could not generate model answer"


# ===== MODIFIED GROQ API CONFIG =====
def ask_groq(prompt):
    """Get AI response with error handling"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"},
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.85,  # Increased randomness
                "max_tokens": 500,
                "frequency_penalty": 0.5  # Anti-repetition
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


# ===== PERSONA CONFIGURATION =====
PERSONAS = {
    "General": {"emoji": "🌐", "prompt": "Ask common interview questions about:"},
    "Technical": {"emoji": "👩💻", "prompt": "Ask technical interview questions about:"},
    "Behavioral": {"emoji": "🗣️", "prompt": "Ask behavioral interview questions about:"},
    "HR": {"emoji": "💼", "prompt": "Ask HR screening questions about:"}
}


# ===== STREAMLIT UI =====
st.title("🤖 Interview Prep Buddy")

# Persona Selector
with st.container(border=True):
    cols = st.columns([3, 1])
    with cols[0]:
        # Get current persona index safely
        current_index = list(PERSONAS.keys()).index(
            st.session_state.get("selected_persona", "General")
        )

        new_persona = st.radio(
            "Interview Type:",
            options=list(PERSONAS.keys()),
            index=current_index,
            format_func=lambda x: f"{x} {PERSONAS[x]['emoji']}",
            horizontal=True,
            key="persona_selector_ui"  # Unique key for this widget
        )
    with cols[1]:
        if st.button("🔄 New Interview"):
            init_session()
            st.rerun()

    if new_persona != st.session_state.get("selected_persona", "General"):
        st.session_state.selected_persona = new_persona
        st.rerun()

# Inputs
jd = st.text_area("Job Description:", height=150, key="jd_text_input",
                  value=st.session_state.get("jd_text", ""))
# Update session state as user types
if jd != st.session_state.get("jd_text", ""):
    st.session_state.jd_text = jd

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

# ===== MODIFIED INTERVIEW LOGIC =====
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
        
    # Modified question generation with anti-repetition
    prompt = f"""
    Act as a {st.session_state.selected_persona.lower()} interviewer. 
    {PERSONAS[st.session_state.selected_persona]["prompt"]}
    Job: {jd}
    Resume: {resume_text if resume_text else "Not provided"}
    
    Previous questions: {st.session_state.asked_questions[-3:] if st.session_state.asked_questions else "None"}
    Generate a unique question that hasn't been asked yet. Do NOT mention that it's a new/unasked question.
    """
    
    for _ in range(3):  # Retry up to 3 times
        question = ask_groq(prompt)
        if question and not is_duplicate_question(question):
            # Store question and embedding
            st.session_state.asked_questions.append(question)
            new_embed = model.encode([question])
            if st.session_state.question_embeddings.size == 0:
                st.session_state.question_embeddings = new_embed
            else:
                st.session_state.question_embeddings = np.vstack(
                    [st.session_state.question_embeddings, new_embed]
                )
            
            break
    
    if question:
        st.session_state.current_ideal_answer = generate_ideal_answer(question)
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
    #st.session_state.scores.append(score)

    # Format score breakdown
    breakdown_content = format_score_breakdown(score, breakdown, user_input)
    
    
    # Generate structured feedback
    feedback_prompt = f"""
    **INSTRUCTIONS**
    You are an expert interview coach. Analyze this candidate response and:

    1. Provide constructive feedback for their answer
    2. Provide 3 concise guidelines for an ideal answer
    3. Generate a model example answer

    **QUESTION:** {st.session_state.current_question}
    **CANDIDATE ANSWER:** {user_input}
    **Ideal Guidelines:** {st.session_state.current_ideal_answer}
   

    **REQUIRED FORMAT:**
    === BEGIN FORMAT ===
    Feedback: [Your constructive feedback here]
    Guidelines: 
    - [Guideline 1]
    - [Guideline 2]
    - [Guideline 3]
    Example Answer: [Your example answer here]
    === END FORMAT ===

    Do not add any additional text outside these sections. Use clear section headers exactly as shown.
    """
    
    raw_feedback = ask_groq(feedback_prompt)
    feedback, guidelines, example = parse_feedback(raw_feedback) 
    
   

    # Combine both formats
    feedback_content = format_feedback_content(
        score=score,
        breakdown_content=breakdown_content,  # Added parameter
        feedback=feedback,
        guidelines=guidelines,
        example=example     
    )
    
    # Store and display
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": feedback_content})
    
    # Generate follow-up
    if score < 8:
        st.session_state.follow_up_count = 0
    elif st.session_state.follow_up_count < 1:
        follow_up = ask_groq(f"Ask 1 short follow-up about: {user_input}")
        st.session_state.messages.append({"role": "assistant", "content": follow_up})
        st.session_state.follow_up_count += 1
        st.session_state.current_question = follow_up
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ Great job! (Score {score:.1f}/10)"
        })
        st.session_state.follow_up_count = 0
    
    st.rerun()














