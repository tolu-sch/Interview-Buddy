import streamlit as st
from PyPDF2 import PdfReader
import requests

# Groq API Config (Free Tier)
GROQ_API_KEY = "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"  # 🔑 Get yours at https://console.groq.com
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"  # Groq's fastest free model

def ask_groq(prompt):
    """Get AI response from Groq's ultra-fast API"""
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=10  # Groq responds in <1s usually
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        if 'response' in locals():
            st.json(response.json())  # Show full error
        return None

# --- Streamlit UI ---
st.title("🤖 Interview Prep Buddy (Groq Powered)")
st.write("Paste a job description and upload your resume:")

# ===== NEW CODE START =====
# Initialize chat history AND reset tracker
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.interview_started = False

# Reset function
def reset_interview():
    st.session_state.messages = []
    st.session_state.interview_started = False
    st.rerun()

# Reset button (top-right corner)
with st.columns(3)[2]:  # Positions button in top-right
    if st.button("🔄 New Interview"):
        reset_interview()
# ===== NEW CODE END =====

# Inputs
jd = st.text_area("Job Description:", height=150)
resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Display conversation
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Interview Logic ---
if jd:
    # Determine button text based on whether we've started
    button_text = "New Question" if st.session_state.messages else "Start Interview"
    
    if st.button(button_text):
        resume_text = ""
        if resume:
            try:
                resume_text = PdfReader(resume).pages[0].extract_text()
            except Exception as e:
                st.warning(f"Couldn't read resume: {str(e)}")
        
        prompt = f"""
        Act as a technical interviewer. Ask one relevant question about:
        Job: {jd}
        Resume: {resume_text if resume_text else "Not provided"}
        """
        
        question = ask_groq(prompt)
        if question:
            st.session_state.messages.append({"role": "assistant", "content": question})
            st.rerun()

    if user_input := st.chat_input("Your answer:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        follow_up = ask_groq(f"Ask a follow-up about: {user_input}")
        
        if follow_up:
            st.session_state.messages.append({"role": "assistant", "content": follow_up})
            st.rerun()

# Show remaining quota
if GROQ_API_KEY.startswith("gsk_"):
    st.caption("Using Groq's free tier (500 requests/day)")