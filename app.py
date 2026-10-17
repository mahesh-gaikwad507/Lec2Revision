from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import re
import secrets
import smtplib
import sqlite3
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import streamlit as st

# Optional packages are imported only where needed, so the app can still start
# when an optional feature dependency has not been installed.
try:
    import requests
except Exception:
    requests = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


# ----------------------------- Configuration -----------------------------

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB = Path("/content/lecture_to_revision/lecture_revision.db")
DB_PATH = Path(os.getenv("LECTURE_REVISION_DB", str(DEFAULT_DB if DEFAULT_DB.parent.exists() else APP_DIR / "lecture_revision.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Lecture to Revision AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>"
    ":root { color-scheme: light; }"
    ".stApp { background: linear-gradient(135deg,#f7f9ff 0%,#eef4ff 52%,#f8f5ff 100%); }"
    "[data-testid=\"stSidebar\"] { background: #101b35; }"
    "[data-testid=\"stSidebar\"] * { color: #f5f7ff !important; }"
    ".hero { padding: 1.35rem 1.5rem; border-radius: 20px; background: linear-gradient(120deg,#1e3a8a,#6d28d9); color: white; margin-bottom: 1rem; box-shadow: 0 12px 30px rgba(40,50,110,.16); }"
    ".hero h1 { color:white; margin:0; }"
    ".hero p { color:#e8eaff; margin:.4rem 0 0 0; }"
    ".metric-card { padding: 1rem; border-radius: 16px; background:white; border:1px solid #e1e8fb; box-shadow:0 5px 18px rgba(30,58,138,.06); }"
    "div.stButton > button { border-radius: 10px; font-weight: 600; }"
    ".small-note { color:#64748b; font-size:.88rem; }"
    "</style>",
    unsafe_allow_html=True,
)


# ----------------------------- Database Setup -----------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT PRIMARY KEY,
            otp_hash TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            title TEXT NOT NULL,
            source_type TEXT,
            content TEXT,
            summary TEXT,
            flashcards TEXT,
            quiz TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ----------------------------- Helper Functions -----------------------------

def get_secret(key: str, default: Any = None) -> Any:
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

def send_email_otp(email: str, otp: str):
    smtp_host = get_secret("SMTP_HOST")
    smtp_port = int(get_secret("SMTP_PORT", 587))
    smtp_user = get_secret("SMTP_USERNAME")
    smtp_pass = get_secret("SMTP_PASSWORD")
    smtp_from = get_secret("SMTP_FROM", smtp_user)

    if not smtp_host or not smtp_user or not smtp_pass:
        return False, "SMTP configuration missing in secrets/environment."

    try:
        msg = EmailMessage()
        msg.set_subject("Your Lecture-to-Revision Verification Code")
        msg.set_from(smtp_from)
        msg.set_to(email)
        msg.set_content(f"Your one-time passcode (OTP) is: {otp}\n\nIt expires in 10 minutes.")

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "OTP sent successfully."
    except Exception as e:
        return False, str(e)

def call_openrouter(prompt: str) -> str:
    api_key = get_secret("OPENROUTER_API_KEY")
    model = get_secret("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        return "Error: OPENROUTER_API_KEY is not configured in secrets."

    if not requests:
        return "Error: 'requests' package is not installed."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Lecture to Revision AI"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"OpenRouter API Error [{response.status_code}]: {response.text}"
    except Exception as e:
        return f"Exception during API call: {str(e)}"

def extract_youtube_transcript(url: str) -> str:
    if not YouTubeTranscriptApi:
        return "Error: youtube_transcript_api package is not installed."
    
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        return "Error: Invalid YouTube URL format."
    
    video_id = match.group(1)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item["text"] for item in transcript_list])
        return full_text
    except Exception as e:
        return f"Error extracting transcript: {str(e)}"

def extract_pdf_text(uploaded_file) -> str:
    if not PdfReader:
        return "Error: pypdf package is not installed."
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"


# ----------------------------- Authentication Flow -----------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = ""

if not st.session_state.authenticated:
    st.sidebar.markdown("## 🔐 Sign In")
    auth_email = st.sidebar.text_input("Email Address", placeholder="name@example.com")
    
    dev_show_otp = str(get_secret("DEV_SHOW_OTP", "false")).lower() == "true"

    if st.sidebar.button("Send OTP"):
        if not auth_email or "@" not in auth_email:
            st.sidebar.error("Please enter a valid email address.")
        else:
            otp_code = f"{random.randint(100000, 999999)}"
            otp_hash = hashlib.sha256(otp_code.encode()).hexdigest()
            expires = datetime.now() + timedelta(minutes=10)

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("REPLACE INTO otps (email, otp_hash, expires_at) VALUES (?, ?, ?)", 
                           (auth_email, otp_hash, expires))
            conn.commit()
            conn.close()

            success, msg = email_sent = send_email_otp(auth_email, otp_code)
            st.session_state.temp_email = auth_email

            if success:
                st.sidebar.success("OTP sent to your email!")
            else:
                st.sidebar.warning(f"Failed to email OTP: {msg}")
            
            if dev_show_otp:
                st.sidebar.info(f"[DEV MODE] OTP Code: {otp_code}")

    if "temp_email" in st.session_state:
        entered_otp = st.sidebar.text_input("Enter 6-digit OTP", type="password")
        if st.sidebar.button("Verify OTP"):
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT otp_hash, expires_at FROM otps WHERE email = ?", (st.session_state.temp_email,))
            row = cursor.fetchone()
            conn.close()

            if row:
                stored_hash, expires_at = row['otp_hash'], datetime.fromisoformat(row['expires_at'])
                entered_hash = hashlib.sha256(entered_otp.encode()).hexdigest()

                if datetime.now() > expires_at:
                    st.sidebar.error("OTP has expired. Please request a new one.")
                elif hmac.compare_digest(stored_hash, entered_hash):
                    st.session_state.authenticated = True
                    st.session_state.user_email = st.session_state.temp_email
                    
                    # Ensure user in db
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("INSERT OR IGNORE INTO users (email) VALUES (?)", (st.session_state.user_email,))
                    conn.commit()
                    conn.close()
                    
                    st.sidebar.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid OTP code.")
            else:
                st.sidebar.error("No OTP request found. Please request a code.")

    st.markdown("""
        <div class="hero">
            <h1>Lecture-to-Revision AI 📚</h1>
            <p>Transform raw lecture transcripts or notes into structured revision summaries, flashcards, and quizzes effortlessly.</p>
        </div>
        <div class="metric-card">
            <h3>Welcome! Please sign in via the sidebar</h3>
            <p class="small-note">Enter your email address to receive your secure one-time login passcode.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# ----------------------------- Main App Dashboard -----------------------------

st.sidebar.markdown(f"👤 **Signed in as:** `{st.session_state.user_email}`")
if st.sidebar.button("Sign Out"):
    st.session_state.authenticated = False
    st.session_state.user_email = ""
    st.rerun()

st.markdown("""
    <div class="hero">
        <h1>Lecture-to-Revision Dashboard</h1>
        <p>Upload your materials, generate intelligent study guides, and review saved content.</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📄 New Intake & Generate", "📚 Saved Revisions", "⚙️ Account Info"])

with tab1:
    st.subheader("Import Lecture Source")
    input_source = st.radio("Choose source type:", ["YouTube URL", "Upload PDF Notes", "Paste Raw Text"])
    
    raw_content = ""
    default_title = "Untitled Lecture"

    if input_source == "YouTube URL":
        yt_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        if yt_url:
            if st.button("Fetch Transcript"):
                with st.spinner("Extracting transcript..."):
                    raw_content = extract_youtube_transcript(yt_url)
                    st.session_state.raw_content = raw_content
                    st.success("Transcript extracted successfully!")
        default_title = "YouTube Lecture"

    elif input_source == "Upload PDF Notes":
        pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
        if pdf_file:
            with st.spinner("Parsing PDF..."):
                raw_content = extract_pdf_text(pdf_file)
                st.session_state.raw_content = raw_content
                st.success("PDF parsed successfully!")
            default_title = pdf_file.name.replace(".pdf", "")

    else:
        raw_content = st.text_area("Paste lecture notes or text here...", height=200)
        st.session_state.raw_content = raw_content
        default_title = "Text Notes Lecture"

    if "raw_content" in st.session_state and st.session_state.raw_content:
        st.markdown("---")
        st.subheader("Generate Revision Materials")
        lecture_title = st.text_input("Revision Title", value=default_title)

        if st.button("Generate AI Revision Suite"):
            with st.spinner("Communicating with OpenRouter AI..."):
                prompt_base = f"Based on the following lecture text, provide a comprehensive output containing: 1. Summary, 2. Flashcards (Q&A format), and 3. Practice Quiz (MCQs).\n\nLecture Text:\n{st.session_state.raw_content[:15000]}"
                
                ai_result = call_openrouter(prompt_base)
                
                # Simple parsing fallback
                summary_text = ai_result
                flashcards_text = "See main summary/content."
                quiz_text = "See main summary/content."

                # Save to DB
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO revisions (user_email, title, source_type, content, summary, flashcards, quiz)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (st.session_state.user_email, lecture_title, input_source, st.session_state.raw_content, summary_text, flashcards_text, quiz_text))
                conn.commit()
                conn.close()

                st.success("Revision suite generated and saved successfully!")
                st.markdown("### Generated Output")
                st.markdown(ai_result)

with tab2:
    st.subheader("Your Saved Revisions")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, source_type, created_at FROM revisions WHERE user_email = ? ORDER BY created_at DESC", (st.session_state.user_email,))
    revisions = cursor.fetchall()
    conn.close()

    if not revisions:
        st.info("No saved revisions found yet. Create one from the first tab!")
    else:
        rev_options = {f"{r['title']} ({r['created_at']})": r['id'] for r in revisions}
        selected_rev_label = st.selectbox("Select a revision to view", list(rev_options.keys()))
        
        if selected_rev_label:
            rev_id = rev_options[selected_rev_label]
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM revisions WHERE id = ?", (rev_id,))
            rev_data = cursor.fetchone()
            conn.close()

            if rev_data:
                st.markdown(f"### {rev_data['title']}")
                st.caption(f"Source: {rev_data['source_type']} | Created: {rev_data['created_at']}")
                st.markdown(rev_data['summary'])

with tab3:
    st.subheader("Account Information")
    st.write(f"**Email:** {st.session_state.user_email}")
    st.write(f"**Database Path:** `{DB_PATH}`")
    st.info("You are currently authenticated using email OTP authentication.")
