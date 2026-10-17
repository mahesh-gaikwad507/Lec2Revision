
import os
import re
import json
import time
import random
import sqlite3
import smtplib
import hashlib
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path

import bcrypt
import requests
import streamlit as st
from pypdf import PdfReader

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "lecture_revision.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

APP_NAME = "Lecture-to-Revision AI"
DEFAULT_MODEL = "openai/gpt-4o-mini"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0d111b;
    color: #f5f7fb;
}
[data-testid="stSidebar"] {
    background: #141a29;
}
h1, h2, h3 {
    color: #ffffff;
}
.stButton > button {
    background: #ff4b55;
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 600;
}
.stButton > button:hover {
    background: #e83b47;
    color: white;
}
.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}
.hero {
    padding: 26px;
    border-radius: 16px;
    background: linear-gradient(120deg, #202944, #38233d);
    margin-bottom: 20px;
}
.muted {
    color: #aeb8ca;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# Existing database is retained. Tables are created if missing.
# ============================================================

def db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revision_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subject TEXT,
                topic TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                purpose TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)


init_db()


# ============================================================
# SECRETS / ENVIRONMENT
# ============================================================

def setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def utcnow():
    return datetime.utcnow()


def now_text():
    return utcnow().isoformat(timespec="seconds")


def get_user_by_email(email):
    with db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE lower(email)=lower(?)",
            (email.strip(),)
        ).fetchone()


def get_user_by_username(username):
    with db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE lower(username)=lower(?)",
            (username.strip(),)
        ).fetchone()


# ============================================================
# PASSWORDS
# Required change: maximum of 8 characters.
# NOTE: This is less secure than permitting longer passwords.
# ============================================================

def validate_password(password):
    if not password:
        return False, "Please enter a password."

    if len(password) > 8:
        return False, "Password must be at most 8 characters."

    if len(password) < 1:
        return False, "Password cannot be empty."

    return True, ""


def hash_password(password):
    valid, message = validate_password(password)
    if not valid:
        raise ValueError(message)

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def check_password(password, stored_hash):
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            bytes(stored_hash)
        )
    except (ValueError, TypeError):
        return False


# ============================================================
# EMAIL OTP
# Configure SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
# SMTP_FROM, and OTP_PEPPER in Streamlit secrets.
# ============================================================

def send_email(to_email, subject, body):
    host = setting("SMTP_HOST")
    port = int(setting("SMTP_PORT", "587"))
    username = setting("SMTP_USER")
    password = setting("SMTP_PASSWORD")
    sender = setting("SMTP_FROM", username)

    if not all([host, username, password, sender]):
        raise RuntimeError(
            "Email is not configured. Add SMTP_HOST, SMTP_USER, "
            "SMTP_PASSWORD and SMTP_FROM to Streamlit secrets."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)


def otp_digest(email, purpose, code):
    pepper = setting("OTP_PEPPER")
    if not pepper:
        raise RuntimeError("OTP_PEPPER is not configured.")
    raw = f"{email.lower()}|{purpose}|{code}|{pepper}"
    return hashlib.sha256(raw.encode()).hexdigest()


def create_and_send_otp(email, purpose):
    # Basic cooldown: do not send another code for 60 seconds.
    with db() as conn:
        recent = conn.execute("""
            SELECT created_at FROM otp_codes
            WHERE lower(email)=lower(?) AND purpose=?
            ORDER BY id DESC LIMIT 1
        """, (email, purpose)).fetchone()

        if recent:
            try:
                created = datetime.fromisoformat(recent["created_at"])
                if (utcnow() - created).total_seconds() < 60:
                    raise ValueError(
                        "Please wait 60 seconds before requesting another code."
                    )
            except ValueError:
                raise

        code = f"{random.SystemRandom().randint(0, 999999):06d}"
        digest = otp_digest(email, purpose, code)
        expires = (utcnow() + timedelta(minutes=10)).isoformat(
            timespec="seconds"
        )

        conn.execute("""
            INSERT INTO otp_codes
            (email, purpose, code_hash, expires_at, attempts, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
        """, (email, purpose, digest, expires, now_text()))

    try:
        send_email(
            email,
            f"{APP_NAME} verification code",
            f"Your verification code is {code}.\n\n"
            "It expires in 10 minutes. Do not share this code."
        )
    except Exception:
        # Remove the unusable code if delivery fails.
        with db() as conn:
            conn.execute("""
                DELETE FROM otp_codes
                WHERE lower(email)=lower(?) AND purpose=?
            """, (email, purpose))
        raise


def verify_otp(email, purpose, code):
    if not re.fullmatch(r"\d{6}", code.strip()):
        return False, "Enter the 6-digit code."

    with db() as conn:
        row = conn.execute("""
            SELECT * FROM otp_codes
            WHERE lower(email)=lower(?) AND purpose=?
            ORDER BY id DESC LIMIT 1
        """, (email.strip(), purpose)).fetchone()

        if not row:
            return False, "Request a new verification code."

        if row["attempts"] >= 5:
            return False, "Too many attempts. Request a new code."

        expires = datetime.fromisoformat(row["expires_at"])
        if utcnow() > expires:
            conn.execute("DELETE FROM otp_codes WHERE id=?", (row["id"],))
            return False, "Code expired. Request a new one."

        conn.execute(
            "UPDATE otp_codes SET attempts=attempts+1 WHERE id=?",
            (row["id"],)
        )

        expected = row["code_hash"]
        actual = otp_digest(email.strip(), purpose, code.strip())

        if not hashlib.compare_digest(expected, actual):
            return False, "Incorrect verification code."

        conn.execute("DELETE FROM otp_codes WHERE id=?", (row["id"],))
        return True, "Email verified."


# ============================================================
# OPENROUTER AI
# Configure OPENROUTER_API_KEY and optionally OPENROUTER_MODEL.
# ============================================================

def ai_generate(prompt, system_prompt=None, temperature=0.3):
    api_key = setting("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing from Streamlit secrets."
        )

    model = setting("OPENROUTER_MODEL", DEFAULT_MODEL)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": setting("APP_URL", "http://localhost:8501"),
            "X-Title": APP_NAME,
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or
                    "You are a careful tutor. Use clear headings and accurate explanations."
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        },
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"AI provider returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("The AI provider returned an unexpected response.")


# ============================================================
# PDF AND YOUTUBE INGESTION
# ============================================================

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages)


def youtube_video_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?v=)([\w-]{11})",
        r"(?:youtu\.be/)([\w-]{11})",
        r"(?:youtube\.com/embed/)([\w-]{11})",
        r"(?:youtube\.com/shorts/)([\w-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def extract_youtube_transcript(url, language="en"):
    video_id = youtube_video_id(url)
    if not video_id:
        raise ValueError("That doesn't look like a valid YouTube video URL.")

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=[language, "en"])
        parts = []

        for item in transcript:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])

        text = " ".join(parts).strip()
        if not text:
            raise RuntimeError("The transcript is empty.")

        return text

    except ImportError:
        raise RuntimeError(
            "Install a compatible youtube-transcript-api package."
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve the transcript. It may be disabled, "
            "unavailable for this language, or blocked by YouTube. "
            f"Details: {str(exc)[:250]}"
        )


def split_chunks(text, chunk_size=1200, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def ingest_uploaded_material(uploaded_file, source_type, language="en"):
    if source_type == "PDF":
        return extract_pdf(uploaded_file)

    if source_type == "YouTube":
        return extract_youtube_transcript(uploaded_file, language)

    raise ValueError("Unsupported source type.")


# ============================================================
# REVISION HISTORY
# ============================================================

def save_revision(user_id, subject, topic, kind, content):
    with db() as conn:
        conn.execute("""
            INSERT INTO revision_history
            (user_id, subject, topic, kind, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id, subject, topic, kind, content, now_text()
        ))


def get_revisions(user_id, limit=100):
    with db() as conn:
        return conn.execute("""
            SELECT * FROM revision_history
            WHERE user_id=?
            ORDER BY id DESC LIMIT ?
        """, (user_id, limit)).fetchall()


def save_chat(user_id, question, answer):
    with db() as conn:
        conn.execute("""
            INSERT INTO chat_history
            (user_id, question, answer, created_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, question, answer, now_text()))


def get_chats(user_id, limit=100):
    with db() as conn:
        return conn.execute("""
            SELECT * FROM chat_history
            WHERE user_id=?
            ORDER BY id DESC LIMIT ?
        """, (user_id, limit)).fetchall()


# ============================================================
# SESSION STATE
# ============================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "Log in"


# ============================================================
# AUTHENTICATION UI
# ============================================================

def register_account(username, email, password, confirm):
    username = username.strip()
    email = email.strip().lower()

    if not re.fullmatch(r"[A-Za-z0-9._-]{3,30}", username):
        raise ValueError(
            "Username must be 3–30 characters using letters, digits, "
            "periods, underscores or hyphens."
        )

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("Enter a valid email address.")

    valid, message = validate_password(password)
    if not valid:
        raise ValueError(message)

    valid, message = validate_password(confirm)
    if not valid:
        raise ValueError("Confirmation password must be at most 8 characters.")

    if password != confirm:
        raise ValueError("Passwords do not match.")

    if get_user_by_username(username):
        raise ValueError("That username is already registered.")

    if get_user_by_email(email):
        raise ValueError("That email address is already registered.")

    password_hash = hash_password(password)

    with db() as conn:
        conn.execute("""
            INSERT INTO users
            (username, email, password_hash, verified, created_at)
            VALUES (?, ?, ?, 0, ?)
        """, (username, email, password_hash, now_text()))

    st.session_state.pending_email = email
    create_and_send_otp(email, "verify")
    return True


def authenticate(username, password):
    user = get_user_by_username(username)
    if not user:
        return None, "Invalid username or password."

    if not check_password(password, user["password_hash"]):
        return None, "Invalid username or password."

    if not user["verified"]:
        st.session_state.pending_email = user["email"]
        return None, "Verify your email before logging in."

    return user, ""


def show_auth():
    st.markdown(
        f"<div class='hero'><h1>📚 {APP_NAME}</h1>"
        "<p class='muted'>Turn lectures into effective revision materials.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "Log in", "Create account", "Verify email", "Forgot password"
    ])

    # ---------------- LOGIN ----------------
    with tabs[0]:
        st.subheader("Welcome back")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input(
                "Password",
                type="password",
                max_chars=8,
            )
            submitted = st.form_submit_button(
                "Log in", use_container_width=True
            )

        if submitted:
            try:
                user, error = authenticate(username, password)
                if user:
                    st.session_state.user = dict(user)
                    st.rerun()
                else:
                    st.error(error)
            except Exception as exc:
                st.error(str(exc))

    # ---------------- REGISTER ----------------
    with tabs[1]:
        st.subheader("Create your account")

        with st.form("register_form"):
            username = st.text_input(
                "Username",
                help="3–30 characters: letters, digits, . _ -",
            )
            email = st.text_input("Email address")

            password = st.text_input(
                "Password (maximum 8 characters)",
                type="password",
                max_chars=8,
            )
            confirm = st.text_input(
                "Confirm password",
                type="password",
                max_chars=8,
            )

            st.caption(
                "Password limit: 8 characters. Short passwords are less secure."
            )

            submitted = st.form_submit_button(
                "Create account", use_container_width=True
            )

        if submitted:
            try:
                register_account(username, email, password, confirm)
                st.success(
                    "Account created. Check your email for the verification code."
                )
                st.info("Open the Verify email tab to enter your code.")
            except sqlite3.IntegrityError:
                st.error("Username or email is already registered.")
            except Exception as exc:
                st.error(str(exc))

    # ---------------- VERIFY EMAIL ----------------
    with tabs[2]:
        st.subheader("Verify your email")

        default_email = st.session_state.get("pending_email", "")
        email = st.text_input(
            "Email address",
            value=default_email,
            key="verify_email",
        )

        if st.button("Send verification code"):
            try:
                user = get_user_by_email(email)
                if not user:
                    st.error("No account was found for that email.")
                elif user["verified"]:
                    st.info("This email is already verified.")
                else:
                    create_and_send_otp(email, "verify")
                    st.success("Verification code sent.")
            except Exception as exc:
                st.error(str(exc))

        code = st.text_input(
            "6-digit verification code",
            max_chars=6,
            key="verify_code",
        )

        if st.button("Verify email"):
            try:
                ok, message = verify_otp(email, "verify", code)
                if ok:
                    with db() as conn:
                        conn.execute("""
                            UPDATE users SET verified=1
                            WHERE lower(email)=lower(?)
                        """, (email.strip(),))
                    st.success("Email verified. You can now log in.")
                else:
                    st.error(message)
            except Exception as exc:
                st.error(str(exc))

    # ---------------- FORGOT PASSWORD ----------------
    with tabs[3]:
        st.subheader("Reset your password")

        email = st.text_input("Registered email", key="reset_email")

        if st.button("Send reset code"):
            try:
                user = get_user_by_email(email)
                if not user:
                    st.error("No account was found for that email.")
                elif not user["verified"]:
                    st.error("Verify your email before resetting the password.")
                else:
                    create_and_send_otp(email, "reset")
                    st.success("Reset code sent to your email.")
            except Exception as exc:
                st.error(str(exc))

        reset_code = st.text_input(
            "6-digit reset code",
            max_chars=6,
            key="reset_code",
        )
        new_password = st.text_input(
            "New password (maximum 8 characters)",
            type="password",
            max_chars=8,
            key="new_password",
        )
        confirm_new = st.text_input(
            "Confirm new password",
            type="password",
            max_chars=8,
            key="confirm_new_password",
        )

        if st.button("Reset password"):
            try:
                valid, message = validate_password(new_password)
                if not valid:
                    raise ValueError(message)

                valid, message = validate_password(confirm_new)
                if not valid:
                    raise ValueError(message)

                if new_password != confirm_new:
                    raise ValueError("Passwords do not match.")

                ok, message = verify_otp(email, "reset", reset_code)
                if not ok:
                    raise ValueError(message)

                with db() as conn:
                    conn.execute("""
                        UPDATE users SET password_hash=?
                        WHERE lower(email)=lower(?)
                    """, (hash_password(new_password), email.strip()))

                st.success("Password reset successfully. You can now log in.")
            except Exception as exc:
                st.error(str(exc))


# ============================================================
# AI REVISION WORKSPACE
# ============================================================

def require_source():
    source = st.session_state.source_text.strip()
    if not source:
        st.warning("Upload a PDF or retrieve a YouTube transcript first.")
        return None
    return source


def generate_revision(kind, subject, topic, language, source):
    instructions = {
        "Summary": (
            "Create clear, structured revision notes. Include key concepts, "
            "definitions, examples, and a concise final recap."
        ),
        "Flashcards": (
            "Create 12 useful flashcards in Question / Answer format. "
            "Keep answers concise and based on the source."
        ),
        "Quiz": (
            "Create 10 multiple-choice questions. Give four options per "
            "question and show the correct answer with a short explanation."
        ),
        "Exam Questions": (
            "Create important exam-style questions, mixing short answers "
            "and long answers. Provide model answers and mark-worthy points."
        ),
        "Mind Map": (
            "Create a hierarchical text mind map using headings, indentation, "
            "and arrows to show relationships between concepts."
        ),
        "Key Points": (
            "Extract the most important facts, terms, formulas, and concepts "
            "as a well-organized checklist."
        ),
        "Translate": (
            "Translate and explain the source content in the requested language "
            "while preserving technical terminology and meaning."
        ),
    }

    prompt = f"""
Subject: {subject or "General"}
Topic: {topic or "Not specified"}
Output language: {language}
Task: {kind}

Instructions:
{instructions.get(kind, "Explain the source accurately.")}

Use the supplied material as the main source. Do not invent facts.
If the material does not contain enough information, state that clearly.

SOURCE MATERIAL:
{source[:24000]}
"""
    return ai_generate(prompt)


def show_workspace():
    user = st.session_state.user

    st.markdown(
        f"<div class='hero'><h1>Welcome, {user['username']} 👋</h1>"
        "<p class='muted'>Your personal AI-powered revision workspace.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("📚 Revision Menu")
        st.write(f"Signed in as **{user['username']}**")
        page = st.radio(
            "Navigate",
            [
                "Learning workspace",
                "AI revision tools",
                "Ask your lecture",
                "My history",
                "Account",
            ],
        )

        st.divider()

        if st.button("Log out", use_container_width=True):
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()

    if page == "Learning workspace":
        show_learning_workspace()

    elif page == "AI revision tools":
        show_revision_tools()

    elif page == "Ask your lecture":
        show_chat()

    elif page == "My history":
        show_history()

    elif page == "Account":
        st.subheader("Account details")
        st.write("**Username:**", user["username"])
        st.write("**Email:**", user["email"])
        st.write(
            "**Email verified:**",
            "Yes" if user["verified"] else "No",
        )


def show_learning_workspace():
    st.header("Learning workspace")

    col1, col2 = st.columns(2)

    with col1:
        subject = st.text_input(
            "Subject",
            key="learning_subject",
            placeholder="e.g. Computer Networks",
        )

    with col2:
        topic = st.text_input(
            "Topic",
            key="learning_topic",
            placeholder="e.g. TCP/IP",
        )

    language = st.selectbox(
        "Output language",
        ["English", "Hindi", "Marathi", "Spanish", "French"],
        key="learning_language",
    )

    source_type = st.radio(
        "Choose your learning material",
        ["PDF", "YouTube"],
        horizontal=True,
    )

    if source_type == "PDF":
        uploaded = st.file_uploader(
            "Upload lecture PDF",
            type=["pdf"],
            key="lecture_pdf",
        )

        if uploaded and st.button("Extract PDF text"):
            try:
                with st.spinner("Reading PDF..."):
                    text = extract_pdf(uploaded)

                if not text.strip():
                    st.error(
                        "No selectable text was found. Scanned PDFs need OCR."
                    )
                else:
                    st.session_state.source_text = text
                    st.session_state.source_name = uploaded.name
                    st.success(
                        f"Extracted text from PDF ({len(text):,} characters)."
                    )
            except Exception as exc:
                st.error(f"Could not read PDF: {exc}")

    else:
        youtube_url = st.text_input(
            "YouTube video URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        transcript_language = st.selectbox(
            "Transcript language",
            ["en", "hi", "mr"],
            key="transcript_language",
        )

        if st.button("Get YouTube transcript"):
            try:
                with st.spinner("Retrieving transcript..."):
                    text = extract_youtube_transcript(
                        youtube_url, transcript_language
                    )

                st.session_state.source_text = text
                st.session_state.source_name = youtube_url
                st.success(
                    f"Transcript retrieved ({len(text):,} characters)."
                )
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.source_text:
        st.divider()
        st.subheader("Loaded material")
        st.caption(st.session_state.source_name)
        chunks = split_chunks(st.session_state.source_text)

        st.write(
            f"Characters: {len(st.session_state.source_text):,} "
            f"| Text chunks: {len(chunks)}"
        )

        with st.expander("Preview extracted text"):
            st.text(st.session_state.source_text[:6000])

        if st.button("Clear loaded material"):
            st.session_state.source_text = ""
            st.session_state.source_name = ""
            st.rerun()

        st.info(
            "Material is held in this Streamlit session. Save any generated "
            "revision output to history if you want to retrieve it later."
        )


def show_revision_tools():
    st.header("AI revision tools")

    subject = st.text_input(
        "Subject",
        value=st.session_state.get("learning_subject", ""),
        key="tools_subject",
    )
    topic = st.text_input(
        "Topic",
        value=st.session_state.get("learning_topic", ""),
        key="tools_topic",
    )
    language = st.selectbox(
        "Output language",
        ["English", "Hindi", "Marathi", "Spanish", "French"],
        key="tools_language",
    )

    kind = st.selectbox(
        "Choose revision output",
        [
            "Summary",
            "Flashcards",
            "Quiz",
            "Exam Questions",
            "Mind Map",
            "Key Points",
            "Translate",
        ],
    )

    if st.button("Generate revision material", use_container_width=True):
        source = require_source()
        if source:
            try:
                with st.spinner(f"Generating {kind.lower()}..."):
                    result = generate_revision(
                        kind, subject, topic, language, source
                    )

                st.session_state.generated_result = result
                st.session_state.generated_kind = kind

                save_revision(
                    st.session_state.user["id"],
                    subject,
                    topic,
                    kind,
                    result,
                )
                st.success("Revision material generated and saved to history.")

            except Exception as exc:
                st.error(f"Generation failed: {exc}")

    if st.session_state.get("generated_result"):
        st.divider()
        st.subheader(
            st.session_state.get("generated_kind", "Revision material")
        )
        st.markdown(st.session_state.generated_result)

        st.download_button(
            "Download revision material",
            data=st.session_state.generated_result,
            file_name="revision_material.md",
            mime="text/markdown",
        )


def show_chat():
    st.header("Ask your lecture")

    source = st.session_state.source_text.strip()
    if not source:
        st.info("Load a PDF or YouTube transcript in the Learning workspace first.")
        return

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    question = st.chat_input("Ask a question about your lecture...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        prompt = f"""
Answer the student's question using the lecture material below.
If the answer is not supported by the lecture, say so.
Be clear and educational.

QUESTION:
{question}

LECTURE MATERIAL:
{source[:24000]}
"""
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ai_generate(prompt)
                st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

            save_chat(
                st.session_state.user["id"],
                question,
                answer,
            )

        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")


def show_history():
    st.header("My revision history")

    user_id = st.session_state.user["id"]
    revisions = get_revisions(user_id)

    st.subheader("Generated revision materials")

    if not revisions:
        st.info("You haven't saved any revision materials yet.")
    else:
        for row in revisions:
            title = (
                f"{row['kind']} — {row['topic'] or row['subject'] or 'General'}"
                f" — {row['created_at']}"
            )
            with st.expander(title):
                st.markdown(row["content"])
                st.download_button(
                    "Download",
                    data=row["content"],
                    file_name=f"revision_{row['id']}.md",
                    mime="text/markdown",
                    key=f"download_revision_{row['id']}",
                )

    st.divider()
    st.subheader("Previous chat conversations")

    chats = get_chats(user_id)

    if not chats:
        st.info("No saved chat conversations yet.")
    else:
        for row in chats:
            with st.expander(
                f"{row['created_at']} — {row['question'][:90]}"
            ):
                st.markdown("**Question**")
                st.write(row["question"])
                st.markdown("**Answer**")
                st.markdown(row["answer"])


# ============================================================
# MAIN
# ============================================================

def main():
    if st.session_state.user is None:
        show_auth()
    else:
        show_workspace()


if __name__ == "__main__":
    main()
