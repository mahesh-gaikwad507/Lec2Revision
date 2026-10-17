import os
import re
import json
import time
import math
import hmac
import hashlib
import secrets
import smtplib
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from email.message import EmailMessage

import bcrypt
import faiss
import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("LTR_DATA_DIR", APP_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "lecture_revision.db"

st.set_page_config(
    page_title="Lecture-to-Revision AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name, default=""):
    """Read a key from Streamlit Secrets or environment variables."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_secret(
    "OPENROUTER_MODEL", "openai/gpt-4o-mini"
)
YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")
SMTP_HOST = get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(get_secret("SMTP_PORT", "587"))
SMTP_USERNAME = get_secret("SMTP_USERNAME")
SMTP_PASSWORD = get_secret("SMTP_PASSWORD")
SMTP_FROM_EMAIL = get_secret("SMTP_FROM_EMAIL", SMTP_USERNAME)

EMBEDDING_MODEL_NAME = get_secret(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(91, 92, 246, .10), transparent 30%),
          radial-gradient(circle at 100% 10%, rgba(20, 184, 166, .08), transparent 25%);
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.035em;
    }

    .hero {
        padding: 1.5rem 0 1rem 0;
    }

    .hero-kicker {
        color: #7776f6;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .76rem;
        font-weight: 800;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 800;
        line-height: 1.1;
        margin: .45rem 0;
    }

    .hero-subtitle {
        color: #777f90;
        font-size: 1.05rem;
        max-width: 720px;
    }

    .panel {
        border: 1px solid rgba(130, 140, 165, .22);
        background: rgba(128, 128, 128, .035);
        border-radius: 18px;
        padding: 1.1rem;
        margin: .3rem 0 .8rem 0;
    }

    .metric {
        border: 1px solid rgba(130, 140, 165, .22);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(128, 128, 128, .035);
    }

    .muted { color: #7b8495; }
    div.stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.6rem;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(130, 140, 165, .2);
        border-radius: 16px;
        padding: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS email_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                purpose TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS study_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                score REAL,
                details TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_materials_user
                ON materials(user_id, subject, topic);

            CREATE INDEX IF NOT EXISTS idx_chats_user
                ON chats(user_id, subject, topic, created_at);

            CREATE INDEX IF NOT EXISTS idx_codes_email
                ON email_codes(email, purpose, created_at);
            """
        )


init_db()


# ---------------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------------

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def normalize_email(email):
    return email.strip().lower()


def valid_email(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))


def valid_password(password):
    return len(password) >= 10


def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")


def check_password(password, hashed):
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), hashed.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def add_progress(user_id, subject, topic, activity, score=None, details=""):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO study_progress
            (user_id, subject, topic, activity_type, score, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, subject, topic, activity,
                score, details, now_iso()
            ),
        )


# ---------------------------------------------------------
# EMAIL OTP
# ---------------------------------------------------------

def send_email(to_email, subject, body):
    if not all([SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL]):
        raise RuntimeError(
            "Email is not configured. Add SMTP settings to Streamlit Secrets."
        )

    msg = EmailMessage()
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)


def issue_email_code(email, purpose):
    email = normalize_email(email)
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires = (datetime.utcnow() + timedelta(minutes=10)).isoformat(
        timespec="seconds"
    )

    # Use a keyed hash so the OTP is not stored as plaintext.
    pepper = get_secret("OTP_PEPPER")
    if not pepper:
        raise RuntimeError(
            "OTP_PEPPER is missing from configuration. "
            "Set it to a long random secret."
        )

    code_hash = hmac.new(
        pepper.encode(),
        f"{email}:{purpose}:{code}".encode(),
        hashlib.sha256,
    ).hexdigest()

    with db() as conn:
        conn.execute(
            "DELETE FROM email_codes WHERE email=? AND purpose=?",
            (email, purpose),
        )
        conn.execute(
            """
            INSERT INTO email_codes
            (email, purpose, code_hash, expires_at, attempts, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (email, purpose, code_hash, expires, now_iso()),
        )

    subject = (
        "Your Lecture-to-Revision verification code"
        if purpose == "verify"
        else "Reset your Lecture-to-Revision password"
    )

    body = (
        f"Your verification code is: {code}\n\n"
        "This code expires in 10 minutes. If you did not request it, "
        "you can ignore this email."
    )

    try:
        send_email(email, subject, body)
    except Exception:
        with db() as conn:
            conn.execute(
                "DELETE FROM email_codes WHERE email=? AND purpose=?",
                (email, purpose),
            )
        raise


def verify_email_code(email, purpose, code):
    email = normalize_email(email)
    pepper = get_secret("OTP_PEPPER")
    if not pepper:
        return False, "Email verification is not configured."

    with db() as conn:
        row = conn.execute(
            """
            SELECT * FROM email_codes
            WHERE email=? AND purpose=?
            ORDER BY id DESC LIMIT 1
            """,
            (email, purpose),
        ).fetchone()

        if not row:
            return False, "Request a new verification code."

        if row["attempts"] >= 5:
            return False, "Too many attempts. Request a new code."

        if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
            conn.execute("DELETE FROM email_codes WHERE id=?", (row["id"],))
            return False, "This code has expired. Request another."

        expected = hmac.new(
            pepper.encode(),
            f"{email}:{purpose}:{code.strip()}".encode(),
            hashlib.sha256,
        ).hexdigest()

        conn.execute(
            "UPDATE email_codes SET attempts=attempts+1 WHERE id=?",
            (row["id"],),
        )

        if not hmac.compare_digest(expected, row["code_hash"]):
            return False, "Incorrect code."

        conn.execute("DELETE FROM email_codes WHERE id=?", (row["id"],))
        return True, "Verified."


# ---------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------

def current_user():
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None

    with db() as conn:
        return conn.execute(
            "SELECT id, email, username, verified FROM users WHERE id=?",
            (user_id,),
        ).fetchone()


def auth_screen():
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Your personal AI study companion</div>
          <div class="hero-title">Learn smarter.<br>Revise with confidence.</div>
          <div class="hero-subtitle">
            Turn lecture videos and PDFs into summaries, flashcards,
            practice questions and searchable study notes.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab, verify_tab, reset_tab = st.tabs(
        ["Log in", "Create account", "Verify email", "Forgot password"]
    )

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )
            submitted = st.form_submit_button(
                "Log in", type="primary", use_container_width=True
            )

        if submitted:
            email = normalize_email(email)
            with db() as conn:
                user = conn.execute(
                    "SELECT * FROM users WHERE email=?", (email,)
                ).fetchone()

            if user and check_password(password, user["password_hash"]):
                if not user["verified"]:
                    st.warning("Verify your email before logging in.")
                else:
                    st.session_state.user_id = user["id"]
                    st.rerun()
            else:
                st.error("Email or password is incorrect.")

    with signup_tab:
        with st.form("signup_form"):
            username = st.text_input("Username", max_chars=30)
            email = st.text_input("Email address", key="signup_email")
            password = st.text_input(
                "Password (at least 10 characters)",
                type="password",
                key="signup_password",
            )
            confirm = st.text_input(
                "Confirm password", type="password"
            )
            submitted = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )

        if submitted:
            email = normalize_email(email)
            username = username.strip()

            if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
                st.error("Username must be 3–30 characters: letters, digits, . _ -")
            elif not valid_email(email):
                st.error("Enter a valid email address.")
            elif not valid_password(password):
                st.error("Use a password with at least 10 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    with db() as conn:
                        conn.execute(
                            """
                            INSERT INTO users
                            (email, username, password_hash, verified, created_at)
                            VALUES (?, ?, ?, 0, ?)
                            """,
                            (email, username, hash_password(password), now_iso()),
                        )
                    st.session_state.pending_verification_email = email
                    issue_email_code(email, "verify")
                    st.success(
                        "Account created. Check your email for the six-digit code."
                    )
                except sqlite3.IntegrityError:
                    st.error("That email or username is already registered.")
                except Exception as exc:
                    st.error(f"Could not send verification email: {exc}")

    with verify_tab:
        st.caption("Verify a new account using the code sent to your email.")
        email = st.text_input(
            "Account email", key="verify_email",
            value=st.session_state.get("pending_verification_email", "")
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send / resend code", use_container_width=True):
                email = normalize_email(email)
                with db() as conn:
                    user = conn.execute(
                        "SELECT verified FROM users WHERE email=?", (email,)
                    ).fetchone()
                if not valid_email(email) or not user:
                    st.error("No account found for that email.")
                elif user["verified"]:
                    st.info("This account is already verified.")
                else:
                    try:
                        issue_email_code(email, "verify")
                        st.success("A new code has been sent.")
                    except Exception as exc:
                        st.error(f"Email could not be sent: {exc}")

        with col2:
            code = st.text_input("Six-digit code", max_chars=6, key="verify_code")
            if st.button("Verify account", type="primary", use_container_width=True):
                email = normalize_email(email)
                ok, message = verify_email_code(email, "verify", code)
                if ok:
                    with db() as conn:
                        conn.execute(
                            "UPDATE users SET verified=1 WHERE email=?", (email,)
                        )
                    st.success("Email verified. You can now log in.")
                else:
                    st.error(message)

    with reset_tab:
        st.caption("Request a password-reset code for your registered email.")
        email = normalize_email(
            st.text_input("Registered email", key="reset_email")
        )

        if st.button("Send reset code", use_container_width=True):
            with db() as conn:
                user = conn.execute(
                    "SELECT id FROM users WHERE email=?", (email,)
                ).fetchone()

            # Do not reveal whether an account exists.
            if user and valid_email(email):
                try:
                    issue_email_code(email, "reset")
                except Exception:
                    pass
            st.info(
                "If the account exists and email is configured, "
                "a reset code will be sent."
            )

        code = st.text_input("Reset code", max_chars=6, key="reset_code")
        new_password = st.text_input(
            "New password (at least 10 characters)",
            type="password", key="new_password"
        )
        confirm_password = st.text_input(
            "Confirm new password", type="password", key="new_password_confirm"
        )

        if st.button("Reset password", type="primary", use_container_width=True):
            if not valid_email(email):
                st.error("Enter the email address used for the account.")
            elif not valid_password(new_password):
                st.error("Use a password with at least 10 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = verify_email_code(email, "reset", code)
                if ok:
                    with db() as conn:
                        conn.execute(
                            "UPDATE users SET password_hash=? WHERE email=?",
                            (hash_password(new_password), email),
                        )
                    st.success("Password reset successfully. Please log in.")
                else:
                    st.error(message)


# ---------------------------------------------------------
# AI MODEL AND OPENROUTER
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def call_ai(system_prompt, user_prompt, temperature=0.3, max_tokens=1800):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured in Streamlit Secrets."
        )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Lecture-to-Revision AI",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter error ({response.status_code}): "
            f"{response.text[:700]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("The AI provider returned an unexpected response.")


def split_chunks(text, chunk_size=1100, overlap=180):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve_context(query, documents, top_k=5):
    """Combine dense semantic retrieval with BM25 keyword retrieval."""
    documents = [d.strip() for d in documents if d and d.strip()]
    if not documents:
        return []

    tokenized = [re.findall(r"\w+", d.lower()) for d in documents]
    bm25 = BM25Okapi(tokenized)
    query_tokens = re.findall(r"\w+", query.lower())
    lexical_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)

    model = load_embedding_model()
    doc_vectors = model.encode(
        documents, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    query_vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatIP(doc_vectors.shape[1])
    index.add(doc_vectors)
    dense_scores, dense_ids = index.search(
        query_vector, min(top_k * 2, len(documents))
    )

    # Reciprocal-rank fusion combines keyword and semantic rankings.
    fused = {}
    for rank, idx in enumerate(dense_ids[0], start=1):
        if idx >= 0:
            fused[int(idx)] = fused.get(int(idx), 0) + 1 / (60 + rank)

    lexical_order = np.argsort(lexical_scores)[::-1][:top_k * 2]
    for rank, idx in enumerate(lexical_order, start=1):
        fused[int(idx)] = fused.get(int(idx), 0) + 1 / (60 + rank)

    ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [documents[i] for i in ordered]


# ---------------------------------------------------------
# PDF AND YOUTUBE INGESTION
# ---------------------------------------------------------

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_youtube_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_transcript(url):
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError("This does not look like a valid YouTube video URL.")

    # youtube-transcript-api 0.x interface
    transcript = YouTubeTranscriptApi.get_transcript(
        video_id, languages=["en", "hi"]
    )
    return " ".join(item.get("text", "") for item in transcript).strip()


def search_youtube_videos(query, max_results=5):
    if not YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is not configured.")

    response = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results, 10),
            "key": YOUTUBE_API_KEY,
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"YouTube search error ({response.status_code}): "
            f"{response.text[:500]}"
        )

    items = response.json().get("items", [])
    ids = [
        item.get("id", {}).get("videoId")
        for item in items
        if item.get("id", {}).get("videoId")
    ]

    stats = {}
    if ids:
        stat_response = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "statistics,contentDetails",
                "id": ",".join(ids),
                "key": YOUTUBE_API_KEY,
            },
            timeout=30,
        )
        if stat_response.ok:
            stats = {
                item["id"]: item
                for item in stat_response.json().get("items", [])
            }

    results = []
    for item in items:
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet", {})
        stat = stats.get(video_id, {}).get("statistics", {})
        results.append({
            "title": snippet.get("title", "Untitled video"),
            "channel": snippet.get("channelTitle", "Unknown channel"),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "views": int(stat.get("viewCount", 0)),
            "likes": int(stat.get("likeCount", 0)),
            "comments": int(stat.get("commentCount", 0)),
            "description": snippet.get("description", ""),
        })
    return results


def save_material(user_id, subject, topic, source_type, source_name, content):
    if not content or len(content.strip()) < 40:
        raise ValueError("The extracted text is too short to save.")

    with db() as conn:
        conn.execute(
            """
            INSERT INTO materials
            (user_id, subject, topic, source_type, source_name, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, subject.strip(), topic.strip(), source_type,
                source_name[:500], content, now_iso()
            ),
        )


def load_materials(user_id, subject, topic):
    with db() as conn:
        rows = conn.execute(
            """
            SELECT id, source_type, source_name, content, created_at
            FROM materials
            WHERE user_id=? AND subject=? AND topic=?
            ORDER BY id DESC
            """,
            (user_id, subject, topic),
        ).fetchall()
    return rows


def delete_material(user_id, material_id):
    with db() as conn:
        conn.execute(
            "DELETE FROM materials WHERE id=? AND user_id=?",
            (material_id, user_id),
        )


# ---------------------------------------------------------
# SAVED CHAT HISTORY
# ---------------------------------------------------------

def save_chat(user_id, subject, topic, role, message):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO chats
            (user_id, subject, topic, role, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, subject, topic, role, message, now_iso()),
        )


def load_chat(user_id, subject, topic, limit=100):
    with db() as conn:
        return conn.execute(
            """
            SELECT role, message, created_at
            FROM chats
            WHERE user_id=? AND subject=? AND topic=?
            ORDER BY id DESC LIMIT ?
            """,
            (user_id, subject, topic, limit),
        ).fetchall()[::-1]


# ---------------------------------------------------------
# REVISION GENERATION
# ---------------------------------------------------------

def generate_revision(tool, subject, topic, context, language):
    instructions = {
        "Summary": (
            "Write a structured study summary with headings, key concepts, "
            "definitions, examples, and a short final recap."
        ),
        "Flashcards": (
            "Create 12 useful flashcards. Format each as "
            "'Q: ...' followed by 'A: ...'. Cover different difficulty levels."
        ),
        "Quiz": (
            "Create 10 multiple-choice questions with four options each. "
            "Mark the correct answer and explain why it is correct."
        ),
        "Exam questions": (
            "Create 10 exam-style questions of mixed short-answer and "
            "long-answer difficulty. Include a concise model answer and "
            "marks for each question."
        ),
        "Mind map": (
            "Create a hierarchical text mind map using indentation, "
            "branches and arrows. Group related concepts together."
        ),
        "Key points": (
            "Extract the 15 most important facts, definitions, formulas "
            "or principles. Do not add unsupported facts."
        ),
    }

    return call_ai(
        system_prompt=(
            "You are an accurate academic revision assistant. Base answers "
            "on the supplied study material. If information is missing, "
            "say so instead of inventing facts. Use clear Markdown."
        ),
        user_prompt=(
            f"Subject: {subject}\nTopic: {topic}\n"
            f"Output language: {language}\n"
            f"Task: {instructions[tool]}\n\n"
            f"Study material:\n{context[:18000]}"
        ),
        temperature=0.35,
        max_tokens=2500,
    )


# ---------------------------------------------------------
# MAIN WORKSPACE
# ---------------------------------------------------------

def workspace(user):
    st.sidebar.markdown("## 📚 Lecture-to-Revision")
    st.sidebar.caption(f"Signed in as **{user['username']}**")

    if st.sidebar.button("Log out", use_container_width=True):
        for key in ["user_id", "active_subject", "active_topic"]:
            st.session_state.pop(key, None)
        st.rerun()

    page = st.sidebar.radio(
        "Workspace",
        [
            "Dashboard",
            "Study materials",
            "Ask AI",
            "Revision studio",
            "YouTube finder",
            "Progress",
        ],
    )

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Your learning workspace</div>
          <div class="hero-title">Make every lecture count.</div>
          <div class="hero-subtitle">
            Upload course materials, ask focused questions, and build a
            revision system around your subjects.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input(
            "Subject",
            value=st.session_state.get("active_subject", ""),
            placeholder="e.g. Computer Networks",
        )
    with col2:
        topic = st.text_input(
            "Topic",
            value=st.session_state.get("active_topic", ""),
            placeholder="e.g. TCP congestion control",
        )

    subject = subject.strip()
    topic = topic.strip()

    if subject and topic:
        st.session_state.active_subject = subject
        st.session_state.active_topic = topic
    else:
        st.info("Enter a subject and topic to start a study session.")

    ready = bool(subject and topic)

    if page == "Dashboard":
        with db() as conn:
            material_count = conn.execute(
                "SELECT COUNT(*) FROM materials WHERE user_id=?",
                (user["id"],),
            ).fetchone()[0]
            chat_count = conn.execute(
                "SELECT COUNT(*) FROM chats WHERE user_id=?",
                (user["id"],),
            ).fetchone()[0]
            progress_count = conn.execute(
                "SELECT COUNT(*) FROM study_progress WHERE user_id=?",
                (user["id"],),
            ).fetchone()[0]

        a, b, c = st.columns(3)
        for col, value, label in [
            (a, material_count, "Saved materials"),
            (b, chat_count, "Chat messages"),
            (c, progress_count, "Revision activities"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric"><h2>{value}</h2>'
                    f'<div class="muted">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.subheader("A simple study workflow")
        st.markdown(
            """
            <div class="panel">
            <b>01 · Add materials</b><br>
            Upload a PDF or import a YouTube transcript.
            <hr>
            <b>02 · Ask questions</b><br>
            Search your own materials with semantic and keyword retrieval.
            <hr>
            <b>03 · Practice retrieval</b><br>
            Generate flashcards, quizzes, summaries and exam-style questions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif page == "Study materials":
        if not ready:
            return

        st.subheader("Add learning materials")
        st.caption("Extracted text is saved to your account's local database.")

        mode = st.radio(
            "Source",
            ["Upload PDF", "YouTube URLs"],
            horizontal=True,
        )

        if mode == "Upload PDF":
            uploads = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if st.button("Import PDF files", type="primary"):
                if not uploads:
                    st.warning("Choose at least one PDF.")
                else:
                    successes = 0
                    for uploaded in uploads:
                        try:
                            content = extract_pdf_text(uploaded)
                            if not content:
                                st.warning(
                                    f"No selectable text found in {uploaded.name}. "
                                    "Scanned PDFs need OCR before importing."
                                )
                                continue
                            save_material(
                                user["id"], subject, topic, "PDF",
                                uploaded.name, content
                            )
                            successes += 1
                        except Exception as exc:
                            st.error(f"{uploaded.name}: {exc}")
                    if successes:
                        add_progress(
                            user["id"], subject, topic,
                            "Imported PDF", details=f"{successes} file(s)"
                        )
                        st.success(f"Imported {successes} PDF file(s).")
                        st.rerun()

        else:
            urls_text = st.text_area(
                "Paste up to five YouTube video URLs",
                placeholder="https://www.youtube.com/watch?v=...\n"
                            "https://youtu.be/...",
                height=130,
            )
            if st.button("Import transcripts", type="primary"):
                urls = [
                    line.strip() for line in urls_text.splitlines()
                    if line.strip()
                ]
                if not urls:
                    st.warning("Enter at least one video URL.")
                elif len(urls) > 5:
                    st.error("Please import no more than five videos at once.")
                else:
                    imported = 0
                    for url in urls:
                        try:
                            transcript = fetch_youtube_transcript(url)
                            save_material(
                                user["id"], subject, topic, "YouTube",
                                url, transcript
                            )
                            imported += 1
                        except Exception as exc:
                            st.warning(f"{url}: {exc}")
                    if imported:
                        add_progress(
                            user["id"], subject, topic,
                            "Imported transcripts", details=f"{imported} video(s)"
                        )
                        st.success(f"Imported {imported} transcript(s).")
                        st.rerun()

        st.subheader("Saved materials for this topic")
        materials = load_materials(user["id"], subject, topic)
        if not materials:
            st.info("No materials have been added to this topic yet.")
        for item in materials:
            with st.expander(
                f"{item['source_type']} · {item['source_name'][:100]}"
            ):
                st.caption(f"Added {item['created_at']}")
                st.write(item["content"][:1500] + (
                    "…" if len(item["content"]) > 1500 else ""
                ))
                if st.button(
                    "Delete material",
                    key=f"delete_material_{item['id']}",
                ):
                    delete_material(user["id"], item["id"])
                    st.rerun()

    elif page == "Ask AI":
        if not ready:
            return

        materials = load_materials(user["id"], subject, topic)
        if not materials:
            st.warning("Add a PDF or YouTube transcript before asking questions.")
            return

        chunks = []
        for material in materials:
            for chunk in split_chunks(material["content"]):
                chunks.append(
                    f"Source: {material['source_name']}\n{chunk}"
                )

        st.subheader("Ask questions about your materials")
        st.caption(
            f"{len(materials)} source(s) · {len(chunks)} searchable text chunks"
        )

        history = load_chat(user["id"], subject, topic)
        for row in history:
            with st.chat_message(
                row["role"] if row["role"] in ("user", "assistant") else "assistant"
            ):
                st.markdown(row["message"])

        question = st.chat_input("Ask a question about your study materials…")
        if question:
            save_chat(user["id"], subject, topic, "user", question)
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching your materials and preparing an answer…"):
                    try:
                        context_chunks = retrieve_context(question, chunks)
                        context = "\n\n---\n\n".join(context_chunks)
                        answer = call_ai(
                            system_prompt=(
                                "You are a careful academic tutor. Answer using "
                                "the provided excerpts first. Cite sources by "
                                "their source name where possible. Clearly state "
                                "when the excerpts do not contain the answer. "
                                "Do not fabricate citations."
                            ),
                            user_prompt=(
                                f"Subject: {subject}\nTopic: {topic}\n\n"
                                f"Relevant excerpts:\n{context}\n\n"
                                f"Question: {question}"
                            ),
                        )
                        st.markdown(answer)
                        save_chat(
                            user["id"], subject, topic, "assistant", answer
                        )
                        add_progress(
                            user["id"], subject, topic,
                            "Asked AI question", details=question[:250]
                        )
                    except Exception as exc:
                        st.error(f"Could not generate an answer: {exc}")

    elif page == "Revision studio":
        if not ready:
            return

        materials = load_materials(user["id"], subject, topic)
        if not materials:
            st.warning("Import study materials before generating revision content.")
            return

        language = st.selectbox(
            "Output language",
            ["English", "Hindi", "Bilingual (Hindi + English)"],
        )
        tool = st.selectbox(
            "What would you like to create?",
            [
                "Summary",
                "Flashcards",
                "Quiz",
                "Exam questions",
                "Mind map",
                "Key points",
            ],
        )

        if st.button("Generate revision content", type="primary"):
            all_chunks = []
            for item in materials:
                all_chunks.extend(split_chunks(item["content"]))

            with st.spinner(f"Creating your {tool.lower()}…"):
                try:
                    query = f"{tool} for {subject}, topic {topic}"
                    selected = retrieve_context(query, all_chunks, top_k=8)
                    context = "\n\n".join(selected)
                    output = generate_revision(
                        tool, subject, topic, context, language
                    )
                    st.session_state.last_revision = {
                        "subject": subject,
                        "topic": topic,
                        "tool": tool,
                        "output": output,
                    }
                    add_progress(
                        user["id"], subject, topic, tool,
                        details=f"Generated {tool}"
                    )
                except Exception as exc:
                    st.error(f"Could not generate content: {exc}")

        result = st.session_state.get("last_revision")
        if result and result["subject"] == subject and result["topic"] == topic:
            st.markdown("---")
            st.subheader(result["tool"])
            st.markdown(result["output"])
            st.download_button(
                "Download revision notes",
                data=result["output"],
                file_name=(
                    f"{re.sub(r'[^A-Za-z0-9_-]+', '_', subject)}_"
                    f"{re.sub(r'[^A-Za-z0-9_-]+', '_', topic)}_"
                    f"{result['tool'].lower().replace(' ', '_')}.md"
                ),
                mime="text/markdown",
            )

    elif page == "YouTube finder":
        st.subheader("Discover related lectures")
        search_query = st.text_input(
            "Search topic",
            value=f"{subject} {topic}".strip() if ready else "",
        )
        count = st.slider("Number of videos", 3, 10, 5)

        if st.button("Find videos", type="primary"):
            try:
                with st.spinner("Searching YouTube…"):
                    results = search_youtube_videos(search_query, count)
                st.session_state.youtube_results = results
            except Exception as exc:
                st.error(str(exc))

        for video in st.session_state.get("youtube_results", []):
            with st.container(border=True):
                st.markdown(f"### [{video['title']}]({video['url']})")
                st.caption(f"Channel: {video['channel']}")
                st.write(
                    f"Views: {video['views']:,} · "
                    f"Likes: {video['likes']:,} · "
                    f"Comments: {video['comments']:,}"
                )
                st.write(video["description"][:350])
                if ready and st.button(
                    "Import transcript",
                    key="import_" + video["url"],
                ):
                    try:
                        transcript = fetch_youtube_transcript(video["url"])
                        save_material(
                            user["id"], subject, topic, "YouTube",
                            video["url"], transcript
                        )
                        add_progress(
                            user["id"], subject, topic,
                            "Imported transcript", details=video["title"]
                        )
                        st.success("Transcript saved to this topic.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not import transcript: {exc}")

    elif page == "Progress":
        with db() as conn:
            rows = conn.execute(
                """
                SELECT subject, topic, activity_type, score, details, created_at
                FROM study_progress
                WHERE user_id=?
                ORDER BY id DESC LIMIT 100
                """,
                (user["id"],),
            ).fetchall()

        st.subheader("Your recent study activity")
        if not rows:
            st.info("Your study activity will appear here.")
        else:
            for row in rows:
                with st.container(border=True):
                    st.markdown(
                        f"**{row['activity_type']}** · "
                        f"{row['subject']} / {row['topic']}"
                    )
                    if row["details"]:
                        st.caption(row["details"])
                    st.caption(row["created_at"])


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

user = current_user()

if user and user["verified"]:
    workspace(user)
else:
    auth_screen()
