"""
Lecture-to-Revision AI — Streamlit single-file app
=================================================
Run:
    streamlit run app.py

Optional Streamlit secrets (.streamlit/secrets.toml):
    OPENROUTER_API_KEY = "your-key"
    OPENROUTER_MODEL = "openai/gpt-4o-mini"
    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USERNAME = "your-email@gmail.com"
    SMTP_PASSWORD = "your-app-password"
    SMTP_FROM = "your-email@gmail.com"
    # For local development only; leave false in production:
    DEV_SHOW_OTP = true

OTP hashing intentionally does not use OTP_PEPPER, as requested.
The app uses the existing database path in Colab when it exists.
"""

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
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: linear-gradient(135deg,#f7f9ff 0%,#eef4ff 52%,#f8f5ff 100%); }
      [data-testid="stSidebar"] { background: #101b35; }
      [data-testid="stSidebar"] * { color: #f5f7ff !important; }
      .hero { padding: 1.35rem 1.5rem; border-radius: 20px;
        background: linear-gradient(120deg,#1e3a8a,#6d28d9); color: white;
        margin-bottom: 1rem; box-shadow: 0 12px 30px rgba(40,50,110,.16); }
      .hero h1 { color:white; margin:0; }
      .hero p { color:#e8eaff; margin:.4rem 0 0 0; }
      .metric-card { padding: 1rem; border-radius: 16px; background:white;
        border:1px solid #e1e8fb; box-shadow:0 5px 18px rgba(30,58,138,.06); }
      div.stButton > button { border-radius: 10px; font-weight: 600; }
      .small-note { color:#64748b; font-size:.88rem; }
    </style>
