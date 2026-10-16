
import os
import re
import json
import sqlite3
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import streamlit as st
import fitz
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "app_data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "app.db"

st.set_page_config(
    page_title="Lecture to Revision AI",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #14112b);
    color: #f4f4ff;
}

[data-testid="stSidebar"] {
    background: #10152a;
}

h1, h2, h3 {
    color: #c4b5fd;
}

.stButton button {
    background: linear-gradient(90deg, #6d5dfc, #4387f5);
    color: white;
    border: 0;
    border-radius: 10px;
}

div[data-testid="stChatMessage"] {
    border: 1px solid #35345a;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATABASE
# =========================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def save_document(source, title, content):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO documents (source, title, content)
            VALUES (?, ?, ?)
            """,
            (source, title, content)
        )


def load_documents():
    with db() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM documents ORDER BY id DESC"
            ).fetchall()
        ]


# =========================================================
# OPENROUTER LLM
# =========================================================

def get_client():
    key = None

    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        key = os.getenv("OPENROUTER_API_KEY")

    if not key:
        st.error(
            "OpenRouter API key is missing. Add "
            "OPENROUTER_API_KEY to Streamlit Cloud Secrets."
        )
        st.stop()

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )


def ask_model(system_prompt, user_prompt):
    client = get_client()

    model = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    try:
        model = st.secrets.get("OPENROUTER_MODEL", model)
    except Exception:
        pass

    result = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return result.choices[0].message.content or ""


# =========================================================
# YOUTUBE URL VALIDATION
# =========================================================

def youtube_id(url):
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()

    video_id = None

    if host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com"
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(
                parsed.query
            ).get("v", [None])[0]

        elif parsed.path.startswith(
            ("/embed/", "/shorts/", "/live/")
        ):
            segments = parsed.path.strip("/").split("/")

            if len(segments) >= 2:
                video_id = segments[1]

    elif host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]

    if not video_id or not re.fullmatch(
        r"[A-Za-z0-9_-]{11}", video_id
    ):
        raise ValueError(
            "Enter a valid YouTube video URL."
        )

    return video_id


# =========================================================
# YOUTUBE TRANSCRIPT RETRIEVAL
# =========================================================

def transcript_text(url, language):
    vid = youtube_id(url)

    codes = {
        "English": ["en"],
        "Hindi": ["hi", "en"],
        "Marathi": ["mr", "hi", "en"]
    }

    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(
            vid,
            languages=codes[language]
        )

        parts = []

        for item in transcript:
            text = item.text.strip()

            if text:
                timestamp = int(item.start)

                parts.append(
                    f"[{timestamp // 60:02d}:"
                    f"{timestamp % 60:02d}] {text}"
                )

        if not parts:
            raise ValueError(
                "No transcript text was found."
            )

        return vid, "\n".join(parts)

    except Exception as exc:
        raise RuntimeError(
            "Automatic transcript retrieval failed. YouTube may be "
            "blocking the app's cloud IP, or captions may be unavailable "
            "in the selected language. Try another language or upload a "
            "TXT/SRT transcript below. Details: "
            + str(exc)[:300]
        ) from exc


# =========================================================
# MANUAL TRANSCRIPT FILE PROCESSING
# =========================================================

def transcript_file_text(upload):
    filename = upload.name.lower()

    try:
        text = upload.getvalue().decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Could not decode the file. Save it as UTF-8 text "
            "and upload it again."
        ) from exc

    if not text:
        raise ValueError("The transcript file is empty.")

    if filename.endswith(".srt"):
        # Remove SRT subtitle sequence numbers.
        text = re.sub(
            r"(?m)^\s*\d+\s*$",
            "",
            text
        )

        # Remove SRT timestamps.
        text = re.sub(
            r"(?m)^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
            r"\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
            r".*$",
            "",
            text
        )

        # Remove common WebVTT-style tags if present.
        text = re.sub(r"<[^>]+>", "", text)

    # Normalize excess whitespace without joining every line.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError(
            "No readable transcript text was found."
        )

    return text


# =========================================================
# PDF PROCESSING
# =========================================================

def pdf_text(upload):
    doc = fitz.open(
        stream=upload.getvalue(),
        filetype="pdf"
    )

    pages = []

    for number, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if text:
            pages.append(
                f"[Page {number}]\n{text}"
            )

    doc.close()

    if not pages:
        raise ValueError(
            "No selectable text found in this PDF. "
            "Scanned PDFs may require OCR."
        )

    return "\n\n".join(pages)


# =========================================================
# TEXT CHUNKING AND RETRIEVAL
# =========================================================

def split_chunks(text, size=1100, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def retrieve(text, query, count=5):
    chunks = split_chunks(text)

    terms = set(
        re.findall(r"\w+", query.lower())
    )

    scored = []

    for chunk in chunks:
        words = re.findall(
            r"\w+",
            chunk.lower()
        )

        wordset = set(words)

        score = sum(
            1 for term in terms
            if term in wordset
        )

        if score:
            scored.append((score, chunk))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        chunk for _, chunk in scored[:count]
    ]


# =========================================================
# APPLICATION HEADER AND SIDEBAR
# =========================================================

st.title("🎓 Lecture to Revision AI")

st.caption(
    "Turn lectures and PDFs into revision notes, "
    "answers and practice."
)

with st.sidebar:
    st.header("⚙️ Study settings")

    language = st.selectbox(
        "Response language",
        ["English", "Hindi", "Marathi"]
    )

    st.divider()

    st.caption(
        "Study materials are stored in this app's local database. "
        "On Streamlit Cloud, local storage may not be permanent."
    )


# =========================================================
# TABS
# =========================================================

tab_add, tab_study, tab_notes, tab_cards, tab_quiz = st.tabs([
    "📥 Add material",
    "💬 Ask AI",
    "📝 Revision notes",
    "🃏 Flashcards",
    "✅ MCQ practice"
])


# =========================================================
# TAB 1: ADD MATERIAL
# =========================================================

with tab_add:

    # -----------------------------------------------------
    # YOUTUBE VIDEO
    # -----------------------------------------------------

    st.subheader("Add a YouTube lecture")

    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if st.button("Process YouTube lecture"):
        try:
            with st.spinner("Fetching transcript..."):
                vid, text = transcript_text(
                    url,
                    language
                )

                save_document(
                    "youtube",
                    vid,
                    text
                )

            st.success(
                f"Transcript indexed successfully "
                f"({len(text):,} characters)."
            )

            st.rerun()

        except Exception as exc:
            st.error(
                f"Could not process this video: {exc}"
            )

            st.info(
                "If YouTube is blocking automatic retrieval, "
                "use the transcript upload option below."
            )

    # -----------------------------------------------------
    # MANUAL TRANSCRIPT UPLOAD
    # -----------------------------------------------------

    st.divider()

    st.subheader("Upload a transcript instead")

    st.caption(
        "Use this option if automatic YouTube transcript retrieval "
        "fails. Supported formats: TXT and SRT."
    )

    transcript_file = st.file_uploader(
        "Choose a transcript file",
        type=["txt", "srt"],
        key="transcript_upload"
    )

    if transcript_file and st.button(
        "Process transcript file"
    ):
        try:
            with st.spinner("Processing transcript..."):
                text = transcript_file_text(
                    transcript_file
                )

                save_document(
                    "transcript_upload",
                    transcript_file.name,
                    text
                )

            st.success(
                f"Transcript imported successfully "
                f"({len(text):,} characters)."
            )

            st.rerun()

        except Exception as exc:
            st.error(
                f"Could not process transcript file: {exc}"
            )

    # -----------------------------------------------------
    # PDF UPLOAD
    # -----------------------------------------------------

    st.divider()

    st.subheader("Upload a PDF")

    uploaded = st.file_uploader(
        "Choose a text-based PDF",
        type=["pdf"],
        key="pdf_upload"
    )

    if uploaded and st.button("Process PDF"):
        try:
            with st.spinner("Extracting PDF text..."):
                text = pdf_text(uploaded)

                save_document(
                    "pdf",
                    uploaded.name,
                    text
                )

            st.success(
                f"PDF processed ({len(text):,} characters)."
            )

            st.rerun()

        except Exception as exc:
            st.error(
                f"Could not process PDF: {exc}"
            )

    # -----------------------------------------------------
    # INDEXED MATERIALS
    # -----------------------------------------------------

    st.divider()

    st.subheader("Indexed materials")

    documents = load_documents()

    if not documents:
        st.info(
            "No materials yet. Add a YouTube lecture, "
            "transcript, or PDF above."
        )

    for doc in documents:
        st.markdown(
            f"**{doc['title']}** · {doc['source']}"
        )


# =========================================================
# TAB 2: ASK AI
# =========================================================

with tab_study:

    st.subheader(
        "Ask questions about your material"
    )

    documents = load_documents()

    if not documents:
        st.info(
            "Add a lecture, transcript, or PDF first."
        )

    else:
        options = {
            f"{d['title']} · {d['source']} · #{d['id']}": d
            for d in documents
        }

        selected_label = st.selectbox(
            "Choose material",
            list(options)
        )

        selected = options[selected_label]

        if "chat" not in st.session_state:
            st.session_state.chat = []

        # Keep conversation history separate for each document.
        chat_key = f"chat_{selected['id']}"

        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask a question about this lecture..."
        )

        if question:
            relevant = retrieve(
                selected["content"],
                question
            )

            context = "\n\n".join(relevant)

            if not context:
                st.warning(
                    "No matching passage found. "
                    "Try different keywords."
                )

            else:
                st.session_state[chat_key].append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner(
                        "Finding evidence and preparing an answer..."
                    ):
                        try:
                            answer = ask_model(
                                "You are a student revision assistant. "
                                "Answer only from the supplied evidence. "
                                f"Respond in {language}. "
                                "If evidence is insufficient, say so. "
                                "Do not invent facts, citations, "
                                "or timestamps.",

                                f"Evidence:\n{context}\n\n"
                                f"Question: {question}"
                            )

                            st.markdown(answer)

                            st.session_state[chat_key].append(
                                {
                                    "role": "assistant",
                                    "content": answer
                                }
                            )

                            with st.expander(
                                "Retrieved source passages"
                            ):
                                for i, chunk in enumerate(
                                    relevant, 1
                                ):
                                    st.markdown(
                                        f"**Source {i}**"
                                    )
                                    st.write(chunk)

                        except Exception as exc:
                            st.error(
                                f"Answer generation failed: {exc}"
                            )


# =========================================================
# TAB 3: REVISION NOTES
# =========================================================

with tab_notes:

    st.subheader("Generate revision notes")

    documents = load_documents()

    if documents:
        options = {
            f"{d['title']} · #{d['id']}": d
            for d in documents
        }

        label = st.selectbox(
            "Material for notes",
            list(options),
            key="notes_doc"
        )

        doc = options[label]

        if st.button("Generate notes"):
            with st.spinner(
                "Creating student-friendly notes..."
            ):
                try:
                    chunks = split_chunks(
                        doc["content"]
                    )

                    context = "\n\n".join(
                        chunks[:15]
                    )

                    notes = ask_model(
                        "Create clear revision notes in "
                        + language
                        + ". Use headings, definitions, key points "
                        "and a recap. Use only the supplied material.",
                        context
                    )

                    st.session_state.latest_notes = notes

                except Exception as exc:
                    st.error(
                        f"Could not generate notes: {exc}"
                    )

        if st.session_state.get("latest_notes"):
            st.markdown(
                st.session_state.latest_notes
            )

            st.download_button(
                "Download notes as Markdown",
                st.session_state.latest_notes,
                file_name="revision_notes.md",
                mime="text/markdown"
            )

    else:
        st.info(
            "Add a lecture, transcript, or PDF first."
        )


# =========================================================
# TAB 4: FLASHCARDS
# =========================================================

with tab_cards:

    st.subheader("Create flashcards")

    documents = load_documents()

    if documents:
        options = {
            f"{d['title']} · #{d['id']}": d
            for d in documents
        }

        label = st.selectbox(
            "Material for cards",
            list(options),
            key="cards_doc"
        )

        doc = options[label]

        count = st.slider(
            "Number of cards",
            3,
            20,
            10
        )

        if st.button("Generate flashcards"):
            try:
                with st.spinner(
                    "Creating flashcards..."
                ):
                    raw = ask_model(
                        "Create educational flashcards from the material. "
                        "Respond in "
                        + language
                        + ". Return valid JSON only: an array of objects "
                        "with keys question and answer. "
                        "Do not invent facts.",

                        f"Create {count} flashcards from:\n"
                        + "\n\n".join(
                            split_chunks(doc["content"])[:15]
                        )
                    )

                    raw = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        raw.strip()
                    )

                    cards = json.loads(raw)

                    if not isinstance(cards, list):
                        raise ValueError(
                            "The model did not return a JSON array."
                        )

                    st.session_state.cards = cards[:count]

            except Exception as exc:
                st.error(
                    f"Could not create flashcards: {exc}"
                )

        for i, card in enumerate(
            st.session_state.get("cards", []), 1
        ):
            with st.expander(
                f"Card {i}: {card.get('question', 'Question')}"
            ):
                st.write(
                    card.get("answer", "")
                )

    else:
        st.info(
            "Add a lecture, transcript, or PDF first."
        )


# =========================================================
# TAB 5: MCQ PRACTICE
# =========================================================

with tab_quiz:

    st.subheader(
        "Practice with multiple-choice questions"
    )

    documents = load_documents()

    if documents:
        options = {
            f"{d['title']} · #{d['id']}": d
            for d in documents
        }

        label = st.selectbox(
            "Material for quiz",
            list(options),
            key="quiz_doc"
        )

        doc = options[label]

        count = st.slider(
            "Number of questions",
            3,
            15,
            5
        )

        if st.button("Generate MCQ quiz"):
            try:
                with st.spinner(
                    "Preparing your quiz..."
                ):
                    raw = ask_model(
                        "Create multiple-choice questions in "
                        + language
                        + " based only on the material. "
                        "Return valid JSON only, an array with "
                        "question, options (four strings), "
                        "correct_answer and explanation. "
                        "correct_answer must exactly match an option.",

                        f"Create {count} questions from:\n"
                        + "\n\n".join(
                            split_chunks(doc["content"])[:15]
                        )
                    )

                    raw = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        raw.strip()
                    )

                    quiz = json.loads(raw)

                    if not isinstance(quiz, list):
                        raise ValueError(
                            "The model did not return a JSON array."
                        )

                    st.session_state.quiz = quiz[:count]

            except Exception as exc:
                st.error(
                    f"Could not generate quiz: {exc}"
                )

        if st.session_state.get("quiz"):
            with st.form("quiz_form"):
                answers = {}

                for i, q in enumerate(
                    st.session_state.quiz
                ):
                    st.markdown(
                        f"**{i + 1}. {q['question']}**"
                    )

                    answers[i] = st.radio(
                        "Choose an answer",
                        q["options"],
                        key=f"quiz_answer_{i}",
                        index=None
                    )

                submitted = st.form_submit_button(
                    "Check answers"
                )

            if submitted:
                score = 0

                for i, q in enumerate(
                    st.session_state.quiz
                ):
                    correct = (
                        answers[i] == q["correct_answer"]
                    )

                    score += int(correct)

                    st.markdown(
                        f"**Question {i + 1}:** "
                        + (
                            "✅ Correct"
                            if correct
                            else (
                                "❌ Correct answer: "
                                + str(q["correct_answer"])
                            )
                        )
                    )

                    st.caption(
                        q.get("explanation", "")
                    )

                st.metric(
                    "Your score",
                    f"{score}/{len(st.session_state.quiz)}"
                )

    else:
        st.info(
            "Add a lecture, transcript, or PDF first."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Lecture to Revision AI · Verify important answers "
    "against the original material."
)
