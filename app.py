import os
import re
import time
import requests
import streamlit as st

from urllib.parse import urlparse, parse_qs
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lecture to Revision AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top left,
            rgba(109, 93, 252, 0.16), transparent 40%),
            #0b0e14;
        color: #f3f4f6;
    }

    [data-testid="stSidebar"] {
        background-color: #111620;
        border-right: 1px solid #252b3a;
    }

    .hero {
        padding: 25px;
        border: 1px solid #39315f;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            rgba(109, 93, 252, 0.20),
            rgba(67, 135, 245, 0.10)
        );
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #b7bfd1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid #292f40;
        padding: 15px;
        border-radius: 12px;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #292f40;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    .muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>🎓 Lecture to Revision AI</h1>
        <p>
            Convert YouTube lectures into searchable learning material.
            Ask questions, generate revision notes, and summarize videos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "messages": [],
    "vectorstore": None,
    "documents": [],
    "video_id": None,
    "video_url": None,
    "video_language": None,
    "segment_count": 0,
    "chunk_count": 0,
    "full_summary": None,
    "video_url_input": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SECRETS AND ENVIRONMENT VARIABLES
# ============================================================

def get_secret(name, default=None):
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return value or os.environ.get(name, default)


# ============================================================
# YOUTUBE URL VALIDATION
# ============================================================

def extract_video_id(url):
    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(
                parsed.query
            ).get("v", [""])[0]

        elif parsed.path.startswith(
            ("/embed/", "/shorts/", "/live/")
        ):
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""

        else:
            video_id = ""

    else:
        video_id = ""

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{11}", video_id or ""
    ):
        raise ValueError(
            "Invalid YouTube URL. Please enter a valid video link."
        )

    return video_id


# ============================================================
# SUPADATA TRANSCRIPT API
# ============================================================

def supadata_request(path, params=None, timeout=60):
    api_key = get_secret("SUPADATA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "SUPADATA_API_KEY is missing. Add it to "
            "Streamlit Cloud → Settings → Secrets."
        )

    response = requests.get(
        f"https://api.supadata.ai/v1{path}",
        headers={"x-api-key": api_key},
        params=params,
        timeout=timeout,
    )

    try:
        data = response.json()
    except ValueError:
        data = {"message": response.text[:500]}

    if not response.ok:
        message = (
            data.get("message")
            or data.get("error")
            or str(data)
        )

        if response.status_code == 401:
            message = "Invalid Supadata API key."
        elif response.status_code == 402:
            message = (
                "Supadata credits are exhausted or billing "
                "is required."
            )
        elif response.status_code == 429:
            message = (
                "Supadata rate limit reached. Please try later."
            )

        raise RuntimeError(
            f"Transcript API error ({response.status_code}): "
            f"{message}"
        )

    return data


def poll_supadata_job(job_id, timeout_seconds=180):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        time.sleep(2)

        result = supadata_request(
            f"/transcript/{job_id}"
        )

        status = str(
            result.get("status", "")
        ).lower()

        if status in {
            "completed",
            "complete",
            "success",
            "succeeded",
        }:
            return result.get("result", result)

        if status in {"failed", "error"}:
            raise RuntimeError(
                "Transcript job failed: "
                + str(
                    result.get("error")
                    or result.get("message")
                    or "Unknown error"
                )
            )

    raise TimeoutError(
        "Transcript generation timed out. Please try again."
    )


def parse_transcript(data):
    payload = (
        data.get("result", data)
        if isinstance(data, dict)
        else data
    )

    if isinstance(payload, str):
        text = payload.strip()

        if not text:
            raise RuntimeError(
                "The transcript returned by the provider is empty."
            )

        return [
            {
                "text": text,
                "start": 0.0,
                "end": 0.0,
            }
        ], "auto"

    content = (
        payload.get("content", [])
        if isinstance(payload, dict)
        else payload
    )

    language = (
        payload.get("lang")
        or payload.get("language")
        or "auto"
    ) if isinstance(payload, dict) else "auto"

    if isinstance(content, str):
        content = [
            {
                "text": content,
                "start": 0,
                "duration": 0,
            }
        ]

    segments = []

    if isinstance(content, list):
        for item in content:

            if isinstance(item, str):
                text = item.strip()
                start = 0.0
                end = 0.0

            elif isinstance(item, dict):
                text = str(
                    item.get("text", "")
                ).strip()

                start = float(
                    item.get(
                        "start",
                        item.get("offset", 0),
                    ) or 0
                )

                duration = float(
                    item.get("duration", 0) or 0
                )

                end = float(
                    item.get(
                        "end",
                        start + duration,
                    ) or (start + duration)
                )

                if start > 100000:
                    start /= 1000

                if end > 100000:
                    end /= 1000

            else:
                continue

            if text:
                segments.append({
                    "text": text,
                    "start": start,
                    "end": end,
                })

    if not segments:
        raise RuntimeError(
            "No usable transcript segments were returned."
        )

    return segments, language


def fetch_supadata_transcript(video_url, mode="native"):
    params = {
        "url": video_url,
        "text": "true",
        "mode": mode,
    }

    data = supadata_request(
        "/transcript",
        params=params,
    )

    if isinstance(data, dict) and data.get("jobId"):
        data = poll_supadata_job(data["jobId"])

    return parse_transcript(data)


def fetch_youtube_transcript(video_id):
    video_url = (
        f"https://www.youtube.com/watch?v={video_id}"
    )

    errors = []

    # First attempt: existing captions.
    try:
        return fetch_supadata_transcript(
            video_url,
            mode="native",
        )

    except Exception as error:
        errors.append(
            f"Existing captions: {error}"
        )

    # Second attempt: provider-generated transcript from audio.
    try:
        return fetch_supadata_transcript(
            video_url,
            mode="generate",
        )

    except Exception as error:
        errors.append(
            f"Audio transcription: {error}"
        )

    raise RuntimeError(
        "Unable to retrieve or generate a transcript. "
        "Check video accessibility, API key, provider support, "
        "and available credits.\n\n"
        + "\n".join(errors)
    )


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


# ============================================================
# BUILD TRANSCRIPT SEARCH INDEX
# ============================================================

def build_index(video_url):
    video_id = extract_video_id(video_url)

    segments, language = fetch_youtube_transcript(
        video_id
    )

    full_text_parts = []
    segment_ranges = []
    position = 0

    for segment in segments:
        text = segment["text"].strip()

        if not text:
            continue

        segment_ranges.append({
            "char_start": position,
            "char_end": position + len(text),
            "start": segment["start"],
            "end": segment["end"],
        })

        full_text_parts.append(text)
        position += len(text) + 1

    full_text = " ".join(full_text_parts)

    if not full_text.strip():
        raise RuntimeError(
            "The retrieved transcript is empty."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise RuntimeError(
            "Could not create searchable transcript chunks."
        )

    documents = []
    search_position = 0

    for index, chunk in enumerate(chunks):

        chunk_start = full_text.find(
            chunk,
            search_position,
        )

        if chunk_start == -1:
            chunk_start = full_text.find(chunk)

        if chunk_start == -1:
            chunk_start = 0

        chunk_end = chunk_start + len(chunk)

        search_position = max(
            chunk_start + 1,
            chunk_end - 150,
        )

        overlapping = [
            segment for segment in segment_ranges
            if (
                segment["char_end"] >= chunk_start
                and segment["char_start"] <= chunk_end
            )
        ]

        start_seconds = (
            int(overlapping[0]["start"])
            if overlapping else 0
        )

        end_seconds = (
            int(overlapping[-1]["end"])
            if overlapping else 0
        )

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "video_id": video_id,
                    "chunk_index": index,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "language": language,
                },
            )
        )

    vectorstore = FAISS.from_documents(
        documents,
        get_embeddings(),
    )

    return {
        "video_id": video_id,
        "video_url": (
            f"https://www.youtube.com/watch?v={video_id}"
        ),
        "language": language,
        "segment_count": len(full_text_parts),
        "chunk_count": len(documents),
        "vectorstore": vectorstore,
        "documents": documents,
    }


# ============================================================
# OPENROUTER LANGUAGE MODEL
# ============================================================

def get_llm():
    api_key = get_secret("OPENROUTER_API_KEY")

    model = get_secret(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini",
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to "
            "Streamlit Cloud → Settings → Secrets."
        )

    # OpenRouter provides an OpenAI-compatible API.
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

qa_prompt = ChatPromptTemplate.from_template(
    """
You are an intelligent YouTube lecture assistant.

Answer using the transcript context below.

Rules:
- Answer directly and clearly.
- Use relevant evidence from the transcript.
- Do not invent facts.
- If the transcript does not contain the answer,
  explain that the information is unavailable.
- Use headings and bullet points when useful.
- Respond in the same language as the question when possible.

Transcript context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# SUMMARIZE THE ENTIRE VIDEO
# ============================================================

def summarize_entire_video(
    documents,
    llm,
    batch_size=8,
):
    if not documents:
        return "No transcript content is available."

    partial_summaries = []

    total_batches = (
        len(documents) + batch_size - 1
    ) // batch_size

    progress = st.progress(0)
    status = st.empty()

    for start in range(
        0,
        len(documents),
        batch_size,
    ):
        batch = documents[
            start:start + batch_size
        ]

        context = "\n\n".join(
            document.page_content
            for document in batch
        )

        prompt = ChatPromptTemplate.from_template(
            """
Summarize this portion of a lecture transcript.

Include important concepts, explanations, examples,
facts, and conclusions. Do not invent information.

Part {part_number} of {total_parts}

Transcript:
{context}

Partial summary:
"""
        )

        response = (prompt | llm).invoke({
            "context": context,
            "part_number": start // batch_size + 1,
            "total_parts": total_batches,
        })

        partial_summaries.append(
            str(response.content)
        )

        completed = min(
            start + batch_size,
            len(documents),
        )

        progress.progress(
            completed / len(documents)
        )

        status.caption(
            f"Summarizing: {completed}/{len(documents)} chunks"
        )

    progress.empty()
    status.empty()

    # Combine partial summaries in smaller groups.
    summaries = partial_summaries

    while len(summaries) > 1:
        combined_summaries = []

        for start in range(
            0,
            len(summaries),
            4,
        ):
            group = summaries[start:start + 4]

            if len(group) == 1:
                combined_summaries.append(group[0])
                continue

            prompt = ChatPromptTemplate.from_template(
                """
Combine these lecture summaries.

Preserve important concepts, facts, examples,
and conclusions. Remove repetition without losing
meaningful information. Do not invent facts.

Summaries:
{summaries}

Combined summary:
"""
            )

            response = (prompt | llm).invoke({
                "summaries": "\n\n".join(group)
            })

            combined_summaries.append(
                str(response.content)
            )

        summaries = combined_summaries

    final_prompt = ChatPromptTemplate.from_template(
        """
Create a comprehensive revision summary of the lecture.

Include:
1. Main topic and purpose.
2. Major concepts in logical order.
3. Important definitions and explanations.
4. Examples mentioned in the lecture.
5. Key takeaways for revision.

Use clear headings and bullet points.
Do not invent information.

Lecture summaries:
{summaries}

Final revision summary:
"""
    )

    response = (final_prompt | llm).invoke({
        "summaries": summaries[0]
    })

    return str(response.content)


def is_summary_question(question):
    keywords = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "main points",
        "key points",
        "what is this video about",
        "explain the video",
        "recap",
        "revision notes",
        "summarize the lecture",
    ]

    question = question.lower()

    return any(
        keyword in question
        for keyword in keywords
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎬 Video input")

    st.caption(
        "Paste a YouTube URL to begin."
    )

    video_url_input = st.text_input(
        "YouTube video URL",
        key="video_url_input",
        placeholder=(
            "https://www.youtube.com/watch?v=..."
        ),
    )

    process_button = st.button(
        "⚡ Process video",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        "1. Paste a YouTube URL\n"
        "2. Retrieve or generate a transcript\n"
        "3. Build a searchable index\n"
        "4. Ask questions or summarize the lecture"
    )

    st.caption(
        "Transcript generation depends on provider "
        "support, video accessibility, and available credits."
    )

    if st.session_state.video_id:
        st.divider()
        st.markdown("### Current video")

        st.markdown(
            f"[Open on YouTube]({st.session_state.video_url})"
        )

        if st.button(
            "🧹 Clear current video",
            use_container_width=True,
        ):
            for key, value in DEFAULTS.items():
                st.session_state[key] = value

            st.rerun()


# ============================================================
# PROCESS VIDEO
# ============================================================

if process_button:
    if not video_url_input.strip():
        st.sidebar.warning(
            "Please paste a YouTube URL first."
        )

    else:
        try:
            with st.spinner(
                "Retrieving transcript and building search index..."
            ):
                result = build_index(
                    video_url_input
                )

            st.session_state.vectorstore = (
                result["vectorstore"]
            )

            st.session_state.documents = (
                result["documents"]
            )

            st.session_state.video_id = (
                result["video_id"]
            )

            st.session_state.video_url = (
                result["video_url"]
            )

            st.session_state.video_language = (
                result["language"]
            )

            st.session_state.segment_count = (
                result["segment_count"]
            )

            st.session_state.chunk_count = (
                result["chunk_count"]
            )

            st.session_state.messages = []
            st.session_state.full_summary = None

            st.success(
                "Video processed successfully!"
            )

        except Exception as error:
            st.error(
                f"Could not process this video: {error}"
            )


# ============================================================
# MAIN UI
# ============================================================

if st.session_state.vectorstore is None:

    left, right = st.columns(
        [1.3, 1]
    )

    with left:
        st.markdown("### 👋 Welcome!")

        st.write(
            "Start by entering a YouTube URL in the sidebar "
            "and clicking **Process video**."
        )

        st.markdown(
            """
            **Features**
            - Direct YouTube URL processing
            - Transcript-based question answering
            - Semantic search using FAISS
            - Whole-video summaries
            - English, Hindi, and Marathi text support
            """
        )

    with right:
        st.markdown("### 💡 Example questions")

        st.info(
            "What is this video about?\n\n"
            "What are the key concepts?\n\n"
            "Explain the main example.\n\n"
            "Summarize the complete lecture."
        )

else:

    st.markdown("### 📚 Your lecture workspace")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transcript segments",
        st.session_state.segment_count,
    )

    col2.metric(
        "Search chunks",
        st.session_state.chunk_count,
    )

    col3.metric(
        "Language",
        st.session_state.video_language or "auto",
    )

    st.video(
        st.session_state.video_url
    )

    st.divider()

    st.markdown("### 💬 Ask about your lecture")

    st.caption(
        "Ask a specific question or request a complete summary."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT AND RESPONSE
# ============================================================

question = st.chat_input(
    "Ask a question about the video..."
)

if question:

    if st.session_state.vectorstore is None:
        st.warning(
            "Please process a YouTube video first."
        )

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                llm = get_llm()

                if is_summary_question(question):

                    with st.spinner(
                        "Summarizing the entire lecture..."
                    ):
                        if (
                            st.session_state.full_summary
                            is None
                        ):
                            st.session_state.full_summary = (
                                summarize_entire_video(
                                    st.session_state.documents,
                                    llm,
                                    batch_size=8,
                                )
                            )

                        answer = (
                            st.session_state.full_summary
                        )

                else:
                    with st.spinner(
                        "Searching the transcript..."
                    ):
                        retrieved_docs = (
                            st.session_state.vectorstore
                            .similarity_search(
                                question,
                                k=min(
                                    8,
                                    len(
                                        st.session_state.documents
                                    ),
                                ),
                            )
                        )

                        context = "\n\n".join(
                            doc.page_content
                            for doc in retrieved_docs
                        )

                    if not context.strip():
                        answer = (
                            "I couldn't find relevant information "
                            "in the lecture transcript."
                        )

                    else:
                        with st.spinner(
                            "Generating answer..."
                        ):
                            response = (
                                qa_prompt | llm
                            ).invoke({
                                "context": context,
                                "question": question,
                            })

                            answer = str(
                                response.content
                            )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

            except Exception as error:
                st.error(
                    f"Could not generate an answer: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lecture to Revision AI · Verify important information "
    "against the original lecture."
)
