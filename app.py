
import os
import re
import requests
import time
import streamlit as st

from urllib.parse import urlparse, parse_qs
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openrouter import ChatOpenRouter


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)


# ==================================================
# DARK THEME
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
    }

    .subtitle {
        color: #aab4c3;
        margin-bottom: 1.5rem;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-title">🎥 YouTube RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Paste a YouTube URL and ask questions about the video.'
    '</div>',
    unsafe_allow_html=True,
)


# ==================================================
# EXTRACT YOUTUBE VIDEO ID
# ==================================================

def extract_video_id(url):
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
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

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(
            "Invalid YouTube URL. Paste a valid video link."
        )

    return video_id


# ==================================================
# FETCH TRANSCRIPT THROUGH HOSTED API
# ==================================================

def _supadata_request(path, params=None, timeout=60):
    """Call Supadata using the API key stored in Streamlit Secrets."""
    try:
        api_key = st.secrets.get("SUPADATA_API_KEY", "")
    except Exception:
        api_key = os.environ.get("SUPADATA_API_KEY", "")
    api_key = api_key or os.environ.get("SUPADATA_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "SUPADATA_API_KEY is missing. Add it in Streamlit Cloud → "
            "Settings → Secrets."
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
        message = data.get("message") or data.get("error") or str(data)
        if response.status_code == 401:
            message = "Supadata API key is invalid. " + str(message)
        elif response.status_code == 402:
            message = "Supadata credits may be exhausted. " + str(message)
        elif response.status_code == 429:
            message = "Supadata rate limit reached. " + str(message)
        raise RuntimeError(f"Supadata HTTP {response.status_code}: {message}")
    return data


def _poll_supadata_job(job_id, timeout_seconds=180):
    """Wait for a Supadata asynchronous transcript job."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(2)
        result = _supadata_request(f"/transcript/{job_id}")
        status = str(result.get("status", "")).lower()
        if status in {"completed", "complete", "success", "succeeded"}:
            return result.get("result", result)
        if status in {"failed", "error"}:
            raise RuntimeError(
                "Supadata transcript job failed: "
                + str(result.get("error") or result.get("message") or "unknown error")
            )
    raise TimeoutError("Supadata transcript job timed out. Please try again.")


def _parse_supadata_response(data):
    """Normalize transcript response data into timestamped segments."""
    payload = data.get("result", data) if isinstance(data, dict) else data
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise RuntimeError("Supadata returned an empty transcript.")
        return [{"text": text, "start": 0.0, "end": 0.0}], "auto"

    content = payload.get("content", []) if isinstance(payload, dict) else payload
    language = (
        payload.get("lang") or payload.get("language")
        or data.get("lang", data.get("language", "auto"))
    ) if isinstance(payload, dict) and isinstance(data, dict) else "auto"

    if isinstance(content, str):
        content = [{"text": content, "start": 0, "duration": 0}]

    segments = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                text, start, end = item.strip(), 0.0, 0.0
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                start = float(item.get("start", item.get("offset", 0)) or 0)
                duration = float(item.get("duration", 0) or 0)
                end = float(item.get("end", start + duration) or (start + duration))
                # Offsets may be supplied in milliseconds.
                if start > 100000:
                    start /= 1000
                if end > 100000:
                    end /= 1000
            else:
                continue
            if text:
                segments.append({"text": text, "start": start, "end": end})

    if not segments:
        raise RuntimeError("Supadata returned no usable transcript segments.")
    return segments, language


def _fetch_supadata_transcript(video_url, mode, language="en"):
    params = {"url": video_url, "text": "true", "mode": mode}
    if language and language != "auto":
        params["lang"] = language
    data = _supadata_request("/transcript", params=params)
    if isinstance(data, dict) and data.get("jobId"):
        data = _poll_supadata_job(data["jobId"])
    return _parse_supadata_response(data)


def fetch_youtube_transcript(video_id):
    """Fetch captions first, then request provider-side AI audio transcription."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    errors = []

    # First attempt: retrieve existing captions.
    try:
        return _fetch_supadata_transcript(video_url, mode="native", language="en")
    except Exception as error:
        errors.append(f"Caption retrieval: {error}")

    # Fallback: ask Supadata to generate a transcript from the video's audio.
    # This does not require the user to upload a transcript or audio file.
    try:
        return _fetch_supadata_transcript(video_url, mode="generate", language="en")
    except Exception as error:
        errors.append(f"AI audio transcription: {error}")

    raise RuntimeError(
        "Could not retrieve captions or generate a transcript. Check that the "
        "video is publicly accessible, SUPADATA_API_KEY is valid, and credits "
        "are available. Details: " + " | ".join(errors)
    )


# ==================================================
# EMBEDDINGS
# ==================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


# ==================================================
# BUILD VECTOR INDEX
# ==================================================

def build_index(video_url):
    video_id = extract_video_id(video_url)

    segments, language = fetch_youtube_transcript(video_id)

    # Keep all transcript segments for full-video summaries.
    full_text_parts = []
    segment_ranges = []
    position = 0

    for segment in segments:
        text = segment["text"]

        segment_ranges.append(
            {
                "char_start": position,
                "char_end": position + len(text),
                "start": segment["start"],
                "end": segment["end"],
            }
        )

        full_text_parts.append(text)
        position += len(text) + 1

    full_text = " ".join(full_text_parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise RuntimeError(
            "Could not split the transcript into chunks."
        )

    documents = []
    search_position = 0

    for index, chunk in enumerate(chunks):
        chunk_start = full_text.find(
            chunk, search_position
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
            item for item in segment_ranges
            if (
                item["char_end"] >= chunk_start
                and item["char_start"] <= chunk_end
            )
        ]

        if overlapping:
            start_seconds = int(overlapping[0]["start"])
            end_seconds = int(overlapping[-1]["end"])
        else:
            start_seconds = 0
            end_seconds = 0

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
        "segment_count": len(segments),
        "chunk_count": len(documents),
        "vectorstore": vectorstore,
        "documents": documents,
    }


# ==================================================
# OPENROUTER LLM
# ==================================================

def get_llm():
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get(
                "OPENROUTER_API_KEY"
            )
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Configure it in Streamlit Cloud Secrets."
        )

    return ChatOpenRouter(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0,
    )


# ==================================================
# QUESTION ANSWERING PROMPT
# ==================================================

qa_prompt = ChatPromptTemplate.from_template(
    """
You are an intelligent YouTube video assistant.

Answer using the transcript context provided below.

Rules:
- Answer the question directly and clearly.
- Use all relevant information in the context.
- For broad questions, give a useful answer based on
  the available evidence.
- Do not invent facts or pretend to know content absent
  from the transcript.
- If some information is missing, answer the parts you
  can and briefly explain the limitation.
- Use headings or bullet points when useful.
- Do not include source links.

Transcript context:
{context}

Question:
{question}

Answer:
"""
)


# ==================================================
# FULL VIDEO SUMMARY
# ==================================================

def summarize_entire_video(documents, llm, batch_size=8):
    """
    Summarize every transcript chunk in batches, then combine
    the partial summaries into a final text-only answer.
    """

    if not documents:
        return "No transcript content is available."

    partial_summaries = []

    total_batches = (
        len(documents) + batch_size - 1
    ) // batch_size

    progress = st.progress(0)
    status = st.empty()

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]

        batch_context = "\n\n".join(
            doc.page_content for doc in batch
        )

        batch_prompt = ChatPromptTemplate.from_template(
            """
Summarize this portion of a YouTube transcript.

Capture the key ideas, explanations, examples, facts,
and conclusions. Do not invent information.

This is part {part_number} of {total_parts} of the video.
Preserve important details that may be needed in the
final summary.

Transcript:
{context}

Partial summary:
"""
        )

        chain = batch_prompt | llm

        result = chain.invoke(
            {
                "context": batch_context,
                "part_number": (
                    start // batch_size + 1
                ),
                "total_parts": total_batches,
            }
        )

        partial_summaries.append(result.content)

        completed = min(
            start + batch_size, len(documents)
        )

        progress.progress(
            completed / len(documents)
        )

        status.caption(
            f"Summarizing transcript: "
            f"{completed}/{len(documents)} chunks"
        )

    progress.empty()
    status.empty()

    # Hierarchical reduction prevents the final prompt from
    # growing indefinitely for very long videos.
    current_summaries = partial_summaries

    while len(current_summaries) > 1:
        next_summaries = []

        for start in range(0, len(current_summaries), 4):
            group = current_summaries[start:start + 4]

            if len(group) == 1:
                next_summaries.append(group[0])
                continue

            combined = "\n\n".join(group)

            reduce_prompt = ChatPromptTemplate.from_template(
                """
Combine these partial summaries into one coherent summary.

Preserve the important facts, key topics, examples, and
conclusions. Remove repetition without losing meaningful
information. Do not introduce unsupported facts.

Partial summaries:
{summaries}

Combined summary:
"""
            )

            chain = reduce_prompt | llm

            result = chain.invoke(
                {"summaries": combined}
            )

            next_summaries.append(result.content)

        current_summaries = next_summaries

    final_prompt = ChatPromptTemplate.from_template(
        """
Write a comprehensive summary of the complete YouTube video.

Include:
1. The main topic and purpose.
2. Major points in their original progression.
3. Important explanations and examples.
4. Key conclusions and takeaways.

Use clear headings and bullet points. Keep the summary
informative rather than excessively repetitive. Do not
invent information.

Video summaries:
{summaries}

Final summary:
"""
    )

    result = (
        final_prompt | llm
    ).invoke(
        {"summaries": current_summaries[0]}
    )

    return result.content


# ==================================================
# URL-ONLY SIDEBAR
# ==================================================

with st.sidebar:
    st.header("🎬 Video Input")

    video_url = st.text_input(
        "Paste YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    process_button = st.button(
        "Process Video",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "Paste a video URL, process its transcript, "
        "and ask questions."
    )


# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "messages": [],
    "vectorstore": None,
    "documents": [],
    "video_id": None,
    "video_url": None,
    "video_language": None,
    "segment_count": 0,
    "chunk_count": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# PROCESS VIDEO
# ==================================================

if process_button:
    if not video_url.strip():
        st.sidebar.warning(
            "Please paste a YouTube URL."
        )

    else:
        try:
            with st.spinner(
                "Fetching transcript and building the index..."
            ):
                result = build_index(video_url)

            st.session_state.vectorstore = result["vectorstore"]
            st.session_state.documents = result["documents"]
            st.session_state.video_id = result["video_id"]
            st.session_state.video_url = result["video_url"]
            st.session_state.video_language = result["language"]
            st.session_state.segment_count = result["segment_count"]
            st.session_state.chunk_count = result["chunk_count"]
            st.session_state.messages = []

            st.success(
                "Video processed successfully! "
                "You can now ask questions."
            )

        except Exception as error:
            st.error(str(error))


# ==================================================
# VIDEO INFORMATION
# ==================================================

if st.session_state.vectorstore is not None:
    st.success("✅ Video indexed and ready.")

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
        st.session_state.video_language,
    )

    st.video(st.session_state.video_url)

    st.divider()


# ==================================================
# DISPLAY CHAT HISTORY — TEXT ONLY
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==================================================
# CHAT INPUT AND ANSWERS
# ==================================================

question = st.chat_input(
    "Ask anything about the video..."
)

if question:
    if st.session_state.vectorstore is None:
        st.warning(
            "Please paste and process a YouTube URL first."
        )

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                llm = get_llm()

                question_lower = question.lower()

                summary_keywords = [
                    "summary",
                    "summarize",
                    "summarise",
                    "overview",
                    "main points",
                    "key points",
                    "what is this video about",
                    "explain the video",
                    "recap",
                ]

                is_summary_question = any(
                    keyword in question_lower
                    for keyword in summary_keywords
                )

                if is_summary_question:
                    with st.spinner(
                        "Creating a summary of the entire video..."
                    ):
                        answer = summarize_entire_video(
                            st.session_state.documents,
                            llm,
                            batch_size=8,
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
                                    12,
                                    len(st.session_state.documents),
                                ),
                            )
                        )

                        context = "\n\n".join(
                            doc.page_content
                            for doc in retrieved_docs
                        )

                    with st.spinner(
                        "Generating answer..."
                    ):
                        chain = qa_prompt | llm

                        response = chain.invoke(
                            {
                                "context": context,
                                "question": question,
                            }
                        )

                        answer = response.content

                # Display only the answer.
                st.markdown(answer)

                # Store text only; no sources or links.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:
                st.error(
                    f"Could not generate an answer: {error}"
                )
