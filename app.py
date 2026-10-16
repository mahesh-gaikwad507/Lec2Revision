import os
import re
import time
import json
import hashlib
from typing import List, Dict, Optional

import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="One Night Study",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 One Night Study")
st.caption(
    "Find educational videos, combine their transcripts, and create summaries, "
    "notes, mind maps, flashcards, quizzes, and transcript-grounded answers."
)

# ----------------------------- Secrets and constants -----------------------------
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"
SUPADATA_BASE_URL = "https://api.supadata.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

YOUTUBE_API_KEY = get_secret("YOUTUBE_API_KEY")
SUPADATA_API_KEY = get_secret("SUPADATA_API_KEY")
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")

# ----------------------------- Cached resources/data -----------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run can take a little while)...")
def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load once per Streamlit server process instead of on every rerun."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def load_llm(api_key: str, model_name: str = DEFAULT_MODEL):
    """Reuse the API client across Streamlit reruns."""
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
        max_retries=2,
        timeout=90,
    )

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def youtube_search(subject: str, topic: str, language: str, max_results: int = 5) -> List[Dict]:
    """Cache YouTube search responses for 24 hours."""
    if not YOUTUBE_API_KEY:
        raise RuntimeError("Add YOUTUBE_API_KEY to Streamlit Secrets.")
    query = " ".join(part.strip() for part in [subject, topic] if part.strip())
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max(1, min(max_results, 10)),
        "relevanceLanguage": language[:2].lower() if language else "en",
        "key": YOUTUBE_API_KEY,
        "safeSearch": "moderate",
    }
    response = requests.get(YOUTUBE_API_URL, params=params, timeout=25)
    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"YouTube API error ({response.status_code}): {detail}")
    items = response.json().get("items", [])
    results = []
    for item in items:
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not video_id:
            continue
        results.append({
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": snippet.get("title", "Untitled video"),
            "channel": snippet.get("channelTitle", "Unknown channel"),
            "description": snippet.get("description", ""),
            "published_at": snippet.get("publishedAt", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        })
    return results

@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def fetch_transcript_cached(video_id: str, api_key: str) -> str:
    """Cache transcript text for one week. Uses Supadata transcript endpoint."""
    if not api_key:
        raise RuntimeError("Add SUPADATA_API_KEY to Streamlit Secrets.")
    headers = {"x-api-key": api_key}
    response = requests.get(
        f"{SUPADATA_BASE_URL}/transcript",
        params={"url": f"https://www.youtube.com/watch?v={video_id}", "text": "true"},
        headers=headers,
        timeout=60,
    )
    if response.status_code not in (200, 202):
        try:
            detail = response.json().get("message") or response.json().get("error") or response.text
        except Exception:
            detail = response.text
        raise RuntimeError(f"Supadata transcript error ({response.status_code}): {detail}")
    payload = response.json()

    # Supadata may return transcript immediately or an async job ID.
    if response.status_code == 202 or payload.get("jobId") or payload.get("job_id"):
        job_id = payload.get("jobId") or payload.get("job_id")
        if not job_id:
            raise RuntimeError("Supadata accepted the request but returned no job ID.")
        deadline = time.time() + 150
        while time.time() < deadline:
            time.sleep(3)
            poll = requests.get(
                f"{SUPADATA_BASE_URL}/transcript/{job_id}",
                headers=headers,
                timeout=30,
            )
            if poll.status_code not in (200, 202):
                continue
            payload = poll.json()
            status = str(payload.get("status", "")).lower()
            if status in ("failed", "error"):
                raise RuntimeError(payload.get("error", "Supadata transcript job failed."))
            if status in ("completed", "complete", "success", "succeeded") or payload.get("content"):
                break
        else:
            raise RuntimeError("Timed out waiting for Supadata transcript. Try again later.")

    content = payload.get("content", "")
    if isinstance(content, list):
        segments = []
        for segment in content:
            if isinstance(segment, dict):
                text = segment.get("text", "")
                if text:
                    segments.append(str(text))
            elif isinstance(segment, str):
                segments.append(segment)
        content = " ".join(segments)
    if not content or not str(content).strip():
        raise RuntimeError(
            "No transcript returned. The video may not have captions, or the transcript API response format may have changed."
        )
    return str(content).strip()

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video IDs from watch, shorts, embed, and youtu.be URLs."""
    url = url.strip()
    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def call_llm(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    return str(response.content).strip()

def make_docs(texts: List[Dict]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = []
    for item in texts:
        chunks = splitter.split_text(item["transcript"])
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "video_id": item["video_id"],
                    "title": item.get("title", "YouTube video"),
                    "url": item["url"],
                    "chunk": i,
                },
            ))
    return docs

def stable_cache_key(video_items: List[Dict], language: str, level: str) -> str:
    key_material = json.dumps({
        "videos": sorted((v["video_id"] for v in video_items)),
        "language": language,
        "level": level,
        "version": 2,
    }, sort_keys=True)
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()

@st.cache_resource(show_spinner="Building searchable transcript index...")
def build_vectorstore_cached(cache_key: str, serialized_docs: List[Dict], _embedding_model):
    """Cache FAISS index in process memory; serialized docs make cache key deterministic."""
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in serialized_docs
    ]
    return FAISS.from_documents(docs, _embedding_model)

def documents_to_payload(docs: List[Document]) -> List[Dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

def output_prompt(kind: str, subject: str, topic: str, level: str, language: str, source_text: str) -> str:
    common = f"""
You are an accurate, helpful learning assistant.
Subject: {subject}
Topic: {topic}
Learner level: {level}
Output language: {language}

Use ONLY the supplied transcript material for video-specific facts. If the transcripts
do not cover a detail, clearly say so instead of inventing it. Explain concepts at the
specified learner level. Avoid claiming that the videos say something unless it appears
in the source material.
TRANSCRIPT MATERIAL:
{source_text}
"""
    prompts = {
        "summary": common + """
Create a clear combined summary of all supplied videos. Include:
- a short overview
- the most important concepts
- key examples or explanations found in the transcripts
- a concise final takeaway
Use headings and bullets.
""",
        "notes": common + """
Create detailed study notes that combine the videos without unnecessary repetition.
Use descriptive headings, definitions, numbered processes, examples, common mistakes,
and a final revision checklist. Keep the notes easy to study from.
""",
        "mindmap": common + """
Create a Mermaid mind map for the topic. Output ONLY valid Mermaid code beginning
with `mindmap` and then a root node. Use concise branches and sub-branches. Do not
wrap the code in Markdown fences.
""",
        "flashcards": common + """
Create exactly 20 useful flashcards as a Markdown table with columns:
| # | Front / Question | Back / Answer |
Keep each answer concise but meaningful. Do not include text before or after the table.
""",
        "quiz": common + """
Create 10 multiple-choice questions. For each question include options A-D, then the
correct answer and a short explanation. Spread questions across the supplied material
and vary difficulty. Do not make up unsupported details.
""",
    }
    return prompts[kind]

def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return (cleaned[:60] or "learning_material").lower()

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    level = st.selectbox(
        "Learning level",
        ["Beginner", "School", "Undergraduate", "Advanced"],
        index=0,
    )
    language = st.selectbox(
        "Output language",
        ["English", "Hindi", "Marathi", "Spanish", "French"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### API configuration")
    st.caption("Add API keys in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.caption("Required: YouTube Data API, Supadata, and OpenRouter.")

# ----------------------------- Topic search -----------------------------
st.subheader("1. Find learning videos")
with st.form("topic_form"):
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", placeholder="e.g. Computer Science")
    with col2:
        topic = st.text_input("Topic", placeholder="e.g. Retrieval-Augmented Generation")
    search_clicked = st.form_submit_button("🔎 Find recommended videos", type="primary")

if search_clicked:
    if not subject.strip() or not topic.strip():
        st.warning("Enter both a subject and a topic.")
    else:
        try:
            with st.spinner("Searching YouTube..."):
                st.session_state["recommended_videos"] = youtube_search(
                    subject.strip(), topic.strip(), language, 5
                )
                st.session_state["active_subject"] = subject.strip()
                st.session_state["active_topic"] = topic.strip()
                st.session_state["active_language"] = language
            st.session_state.pop("selected_video_urls", None)
            st.success(f"Found {len(st.session_state['recommended_videos'])} recommended videos.")
        except Exception as exc:
            st.error(str(exc))

recommended = st.session_state.get("recommended_videos", [])
if recommended:
    st.markdown("### Recommended videos")
    selected_ids = []
    for video in recommended:
        with st.container(border=True):
            cols = st.columns([1, 3, 1])
            with cols[0]:
                if video.get("thumbnail"):
                    st.image(video["thumbnail"], use_container_width=True)
            with cols[1]:
                st.markdown(f"**[{video['title']}]({video['url']})**")
                st.caption(f"Channel: {video['channel']} · Published: {video.get('published_at', '')[:10]}")
                if video.get("description"):
                    st.write(video["description"][:240] + ("…" if len(video["description"]) > 240 else ""))
            with cols[2]:
                checked = st.checkbox(
                    "Add video",
                    key=f"select_{video['video_id']}",
                    value=False,
                )
                if checked:
                    selected_ids.append(video["video_id"])

    if st.button("Use selected recommendations", key="use_recommendations"):
        chosen = [v["url"] for v in recommended if v["video_id"] in selected_ids]
        st.session_state["selected_video_urls"] = chosen[:5]
        st.rerun()

# ----------------------------- Video URL input -----------------------------
st.subheader("2. Add up to five YouTube videos")
prefill = st.session_state.get("selected_video_urls", [])
with st.form("video_form"):
    url_inputs = []
    for i in range(5):
        default_value = prefill[i] if i < len(prefill) else ""
        url_inputs.append(st.text_input(
            f"YouTube URL {i + 1}",
            value=default_value,
            placeholder="https://www.youtube.com/watch?v=...",
            key=f"video_url_{i}",
        ))
    process_clicked = st.form_submit_button("🚀 Generate learning materials", type="primary")

if process_clicked:
    urls = [u.strip() for u in url_inputs if u and u.strip()]
    if not urls:
        st.warning("Add at least one YouTube URL.")
    else:
        parsed = []
        seen = set()
        invalid = []
        for url in urls:
            video_id = extract_video_id(url)
            if not video_id:
                invalid.append(url)
            elif video_id not in seen:
                parsed.append({"video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}"})
                seen.add(video_id)
        if invalid:
            st.error("These URLs do not look like supported YouTube video URLs:\n" + "\n".join(invalid))
        elif not parsed:
            st.warning("No unique valid YouTube URLs were provided.")
        else:
            subject_current = st.session_state.get("active_subject", "General learning")
            topic_current = st.session_state.get("active_topic", "Key concepts")
            # Try to use current visible topic fields only when they were submitted via search.
            try:
                subject_current = subject_current or subject
                topic_current = topic_current or topic
            except NameError:
                pass

            transcript_items = []
            failures = []
            progress = st.progress(0)
            status = st.empty()
            for index, item in enumerate(parsed):
                status.info(f"Getting transcript {index + 1} of {len(parsed)}…")
                try:
                    transcript = fetch_transcript_cached(item["video_id"], SUPADATA_API_KEY)
                    title = next(
                        (v["title"] for v in recommended if v["video_id"] == item["video_id"]),
                        f"YouTube video {item['video_id']}",
                    )
                    transcript_items.append({
                        **item,
                        "title": title,
                        "transcript": transcript,
                    })
                except Exception as exc:
                    failures.append(f"{item['url']}: {exc}")
                progress.progress((index + 1) / len(parsed))

            progress.empty()
            status.empty()
            if failures:
                st.warning("Some videos could not be transcribed:\n\n" + "\n\n".join(failures))
            if not transcript_items:
                st.error("No transcripts were retrieved, so learning materials cannot be generated.")
            else:
                try:
                    with st.spinner("Loading AI model and indexing transcripts..."):
                        embedding_model = load_embedding_model()
                        docs = make_docs(transcript_items)
                        cache_key = stable_cache_key(
                            transcript_items, language, level
                        )
                        vectorstore = build_vectorstore_cached(
                            cache_key, documents_to_payload(docs), embedding_model
                        )
                        llm = load_llm(OPENROUTER_API_KEY)
                    transcript_context = "\n\n".join(
                        f"[Video: {item['title']} | {item['url']}]\n{item['transcript'][:18000]}"
                        for item in transcript_items
                    )
                    # Keep prompts within a practical context size.
                    transcript_context = transcript_context[:55000]
                    generated = {}
                    kinds = [
                        ("summary", "Summary"),
                        ("notes", "Study notes"),
                        ("mindmap", "Mind map"),
                        ("flashcards", "Flashcards"),
                        ("quiz", "Quiz"),
                    ]
                    tabs = st.tabs([label for _, label in kinds])
                    for tab, (kind, label) in zip(tabs, kinds):
                        with tab:
                            with st.spinner(f"Generating {label.lower()}..."):
                                try:
                                    content = call_llm(
                                        llm,
                                        output_prompt(
                                            kind,
                                            subject_current,
                                            topic_current,
                                            level,
                                            language,
                                            transcript_context,
                                        ),
                                    )
                                    generated[kind] = content
                                    if kind == "mindmap":
                                        st.code(content, language="mermaid")
                                        st.caption("Copy this Mermaid code into mermaid.live to render or export the diagram.")
                                    else:
                                        st.markdown(content)
                                    st.download_button(
                                        f"Download {label}",
                                        data=content.encode("utf-8"),
                                        file_name=f"{safe_filename(topic_current)}_{kind}.md",
                                        mime="text/markdown",
                                        key=f"download_{kind}_{cache_key}",
                                    )
                                except Exception as exc:
                                    st.error(f"Could not generate {label.lower()}: {exc}")
                    st.session_state["last_vectorstore_key"] = cache_key
                    st.session_state["last_transcript_items"] = transcript_items
                    st.session_state["last_vectorstore_docs"] = documents_to_payload(docs)
                    st.session_state["last_subject"] = subject_current
                    st.session_state["last_topic"] = topic_current
                    st.success(
                        f"Processed {len(transcript_items)} video(s). Transcripts and embeddings are cached for faster reruns."
                    )
                except Exception as exc:
                    st.exception(exc)

# ----------------------------- Transcript-grounded Q&A -----------------------------
st.markdown("---")
st.subheader("3. Ask a question about the videos")
if st.session_state.get("last_transcript_items"):
    question = st.text_input(
        "Ask about the combined video content",
        placeholder="e.g. Explain the main idea with an example",
        key="rag_question",
    )
    if st.button("Ask question", type="secondary"):
        if not question.strip():
            st.warning("Type a question first.")
        elif not OPENROUTER_API_KEY:
            st.error("Add OPENROUTER_API_KEY to Streamlit Secrets.")
        else:
            try:
                embedding_model = load_embedding_model()
                docs_payload = st.session_state.get("last_vectorstore_docs", [])
                cache_key = st.session_state.get("last_vectorstore_key", "qa")
                vectorstore = build_vectorstore_cached(
                    cache_key, docs_payload, embedding_model
                )
                relevant_docs = vectorstore.similarity_search(question, k=5)
                context = "\n\n".join(
                    f"[{d.metadata.get('title', 'Video')}]\n{d.page_content}"
                    for d in relevant_docs
                )
                llm = load_llm(OPENROUTER_API_KEY)
                answer = call_llm(
                    llm,
                    f"""Answer the user's question using only the transcript excerpts below.
If the answer is not supported, say that the provided transcripts do not contain enough information.
Question: {question}
Transcript excerpts:
{context}""",
                )
                st.markdown(answer)
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
else:
    st.caption("Generate learning materials first; then you can ask questions grounded in the selected video transcripts.")

# ----------------------------- Footer -----------------------------
with st.expander("Troubleshooting / performance tips"):
    st.markdown("""
- **First run can be slower:** the sentence-transformer model downloads once, then is reused by Streamlit's resource cache.
- **Repeat searches are faster:** YouTube search results and transcripts are cached.
- **Embeddings/indexing are reused:** FAISS is cached in server memory for the current app process. A server restart clears this cache.
- **Joblib is installed but not required by this implementation:** Streamlit caching handles model and transcript reuse. Use joblib only if you specifically want persistent local disk caching; Streamlit Cloud local files may not persist reliably across rebuilds/restarts.
- If transcript retrieval fails, verify the Supadata API key and check the current Supadata API documentation for endpoint/response changes.
- If OpenRouter returns an authentication/model error, verify the key and selected model.
""")
