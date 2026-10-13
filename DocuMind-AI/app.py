"""DocuMind AI — a small, session-based PDF question-answering app."""

from __future__ import annotations

import hashlib
import html
import json
import os
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from pdf_processor import PDFProcessingError, extract_pdf
from rag_pipeline import (
    EmbeddingModelError,
    GeminiServiceError,
    embed_chunks,
    generate_grounded_answer,
    retrieve_relevant_chunks,
    summarize_documents,
)

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI — Smart Document Q&A",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 30
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
EMBEDDING_MODEL = "all-MiniLM-L6-v2 (local)"


def inject_styles() -> None:
    """Apply the requested branded theme while keeping Streamlit controls native."""
    st.markdown(
        """
        <style>
        :root {
          --ink: #17213a;
          --muted: #69758e;
          --line: #e8eaf3;
          --purple: #684be8;
          --blue: #4f7bf3;
          --surface: #ffffff;
          --soft: #f7f7fc;
        }
        .stApp {
          background:
            radial-gradient(ellipse at 72% 0%, rgba(225, 220, 255, .38), transparent 34rem),
            #f7f8fc;
          color: var(--ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
          max-width: 1440px;
          padding-top: 2rem;
          padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
          background: #fff;
          border-right: 1px solid var(--line);
        }
        .brand-row { display:flex; align-items:center; gap:.7rem; margin:.3rem 0 1.5rem; }
        .brand-icon {
          width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
          background:linear-gradient(140deg, #7156ee, #4b83f4);
          box-shadow:0 7px 18px rgba(91, 87, 220, .24);
        }
        .brand-name { font-size:1.06rem; font-weight:750; letter-spacing:-.035em; color:var(--ink); }
        .brand-name span { color:#7058df; }
        .eyebrow {
          color:#7058df; font-size:.71rem; font-weight:750; letter-spacing:.13em;
          text-transform:uppercase; margin-bottom:.55rem;
        }
        .hero-title {
          color:var(--ink); font-size:clamp(2.5rem, 5vw, 4.25rem); line-height:1.05;
          font-weight:760; letter-spacing:-.065em; margin:.2rem 0 .7rem;
        }
        .hero-title .gradient {
          background:linear-gradient(100deg, #6a4ce8 15%, #4d7ff0 88%);
          -webkit-background-clip:text; background-clip:text; color:transparent;
        }
        .hero-subtitle { color:var(--muted); font-size:1.08rem; line-height:1.65; max-width:630px; }
        .hero-panel {
          position:relative; overflow:hidden; border:1px solid #e9e8f7; border-radius:24px;
          padding:clamp(1.4rem, 4vw, 3rem);
          background:linear-gradient(125deg, rgba(255,255,255,.97), rgba(248,247,255,.91));
          box-shadow:0 14px 40px rgba(37, 42, 77, .045);
        }
        .hero-panel:after {
          content:""; position:absolute; width:260px; height:260px; right:-100px; top:-145px;
          background:radial-gradient(circle, rgba(129,111,246,.18), transparent 68%);
          pointer-events:none;
        }
        .hero-art {
          position:absolute; z-index:0; width:205px; right:clamp(1.4rem, 6vw, 5rem);
          top:50%; transform:translateY(-44%); filter:drop-shadow(0 15px 22px rgba(85, 76, 173, .15));
          pointer-events:none;
        }
        .hero-copy { position:relative; z-index:1; max-width:690px; }
        .feature-card {
          height:100%; min-height:155px; padding:1.2rem; border:1px solid var(--line);
          border-radius:17px; background:rgba(255,255,255,.87);
          box-shadow:0 6px 20px rgba(37, 42, 77, .035);
        }
        .feature-icon {
          width:38px; height:38px; display:grid; place-items:center; border-radius:12px;
          color:#654be2; background:#f0edff; margin-bottom:.85rem;
        }
        .feature-title { color:var(--ink); font-size:.96rem; font-weight:700; margin-bottom:.3rem; }
        .feature-copy { color:var(--muted); font-size:.84rem; line-height:1.5; }
        .section-title { color:var(--ink); font-size:1.15rem; font-weight:720; letter-spacing:-.025em; }
        .section-copy { color:var(--muted); font-size:.87rem; line-height:1.55; }
        .status-pill {
          display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .72rem;
          border:1px solid #e5e3f6; border-radius:999px; color:#5141ae;
          background:#f3f1ff; font-size:.77rem; font-weight:650; white-space:nowrap;
        }
        .status-dot { width:7px; height:7px; border-radius:50%; background:#765de9; }
        .doc-card {
          border:1px solid var(--line); border-radius:16px; background:#fff; padding:1rem;
          box-shadow:0 5px 18px rgba(37, 42, 77, .035);
        }
        .doc-kicker { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
        .setup-box {
          border:1px solid #eadfb7; border-radius:14px; background:#fffaf0;
          padding:.95rem 1.05rem; color:#694d16; line-height:1.55; font-size:.86rem;
        }
        .chat-hint { color:var(--muted); font-size:.83rem; }
        .source-label { color:#606d86; font-size:.77rem; font-weight:600; }
        div[data-testid="stFileUploader"] section {
          border:1.5px dashed #c9c5e9; border-radius:17px; background:#fbfaff;
          padding:1.15rem; transition:border-color .18s, background .18s;
        }
        div[data-testid="stFileUploader"] section:hover {
          border-color:#8271e9; background:#f7f5ff;
        }
        div[data-testid="stButton"] button {
          border-radius:11px; font-weight:650; transition:transform .16s, box-shadow .16s;
        }
        div[data-testid="stButton"] button:hover {
          transform:translateY(-1px); box-shadow:0 5px 14px rgba(89, 76, 197, .13);
        }
        div[data-testid="stChatMessage"] {
          border:1px solid rgba(232,234,243,.8); border-radius:15px;
          background:rgba(255,255,255,.82); padding:.8rem 1rem;
        }
        @media (max-width: 700px) {
          [data-testid="stMainBlockContainer"] { padding:1rem .8rem 2rem; }
          .hero-panel { border-radius:18px; }
          .hero-title { font-size:2.55rem; }
          .hero-art { display:none; }
        }
        @media (min-width: 701px) and (max-width: 980px) { .hero-art { width:145px; right:1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "documents": [],
        "chat_history": [],
        "summary": None,
        "summary_title": None,
        "upload_errors": [],
        "uploader_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def file_size_label(size: int) -> str:
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.summary = None
    st.session_state.summary_title = None


def remove_document(document_id: str) -> None:
    st.session_state.documents = [
        doc for doc in st.session_state.documents if doc["id"] != document_id
    ]
    clear_chat()
    st.session_state.upload_errors = []
    st.session_state.uploader_version += 1


def render_brand() -> None:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M7 3.75h7.2L19 8.6v10.15A2.25 2.25 0 0 1 16.75 21h-9.5A2.25 2.25 0 0 1 5 18.75v-12A3 3 0 0 1 8 3.75Z" stroke="white" stroke-width="1.7"/>
              <path d="M14 4v5h5M8.5 13h7M8.5 16h5" stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="m17.3 12.2.55 1.1 1.2.18-.87.85.2 1.2-1.08-.57-1.08.57.2-1.2-.87-.85 1.2-.18.55-1.1Z" fill="#C9C0FF"/>
            </svg>
          </div>
          <div class="brand-name">DocuMind <span>AI</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(api_key: str | None) -> None:
    with st.sidebar:
        render_brand()
        st.caption("A clearer way to understand your documents.")
        if st.button("＋  New chat", use_container_width=True, type="primary"):
            clear_chat()
            st.rerun()

        st.markdown("---")
        st.markdown("#### Your documents")
        docs = st.session_state.documents
        if not docs:
            st.caption("Your uploaded PDFs will appear here.")
        for doc in docs:
            with st.container():
                st.markdown(f"**{doc['name']}**")
                st.caption(
                    f"{file_size_label(doc['size'])}  ·  {doc['page_count']} pages  ·  "
                    f"{len(doc['chunks'])} passages indexed"
                )
                st.success("Processed", icon=":material/check_circle:")
                if st.button("Remove", key=f"remove_{doc['id']}", use_container_width=True):
                    remove_document(doc["id"])
                    st.rerun()

        if docs and st.button("Remove all documents", use_container_width=True):
            st.session_state.documents = []
            clear_chat()
            st.session_state.upload_errors = []
            st.session_state.uploader_version += 1
            st.rerun()

        st.markdown("---")
        st.markdown("#### AI models")
        st.caption(f"Answers  ·  {GEMINI_MODEL}")
        st.caption(f"Search  ·  {EMBEDDING_MODEL}")
        st.markdown("#### API configuration")
        if api_key:
            st.success("Gemini is ready", icon=":material/verified:")
        else:
            st.warning("Gemini API key needed", icon=":material/key_off:")
        st.caption("Your key is read from Replit Secrets and is never displayed.")


def render_setup_notice() -> None:
    st.markdown(
        """
        <div class="setup-box">
          <strong>Gemini is not configured yet.</strong><br>
          Add <code>GEMINI_API_KEY</code> in Replit → Tools → Secrets, then restart the app.
          The app never shows or stores the key in document or chat history.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_errors() -> None:
    for item in st.session_state.upload_errors:
        st.error(f"{item['name']}: {item['message']}", icon=":material/error:")


def process_uploads(uploaded_files: list[Any]) -> None:
    """Extract, chunk, and embed selected PDFs; retain all data in this session."""
    st.session_state.upload_errors = []
    known_ids = {doc["id"] for doc in st.session_state.documents}

    for uploaded in uploaded_files:
        try:
            raw = uploaded.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in known_ids:
                continue
            if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
                raise PDFProcessingError(
                    f"File exceeds the {MAX_UPLOAD_MB} MB per-file limit."
                )

            with st.spinner(f"Reading and indexing {uploaded.name}…"):
                extracted = extract_pdf(uploaded.name, raw)
                embeddings = embed_chunks(extracted["chunks"])

            document = {
                "id": digest,
                "name": uploaded.name,
                "size": len(raw),
                "page_count": extracted["page_count"],
                "pages": extracted["pages"],
                "chunks": extracted["chunks"],
                "embeddings": embeddings,
            }
            st.session_state.documents.append(document)
            known_ids.add(digest)
        except PDFProcessingError as exc:
            st.session_state.upload_errors.append(
                {"name": uploaded.name, "message": str(exc)}
            )
        except EmbeddingModelError as exc:
            st.session_state.upload_errors.append(
                {"name": uploaded.name, "message": str(exc)}
            )
            break
        except Exception:
            st.session_state.upload_errors.append(
                {
                    "name": uploaded.name,
                    "message": "This PDF could not be processed. Try another file or a text-based PDF.",
                }
            )

    st.session_state.uploader_version += 1
    st.rerun()


def render_uploader(label: str, help_text: str) -> None:
    upload_key = f"pdf_upload_{st.session_state.uploader_version}"
    uploaded_files = st.file_uploader(
        label,
        type=["pdf"],
        accept_multiple_files=True,
        help=help_text,
        key=upload_key,
        label_visibility="collapsed",
    )
    if uploaded_files:
        process_uploads(uploaded_files)


def render_welcome(api_key: str | None) -> None:
    render_upload_errors()
    if not api_key:
        render_setup_notice()
        st.write("")

    st.markdown(
        """
        <div class="hero-panel">
          <div class="hero-copy">
            <div class="eyebrow">Your personal document assistant</div>
            <h1 class="hero-title">Your documents.<br><span class="gradient">Your answers.</span></h1>
            <p class="hero-subtitle">Upload a PDF and let AI help you understand it. Ask questions,
            find important details, and get clear summaries grounded in your source material.</p>
            <div style="margin-top:1.15rem"><span class="status-pill"><span class="status-dot"></span>AI-powered document assistant</span></div>
          </div>
          <div class="hero-art" aria-hidden="true">
            <svg viewBox="0 0 220 220" fill="none">
              <circle cx="113" cy="110" r="95" fill="#F1EFFF"/>
              <circle cx="113" cy="110" r="74" fill="#E9E5FF"/>
              <path d="M71 43h57l35 35v88a13 13 0 0 1-13 13H71a13 13 0 0 1-13-13V56a13 13 0 0 1 13-13Z" fill="white" stroke="#D7D1F4" stroke-width="2"/>
              <path d="M127 44v34h34" stroke="#BDB4ED" stroke-width="2"/>
              <rect x="78" y="94" width="69" height="7" rx="3.5" fill="#C8C1EC"/>
              <rect x="78" y="110" width="58" height="7" rx="3.5" fill="#E0DCF5"/>
              <rect x="78" y="126" width="66" height="7" rx="3.5" fill="#E0DCF5"/>
              <rect x="78" y="148" width="39" height="21" rx="8" fill="#F0EDFF"/>
              <path d="m97.5 151 3.2 5.7 6.4 1-4.8 4.5 1.1 6.4-5.9-3-5.7 3 1.1-6.4-4.8-4.5 6.4-1 3-5.7Z" fill="#735CE4"/>
              <circle cx="163" cy="56" r="22" fill="url(#bubble)"/>
              <path d="m163 43 3.4 7.4 8 1.2-5.8 5.7 1.4 8-7-3.8-7.2 3.8 1.4-8-5.8-5.7 8-1.2L163 43Z" fill="white"/>
              <circle cx="55" cy="145" r="6" fill="#9A8DEB"/>
              <circle cx="169" cy="147" r="5" fill="#9DB7FC"/>
              <defs><linearGradient id="bubble" x1="145" y1="39" x2="181" y2="76" gradientUnits="userSpaceOnUse"><stop stop-color="#8D7BF1"/><stop offset="1" stop-color="#587FEE"/></linearGradient></defs>
            </svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="section-title">Start exploring your documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Add one or more PDFs to build a private workspace for asking questions and finding answers.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="feature-card" style="padding:1.1rem 1.2rem">
          <div class="eyebrow" style="margin-bottom:.35rem">Step 1</div>
          <div class="section-title">Upload your PDFs</div>
          <div class="section-copy">Your documents are processed in this session and are not saved to a database.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    render_uploader(
        "Drop your PDF here, or browse files",
        f"PDF files only. Maximum {MAX_UPLOAD_MB} MB per file. You can select multiple files.",
    )
    st.caption(
        f"PDF files · Up to {MAX_UPLOAD_MB} MB each · Text-based PDFs work best. "
        "Scanned image-only PDFs are not OCR processed."
    )

    st.write("")
    columns = st.columns(3)
    features = [
        (
            "Ask Anything",
            "Get answers grounded in the content of your uploaded documents.",
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M5 5h14v11H9l-4 4V5Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><path d="M8 9h8M8 12h5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
        ),
        (
            "Smart summaries",
            "Make long documents easier to understand, one section at a time.",
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M6 4h12v16H6zM9 8h6M9 12h6M9 16h3" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        ),
        (
            "Source references",
            "See which PDF and page supports each answer.",
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M8.5 14.5 6 17a3 3 0 1 1-4.2-4.2l4-4A3 3 0 0 1 10 8m5.5 1.5L18 7a3 3 0 1 1 4.2 4.2l-4 4A3 3 0 0 1 14 16m-6 0 8-8" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>',
        ),
    ]
    for col, (title, copy, icon) in zip(columns, features):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )


def render_copy_button(text: str, key: str) -> None:
    """Render a small browser-side copy control without sending text to a server."""
    js_text = json.dumps(text, ensure_ascii=True).replace("</", "<\\/")
    component = f"""
    <div style="display:flex;justify-content:flex-end">
      <button id="copy-{key}" aria-label="Copy answer" style="
        cursor:pointer;background:#fff;border:1px solid #e6e8f1;border-radius:9px;
        padding:6px 10px;color:#4c5870;font:600 12px system-ui,sans-serif">
        Copy answer
      </button>
    </div>
    <script>
      const button = document.getElementById("copy-{key}");
      button.addEventListener("click", async () => {{
        const value = {js_text};
        try {{
          await navigator.clipboard.writeText(value);
          button.textContent = "Copied";
          setTimeout(() => button.textContent = "Copy answer", 1400);
        }} catch (_) {{
          const field = document.createElement("textarea");
          field.value = value; document.body.appendChild(field); field.select();
          document.execCommand("copy"); field.remove();
          button.textContent = "Copied";
          setTimeout(() => button.textContent = "Copy answer", 1400);
        }}
      }});
    </script>
    """
    components.html(component, height=42, scrolling=False)


def render_chat_message(message: dict[str, Any], index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        sources = message.get("sources", [])
        if sources:
            source_labels = list(
                dict.fromkeys(
                    f"{item['source']} · page {item['page']}" for item in sources
                )
            )
            st.markdown(
                f'<div class="source-label">Sources: {html.escape("  |  ".join(source_labels))}</div>',
                unsafe_allow_html=True,
            )
            context_items = message.get("context", [])
            if context_items:
                with st.expander("View retrieved context"):
                    for item in context_items:
                        st.caption(f"{item['source']} · page {item['page']}")
                        st.write(item["text"])
        if not message.get("no_copy"):
            render_copy_button(message["content"], f"answer-{index}")


def build_retrieval_query(question: str) -> str:
    """Give short follow-up questions enough conversational context for retrieval."""
    history = st.session_state.chat_history
    # answer_question appends the current question before retrieval; omit it here
    # so only an earlier user turn is used as follow-up context.
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    previous_user = next(
        (item["content"] for item in reversed(history) if item["role"] == "user"),
        "",
    )
    previous_answer = next(
        (
            item["content"]
            for item in reversed(history)
            if item["role"] == "assistant" and item.get("context")
        ),
        "",
    )
    followup_markers = (
        "it",
        "they",
        "them",
        "those",
        "that",
        "this",
        "their",
        "its",
        "what about",
        "how about",
        "why",
        "which one",
    )
    normalized = question.strip().lower()
    if previous_user and (
        len(question.split()) <= 7
        or any(normalized.startswith(marker) for marker in followup_markers)
    ):
        return (
            f"{question}\nRelated question: {previous_user}\n"
            f"Previously retrieved answer: {previous_answer[:350]}"
        )
    return question


def answer_question(question: str, api_key: str) -> None:
    question = question.strip()
    if not question:
        return
    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        with st.spinner("Searching your documents and preparing an answer…"):
            retrieved = retrieve_relevant_chunks(
                st.session_state.documents, build_retrieval_query(question), top_k=3
            )
            if not retrieved:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I couldn’t find information that clearly answers that in "
                            "your uploaded PDFs. Try asking in a different way or name "
                            "a topic from the document."
                        ),
                        "sources": [],
                        "context": [],
                        "no_copy": True,
                    }
                )
                return

            recent_history = st.session_state.chat_history[-9:-1]
            answer = generate_grounded_answer(
                question=question,
                retrieved_chunks=retrieved,
                history=recent_history,
                api_key=api_key,
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved,
                "context": retrieved,
            }
        )
    except EmbeddingModelError as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": str(exc),
                "sources": [],
                "context": [],
                "no_copy": True,
            }
        )
    except GeminiServiceError as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": str(exc),
                "sources": [],
                "context": [],
                "no_copy": True,
            }
        )


def summarize_workspace(api_key: str) -> None:
    progress = st.progress(0, text="Preparing document sections…")

    def update_progress(done: int, total: int, label: str) -> None:
        fraction = min(1.0, done / max(total, 1))
        progress.progress(fraction, text=label)

    try:
        summary = summarize_documents(
            st.session_state.documents,
            api_key=api_key,
            on_progress=update_progress,
        )
        st.session_state.summary = summary
        count = len(st.session_state.documents)
        st.session_state.summary_title = (
            "Workspace summary" if count > 1 else "Document summary"
        )
    except GeminiServiceError as exc:
        st.error(str(exc))
    finally:
        progress.empty()


def render_workspace(api_key: str | None) -> None:
    render_upload_errors()
    if not api_key:
        render_setup_notice()
        st.write("")

    st.markdown(
        '<div class="eyebrow">Your workspace</div><div class="section-title" style="font-size:1.8rem">Understand every page.</div>'
        '<div class="section-copy">Ask a question or choose a prompt. Answers are grounded in your PDFs, with page references.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([0.37, 0.63], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Document library</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-copy">{len(st.session_state.documents)} PDF'
                f'{"s" if len(st.session_state.documents) != 1 else ""} processed and ready.</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            for doc in st.session_state.documents:
                st.markdown(
                    f'<div class="doc-card"><div class="doc-kicker">PDF document</div>'
                    f'<strong>{html.escape(doc["name"])}</strong><br>'
                    f'<span class="section-copy">{file_size_label(doc["size"])} · '
                    f'{doc["page_count"]} pages · {len(doc["chunks"])} passages</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.write("")

            if st.button(
                "Summarize documents" if len(st.session_state.documents) > 1 else "Summarize document",
                type="primary",
                use_container_width=True,
                disabled=not api_key,
                help=None if api_key else "Add GEMINI_API_KEY in Replit Secrets to enable summaries.",
            ):
                summarize_workspace(api_key or "")
                st.rerun()

            if st.session_state.summary:
                st.markdown("---")
                st.markdown(
                    f'<div class="section-title">{html.escape(st.session_state.summary_title or "Summary")}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state.summary)

            st.markdown("---")
            st.markdown("**Add more PDFs**")
            render_uploader(
                "Upload additional PDFs",
                f"PDF files only, up to {MAX_UPLOAD_MB} MB each.",
            )
            if st.button("Replace documents", use_container_width=True):
                st.session_state.documents = []
                clear_chat()
                st.session_state.upload_errors = []
                st.session_state.uploader_version += 1
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Ask your documents</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-copy">Answers include source names and page numbers, so you can check the original.</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            for index, message in enumerate(st.session_state.chat_history):
                render_chat_message(message, index)

            suggestions = [
                "Summarize this document",
                "What are the main points?",
                "Explain the key concepts",
                "What are the important conclusions?",
            ]
            buttons = st.columns(len(suggestions))
            for col, suggestion in zip(buttons, suggestions):
                with col:
                    if st.button(
                        suggestion,
                        key=f"suggest_{suggestion}",
                        use_container_width=True,
                        disabled=not api_key,
                    ):
                        if suggestion == "Summarize this document":
                            summarize_workspace(api_key or "")
                        else:
                            answer_question(suggestion, api_key or "")
                        st.rerun()

            question = st.chat_input(
                "Ask a question about your documents…",
                disabled=not api_key,
            )
            if question:
                answer_question(question, api_key or "")
                st.rerun()

            if not api_key:
                st.caption("Add the Gemini API key to enable questions and summaries.")


def main() -> None:
    initialize_state()
    inject_styles()
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
    render_sidebar(api_key)

    if st.session_state.documents:
        render_workspace(api_key)
    else:
        render_welcome(api_key)


if __name__ == "__main__":
    main()
