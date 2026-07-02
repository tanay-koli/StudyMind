"""
StudyMind — Personal Study Assistant
=====================================
Streamlit single-page app with 3 tabs: Ingest, Ask, Connect Topics.
Powered by Cognee (knowledge graph) + Ollama (local LLM).

Run with:  streamlit run app.py
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `core.*` imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import check_cognee_cloud, get_cloud_status_emoji, DATA_DIR
from core.ingest import (
    ingest_pdf,
    ingest_youtube,
    ingest_url,
    ingest_text,
    forget_source,
    forget_all,
)
from core.query import ask, connect_topics, summarize_topic, ask_scoped, cross_topic_ask
from core.topics import (
    load_topics,
    create_topic,
    delete_topic,
    add_dataset_to_topic,
    get_topic_datasets,
    suggest_topic_name,
    generate_study_guide,
    generate_audio_script,
    synthesize_audio,
    GENERAL_TOPIC_ID,
)

# ---------------------------------------------------------------------------
# Async helper — Streamlit runs synchronously; wrap async calls safely
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code safely.
    
    Avoids using `asyncio.run()` because Cognee spawns background tasks 
    (telemetry, db pools) that cause asyncio.run's cleanup phase to deadlock 
    Streamlit threads.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Intentionally avoiding loop.close() to prevent teardown hangs
        pass


# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyMind — AI Study Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a premium dark look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, .stApp, [class*="st-"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Remove default Streamlit padding for tighter look */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
        border-right: 1px solid #21262d !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e6edf3 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
    }

    /* Source cards in sidebar */
    .source-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-color 0.2s ease;
    }

    .source-card:hover {
        border-color: #7C5CFC;
    }

    .source-card .source-type {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8b949e;
    }

    .source-card .source-title {
        font-size: 14px;
        font-weight: 500;
        color: #e6edf3;
        margin-top: 2px;
        word-break: break-word;
    }

    .source-card .source-time {
        font-size: 11px;
        color: #484f58;
        margin-top: 4px;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        background: #0d1117;
        border: 1px solid #21262d;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .hero-header h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #7C5CFC 0%, #00C9A7 50%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem !important;
    }

    .hero-header p {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Graph container */
    .graph-container {
        border: 1px solid #21262d;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* Spinner text */
    .stSpinner > div {
        color: #8b949e !important;
    }

    /* Info/success/error boxes */
    .stAlert {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "sources" not in st.session_state:
    st.session_state.sources = []  # List of ingestion result dicts

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {role, content} dicts

if "cloud_status" not in st.session_state:
    st.session_state.cloud_status = check_cognee_cloud(timeout=5)

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = "all_topics"

if "suggested_topic" not in st.session_state:
    st.session_state.suggested_topic = None

if "active_study_guide_topic" not in st.session_state:
    st.session_state.active_study_guide_topic = None

if "active_audio_topic" not in st.session_state:
    st.session_state.active_audio_topic = None


# ---------------------------------------------------------------------------
# Sidebar — Knowledge Base & Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🧠</div>
        <div style="
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #7C5CFC 0%, #00C9A7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">StudyMind</div>
        <div style="font-size: 0.8rem; color: #8b949e; margin-top: 2px;">
            AI Study Assistant • Cognee Cloud
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Cognee Cloud status
    status = st.session_state.cloud_status
    status_emoji = get_cloud_status_emoji(status)
    st.markdown(f"**Cognee Cloud:** {status_emoji}")

    if not status["ok"] and status.get("error"):
        st.error(status["error"], icon="⚠️")

    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state.cloud_status = check_cognee_cloud(timeout=5)
        st.rerun()

    st.markdown("---")

    # Load topics
    topics = load_topics()

    # Topics Section
    st.markdown("### 📚 Topics (Notebooks)")

    # "All Topics" selector
    all_selected = st.session_state.selected_topic == "all_topics"
    if st.button(
        "🗂️ All Topics",
        key="btn_all_topics",
        use_container_width=True,
        type="primary" if all_selected else "secondary",
    ):
        st.session_state.selected_topic = "all_topics"
        st.rerun()

    # List each topic with color dot and options
    for tid, topic in topics.items():
        topic_sources = [
            s for s in st.session_state.sources
            if s.get("topic_id") == tid or s.get("dataset_name") in topic.get("datasets", [])
        ]
        count = len(topic_sources)
        is_selected = st.session_state.selected_topic == tid

        col_dot, col_name, col_sg, col_aud = st.columns([0.15, 0.55, 0.15, 0.15])
        with col_dot:
            st.markdown(f"<p style='color: {topic['color']}; margin-top: 0.6rem; font-size: 1.1rem;'>●</p>", unsafe_allow_html=True)
        with col_name:
            if st.button(
                f"{topic['name']} ({count})",
                key=f"sel_{tid}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.selected_topic = tid
                st.rerun()
        with col_sg:
            if st.button("📖", key=f"sg_{tid}", help=f"Study Guide for {topic['name']}"):
                st.session_state.active_study_guide_topic = tid
                st.session_state.active_audio_topic = None
                st.rerun()
        with col_aud:
            if st.button("🎙️", key=f"aud_{tid}", help=f"Audio Overview for {topic['name']}"):
                st.session_state.active_audio_topic = tid
                st.session_state.active_study_guide_topic = None
                st.rerun()

    # New Topic Form
    with st.expander("➕ New Topic"):
        new_topic_name = st.text_input("Topic Name", key="sidebar_new_topic_input")
        if st.button("Create Topic", key="btn_create_topic", use_container_width=True):
            if new_topic_name.strip():
                new_t = create_topic(new_topic_name)
                st.session_state.selected_topic = new_t["id"]
                st.success(f"Topic '{new_topic_name}' created!")
                st.rerun()

    # Render Study Guide Expander below topic list
    if st.session_state.active_study_guide_topic:
        sg_tid = st.session_state.active_study_guide_topic
        sg_topic = topics.get(sg_tid)
        if sg_topic:
            st.markdown("---")
            with st.expander(f"📖 Study Guide: {sg_topic['name']}", expanded=True):
                # Close button
                if st.button("Close Guide", key="close_sg"):
                    st.session_state.active_study_guide_topic = None
                    st.rerun()

                with st.spinner("Generating study guide..."):
                    guide_content = generate_study_guide(sg_tid)

                st.markdown(guide_content)

                if "Could not generate study guide" not in guide_content and "No sources" not in guide_content:
                    st.download_button(
                        label="📥 Download as .md",
                        data=guide_content,
                        file_name=f"{sg_topic['name'].lower().replace(' ', '_')}_study_guide.md",
                        mime="text/markdown",
                        key="btn_download_sg",
                        use_container_width=True,
                    )

    # Render Audio Overview Expander below topic list
    if st.session_state.active_audio_topic:
        aud_tid = st.session_state.active_audio_topic
        aud_topic = topics.get(aud_tid)
        if aud_topic:
            st.markdown("---")
            with st.expander(f"🎙️ Audio Overview: {aud_topic['name']}", expanded=True):
                # Close button
                if st.button("Close Audio", key="close_aud"):
                    st.session_state.active_audio_topic = None
                    st.rerun()

                with st.spinner("Generating audio script..."):
                    script_content = generate_audio_script(aud_tid)

                # Synthesize
                with st.spinner("Synthesizing audio file..."):
                    audio_path = synthesize_audio(aud_tid, script_content)

                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                else:
                    st.info("Install pyttsx3 for audio playback. Displaying script text only.")

                st.markdown("**Podcast Transcript:**")
                st.caption(script_content)

    st.markdown("---")

    # Knowledge Base Sources
    st.markdown("### 📎 Sources in Active Topic")

    # Filter sources list by active topic
    active_tid = st.session_state.selected_topic
    sources = st.session_state.sources
    successful_sources = [s for s in sources if s.get("status") == "success"]

    if active_tid != "all_topics":
        active_topic_name = topics.get(active_tid, {}).get("name", active_tid)
        st.caption(f"Showing sources in **{active_topic_name}**")
        filtered_sources = [
            s for s in successful_sources
            if s.get("topic_id") == active_tid or s.get("dataset_name") in topics.get(active_tid, {}).get("datasets", [])
        ]
    else:
        st.caption("Showing all ingested sources")
        filtered_sources = successful_sources

    if filtered_sources:
        for i, source in enumerate(filtered_sources):
            icon = {
                "pdf": "📄", "youtube": "🎥", "web": "🌐", "note": "📝"
            }.get(source.get("source_type", ""), "📎")

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-type">{icon} {source.get('source_type', 'unknown').upper()}</div>
                    <div class="source-title">{source.get('title', 'Untitled')[:50]}</div>
                    <div class="source-time">{source.get('timestamp', '')[:16]}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("🗑️", key=f"forget_{active_tid}_{i}", help="Remove this source"):
                    with st.spinner("Removing..."):
                        result = run_async(forget_source(source["dataset_name"]))
                    if result["status"] == "success":
                        st.session_state.sources = [
                            s for s in st.session_state.sources
                            if s.get("dataset_name") != source["dataset_name"]
                        ]
                        # Remove from topic lists
                        for tid in topics:
                            if source["dataset_name"] in topics[tid].get("datasets", []):
                                topics[tid]["datasets"].remove(source["dataset_name"])
                                topics[tid]["study_guide"] = None
                                topics[tid]["audio_script"] = None
                        save_topics(topics)
                        st.rerun()
                    else:
                        st.error(result.get("error", "Failed to forget"))

        if active_tid == "all_topics":
            st.markdown("---")
            if st.button("🗑️ Forget Everything", use_container_width=True, type="secondary"):
                with st.spinner("Clearing all knowledge..."):
                    result = run_async(forget_all())
                if result["status"] == "success":
                    st.session_state.sources = []
                    st.session_state.chat_history = []
                    # Reset datasets lists in all topics
                    for tid in topics:
                        topics[tid]["datasets"] = []
                        topics[tid]["study_guide"] = None
                        topics[tid]["audio_script"] = None
                    save_topics(topics)
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed"))
    else:
        st.caption("No sources in this topic.")
        st.markdown(
            '<div style="color: #484f58; font-size: 13px;">'
            "Use the <b>📥 Ingest</b> tab to add material."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

# Hero header
st.markdown("""
<div class="hero-header">
    <h1>🧠 StudyMind</h1>
    <p>Ingest study material • Ask questions • Discover connections</p>
</div>
""", unsafe_allow_html=True)

# Main tabs
tab_ingest, tab_ask, tab_connect = st.tabs([
    "📥 Ingest Material",
    "💬 Ask Questions",
    "🔗 Connect Topics",
])


# ========================== TAB 1: INGEST ==========================
with tab_ingest:
    st.markdown("### Add Study Material")
    st.caption("Ingest PDFs, YouTube videos, web articles, or your own notes into the knowledge graph.")

    # 1. Interactive Topic Suggestion Banner
    if st.session_state.suggested_topic:
        sug = st.session_state.suggested_topic
        st.info(f"💡 **Suggested Topic:** The content from **{sug['source_title']}** seems to belong to topic **'{sug['name']}'**.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Add to suggested Topic", key="accept_suggestion", use_container_width=True, type="primary"):
                # Find or create the topic
                current_topics = load_topics()
                target_tid = None
                for tid, t in current_topics.items():
                    if t["name"].lower() == sug["name"].lower():
                        target_tid = tid
                        break
                if not target_tid:
                    new_t = create_topic(sug["name"])
                    target_tid = new_t["id"]
                
                # Re-associate dataset with this topic in topics.json
                add_dataset_to_topic(target_tid, sug["source_dataset"])
                
                # Update source dict in session state
                for src in st.session_state.sources:
                    if src.get("dataset_name") == sug["source_dataset"]:
                        src["topic_id"] = target_tid
                
                st.session_state.suggested_topic = None
                st.session_state.selected_topic = target_tid
                st.success(f"Successfully added to topic '{sug['name']}'!")
                st.rerun()
        with col_no:
            if st.button("Keep in current topic", key="reject_suggestion", use_container_width=True):
                st.session_state.suggested_topic = None
                st.rerun()

    # 2. Topic Selector for Ingestion
    current_topics = load_topics()
    topic_options = {t["name"]: tid for tid, t in current_topics.items()}
    topic_list = list(topic_options.keys()) + ["➕ New Topic..."]

    # Locate index of currently active topic in sidebar to default to it
    active_tid = st.session_state.selected_topic
    default_idx = 0
    if active_tid != "all_topics":
        active_name = current_topics.get(active_tid, {}).get("name")
        if active_name in topic_options:
            default_idx = list(topic_options.keys()).index(active_name)

    selected_option = st.selectbox(
        "Ingest into topic:",
        topic_list,
        index=default_idx,
        help="Select which topic collection to add this source to."
    )

    ingest_topic_id = None
    if selected_option == "➕ New Topic...":
        new_t_name = st.text_input("New Topic Name:", key="ingest_new_topic_name_input")
        if new_t_name.strip():
            # Check if name already exists
            found_tid = None
            for tid, t in current_topics.items():
                if t["name"].lower() == new_t_name.strip().lower():
                    found_tid = tid
                    break
            if found_tid:
                ingest_topic_id = found_tid
            else:
                if st.button("Confirm and Create Topic", key="btn_confirm_create_topic"):
                    new_t = create_topic(new_t_name)
                    ingest_topic_id = new_t["id"]
                    st.session_state.selected_topic = ingest_topic_id
                    st.success(f"Topic '{new_t_name}' created!")
                    st.rerun()
    else:
        ingest_topic_id = topic_options[selected_option]

    st.markdown("---")

    ingest_col1, ingest_col2 = st.columns(2, gap="large")

    # --- PDF Upload ---
    with ingest_col1:
        st.markdown("#### 📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            if st.button("📄 Ingest PDF", key="btn_ingest_pdf", use_container_width=True):
                save_path = DATA_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("📄 Extracting text & uploading to Cognee Cloud..."):
                    result = run_async(ingest_pdf(str(save_path), title=uploaded_file.name, topic_id=ingest_topic_id))

                if result["status"] == "success":
                    st.session_state.sources.append(result)
                    st.success(
                        f"✅ Ingested **{result['title']}** "
                        f"({result.get('page_count', '?')} pages)",
                        icon="📄",
                    )
                    # Trigger AI topic suggestion
                    sug = suggest_topic_name(result["excerpt"], [t["name"] for t in current_topics.values()])
                    if sug:
                        st.toast(f"💡 Suggested topic: {sug}")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_dataset": result["dataset_name"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")

        st.markdown("---")

        # --- YouTube URL ---
        st.markdown("#### 🎥 YouTube Video")
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input",
            label_visibility="collapsed",
        )
        if yt_url:
            if st.button("🎥 Ingest YouTube", key="btn_ingest_yt", use_container_width=True):
                with st.spinner("🎥 Fetching transcript & uploading to Cognee Cloud..."):
                    result = run_async(ingest_youtube(yt_url, topic_id=ingest_topic_id))

                if result["status"] == "success":
                    st.session_state.sources.append(result)
                    st.success(f"✅ Ingested **{result['title']}**", icon="🎥")
                    # Trigger AI topic suggestion
                    sug = suggest_topic_name(result["excerpt"], [t["name"] for t in current_topics.values()])
                    if sug:
                        st.toast(f"💡 Suggested topic: {sug}")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_dataset": result["dataset_name"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")

    # --- Web Article ---
    with ingest_col2:
        st.markdown("#### 🌐 Web Article")
        web_url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
            key="web_url_input",
            label_visibility="collapsed",
        )
        if web_url:
            if st.button("🌐 Ingest Article", key="btn_ingest_web", use_container_width=True):
                with st.spinner("🌐 Extracting article & uploading to Cognee Cloud..."):
                    result = run_async(ingest_url(web_url, topic_id=ingest_topic_id))

                if result["status"] == "success":
                    st.session_state.sources.append(result)
                    st.success(f"✅ Ingested **{result['title'][:60]}**", icon="🌐")
                    # Trigger AI topic suggestion
                    sug = suggest_topic_name(result["excerpt"], [t["name"] for t in current_topics.values()])
                    if sug:
                        st.toast(f"💡 Suggested topic: {sug}")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_dataset": result["dataset_name"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")

        st.markdown("---")

        # --- Pasted Notes ---
        st.markdown("#### 📝 Paste Notes")
        note_title = st.text_input(
            "Note title",
            placeholder="e.g. Lecture 3: Transformer Architecture",
            key="note_title_input",
        )
        note_text = st.text_area(
            "Your notes",
            placeholder="Paste or type your study notes here...",
            height=180,
            key="note_text_input",
        )
        if note_text:
            if st.button("📝 Ingest Notes", key="btn_ingest_note", use_container_width=True):
                with st.spinner("📝 Processing notes & uploading to Cognee Cloud..."):
                    result = run_async(
                        ingest_text(note_text, title=note_title or "Untitled Note", topic_id=ingest_topic_id)
                    )

                if result["status"] == "success":
                    st.session_state.sources.append(result)
                    st.success(f"✅ Ingested **{result['title']}**", icon="📝")
                    # Trigger AI topic suggestion
                    sug = suggest_topic_name(result["excerpt"], [t["name"] for t in current_topics.values()])
                    if sug:
                        st.toast(f"💡 Suggested topic: {sug}")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_dataset": result["dataset_name"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")


# ========================== TAB 2: ASK ==========================
with tab_ask:
    st.markdown("### 💬 Ask Your Study Material")
    st.caption("Ask questions and get answers synthesized across all your ingested sources.")

    # Check if we have sources
    if not [s for s in st.session_state.sources if s.get("status") == "success"]:
        st.info("📚 Ingest some study material first using the **Ingest** tab.", icon="💡")
    else:
        # Load topics
        current_topics = load_topics()
        ask_options = {"All Topics": "all_topics"}
        for tid, t in current_topics.items():
            ask_options[t["name"]] = tid

        # Find the default index based on sidebar selection
        sidebar_tid = st.session_state.selected_topic
        default_ask_idx = 0
        if sidebar_tid != "all_topics":
            sb_name = current_topics.get(sidebar_tid, {}).get("name")
            if sb_name in ask_options:
                default_ask_idx = list(ask_options.keys()).index(sb_name)

        col_scope, col_cross = st.columns([2, 2])
        with col_scope:
            ask_scope_name = st.selectbox("Ask about:", list(ask_options.keys()), index=default_ask_idx)
            selected_ask_tid = ask_options[ask_scope_name]

        with col_cross:
            # Cross-topic search is only available when a specific topic is selected
            enable_cross = False
            if selected_ask_tid != "all_topics":
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                enable_cross = st.checkbox("🔍 Find connections across topics", value=False)

        st.markdown("---")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg.get("primary_section") or msg.get("cross_section"):
                    st.markdown("### 🎯 From your topic")
                    st.markdown(msg["primary_section"])
                    if msg.get("cross_section"):
                        st.markdown("### 🔗 Connected from other topics")
                        st.markdown(msg["cross_section"])
                else:
                    st.markdown(msg["content"])
                
                if msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for src in msg["sources"]:
                            st.caption(src)

        # Chat input
        question = st.chat_input("Ask a question about your study material...")

        if question:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Querying Cognee Cloud knowledge graph..."):
                    if selected_ask_tid == "all_topics":
                        result = run_async(ask(question))
                    elif enable_cross:
                        result = run_async(cross_topic_ask(question, selected_ask_tid, list(current_topics.keys())))
                    else:
                        result = run_async(ask_scoped(question, selected_ask_tid))

                if result.get("error"):
                    st.error(result["error"])
                    answer_text = f"⚠️ Error: {result['error']}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": [],
                    })
                else:
                    answer_text = result.get("answer", "No answer generated.")
                    
                    if result.get("primary_section") or result.get("cross_section"):
                        # Custom render for cross-topic synthesis
                        st.markdown(f"### 🎯 From your topic ({ask_scope_name})")
                        st.markdown(result["primary_section"])
                        st.markdown("### 🔗 Connected from other topics")
                        if result.get("cross_section"):
                            st.markdown(result["cross_section"])
                        else:
                            st.caption("No connections found.")
                    else:
                        st.markdown(answer_text)

                    if result.get("sources"):
                        with st.expander("📎 Sources"):
                            for src in result["sources"]:
                                st.caption(src)

                    st.caption(f"_Search type: {result.get('search_type', 'unknown')}_")

                    # Save to history with sections
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "primary_section": result.get("primary_section", ""),
                        "cross_section": result.get("cross_section", ""),
                        "sources": result.get("sources", []),
                    })
            st.rerun()

        # Topic summary section
        st.markdown("---")
        st.markdown("### 📋 Topic Summary")
        st.caption("Get a cross-source summary for any topic.")

        summary_topic = st.text_input(
            "Topic to summarize",
            placeholder="e.g. attention mechanism, gradient descent, ...",
            key="summary_topic_input",
        )
        if summary_topic:
            if st.button("📋 Generate Summary", key="btn_summarize", use_container_width=True):
                with st.spinner(f"📋 Summarizing '{summary_topic}' via Cognee Cloud..."):
                    result = run_async(summarize_topic(summary_topic))

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.markdown(result.get("summary", "No summary generated."))
                    if result.get("sources_referenced"):
                        st.markdown("**Sources referenced:**")
                        for src in result["sources_referenced"]:
                            st.caption(f"  • {src}")


# ========================== TAB 3: CONNECT TOPICS ==========================
with tab_connect:
    st.markdown("### 🔗 Connect Topics")
    st.caption(
        "Discover how two topics are connected through your study material. "
        "This traverses the knowledge graph to find relationship paths."
    )

    # Check if we have sources
    if not [s for s in st.session_state.sources if s.get("status") == "success"]:
        st.info("📚 Ingest some study material first using the **Ingest** tab.", icon="💡")
    else:
        col_a, col_mid, col_b = st.columns([2, 1, 2])

        with col_a:
            topic_a = st.text_input(
                "Topic A",
                placeholder="e.g. transformers",
                key="topic_a_input",
                label_visibility="visible",
            )

        with col_mid:
            st.markdown(
                '<div style="text-align: center; padding-top: 2rem; font-size: 1.5rem; color: #7C5CFC;">⟷</div>',
                unsafe_allow_html=True,
            )

        with col_b:
            topic_b = st.text_input(
                "Topic B",
                placeholder="e.g. attention mechanism",
                key="topic_b_input",
                label_visibility="visible",
            )

        if topic_a and topic_b:
            if st.button(
                "🔗 Find Connection",
                key="btn_connect",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner(f"🔗 Finding connections between '{topic_a}' and '{topic_b}' via Cognee Cloud..."):
                    result = run_async(connect_topics(topic_a, topic_b))

                if result.get("error"):
                    st.error(result["error"])
                else:
                    # Graph visualization
                    if result.get("html_graph"):
                        st.markdown(
                            '<div class="graph-container">',
                            unsafe_allow_html=True,
                        )
                        st.components.v1.html(
                            result["html_graph"],
                            height=540,
                            scrolling=False,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Legend
                        st.markdown("""
                        <div style="display: flex; gap: 16px; justify-content: center; padding: 8px 0; flex-wrap: wrap;">
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: #8b949e;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #7C5CFC; display: inline-block;"></span> Topic
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: #8b949e;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #1E90FF; display: inline-block;"></span> Shared Concept
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: #8b949e;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #FF6B6B; display: inline-block;"></span> PDF
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: #8b949e;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #2ED573; display: inline-block;"></span> Web
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; color: #8b949e;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #FFA502; display: inline-block;"></span> Note
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Connection summary
                    if result.get("summary"):
                        st.markdown("---")
                        st.markdown("#### 📖 Connection Explained")
                        st.markdown(result["summary"])

                    # Raw graph data (expandable)
                    if result.get("nodes"):
                        with st.expander(f"🔍 Graph Details ({len(result['nodes'])} nodes, {len(result['edges'])} edges)"):
                            st.markdown("**Nodes:**")
                            for node in result["nodes"]:
                                st.caption(
                                    f"• {node['label']} "
                                    f"({'🔵 ' + node['source_type'] if node.get('source_type') else ''})"
                                )
                            st.markdown("**Edges:**")
                            for edge in result["edges"]:
                                st.caption(
                                    f"• {edge.get('from_label', '?')} "
                                    f"—[{edge.get('relationship', '?')}]→ "
                                    f"{edge.get('to_label', '?')}"
                                )
