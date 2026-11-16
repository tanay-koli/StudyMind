"""
StudyMind — Personal Study Assistant
=====================================
Streamlit single-page app with 4 tabs: Ingest, Agent, Ask, Connect Topics.
A thin client: every action is an HTTP call to the StudyMind API.

Run with:  uvicorn api.main:app   (in one terminal)
           streamlit run ui/app.py (in another)
"""

import json
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `core.*` imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api(method: str, path: str, timeout: int = 300, **kwargs):
    """Call the StudyMind API and return its JSON; show the error and stop on failure."""
    try:
        resp = httpx.request(method, f"{settings.api_url}{path}", timeout=timeout, **kwargs)
    except httpx.TransportError:
        st.error(
            f"Cannot reach the StudyMind API at {settings.api_url}. "
            "Start it with: uvicorn api.main:app"
        )
        st.stop()
    if resp.status_code >= 400:
        st.error(f"API error {resp.status_code}: {resp.text[:300]}")
        st.stop()
    return resp.json()


def _sse_events(lines):
    """Yield (event, data) from SSE lines; data is the parsed JSON (the API never emits raw newlines in it)."""
    event = None
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:") and event:
            yield event, json.loads(line[5:].strip())
            event = None


def suggest(excerpt: str) -> str:
    return api("POST", "/topics/suggest", json={"excerpt": excerpt})["name"]


# Every widget click reruns the page; without the cache each figure in the chat is fetched again.
# ttl bounds a stale image if an old database reuses a deleted chunk's id.
@st.cache_data(ttl=600, show_spinner=False)
def _figure_bytes(url: str) -> bytes:
    resp = httpx.get(url, timeout=httpx.Timeout(30, connect=3))
    resp.raise_for_status()  # errors are not cached, so a figure that failed is retried next rerun
    return resp.content


def _figure(image_url: str, caption: str | None = None) -> None:
    """Fetch figure bytes server-side: under compose the browser cannot resolve the api host."""
    try:
        data = _figure_bytes(settings.api_url + image_url)
    except httpx.HTTPError:
        st.caption(f"{caption or 'Figure'} (image unavailable)")
        return
    st.image(data, caption=caption)


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _render_citations(citations: list[dict], answer: str = "") -> None:
    """One chip per cited chunk, numbered like the [n] markers; click to read the chunk.

    Figures the answer cites as [n] are also shown below it (at most 2).
    """
    if not citations:
        return
    for col, (n, c) in zip(st.columns(len(citations)), enumerate(citations, start=1)):
        kind = c.get("kind", "text")
        page = f" · p.{c['page']}" if c.get("page") else ""
        if kind == "figure":
            label = f"[{n}] :material/image: {c['source_title']}{page}"
        elif kind == "video" and c.get("start_s") is not None:
            label = f"[{n}] :material/smart_display: {c['source_title']} @ {_clock(c['start_s'])}"
        else:
            label = f"[{n}] :material/description: {c['source_title']}{page}"
        with col, st.popover(label, width="stretch"):
            if c.get("image_url"):
                _figure(c["image_url"])
            st.markdown(c["text"])
            if c.get("url"):
                st.link_button(f"Open at {_clock(c['start_s'])}" if c.get("start_s") is not None else "Open",
                               c["url"])
    cited = [(n, c) for n, c in enumerate(citations, start=1) if c.get("image_url") and f"[{n}]" in answer][:2]
    for n, c in cited:
        _figure(c["image_url"],
                f"[{n}] {c['source_title']}" + (f" · p.{c['page']}" if c.get("page") else ""))


# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyMind — AI Study Assistant",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Layout tweaks the theme cannot express (colours and fonts live in .streamlit/config.toml)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 1180px; }
    ::selection { background: rgba(52, 81, 199, 0.22); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {role, content} dicts

if "health" not in st.session_state:
    st.session_state.health = api("GET", "/health")

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
    # LLM status
    for label, status in (
        (f"LLM ({settings.llm_model})", st.session_state.health["llm"]),
        (f"Vision ({settings.vision_model})", st.session_state.health["vision"]),
    ):
        badge = ":green-badge[Connected]" if status["ok"] else ":red-badge[Disconnected]"
        st.markdown(f"{label} {badge}")
        if not status["ok"] and status.get("error"):
            st.error(status["error"], icon=":material/error:")

    if st.button("Refresh status", icon=":material/refresh:", use_container_width=True):
        st.session_state.health = api("GET", "/health")
        st.rerun()

    # Learner progress: what the review would ask about, and what StudyMind remembers about you
    learner = api("GET", "/learner")
    with st.expander("Your progress", icon=":material/trending_up:"):
        st.caption(f"{learner['to_review']} concepts to review")
        for m in learner["mastery"][:5]:
            mark = {"correct": ":green[:material/check_circle:]", "partial": ":orange[:material/timelapse:]",
                    "wrong": ":red[:material/cancel:]"}.get(m["last_verdict"], ":gray[:material/circle:]")
            st.markdown(f"{mark} {m['concept']} — next review {m['next_review']}")
        if not learner["mastery"]:
            st.caption("No quiz answers yet. Try 'quiz me' in the Agent tab.")
        if learner["facts"]:
            st.markdown("**Remembered about you**")
        for f in learner["facts"]:
            col_fact, col_del = st.columns([5, 1])
            col_fact.caption(f["fact"])
            if col_del.button(":material/delete:", key=f"fact_{f['id']}", help="Forget this", type="tertiary"):
                api("DELETE", f"/learner/facts/{f['id']}")
                st.rerun()

    st.markdown("---")

    # Load topics and sources
    topics = api("GET", "/topics")
    sources = api("GET", "/sources")

    # Topics Section
    st.markdown("#### Topics")

    # "All Topics" selector
    all_selected = st.session_state.selected_topic == "all_topics"
    if st.button(
        "All Topics",
        key="btn_all_topics",
        icon=":material/folder_open:",
        use_container_width=True,
        type="primary" if all_selected else "secondary",
    ):
        st.session_state.selected_topic = "all_topics"
        st.rerun()

    for tid, topic in topics.items():
        count = sum(1 for s in sources if s["topic_id"] == tid)
        if st.button(
            f"{topic['name']} ({count})",
            key=f"sel_{tid}",
            use_container_width=True,
            type="primary" if st.session_state.selected_topic == tid else "secondary",
        ):
            st.session_state.selected_topic = tid
            st.rerun()

    # Study Guide and Audio Overview act on the selected topic: labelled buttons instead of tiny icons per row
    sel_tid = st.session_state.selected_topic
    if sel_tid in topics:
        if st.button("Study Guide", key=f"sg_{sel_tid}", icon=":material/menu_book:", use_container_width=True,
                     help=f"Study Guide for {topics[sel_tid]['name']}"):
            st.session_state.active_study_guide_topic = sel_tid
            st.session_state.active_audio_topic = None
            st.rerun()
        if st.button("Audio Overview", key=f"aud_{sel_tid}", icon=":material/podcasts:", use_container_width=True,
                     help=f"Audio Overview for {topics[sel_tid]['name']}"):
            st.session_state.active_audio_topic = sel_tid
            st.session_state.active_study_guide_topic = None
            st.rerun()

    # New Topic Form
    with st.expander("New topic", icon=":material/add:"):
        new_topic_name = st.text_input("Topic Name", key="sidebar_new_topic_input")
        if st.button("Create Topic", key="btn_create_topic", use_container_width=True):
            if new_topic_name.strip():
                new_t = api("POST", "/topics", json={"name": new_topic_name})
                st.session_state.selected_topic = new_t["id"]
                st.success(f"Topic '{new_topic_name}' created!")
                st.rerun()

    # Render Study Guide Expander below topic list
    if st.session_state.active_study_guide_topic:
        sg_tid = st.session_state.active_study_guide_topic
        sg_topic = topics.get(sg_tid)
        if sg_topic:
            st.markdown("---")
            with st.expander(f"Study Guide: {sg_topic['name']}", expanded=True, icon=":material/menu_book:"):
                # Close button
                if st.button("Close Guide", key="close_sg"):
                    st.session_state.pop(f"sg_{sg_tid}", None)
                    st.session_state.active_study_guide_topic = None
                    st.rerun()

                sg_key = f"sg_{sg_tid}"
                regen = st.button("Regenerate", key="regen_sg")
                if regen or sg_key not in st.session_state:
                    with st.spinner("Generating study guide..."):
                        st.session_state[sg_key] = api(
                            "POST", f"/topics/{sg_tid}/study-guide", params={"force": "true"} if regen else None,
                            timeout=900)
                guide_content = st.session_state[sg_key]["guide"]

                st.markdown(guide_content)

                if "Could not generate study guide" not in guide_content and "No sources" not in guide_content:
                    st.download_button(
                        label="Download as .md",
                        icon=":material/download:",
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
            with st.expander(f"Audio Overview: {aud_topic['name']}", expanded=True, icon=":material/podcasts:"):
                # Close button
                if st.button("Close Audio", key="close_aud"):
                    st.session_state.active_audio_topic = None
                    st.rerun()

                with st.spinner("Generating audio script..."):
                    script_content = api("POST", f"/topics/{aud_tid}/audio-script")["script"]

                # Synthesize once per script: every widget click reruns this page
                wav_key = (aud_tid, script_content)
                if st.session_state.get("audio_wav", (None, None))[0] != wav_key:
                    with st.spinner("Synthesizing audio file..."):
                        audio_resp = httpx.post(f"{settings.api_url}/topics/{aud_tid}/audio", timeout=300)
                    ok = audio_resp.status_code == 200
                    st.session_state.audio_wav = (wav_key, audio_resp.content if ok else None)

                if wav := st.session_state.audio_wav[1]:
                    st.audio(wav, format="audio/wav")
                else:
                    st.info("Audio unavailable (no script yet, or pyttsx3 not installed). "
                            "Showing the script text only.")

                st.markdown("**Podcast Transcript:**")
                st.caption(script_content)

    st.markdown("---")

    # Knowledge Base Sources
    st.markdown("#### Sources")

    # Filter sources list by active topic
    active_tid = st.session_state.selected_topic

    if active_tid != "all_topics":
        active_topic_name = topics.get(active_tid, {}).get("name", active_tid)
        st.caption(f"Showing sources in **{active_topic_name}**")
        filtered_sources = [s for s in sources if s["topic_id"] == active_tid]
    else:
        st.caption("Showing all ingested sources")
        filtered_sources = sources

    if filtered_sources:
        for i, source in enumerate(filtered_sources):
            icon = {
                "pdf": "picture_as_pdf", "youtube": "smart_display", "web": "language", "note": "edit_note"
            }.get(source.get("source_type", ""), "draft")

            col1, col2 = st.columns([5, 1], vertical_alignment="center")
            with col1:
                st.markdown(f":material/{icon}: {source.get('title', 'Untitled')[:50]}")
                added = source.get("created_at", "")[:16].replace("T", " ")
                st.caption(f"{source.get('source_type', 'unknown')} · {added}")

            with col2:
                if st.button(":material/delete:", key=f"forget_{active_tid}_{i}", help="Remove this source",
                             type="tertiary"):
                    with st.spinner("Removing..."):
                        api("DELETE", f"/sources/{source['id']}")
                    st.rerun()

        if active_tid == "all_topics":
            st.markdown("---")
            if st.button("Forget everything", icon=":material/delete_forever:", use_container_width=True):
                with st.spinner("Clearing all knowledge..."):
                    api("DELETE", "/sources")
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.caption("No sources in this topic. Use the **Ingest Material** tab to add some.")


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

# Hero header
st.title("StudyMind", anchor=False)
st.caption("Ingest study material · Ask questions · Discover connections")

# Main tabs
tab_ingest, tab_agent, tab_ask, tab_connect = st.tabs([
    ":material/upload_file: Ingest Material",
    ":material/smart_toy: Agent",
    ":material/forum: Ask Questions",
    ":material/hub: Connect Topics",
])


# ========================== TAB 1: INGEST ==========================
with tab_ingest:
    st.markdown("### Add Study Material")
    st.caption("Ingest PDFs, YouTube videos, web articles, or your own notes into the knowledge graph.")

    # 1. Interactive Topic Suggestion Banner
    if st.session_state.suggested_topic:
        sug = st.session_state.suggested_topic
        st.info(f"**Suggested topic:** The content from **{sug['source_title']}** "
                f"seems to belong to topic **'{sug['name']}'**.", icon=":material/lightbulb:")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Add to suggested Topic", key="accept_suggestion", use_container_width=True, type="primary"):
                # Find or create the topic (the API returns an existing one by name)
                target_tid = api("POST", "/topics", json={"name": sug["name"]})["id"]

                # Re-associate the source with this topic
                api("PATCH", f"/sources/{sug['source_id']}", json={"topic_id": target_tid})

                st.session_state.suggested_topic = None
                st.session_state.selected_topic = target_tid
                st.success(f"Successfully added to topic '{sug['name']}'!")
                st.rerun()
        with col_no:
            if st.button("Keep in current topic", key="reject_suggestion", use_container_width=True):
                st.session_state.suggested_topic = None
                st.rerun()

    # 2. Topic Selector for Ingestion
    current_topics = topics
    topic_options = {t["name"]: tid for tid, t in current_topics.items()}
    topic_list = list(topic_options.keys()) + ["New topic…"]

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
    if selected_option == "New topic…":
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
                    new_t = api("POST", "/topics", json={"name": new_t_name})
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
        st.markdown("#### Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            if st.button("Ingest PDF", key="btn_ingest_pdf", icon=":material/upload:", use_container_width=True):
                form = {"title": uploaded_file.name}
                if ingest_topic_id:
                    form["topic_id"] = ingest_topic_id

                with st.spinner("Extracting, chunking and indexing…"):
                    result = api(
                        "POST",
                        "/ingest/pdf",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        data=form,
                    )

                if result["status"] == "success":
                    st.success(
                        f"Ingested **{result['title']}** "
                        f"({result.get('page_count', '?')} pages)",
                        icon=":material/check_circle:",
                    )
                    if result.get("figures_queued"):
                        if st.session_state.health["vision"]["ok"]:
                            st.info(f"{result['figures_queued']} figures queued for captioning — "
                                    "searchable in a few minutes.", icon=":material/image:")
                        else:
                            st.warning(f"{result['figures_queued']} figures are waiting for the vision model: "
                                       f"{st.session_state.health['vision']['error']}",
                                       icon=":material/image:")
                    # Trigger AI topic suggestion
                    sug = suggest(result["excerpt"])
                    if sug:
                        st.toast(f"Suggested topic: {sug}", icon=":material/lightbulb:")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_id": result["source_id"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(result.get("error", "Unknown error"), icon=":material/error:")

        st.markdown("---")

        # --- YouTube URL ---
        st.markdown("#### YouTube Video")
        yt_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input",
            label_visibility="collapsed",
        )
        if yt_url:
            if st.button("Ingest YouTube", key="btn_ingest_yt", icon=":material/upload:", use_container_width=True):
                with st.spinner("Fetching transcript and indexing…"):
                    result = api("POST", "/ingest/youtube", json={"url": yt_url, "topic_id": ingest_topic_id})

                if result["status"] == "success":
                    st.success(f"Ingested **{result['title']}**", icon=":material/check_circle:")
                    # Trigger AI topic suggestion
                    sug = suggest(result["excerpt"])
                    if sug:
                        st.toast(f"Suggested topic: {sug}", icon=":material/lightbulb:")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_id": result["source_id"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(result.get("error", "Unknown error"), icon=":material/error:")

    # --- Web Article ---
    with ingest_col2:
        st.markdown("#### Web Article")
        web_url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
            key="web_url_input",
            label_visibility="collapsed",
        )
        if web_url:
            if st.button("Ingest Article", key="btn_ingest_web", icon=":material/upload:", use_container_width=True):
                with st.spinner("Extracting article and indexing…"):
                    result = api("POST", "/ingest/url", json={"url": web_url, "topic_id": ingest_topic_id})

                if result["status"] == "success":
                    st.success(f"Ingested **{result['title'][:60]}**", icon=":material/check_circle:")
                    # Trigger AI topic suggestion
                    sug = suggest(result["excerpt"])
                    if sug:
                        st.toast(f"Suggested topic: {sug}", icon=":material/lightbulb:")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_id": result["source_id"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(result.get("error", "Unknown error"), icon=":material/error:")

        st.markdown("---")

        # --- Pasted Notes ---
        st.markdown("#### Paste Notes")
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
            if st.button("Ingest Notes", key="btn_ingest_note", icon=":material/upload:", use_container_width=True):
                with st.spinner("Indexing notes…"):
                    result = api("POST", "/ingest/text", json={
                        "text": note_text,
                        "title": note_title or "Untitled Note",
                        "topic_id": ingest_topic_id,
                    })

                if result["status"] == "success":
                    st.success(f"Ingested **{result['title']}**", icon=":material/check_circle:")
                    # Trigger AI topic suggestion
                    sug = suggest(result["excerpt"])
                    if sug:
                        st.toast(f"Suggested topic: {sug}", icon=":material/lightbulb:")
                        st.session_state.suggested_topic = {
                            "name": sug,
                            "source_id": result["source_id"],
                            "source_title": result["title"]
                        }
                    st.rerun()
                else:
                    st.error(result.get("error", "Unknown error"), icon=":material/error:")


# ========================== TAB: AGENT ==========================
with tab_agent:
    st.markdown("### Study Agent")
    st.caption("One chat that decides what to do: answer, explain, or connect topics. "
               "It retries weak searches and checks its answer against your sources.")

    if "agent_thread_id" not in st.session_state:
        st.session_state.agent_thread_id = str(uuid.uuid4())
        st.session_state.agent_messages = None
    if st.session_state.agent_messages is None:
        st.session_state.agent_messages = api("GET", f"/agent/{st.session_state.agent_thread_id}")["messages"]

    # A connect turn clears the picker; otherwise every later message would route to connect
    if st.session_state.pop("agent_connect_reset", False):
        st.session_state.agent_connect = []
    col_connect, col_new = st.columns([3, 1], vertical_alignment="bottom")
    with col_connect:
        connect_with = st.multiselect(
            "Connect with… (optional, one message)",
            [t["name"] for t in topics.values()],
            max_selections=2,
            key="agent_connect",
            help="Pick two topics to ask how they relate.",
        )
    with col_new:
        if st.button("New conversation", key="agent_new", icon=":material/add_comment:",
                     use_container_width=True):
            st.session_state.agent_thread_id = str(uuid.uuid4())
            st.session_state.agent_messages = []
            st.rerun()

    def _render_agent_reply(msg: dict) -> None:
        st.markdown(msg["content"])
        if msg.get("warning"):
            st.warning(msg["warning"])
        _render_citations(msg.get("citations") or [], msg["content"])
        if msg.get("graph"):
            st.components.v1.html(msg["graph"], height=500, scrolling=False)

    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            _render_agent_reply(msg)

    agent_question = st.chat_input("Ask, explain, or connect…", key="agent_input")
    if agent_question:
        st.session_state.agent_messages.append({"role": "user", "content": agent_question})
        with st.chat_message("user"):
            st.markdown(agent_question)

        sel = st.session_state.selected_topic
        body = {
            "message": agent_question,
            "thread_id": st.session_state.agent_thread_id,
            "topic_id": None if sel == "all_topics" else sel,
        }
        if len(connect_with) == 2:
            body["topics"] = connect_with
            st.session_state.agent_connect_reset = True

        with st.chat_message("assistant"):
            reply = None
            answer_box = st.empty()
            with st.status("Thinking…", expanded=True) as status:
                buffer, attempt = "", 1
                try:
                    with httpx.stream("POST", f"{settings.api_url}/agent", json=body, timeout=900) as resp:
                        if resp.status_code >= 400:
                            resp.read()
                            status.update(label="Failed", state="error")
                            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                        for line in resp.iter_lines() if resp.status_code < 400 else []:
                            if not line:
                                continue
                            event = json.loads(line)
                            if event["event"] == "node":
                                step = event["label"] + (f' → "{event["query"]}"' if event.get("query") else "")
                                st.write(step)
                                status.update(label=event["label"].capitalize() + "…")
                            elif event["event"] == "token":
                                if event["attempt"] != attempt:  # self-check regenerated the answer
                                    buffer, attempt = "", event["attempt"]
                                buffer += event["text"]
                                answer_box.markdown(buffer)
                            elif event["event"] == "done":
                                status.update(label="Done", state="complete", expanded=False)
                                answer_box.empty()
                                reply = event
                            else:
                                status.update(label="Failed", state="error")
                                st.error(event["error"])
                except httpx.TransportError:
                    status.update(label="Failed", state="error")
                    st.error(
                        f"Cannot reach the StudyMind API at {settings.api_url}. "
                        "Start it with: uvicorn api.main:app"
                    )
            if reply:
                msg = {"role": "assistant", "content": reply["answer"], "warning": reply.get("warning"),
                       "citations": reply.get("citations"), "graph": reply.get("graph")}
                _render_agent_reply(msg)
                st.session_state.agent_messages.append(msg)


# ========================== TAB 2: ASK ==========================
with tab_ask:
    st.markdown("### Ask Your Study Material")
    st.caption("Ask questions and get answers synthesized across all your ingested sources.")

    # Check if we have sources
    if not sources:
        st.info("Ingest some study material first using the **Ingest Material** tab.", icon=":material/info:")
    else:
        current_topics = topics
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

        col_scope, col_cross = st.columns(2, vertical_alignment="bottom")
        with col_scope:
            ask_scope_name = st.selectbox("Ask about:", list(ask_options.keys()), index=default_ask_idx)
            selected_ask_tid = ask_options[ask_scope_name]

        with col_cross:
            # Cross-topic search is only available when a specific topic is selected
            enable_cross = False
            if selected_ask_tid != "all_topics":
                enable_cross = st.checkbox("Find connections across topics", value=False)

        st.markdown("---")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg.get("primary_section") or msg.get("cross_section"):
                    st.markdown("#### From your topic")
                    st.markdown(msg["primary_section"])
                    if msg.get("cross_section"):
                        st.markdown("#### Connected from other topics")
                        st.markdown(msg["cross_section"])
                else:
                    st.markdown(msg["content"])

                _render_citations(msg.get("citations", []), msg.get("content", ""))
                if msg.get("sources") and not msg.get("citations"):
                    with st.expander("Sources", icon=":material/attach_file:"):
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
                if selected_ask_tid == "all_topics":
                    body = {"question": question, "mode": "global"}
                else:
                    mode = "cross" if enable_cross else "scoped"
                    body = {"question": question, "mode": mode, "topic_id": selected_ask_tid}
                if body["mode"] == "cross":
                    with st.spinner("Searching your material…"):
                        result = api("POST", "/ask", json=body)
                else:
                    result, box, buffer = None, st.empty(), ""
                    try:
                        with st.spinner("Searching your material…"), httpx.stream(
                            "POST", f"{settings.api_url}/ask/stream", json=body, timeout=900
                        ) as resp:
                            if resp.status_code >= 400:
                                resp.read()
                                st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                                st.stop()
                            for ev, data in _sse_events(resp.iter_lines()):
                                if ev == "token":
                                    buffer += data
                                    box.markdown(buffer)
                                elif ev == "done":
                                    result = data
                                elif ev == "error":
                                    result = {"error": data}
                    except httpx.TransportError:
                        st.error(
                            f"Cannot reach the StudyMind API at {settings.api_url}. "
                            "Start it with: uvicorn api.main:app"
                        )
                        st.stop()
                    box.empty()
                    if result is None:
                        result = {"error": "The answer stream ended unexpectedly."}

                if result.get("error"):
                    st.error(result["error"])
                    answer_text = f"Error: {result['error']}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": [],
                    })
                else:
                    answer_text = result.get("answer", "No answer generated.")
                    
                    if result.get("primary_section") or result.get("cross_section"):
                        # Custom render for cross-topic synthesis
                        st.markdown(f"#### From your topic ({ask_scope_name})")
                        st.markdown(result["primary_section"])
                        st.markdown("#### Connected from other topics")
                        if result.get("cross_section"):
                            st.markdown(result["cross_section"])
                        else:
                            st.caption("No connections found.")
                    else:
                        st.markdown(answer_text)

                    _render_citations(result.get("citations", []), answer_text)
                    if result.get("sources") and not result.get("citations"):
                        with st.expander("Sources", icon=":material/attach_file:"):
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
                        "citations": result.get("citations", []),
                    })
            st.rerun()

        # Topic summary section
        st.markdown("---")
        st.markdown("### Topic Summary")
        st.caption("Get a cross-source summary for any topic.")

        summary_topic = st.text_input(
            "Topic to summarize",
            placeholder="e.g. attention mechanism, gradient descent, ...",
            key="summary_topic_input",
        )
        if summary_topic:
            if st.button("Generate Summary", key="btn_summarize", icon=":material/summarize:",
                         use_container_width=True):
                with st.spinner(f"Summarizing '{summary_topic}'…"):
                    result = api("POST", "/summarize", json={"topic": summary_topic})

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.markdown(result.get("summary", "No summary generated."))
                    _render_citations(result.get("citations", []), result.get("summary", ""))


# ========================== TAB 3: CONNECT TOPICS ==========================
with tab_connect:
    st.markdown("### Connect Topics")
    st.caption(
        "Discover how two topics are connected through your study material. "
        "This traverses the knowledge graph to find relationship paths."
    )

    # Check if we have sources
    if not sources:
        st.info("Ingest some study material first using the **Ingest Material** tab.", icon=":material/info:")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            topic_a = st.text_input(
                "Topic A",
                placeholder="e.g. transformers",
                key="topic_a_input",
                label_visibility="visible",
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
                "Find Connection",
                key="btn_connect",
                icon=":material/hub:",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner(f"Finding connections between '{topic_a}' and '{topic_b}'…"):
                    result = api("POST", "/connect", json={"topic_a": topic_a, "topic_b": topic_b})

                if result.get("error"):
                    st.error(result["error"])
                else:
                    # Graph visualization
                    if result.get("html_graph"):
                        st.components.v1.html(
                            result["html_graph"],
                            height=540,
                            scrolling=False,
                        )

                        # Legend
                        st.markdown("""
                        <div style="display: flex; gap: 16px; justify-content: center; padding: 8px 0; flex-wrap: wrap;">
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; opacity: 0.75;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #7C5CFC; display: inline-block;"></span> Topic
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; opacity: 0.75;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #1E90FF; display: inline-block;"></span> Shared Concept
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; opacity: 0.75;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #FF6B6B; display: inline-block;"></span> PDF
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; opacity: 0.75;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #2ED573; display: inline-block;"></span> Web
                            </span>
                            <span style="display: flex; align-items: center; gap: 4px; font-size: 12px; opacity: 0.75;">
                                <span style="width: 10px; height: 10px; border-radius: 50%; background: #FFA502; display: inline-block;"></span> Note
                            </span>
                        </div>
                        """, unsafe_allow_html=True)  # noqa: E501

                    # Connection summary
                    if result.get("summary"):
                        st.markdown("---")
                        st.markdown("#### Connection Explained")
                        st.markdown(result["summary"])

                    # Raw graph data (expandable)
                    if result.get("nodes"):
                        n_nodes, n_edges = len(result["nodes"]), len(result["edges"])
                        with st.expander(f"Graph details ({n_nodes} nodes, {n_edges} edges)"):
                            st.markdown("**Nodes:**")
                            for node in result["nodes"]:
                                st.caption(
                                    f"• {node['label']} "
                                    f"({node['source_type'] if node.get('source_type') else ''})"
                                )
                            st.markdown("**Edges:**")
                            for edge in result["edges"]:
                                st.caption(
                                    f"• {edge.get('from_label', '?')} "
                                    f"—[{edge.get('relationship', '?')}]→ "
                                    f"{edge.get('to_label', '?')}"
                                )
