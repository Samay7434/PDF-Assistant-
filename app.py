"""
app.py :-

Streamlit web UI for the AI Research Paper Assistant.

Run:
    streamlit run app.py

Features:
  - Upload one or more PDFs via the sidebar
  - Ask natural-language questions → get answers with cited sources
  - View exact source paragraphs (highlighted)
  - Summarise individual papers
  - Compare how multiple papers discuss a topic
"""

import os
import tempfile
from pathlib import Path
import streamlit as st
import llm as llm_module

from auth import authenticate_user, create_user, normalize_username
from ingest import extract_text_from_pdf, chunk_pages, build_index
from retriever import Retriever
from embeddings import EmbeddingModelError
from llm import (
    ask_llm_stream,
    summarise_pdf,
    generate_notes,
    generate_flashcards,
    ollama_is_running,
    openai_is_configured,
)
from utils import save_chat, load_chat, clear_session, index_exists, user_index_dir
from config import (
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
)

# Keep the UI compatible with an already-running Streamlit process that has
# loaded an older llm.py module before the model-list helper was added.
ollama_available_models = getattr(llm_module, "ollama_available_models", lambda: [])

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="DocMind", layout="wide")

# ─────────────────────────────────────────
# DOCMIND CHAT THEME
# ─────────────────────────────────────────
st.markdown("""
<style>
:root { color-scheme: dark; }
body, .stApp, [data-testid="stAppViewContainer"] {
    background: #0b0d10 !important;
    color: #f3f4f6 !important;
}
[data-testid="stHeader"] { background: rgba(11, 13, 16, .92); }
[data-testid="stMainBlockContainer"] {
    max-width: 1040px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}
[data-testid="stSidebar"] { background: #12151a; }
[data-testid="stSidebar"] * { color: #e5e7eb; }
h1 { font-size: 2.15rem !important; line-height: 1.2 !important; margin-bottom: 1.25rem !important; }
h2, h3, h4, p, label, [data-testid="stWidgetLabel"] *,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { color: #e5e7eb; }
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: .75rem 0 !important;
    gap: .8rem;
    align-items: flex-start;
}
[data-testid="stChatMessageContent"] { min-width: 0; line-height: 1.65; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    display: flex;
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    width: fit-content;
    max-width: min(82%, 760px);
    background: #222831;
    border: 1px solid #303844;
    border-radius: 18px 7px 18px 18px;
    padding: .65rem 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { margin: 0; color: #f9fafb !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p { color: #e5e7eb !important; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    border-radius: 12px;
    width: 2.3rem;
    height: 2.3rem;
}
[data-testid="stBottom"] { background: linear-gradient(transparent, #0b0d10 22%) !important; }
[data-testid="stBottom"] [data-testid="stChatInput"],
[data-testid="stBottom"] [data-testid="stChatInput"] > div,
[data-testid="stBottom"] [data-testid="stChatInput"] > div > div {
    background: #171b21 !important;
    border: 1px solid #343b46 !important;
    border-radius: 16px !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f3f4f6 !important;
    border: 0 !important;
    min-height: 3.25rem;
}
input, textarea, [data-baseweb="select"] > div {
    background: #1d2229 !important;
    color: #f3f4f6 !important;
    border-color: #343b46 !important;
}
input::placeholder, textarea::placeholder { color: #9ca3af !important; }
button[kind], [data-testid="stFormSubmitButton"] button { border-radius: 10px !important; }
[data-testid="stFileUploader"] section { background: #1d2229; border-color: #343b46; }
hr { border-color: #252a32; }
@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 6rem; }
    h1 { font-size: 1.7rem !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] { max-width: 92%; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────
def openai_api_key() -> str | None:
    try:
        openai_config = st.secrets.get("openai", {})
        return openai_config.get("api_key") or st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


def load_display_chat(username: str) -> list:
    """Drop transient provider failures left in chat history by older runs."""
    return [
        (role, message)
        for role, message in load_chat(username)
        if not (
            role == "assistant"
            and isinstance(message, str)
            and message.startswith("Error contacting LLM:")
        )
    ]


if "username" not in st.session_state:
    st.session_state.username = None
if "display_name" not in st.session_state:
    st.session_state.display_name = None
if not st.session_state.username:
    st.title("DocMind")

    login_tab, signup_tab = st.tabs(["Login", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            username = normalize_username(login_username)
            if authenticate_user(username, login_password):
                st.session_state.username = username
                st.session_state.display_name = username
                st.session_state.messages = load_display_chat(username)
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with signup_tab:
        with st.form("signup_form"):
            signup_username = st.text_input("Choose a username")
            signup_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True)

        if submitted:
            if signup_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(signup_username, signup_password)
                if ok:
                    st.session_state.username = message
                    st.session_state.display_name = message
                    st.session_state.messages = []
                    save_chat([], message)
                    st.rerun()
                else:
                    st.error(message)

    st.stop()

current_user = st.session_state.username

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "retriever" not in st.session_state:
    if index_exists(current_user):
        try:
            st.session_state.retriever = Retriever(index_dir=user_index_dir(current_user))
        except Exception as e:
            st.session_state.retriever = None
            st.session_state.index_load_error = str(e)
    else:
        st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = load_display_chat(current_user)

if "llm_provider" not in st.session_state:
    default_provider = DEFAULT_LLM_PROVIDER
    if default_provider == "openai" and not openai_is_configured(openai_api_key()):
        default_provider = "ollama"
    st.session_state.llm_provider = default_provider

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = DEFAULT_OLLAMA_MODEL

if "openai_model" not in st.session_state:
    st.session_state.openai_model = DEFAULT_OPENAI_MODEL

# ─────────────────────────────────────────
# SIDEBAR (IMPROVED STRUCTURE)
# ─────────────────────────────────────────
with st.sidebar:
    st.title("DocMind")
    st.caption(f"Signed in as {st.session_state.display_name or current_user}")

    if st.button("Logout", use_container_width=True):
        st.session_state.username = None
        st.session_state.display_name = None
        st.session_state.messages = []
        st.session_state.pop("retriever", None)
        st.session_state.pop("index_load_error", None)
        st.rerun()

    # Model
    st.subheader("Model")
    st.radio(
        "Provider",
        ["openai", "ollama"],
        index=0 if st.session_state.llm_provider == "openai" else 1,
        horizontal=True,
        key="llm_provider",
    )

    api_key = openai_api_key()
    if st.session_state.llm_provider == "openai":
        if openai_is_configured(api_key):
            st.success("OpenAI configured")
        else:
            st.error("Add OPENAI_API_KEY in Streamlit secrets")
        st.text_input(
            "OpenAI model",
            value=st.session_state.openai_model,
            key="openai_model",
        )
    else:
        if ollama_is_running():
            st.success("Ollama running")
            available_models = ollama_available_models()
            if available_models:
                if st.session_state.ollama_model not in available_models:
                    st.session_state.ollama_model = available_models[0]
                st.selectbox("Ollama model", available_models, key="ollama_model")
            else:
                st.warning("No Ollama models are installed. In a terminal, run `ollama pull mistral`, then refresh this page.")
                st.text_input("Ollama model", key="ollama_model")
        else:
            st.error("Run: ollama serve")
            st.text_input("Ollama model", key="ollama_model")

    # ───────────────
    # DOCUMENTS
    # ───────────────
    st.divider()
    st.subheader("Documents")

    if st.session_state.retriever:
        st.caption(f"Indexed chunks: {len(st.session_state.retriever.chunks)}")
    elif index_exists(current_user) and "index_load_error" in st.session_state:
        st.warning(f"Could not load saved index: {st.session_state.index_load_error}")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📥 Index Papers", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF")
        else:
            all_chunks = []

            with st.spinner("Processing PDFs..."):
                for file in uploaded_files:
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = Path(tmp.name)

                        pages = extract_text_from_pdf(tmp_path)
                        for page in pages:
                            page["source"] = file.name
                        all_chunks.extend(chunk_pages(pages))
                    except Exception as e:
                        st.warning(f"Could not process {file.name}: {e}")
                    finally:
                        if tmp_path:
                            tmp_path.unlink(missing_ok=True)

            if not all_chunks:
                st.error("No text extracted")
            else:
                try:
                    build_index(all_chunks, index_dir=user_index_dir(current_user))
                    st.session_state.retriever = Retriever(index_dir=user_index_dir(current_user))
                    st.session_state.pop("index_load_error", None)
                    st.session_state.messages = []
                    save_chat([], current_user)
                    st.success("Indexed successfully")
                except EmbeddingModelError as e:
                    st.session_state.retriever = None
                    st.session_state.index_load_error = str(e)
                    st.error(str(e))
                except Exception as e:
                    st.session_state.retriever = None
                    st.session_state.index_load_error = str(e)
                    st.error(f"Indexing failed: {e}")

    # ───────────────
    # AI TOOLS
    # ───────────────
    st.divider()
    st.subheader("AI Tools")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Summarize", use_container_width=True):
            if st.session_state.retriever:
                summary = summarise_pdf(
                    st.session_state.retriever.chunks,
                    model=st.session_state.openai_model if st.session_state.llm_provider == "openai" else st.session_state.ollama_model,
                    provider=st.session_state.llm_provider,
                    api_key=openai_api_key(),
                )
                st.session_state.messages.append(("assistant", summary))
                save_chat(st.session_state.messages, current_user)
            else:
                st.warning("Upload and index PDFs first.")

    with col2:
        if st.button("📒 Notes", use_container_width=True):
            if st.session_state.retriever:
                notes = generate_notes(
                    st.session_state.retriever.chunks,
                    model=st.session_state.openai_model if st.session_state.llm_provider == "openai" else st.session_state.ollama_model,
                    provider=st.session_state.llm_provider,
                    api_key=openai_api_key(),
                )
                st.session_state.messages.append(("assistant", notes))
                save_chat(st.session_state.messages, current_user)
            else:
                st.warning("Upload and index PDFs first.")

    if st.button("🃏 Flash Cards", use_container_width=True):
        if st.session_state.retriever:
            with st.spinner("Generating flash cards..."):
                cards = generate_flashcards(
                    st.session_state.retriever.chunks,
                    model=st.session_state.openai_model if st.session_state.llm_provider == "openai" else st.session_state.ollama_model,
                    provider=st.session_state.llm_provider,
                    api_key=openai_api_key(),
                )
            error_card = next((card for card in cards if card.get("q") == "Error generating flash cards"), None)
            if error_card:
                st.error(error_card.get("a", "Flash card generation failed."))
            else:
                cards_markdown = "## Flash Cards\n\n" + "\n\n".join(
                    f"**{index}. {card['q']}**\n\n{card['a']}"
                    for index, card in enumerate(cards, start=1)
                )
                st.session_state.messages.append(("assistant", cards_markdown))
                save_chat(st.session_state.messages, current_user)
        else:
            st.warning("Upload and index PDFs first.")

    # ───────────────
    # SYSTEM
    # ───────────────
    st.divider()
    st.subheader("System")

    if st.button("🗑 Clear Chat", use_container_width=True):
        clear_session(current_user)
        st.session_state.messages = []
        st.session_state.pop("index_load_error", None)
        st.rerun()

    if st.button("Clear Documents", use_container_width=True):
        clear_session(current_user, clear_index=True)
        st.session_state.retriever = None
        st.session_state.pop("index_load_error", None)
        st.rerun()

# ─────────────────────────────────────────
# MAIN CHAT
# ─────────────────────────────────────────
st.title("Chat with your PDF")

if st.session_state.retriever is None:
    if index_exists(current_user) and "index_load_error" in st.session_state:
        st.error("A saved index exists, but it could not be loaded. Clear chat or re-index your PDFs.")
    else:
        st.info("Upload and index PDFs first")
else:
    # Display messages
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(msg)

    # Input
    question = st.chat_input("Ask anything...")

    if question:
        st.session_state.messages.append(("user", question))

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            response_box = st.empty()
            full = ""

            question_lower = question.lower()
            asks_for_multiple_details = any(
                term in question_lower
                for term in ("all", "more", "other", "instruction", "rules", "list", "summarize")
            )
            results = st.session_state.retriever.search(
                question, k=20 if asks_for_multiple_details else 12
            )

            for token in ask_llm_stream(
                question,
                results,
                st.session_state.messages,
                model=st.session_state.openai_model if st.session_state.llm_provider == "openai" else st.session_state.ollama_model,
                provider=st.session_state.llm_provider,
                api_key=openai_api_key(),
            ):
                full += token
                response_box.markdown(full + "▌")

            response_box.markdown(full)

        st.session_state.messages.append(("assistant", full))
        save_chat(st.session_state.messages, current_user)
