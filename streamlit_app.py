"""
streamlit_app.py
------------------
Deployment layer requested by the assignment ("Deployment using
Streamlit"). This is a thin UI wrapper -- it does NOT reimplement any
logic. All prompt engineering, domain grounding, and generation still
happen in core/ and domain/, exactly as in cli.py. This file only
handles: rendering the chat, session state (history + mode), and a
toggle between the mock backend (no GPU needed, good for quickly
demoing the UI itself) and the real LLM.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

from domain.registry import get_domain, DOMAIN_REGISTRY
from config import MODEL_CONFIG, GEN_CONFIG, DEFAULT_DOMAIN

st.set_page_config(page_title="RegBot — Domain Chatbot", page_icon="🎓", layout="centered")


# ----------------------------------------------------------------------
# Session state init (Streamlit reruns the whole script on every
# interaction, so anything that must persist across turns -- chat
# history, the loaded model, current mode -- lives in st.session_state)
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []          # real conversation memory
if "mode" not in st.session_state:
    st.session_state.mode = "few_shot"
if "domain_key" not in st.session_state:
    st.session_state.domain_key = DEFAULT_DOMAIN
if "use_real_model" not in st.session_state:
    st.session_state.use_real_model = False
if "bot" not in st.session_state:
    st.session_state.bot = None


@st.cache_resource(show_spinner=False)
def _load_real_bot_engine():
    """
    Cached ONCE regardless of domain -- loading the model is the slow
    part (GPU + disk), switching domains is not. We load with a
    placeholder domain and swap `.domain` afterwards, the same way
    cli.py's /domain command does, so custom domains built at runtime
    never trigger a model reload.
    """
    from core.chatbot import DomainChatbot
    placeholder_domain = get_domain(DEFAULT_DOMAIN)
    bot = DomainChatbot(placeholder_domain, MODEL_CONFIG, GEN_CONFIG)
    bot._load_model()
    return bot


def _get_bot():
    domain_key = st.session_state.domain_key
    domain = get_domain(domain_key)
    if st.session_state.use_real_model:
        with st.spinner(f"Loading {MODEL_CONFIG.model_name} onto GPU... (first message only)"):
            bot = _load_real_bot_engine()
        bot.domain = domain
        return bot
    else:
        from core.mock_chatbot import MockDomainChatbot
        return MockDomainChatbot(domain)


# ----------------------------------------------------------------------
# Sidebar: controls (mirror of the CLI's /mode, /domain, /reset commands)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    st.session_state.use_real_model = st.toggle(
        "Use real LLM",
        value=st.session_state.use_real_model,
        help="Off = instant mock backend (no GPU, tests the app itself). "
             "On = loads the real quantized model onto your GPU.",
    )

    st.session_state.domain_key = st.selectbox(
        "Domain",
        options=list(DOMAIN_REGISTRY.keys()),
        index=list(DOMAIN_REGISTRY.keys()).index(st.session_state.domain_key),
    )

    st.session_state.mode = st.radio(
        "Prompting technique",
        options=["zero_shot", "few_shot", "cot"],
        index=["zero_shot", "few_shot", "cot"].index(st.session_state.mode),
        help="Compare how the same question is answered under each "
             "prompt-engineering technique.",
    )

    if st.button("Reset conversation"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    domain_obj = get_domain(st.session_state.domain_key)
    with st.expander("Knowledge base used (no retrieval — injected in full)"):
        st.text(domain_obj.knowledge_base)

    st.divider()
    with st.expander("➕ Build a custom domain"):
        st.caption(
            "Swap in a completely different domain — no code changes. "
            "The same zero-shot/few-shot/CoT logic will run against "
            "whatever you provide here."
        )
        custom_name = st.text_input("Domain name", placeholder="e.g. company HR policy")
        custom_persona = st.text_area(
            "Persona / behavior instructions",
            placeholder="e.g. You are HRBot, an assistant for company leave and benefits policy...",
            height=80,
        )

        kb_source = st.radio("Knowledge base source", ["Paste text", "Upload file"], horizontal=True)
        kb_text = ""
        if kb_source == "Paste text":
            kb_text = st.text_area("Knowledge base text", height=150)
        else:
            uploaded = st.file_uploader("Upload .pdf or .txt", type=["pdf", "txt"])
            if uploaded is not None:
                from domain.custom_loader import extract_text_from_pdf, extract_text_from_txt
                if uploaded.name.lower().endswith(".pdf"):
                    kb_text = extract_text_from_pdf(uploaded.read())
                else:
                    kb_text = extract_text_from_txt(uploaded.read())
                st.text_area("Extracted text (preview/edit)", value=kb_text, height=150, key="kb_preview")
                kb_text = st.session_state.get("kb_preview", kb_text)

        few_shot_raw = st.text_area(
            "Few-shot examples (optional)",
            placeholder="Q: What is the return window?\nA: 30 days from purchase with a receipt.\n\nQ: ...\nA: ...",
            height=100,
        )

        if st.button("Activate this domain"):
            from domain.custom_loader import build_custom_domain, parse_few_shot_block
            few_shot = parse_few_shot_block(few_shot_raw) if few_shot_raw.strip() else []
            config, warning = build_custom_domain(custom_name, custom_persona, kb_text, few_shot)
            key = f"custom: {config.name}"
            DOMAIN_REGISTRY[key] = config
            st.session_state.domain_key = key
            st.session_state.history = []
            if warning:
                st.warning(warning)
            st.success(f"Activated domain '{config.name}'")
            st.rerun()


# ----------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------
st.title("🎓 RegBot")
st.caption(
    f"LLM-based domain chatbot (no retrieval) — currently answering from "
    f"**{domain_obj.name}** using **{st.session_state.mode}** prompting."
)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_input = st.chat_input("Ask a question about college academic regulations...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    bot = _get_bot()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = bot.ask(
                user_input,
                mode=st.session_state.mode,
                history=st.session_state.history,
            )
        st.markdown(result["response"])

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": result["response"]})
    # cap history length, same reasoning as cli.py
    st.session_state.history = st.session_state.history[-12:]
