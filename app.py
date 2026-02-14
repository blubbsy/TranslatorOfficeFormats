"""
OfficeTranslator - Privacy-First Document Translation Tool

A Streamlit application for translating documents locally using LLM backends.
"""

import io
import logging
import zipfile
import uuid
import atexit
import threading
import time
from typing import List

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from settings import settings
from processors import ProcessorFactory
from services.queue_manager import GlobalQueueManager, Job
from services.job_runner import run_translation_job
from utils.logging_handler import setup_logging, get_log_entries, clear_logs
from utils.text_utils import format_size
from utils.i18n import init_i18n, t, AVAILABLE_LANGUAGES, set_language
from utils.styles import CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="OfficeTranslator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logging(settings.log_level, "OfficeTranslator")

@st.cache_resource
def get_queue_manager() -> GlobalQueueManager:
    """Get the singleton global queue manager."""
    manager = GlobalQueueManager(processor_func=run_translation_job)
    atexit.register(manager.shutdown)
    return manager

def check_llm_connection_async():
    """Run connection check in background."""
    try:
        from services import TranslationService
        service = TranslationService()
        is_ready, msg = service.check_ready()
        
        st.session_state.llm_status = "ready" if is_ready else "error"
        st.session_state.llm_message = msg
    except Exception as e:
        st.session_state.llm_status = "error"
        st.session_state.llm_message = f"Error: {str(e)}"

def init_session():
    """Initialize session state and persistent user ID."""
    init_i18n()
    
    # Check for UID in query params for session persistence
    query_params = st.query_params
    uid = query_params.get("uid")
    
    if not uid:
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        # Persist to URL
        st.query_params["uid"] = st.session_state.session_id
    else:
        st.session_state.session_id = uid

    if "llm_status" not in st.session_state:
        st.session_state.llm_status = "checking"
    
    if "llm_message" not in st.session_state:
        st.session_state.llm_message = t("status_connecting")

@st.fragment(run_every=5)
def render_connection_status():
    """Render connection status in sidebar."""
    if st.session_state.llm_status == "checking":
        st.info(f"🔄 {t('status_connecting')}")
        if "check_thread_started" not in st.session_state:
            st.session_state.check_thread_started = True
            thread = threading.Thread(target=check_llm_connection_async)
            add_script_run_ctx(thread)
            thread.daemon = True
            thread.start()
    elif st.session_state.llm_status == "ready":
        st.success(st.session_state.llm_message)
    else:
        col_err, col_retry = st.columns([4, 1])
        with col_err:
            st.error(st.session_state.llm_message)
        with col_retry:
            if st.button("🔄", help=t("btn_retry")):
                st.session_state.llm_status = "checking"
                if "check_thread_started" in st.session_state:
                    del st.session_state.check_thread_started
                st.rerun()

def render_sidebar():
    """Render the settings sidebar."""
    with st.sidebar:
        # Language Selector
        st.subheader("Language")
        current_lang = st.session_state.get("ui_language", "en")
        
        lang_options = list(AVAILABLE_LANGUAGES.keys())
        lang_names = [AVAILABLE_LANGUAGES[l] for l in lang_options]
        
        try:
            current_index = lang_options.index(current_lang)
        except ValueError:
            current_index = 0
            
        selected_name = st.selectbox(
            "Interface Language", 
            options=lang_names, 
            index=current_index,
            label_visibility="collapsed"
        )
        
        selected_code = lang_options[lang_names.index(selected_name)]
        if selected_code != current_lang:
            set_language(selected_code)
            st.rerun()
            
        st.divider()
        
        # Connection status fragment
        render_connection_status()

        st.title(f"⚙️ {t('settings_title')}")
        
        # Session Info
        with st.expander("🆔 Session Info"):
            st.caption("Bookmark this page to save your queue.")
            st.code(st.session_state.session_id, language="text")
        
        with st.expander(f"🔧 {t('console_title')}", expanded=False):
            if st.button(t('console_clear')):
                clear_logs()
                st.rerun()
            
            logs = get_log_entries()
            if logs:
                log_text = "\n".join([
                    f"[{log['timestamp']}] {log['level']}: {log['message']}"
                    for log in logs
                ])
                st.code(log_text, language="log")
            else:
                st.info(t('console_no_logs'))
        
        st.divider()
        st.subheader(t('global_settings'))
        
        translate_images = st.checkbox(
            t('translate_images'),
            value=settings.translate_images
        )
        
        with st.expander(t('adv_settings')):
            method_options = [t('method_ocr'), t('method_vlm')]
            method_mapping = {t('method_ocr'): "ocr", t('method_vlm'): "vlm"}
            
            current_method = settings.image_translation_method
            default_index = 1 if current_method == "vlm" else 0
                
            selected_method_label = st.radio(
                t('image_method'),
                options=method_options,
                index=default_index,
            )
            
            preserve_fmt = st.checkbox(
                t('preserve_fmt'),
                value=settings.preserve_formatting
            )
            
            context_size = st.slider(
                t('context_window'),
                min_value=0,
                max_value=5,
                value=settings.context_window_size
            )
        
        st.divider()
        st.subheader(t('supported_formats'))
        extensions = ProcessorFactory.supported_extensions()
        st.write(", ".join(extensions) if extensions else "None loaded")
        
        return {
            "translate_images": translate_images,
            "image_translation_method": method_mapping[selected_method_label],
            "preserve_formatting": preserve_fmt,
            "context_window_size": context_size,
        }

@st.fragment(run_every=1)
def render_queue_fragment(queue_manager: GlobalQueueManager, session_id: str):
    """
    Render the job queue. This fragment auto-refreshes every 1 second.
    """
    jobs = queue_manager.get_user_jobs(session_id)
    
    if not jobs:
        return

    # Toolbar
    c_info, c_spacer, c_actions = st.columns([2, 4, 4])
    with c_info:
        st.markdown(f"### Queue ({len(jobs)})")
        
    with c_actions:
        # Batch Actions (Right aligned via columns)
        b1, b2 = st.columns([1, 1])
        with b1:
            completed_jobs = [j for j in jobs if j.status == "done" and j.result_data]
            if completed_jobs:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for job in completed_jobs:
                        if job.result_name and job.result_data:
                            zf.writestr(job.result_name, job.result_data)
                
                st.download_button(
                    f"📦 {t('btn_download_zip')}",
                    data=zip_buffer.getvalue(),
                    file_name="translations.zip",
                    mime="application/zip",
                    use_container_width=True
                )
        with b2:
            if st.button(f"🧹 {t('btn_clear_all')}", use_container_width=True):
                queue_manager.clear_user_jobs(session_id)
                st.rerun()

    # Job List
    for idx, job in enumerate(jobs):
        with st.container(border=True):
            # Dense Layout: [Icon] [File] [Progress] [Lang] [Actions]
            c1, c2, c3, c4, c5 = st.columns([0.5, 3.5, 2.5, 2, 3.5])
            
            # 1. Icon
            with c1:
                if job.status == "done":
                    st.write("✅")
                elif job.status == "error":
                    st.write("❌")
                elif job.status == "processing":
                    st.write("🔄")
                else:
                    st.write("⏳")

            # 2. Filename & Size
            with c2:
                st.markdown(f"**{job.filename}**")
                st.caption(format_size(len(job.file_data)))

            # 3. Status / Progress
            with c3:
                if job.status == "processing":
                    st.progress(job.progress)
                    st.caption(f"{job.status_msg} ({int(job.progress * 100)}%)")
                elif job.status == "error":
                    st.error(job.status_msg, icon="🚨")
                else:
                    st.caption(job.status_msg)

            # 4. Language (Compact)
            with c4:
                st.text_input(
                    "Lang", 
                    value=job.target_lang, 
                    disabled=True, 
                    key=f"lang_{job.id}",
                    label_visibility="collapsed"
                )

            # 5. Actions (Toolbar)
            with c5:
                # We use small columns for buttons to keep them inline
                a1, a2, a3, a4 = st.columns(4)
                
                # Download
                if job.status == "done" and job.result_data:
                    with a1:
                        st.download_button(
                            "📥",
                            data=job.result_data,
                            file_name=job.result_name,
                            key=f"dl_{job.id}",
                            help=t('btn_download')
                        )
                
                # Pause / Resume / Cancel
                if job.status == "processing":
                    with a2:
                        if st.button("⏸️", key=f"pause_{job.id}", help=t('btn_pause')):
                            queue_manager.pause_job(job.id)
                    with a4:
                        if st.button("❌", key=f"cancel_{job.id}", help="Cancel"):
                            queue_manager.cancel_job(job.id)
                            st.rerun()
                elif job.status == "pending":
                    with a1:
                        if st.button("⬆️", key=f"up_{job.id}", disabled=(idx == 0), help="Move Up"):
                            queue_manager.reorder_job(job.id, "up")
                            st.rerun()
                    with a2:
                        if st.button("⬇️", key=f"dn_{job.id}", disabled=(idx == len(jobs)-1), help="Move Down"):
                            queue_manager.reorder_job(job.id, "down")
                            st.rerun()
                    with a3:
                        if st.button("⏸️", key=f"pause_pend_{job.id}", help=t('btn_pause')):
                            queue_manager.pause_job(job.id)
                    with a4:
                        if st.button("🗑️", key=f"cl_{job.id}", help="Cancel"):
                            queue_manager.cancel_job(job.id)
                            st.rerun()
                elif job.status in ["paused", "error", "cancelled"]:
                    with a1:
                         if st.button("▶️", key=f"resume_{job.id}", help=t("btn_resume")):
                             queue_manager.resume_job(job.id)
                             st.rerun()
                    with a4:
                        if st.button("🗑️", key=f"del_{job.id}", help="Remove"):
                            queue_manager.cancel_job(job.id)
                            st.rerun()

def get_languages() -> List[str]:
    """Get list of supported languages."""
    langs = [
        "English", "German", "French", "Spanish", "Italian", 
        "Portuguese", "Chinese", "Japanese", "Korean", "Russian", "Arabic"
    ]
    # Ensure default is in list
    default_lang = settings.target_language
    if default_lang not in langs:
        langs.append(default_lang)
    return langs

def render_main_interface(user_settings, queue_manager: GlobalQueueManager):
    """Render the main upload and job interface."""
    st.header(f"📄 {t('upload_header')}")
    
    # Target Language Selection (Global for new uploads)
    languages = get_languages()
    if "current_target_lang" not in st.session_state:
        st.session_state.current_target_lang = settings.target_language
    
    try:
        current_idx = languages.index(st.session_state.current_target_lang)
    except ValueError:
        current_idx = 0

    c_lang, c_upload = st.columns([1, 3])
    
    with c_lang:
        st.caption(t('lbl_language'))
        selected_lang = st.selectbox(
            t('lbl_language'),
            options=languages,
            index=current_idx,
            key="main_target_lang",
            label_visibility="collapsed"
        )
        st.session_state.current_target_lang = selected_lang

    with c_upload:
        st.caption(t('msg_upload_instruction'))
        extensions = ProcessorFactory.supported_extensions()
        uploaded_files = st.file_uploader(
            t('msg_upload_instruction'),
            type=[ext.lstrip(".") for ext in extensions],
            help=f"Supported formats: {', '.join(extensions)}",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    
    session_id = st.session_state.session_id
    
    if uploaded_files:
        current_jobs = queue_manager.get_user_jobs(session_id)
        current_files = {(j.filename, len(j.file_data)) for j in current_jobs}
        
        added_new = False
        for file in uploaded_files:
            file_sig = (file.name, file.size)
            if file_sig not in current_files:
                queue_manager.add_job(
                    filename=file.name,
                    file_data=file.getvalue(),
                    target_lang=st.session_state.current_target_lang,
                    settings=user_settings,
                    session_id=session_id
                )
                added_new = True
        
        if added_new:
            st.toast("Documents added to queue!", icon="✅")

    # Render the queue as an auto-refreshing fragment
    render_queue_fragment(queue_manager, session_id)

def main():
    init_session()
    
    # Inject Custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.title(f"🌐 {t('title')}")
    st.markdown(f"*{t('subtitle')}*")
    
    queue_manager = get_queue_manager()
    user_settings = render_sidebar()
    render_main_interface(user_settings, queue_manager)

if __name__ == "__main__":
    main()