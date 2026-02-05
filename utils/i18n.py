"""
Internationalization (i18n) support for the OfficeTranslator UI.
"""

import locale
import logging
import streamlit as st
from typing import Dict

logger = logging.getLogger("OfficeTranslator.i18n")

# UI Translations
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "OfficeTranslator",
        "subtitle": "Privacy-first document translation using local LLM",
        "upload_header": "Upload Documents",
        "settings_title": "Settings",
        "global_settings": "Global Settings",
        "adv_settings": "Advanced Settings",
        "translate_images": "Translate Images (OCR)",
        "preserve_fmt": "Preserve Formatting",
        "context_window": "Context Window",
        "supported_formats": "Supported Formats",
        "status_connecting": "Connecting to LLM...",
        "status_ready": "System Ready",
        "status_error": "LLM Error",
        "btn_translate_all": "Translate All",
        "btn_clear_all": "Clear All",
        "btn_download_zip": "Download All (ZIP)",
        "btn_translate": "Translate",
        "btn_download": "Download",
        "btn_retry": "Retry",
        "status_pending": "Pending",
        "status_processing": "Processing",
        "status_done": "Complete",
        "status_error_job": "Error",
        "msg_upload_instruction": "Choose files to translate",
        "msg_empty_queue": "Upload documents to begin.",
        "lbl_language": "Language",
        "lbl_size": "Size",
        "lbl_status": "Status",
        "console_title": "Debug Console",
        "console_clear": "Clear Logs",
        "console_no_logs": "No logs yet",
        "image_method": "Image Translation Method",
        "method_ocr": "Standard (OCR)",
        "method_vlm": "Experimental (VLM)",
    },
    "de": {
        "title": "OfficeTranslator",
        "subtitle": "Datenschutzfreundliche Dokumentenübersetzung mit lokalem LLM",
        "upload_header": "Dokumente hochladen",
        "settings_title": "Einstellungen",
        "global_settings": "Allgemeine Einstellungen",
        "adv_settings": "Erweiterte Einstellungen",
        "translate_images": "Bilder übersetzen (OCR)",
        "preserve_fmt": "Formatierung beibehalten",
        "context_window": "Kontext-Fenster",
        "supported_formats": "Unterstützte Formate",
        "status_connecting": "Verbinde mit LLM...",
        "status_ready": "System Bereit",
        "status_error": "LLM Fehler",
        "btn_translate_all": "Alle übersetzen",
        "btn_clear_all": "Alle löschen",
        "btn_download_zip": "Alle herunterladen (ZIP)",
        "btn_translate": "Übersetzen",
        "btn_download": "Herunterladen",
        "btn_retry": "Wiederholen",
        "status_pending": "Ausstehend",
        "status_processing": "Verarbeite",
        "status_done": "Fertig",
        "status_error_job": "Fehler",
        "msg_upload_instruction": "Dateien zum Übersetzen auswählen",
        "msg_empty_queue": "Laden Sie Dokumente hoch, um zu beginnen.",
        "lbl_language": "Sprache",
        "lbl_size": "Größe",
        "lbl_status": "Status",
        "console_title": "Debug Konsole",
        "console_clear": "Logs löschen",
        "console_no_logs": "Noch keine Logs",
        "image_method": "Bildübersetzungsmethode",
        "method_ocr": "Standard (OCR)",
        "method_vlm": "Experimentell (VLM)",
    },
}

AVAILABLE_LANGUAGES = {
    "en": "English",
    "de": "Deutsch",
    # Placeholders for others using English fallback for now
    "fr": "Français",
    "es": "Español",
    "zh": "中文"
}

def get_system_language() -> str:
    """Detect system language code (e.g., 'en', 'de'). Defaults to 'en'."""
    try:
        sys_lang = locale.getdefaultlocale()[0]
        if sys_lang:
            lang_code = sys_lang.split("_")[0].lower()
            if lang_code in AVAILABLE_LANGUAGES:
                return lang_code
    except Exception as e:
        logger.warning(f"Failed to detect system language: {e}")
    return "en"

def t(key: str) -> str:
    """Get translation for key based on session language."""
    lang = st.session_state.get("ui_language", "en")
    # Fallback to English if language or key not found
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_dict.get(key, TRANSLATIONS["en"].get(key, key))

def set_language(lang_code: str):
    """Set the UI language."""
    if lang_code in AVAILABLE_LANGUAGES:
        st.session_state.ui_language = lang_code

def init_i18n():
    """Initialize UI language in session state."""
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = get_system_language()
