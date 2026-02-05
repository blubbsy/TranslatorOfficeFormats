Project Specification: Privacy-Friendly Translation Tool (v5.0)
1. Project Overview

Goal: Develop a locally hosted, privacy-first web application for translating documents (.docx, .pptx). Core Philosophy:

    Modular: Easily extensible for new file types or LLM backends.

    Private: No data leaves the local network (unless using a specific cloud config for testing).

    Robust: Handles errors gracefully and provides transparent user feedback. Key Features:

    "Google Lens" Style Images: Detects text in images, translates it, and overlays it on a semi-transparent background.

    Text Expansion Handling: Automatically resizes text to prevent layout breakage.

    Toggleable Complexity: Formatting preservation (bold/italic) is configurable (feature flagged).

2. System Architecture
A. The "Plugin" Pattern (File Handling)

The system uses a Factory Pattern to decouple the main app from specific file logic.

    FileProcessorFactory: Detects file extension (.docx, .pptx) → Instantiates the correct Processor.

    BaseFileProcessor (Interface): The abstract base class all plugins must inherit from.

        Must implement: load(), extract_content_generator() (yields chunks for progress bars), apply_translation(), save().

    Plugins:

        DocxProcessor: Handles paragraphs, tables, and inline shapes.

        PptxProcessor: Handles slides, text boxes, and shape trees.

B. The "Provider" Pattern (LLM Backend)

    Standard: OpenAI API Specification.

    Client: Uses the openai Python library, configured via .env to point to either:

        Local: LM Studio / AnythingLLM (http://localhost:1234/v1).

        Cloud: OpenAI / Gemini / Groq (for testing).

C. The "Vision" Service (Image Translation)

    OCR: EasyOCR (Local CPU/GPU).

    Overlay Strategy: Instead of complex in-painting, use a Semi-Transparent Text Box approach. This guarantees readability on any background (gradients, photos, patterns) without visual artifacts.

3. Configuration & Environment

The application is driven by a .env file. The code must include a settings.py to load these safely.

Required .env Structure:
Ini, TOML

# --- LLM CONNECTION ---
# Example for Local (LM Studio):
LLM_BASE_URL="http://localhost:1234/v1"
LLM_API_KEY="lm-studio"
LLM_MODEL="llama-3-8b-instruct"

# --- APP BEHAVIOR ---
TARGET_LANGUAGE="English"
TRANSLATE_IMAGES=True

# --- ADVANCED FEATURES ---
# If False: Translate plain text (faster, safer).
# If True: Logic to parse/reconstruct 'runs' for bold/italic preservation.
PRESERVE_FORMATTING=False 

# Number of previous paragraphs to send as context (improves consistency)
CONTEXT_WINDOW_SIZE=1 

# --- SYSTEM ---
LOG_LEVEL="INFO"
MAX_RETRIES=3

4. Implementation Steps
Phase 1: Robust Core Services

    services/llm_client.py:

        Implement Exponential Backoff (wait 1s, 2s, 4s on failure) using the tenacity library.

        Context Injection: When calling the translation function, allow an optional previous_text argument. Append this to the system prompt as: "Context from previous sentence (do not translate, just use for tone): ..."

    services/vision_engine.py:

        OCR: Scan image -> Get Bounding Box (bbox).

        Overlay:

            Create a new generic image layer.

            Draw a Black Rectangle with 50% Opacity (RGBA: 0, 0, 0, 128) over the bbox.

            Draw White Text centered in the rect.

        Auto-Fit Logic: Before drawing text, calculate width. If text_width > bbox_width, reduce font size by 1pt loops until it fits or hits 8pt minimum.

Phase 2: Modular Processors (The Plugins)

    processors/docx_processor.py & pptx_processor.py:

        Formatting Toggle:

            Check settings.PRESERVE_FORMATTING.

            If False: Extract paragraph.text, translate, replace paragraph.text. (Wipes formatting, but is robust).

            If True: (Prepare code structure only) Parse runs, insert pseudo-tags (e.g., <b>), translate, rebuild runs.

        Error Handling: Wrap every paragraph/slide iteration in a try/except. If one fails, log [ERROR] Slide 4 failed, skip it, and continue. Do not crash.

Phase 3: Observability & UI (app.py)

    Real-Time Logs: Implement a custom logging.Handler that writes to st.session_state.

    UI Components:

        Status Badge: "🟢 System Ready" or "🔴 LLM Disconnected".

        Detailed Progress: "Processing Slide 3 of 15..."

        Debug Console: A collapsed st.expander named "Show Logs" that displays the live log stream.

5. Documentation Requirements

The following documentation must be generated and stored in a /docs folder.
A. README.md (User Guide)

    Quick Start:

        Install: pip install -r requirements.txt

        Run Local LLM (e.g., LM Studio).

        Run App: streamlit run app.py

    Configuration: clear table explaining the .env variables.

B. DEVELOPER.md (Extensibility)

    "How to Add a File Format": * Copy processors/template.py.

        Inherit BaseFileProcessor.

        Register in processors/factory.py.

    "How to Enable Formatting": Instructions on switching the PRESERVE_FORMATTING flag and extending the tag_parser logic.

C. Architecture Graph

    Provide a MermaidJS definition in docs/architecture.mermaid illustrating the relationship between Streamlit UI -> Factory -> Plugins -> LLM Service.

6. Technical Stack Summary

    Language: Python 3.10+

    Frontend: Streamlit

    LLM Integration: openai (Python SDK), tenacity (Retries)

    Vision: easyocr (Text detection), opencv-python-headless (Image proc), Pillow (Drawing)

    Document Ops: python-docx, python-pptx

    Config: python-dotenv

    Logging: Standard logging module with custom Streamlit handler.