# Translation Backends

OfficeTranslator now supports two distinct translation engines. You can choose which one to use by setting `TRANSLATION_BACKEND` in your `.env` file.

## 1. Argos Translate (Default)
**Setting:** `TRANSLATION_BACKEND=argos`

Argos Translate is an open-source, offline Neural Machine Translation (NMT) library based on OpenNMT.

### Pros:
- **Fast and Lightweight:** Runs very efficiently on standard CPUs without requiring a dedicated GPU.
- **100% Offline & Private:** No data is sent to the cloud, and no local LLM server needs to be running.
- **Deterministic:** Only outputs the translated text. It will never output conversational text or "thinking" tags.
- **Easy Setup:** Models are automatically downloaded on demand for the language pairs you request.

### Cons & Limitations:
- **Translation Quality:** While good, it generally lags behind modern LLMs, especially for complex technical documents or idiomatic phrasing.
- **No Context:** Translates chunk-by-chunk. It does not look at previous paragraphs to maintain consistent terminology.
- **Formatting Issues:** Traditional NMT models struggle with embedded tags (like `<b>`, `<i>`). Formatting preservation might not work as well in `.docx` and `.pptx` files compared to the LLM backend.
- **Language Pairs:** It requires specific translation models for language pairs. If a direct model (e.g., German to Spanish) doesn't exist, it will "pivot" through English (German -> English -> Spanish), which can reduce quality and double the processing time.
- **Image Translation:** Does not support the experimental VLM (Vision-Language Model) direct image translation. It relies entirely on the OCR fallback pipeline.

---

## 2. LLM Provider (High Quality)
**Setting:** `TRANSLATION_BACKEND=llm`

This uses an OpenAI-compatible API to connect to either a local LLM (like LM Studio or Ollama) or a cloud provider (like OpenAI).

### Pros:
- **Superior Quality:** Modern LLMs excel at natural, fluent translations and understanding context.
- **Context-Aware:** Uses a sliding context window to maintain tone and terminology throughout a document.
- **Formatting Preservation:** Much better at understanding and preserving XML/HTML formatting tags in Word and PowerPoint documents.
- **Glossary Support:** Can dynamically inject custom glossary terms into the prompt to force specific translations.
- **VLM Support:** If using a Vision-Language Model, it can directly translate text within images, bypassing standard OCR.

### Cons & Limitations:
- **Resource Intensive:** Running a good LLM locally requires significant RAM and ideally a dedicated GPU.
- **Slower:** Inference is generally slower than Argos Translate, especially for large documents.
- **Unpredictability:** LLMs can sometimes hallucinate, output conversational text, or refuse to translate certain content.
- **Setup Complexity:** Requires the user to have a separate LLM server running and correctly configured in the `.env` file.
