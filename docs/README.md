# OfficeTranslator

Privacy-first document translation tool using local LLM backends.

![Status](https://img.shields.io/badge/status-beta-yellow)
![Python](https://img.shields.io/badge/python-3.10+-blue)

## Features

- 🔒 **Privacy-First**: All processing happens locally
- 📄 **Multiple Formats**: DOCX, PPTX, XLSX, PDF
- 🖼️ **Image Translation**: OCR with "Google Lens" style overlays
- 🔄 **Context Awareness**: Uses previous paragraphs for consistent translations
- 📖 **Glossary Support**: Consistent technical term translations
- 🎨 **Smart Overlays**: Adaptive colors for image text

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### 3. Start Local LLM

Start your preferred LLM backend (e.g., LM Studio, Ollama):

```bash
# LM Studio: Open app, load model, start server
# Ollama: ollama serve
```

### 4. Run Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | `lm-studio` | API key (use `lm-studio` for local) |
| `LLM_MODEL` | `llama-3-8b-instruct` | Model identifier |
| `TARGET_LANGUAGE` | `English` | Translation target language |
| `TRANSLATE_IMAGES` | `true` | Enable OCR for images |
| `PRESERVE_FORMATTING` | `false` | Preserve bold/italic (experimental) |
| `CONTEXT_WINDOW_SIZE` | `1` | Previous paragraphs for context |
| `GLOSSARY_FILE` | *(empty)* | Path to glossary JSON |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MAX_RETRIES` | `3` | LLM retry attempts |

## Glossary Format

Create a JSON file with source → target mappings:

```json
{
  "Antragsstellung": "Application submission",
  "Bescheid": "Official notice",
  "Fördermittel": "Funding"
}
```

Set `GLOSSARY_FILE=path/to/glossary.json` in your `.env`.

## Supported Backends

Any OpenAI-compatible API:

- **LM Studio** (recommended for local)
- **Ollama** with OpenAI compatibility layer
- **AnythingLLM**
- **OpenAI** (cloud, for testing)
- **Groq** (cloud, for testing)

## Troubleshooting

### "LLM Disconnected" Error

1. Ensure your LLM server is running
2. Check `LLM_BASE_URL` in `.env`
3. Verify the model is loaded

### OCR Not Working

1. First run downloads EasyOCR models (~100MB)
2. Check logs for CUDA/GPU errors
3. Try setting `gpu=False` in vision_engine.py

### PDF Output Looks Different

PDF translation uses an overlay approach to preserve layout. Original text is covered with translated text boxes.

## License

MIT
