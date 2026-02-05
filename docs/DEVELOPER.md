# Developer Guide

Extending OfficeTranslator with new file formats and features.

## Architecture

```
app.py (Streamlit UI)
    ↓
TranslationService
    ├── LLMClient (translation)
    └── VisionEngine (OCR)
    ↓
ProcessorFactory
    ├── DocxProcessor
    ├── PptxProcessor
    ├── XlsxProcessor
    └── PdfProcessor
```

## Adding a New File Format

### 1. Create Processor

Copy the template and implement the interface:

```python
# processors/myformat_processor.py
from .base import BaseFileProcessor, ContentChunk, ContentType

class MyFormatProcessor(BaseFileProcessor):
    SUPPORTED_EXTENSIONS = ["myf"]
    
    def load(self, file_path):
        # Load your document format
        pass
    
    def extract_content_generator(self):
        # Yield ContentChunk objects
        for i, text in enumerate(self.texts):
            yield ContentChunk(
                id=f"item_{i}",
                content_type=ContentType.TEXT,
                text=text,
                location=f"Item {i+1}",
            )
    
    def apply_translation(self, chunk_id, translated_content):
        # Apply translation back to document
        pass
    
    def save(self, output_path):
        # Save modified document
        pass
```

### 2. Register Processor

Add to `processors/factory.py`:

```python
def _auto_register():
    # ... existing registrations ...
    
    try:
        from .myformat_processor import MyFormatProcessor
        register_processor("myf", MyFormatProcessor)
    except ImportError:
        pass
```

### 3. Update Dependencies

Add required libraries to `requirements.txt`.

## Content Types

| Type | Use For | Fields |
|------|---------|--------|
| `TEXT` | Paragraphs, headings | `text` |
| `IMAGE` | Photos, diagrams | `image_data` (bytes) |
| `TABLE_CELL` | Spreadsheet cells | `text` |

## Formatting Preservation

When `PRESERVE_FORMATTING=true`:

1. Parser extracts runs with formatting tags: `<b>bold</b>`
2. LLM translates while preserving tags
3. Parser rebuilds runs from tagged output

To extend formatting support, modify:
- `_has_formatting()` in processor
- `_apply_with_formatting()` in processor
- System prompt in `llm_client.py`

## Vision Engine Customization

### Custom OCR Languages

```python
from services.vision_engine import VisionEngine

engine = VisionEngine(languages=['en', 'de', 'fr'])
```

### Disable Smart Colors

```python
result, regions = engine.process_image(
    image_data,
    translate_func,
    smart_colors=False  # Always use black background
)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Mocking LLM for Tests

```python
from unittest.mock import Mock, patch

@patch('services.llm_client.LLMClient.translate')
def test_translation(mock_translate):
    mock_translate.return_value = "Translated text"
    # Your test code
```

## Project Structure

```
OfficeTranslator/
├── app.py              # Streamlit entry
├── settings.py         # Pydantic config
├── processors/
│   ├── base.py         # Abstract interface
│   ├── factory.py      # Registry
│   └── *_processor.py  # Format handlers
├── services/
│   ├── llm_client.py   # LLM with retry
│   ├── vision_engine.py # OCR + overlay
│   └── translation_service.py
├── utils/
│   ├── logging_handler.py
│   └── text_utils.py
├── docs/
└── tests/
```
