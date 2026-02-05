# LM Studio 0.4.1+ Compatibility Guide

With the release of LM Studio 0.4.1, a new "REST API v1" was introduced. This document clarifies how **OfficeTranslator** interacts with it.

## Compatibility Status: ✅ Fully Compatible

OfficeTranslator uses the standard OpenAI Python client library (`openai`). LM Studio's new REST API v1 is designed to be fully compatible with this standard. Therefore, **no code changes are required** to use OfficeTranslator with LM Studio 0.4.1+.

## Configuration

Ensure your `.env` file points to the correct local server address. By default, LM Studio uses port `1234`.

```ini
# Standard LM Studio configuration
LLM_BASE_URL="http://localhost:1234/v1"
LLM_API_KEY="lm-studio"
```

## Key Features Supported

*   **Chat Completions**: The core translation engine uses the `/v1/chat/completions` endpoint, which is the heart of the new API.
*   **Structured Outputs**: OfficeTranslator's experimental VLM mode requests JSON output. LM Studio's new engine has improved support for JSON schemas, making this feature more reliable.
*   **Vision**: When using "Experimental (Qwen-VL)" mode, the app sends base64-encoded images. This follows the OpenAI Vision spec supported by LM Studio.

## Troubleshooting

If you encounter connection errors:
1.  **Check Port**: Ensure LM Studio is running and the server is started (green "Start Server" button).
2.  **Model Loading**: Ensure a model is actually loaded in LM Studio.
3.  **Firewall**: Check if Windows Firewall is blocking the connection to `localhost:1234`.
