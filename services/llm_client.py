"""
LLM client with OpenAI-compatible API and retry logic.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from settings import settings

logger = logging.getLogger("OfficeTranslator.LLMClient")

# Exception types that warrant a retry
_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    OSError,
)


class LLMClient:
    """
    OpenAI-compatible LLM client with exponential backoff retry logic.
    Supports local LLM backends (LM Studio, AnythingLLM) and cloud providers.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
    ):
        self.base_url: str = base_url or settings.llm_base_url
        self.api_key: str = api_key or settings.llm_api_key
        self.model: str = model or settings.llm_model
        self.max_retries: int = max_retries or settings.max_retries
        self.context_window: int = settings.llm_context_window

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=60.0,
            max_retries=0,  # we handle retries ourselves via tenacity
        )

        self.connected_model_id: Optional[str] = None

        # Load glossary if configured
        self.glossary: dict[str, str] = {}
        self._load_glossary()

        logger.info(f"LLM Client initialised: {self.base_url} using {self.model}")

    # ------------------------------------------------------------------
    # Glossary
    # ------------------------------------------------------------------

    def _load_glossary(self) -> None:
        """Load translation glossary from file if configured."""
        if settings.glossary_file:
            try:
                path = Path(settings.glossary_file)
                with open(path, "r", encoding="utf-8") as f:
                    self.glossary = json.load(f)
                logger.info(f"Loaded glossary with {len(self.glossary)} entries")
            except Exception as e:
                logger.warning(f"Failed to load glossary: {e}")

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            models = self.client.models.list(timeout=10.0)

            if models.data:
                first_model = models.data[0]
                self.connected_model_id = first_model.id

                model_data = (
                    first_model.model_dump()
                    if hasattr(first_model, "model_dump")
                    else first_model.__dict__
                )

                for key in (
                    "context_window",
                    "n_ctx",
                    "max_context_length",
                    "max_tokens",
                ):
                    val = model_data.get(key)
                    if val:
                        try:
                            new_window = int(val)
                            if new_window > 0 and new_window != self.context_window:
                                self.context_window = new_window
                                logger.info(
                                    f"Auto-detected context window: {new_window}"
                                )
                                break
                        except (ValueError, TypeError):
                            pass

            logger.debug(
                f"LLM connection OK. Models: {[m.id for m in models.data]}"
            )
            return True
        except Exception as e:
            logger.error(f"LLM connection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Text translation
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def translate(
        self,
        text: str,
        target_language: Optional[str] = None,
        context: Optional[str] = None,
        preserve_formatting: bool = False,
    ) -> str:
        """Translate text to *target_language* using the LLM."""
        if not text or not text.strip():
            return text

        target_lang = target_language or settings.target_language
        system_prompt = self._build_system_prompt(
            target_lang, context, preserve_formatting
        )

        user_prompt = text
        if self.glossary:
            hints = self._get_relevant_glossary_hints(text)
            if hints:
                user_prompt = f"{text}\n\n[Glossary hints: {hints}]"

        logger.debug(f"Translating: {text[:50]}...")

        # Token budget
        estimated_input_tokens = (len(text) // 3) + 250
        available_for_output = max(100, self.context_window - estimated_input_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=min(available_for_output, max(len(text) * 2, 200)),
            )

            # Defensive: handle missing/empty response
            if not response.choices:
                logger.warning("LLM returned empty choices list")
                return text

            content = response.choices[0].message.content
            if content is None:
                logger.warning("LLM returned None content")
                return text

            translated = content.strip()
            logger.debug(f"Translated to: {translated[:50]}...")
            return translated

        except _RETRYABLE:
            raise  # let tenacity handle
        except Exception as e:
            logger.error(f"Translation failed (non-retryable): {e}")
            raise

    # ------------------------------------------------------------------
    # Vision / VLM translation
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def translate_image(
        self,
        base64_image: str,
        target_language: Optional[str] = None,
    ) -> str:
        """
        Analyse and translate text in an image using a VLM.
        Returns the raw JSON string from the model.
        """
        target_lang = target_language or settings.target_language

        system_prompt = (
            f"You are a sophisticated OCR and translation engine. "
            f"Detect all text in the image and translate it to {target_lang}. "
            f"Preserve proper nouns, technical terms, and named entities. "
            f"Return a JSON object with a single key 'regions', which is a list. "
            f"Each item must have: "
            f"'bbox' [x1, y1, x2, y2] (integers, pixel coordinates), "
            f"'text' (the translated string). "
            f"Do not include the original text. Return ONLY valid JSON."
        )

        user_content = [
            {"type": "text", "text": "Locate and translate all text in this image."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]

        # Try with response_format first, then without
        for attempt, kwargs in enumerate(
            [
                {"response_format": {"type": "json_object"}},
                {},
            ]
        ):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    **kwargs,
                )

                content = response.choices[0].message.content if response.choices else None
                if content:
                    return content

            except _RETRYABLE:
                raise  # let tenacity decide
            except Exception as e:
                if attempt == 0:
                    logger.warning(
                        f"VLM call with json_object failed ({e}), "
                        f"retrying without strict format..."
                    )
                    continue
                raise

        # Should not reach here, but just in case
        raise RuntimeError("VLM image translation returned no content")

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        target_language: str,
        context: Optional[str],
        preserve_formatting: bool,
    ) -> str:
        parts = [
            f"You are a professional translator. Translate the following text to {target_language}.",
            "Maintain the original meaning, tone, and style.",
            "Preserve proper nouns, technical terms, and named entities in their original language.",
            "Only output the translation, nothing else.",
        ]

        if preserve_formatting:
            parts.append(
                "Preserve any formatting tags like <b>, </b>, <i>, </i> "
                "in their exact positions relative to the text."
            )

        if context:
            parts.append(
                f"\nContext from previous text (do not translate, "
                f"use for tone and consistency): {context}"
            )

        return "\n".join(parts)

    def _get_relevant_glossary_hints(self, text: str) -> str:
        hints = []
        text_lower = text.lower()
        for source, target in self.glossary.items():
            if source.lower() in text_lower:
                hints.append(f'"{source}" → "{target}"')
        return ", ".join(hints) if hints else ""

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch_translate(
        self,
        texts: list[str],
        target_language: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> list[str]:
        """Translate multiple texts with a sliding context window."""
        window_size = (
            context_window if context_window is not None else settings.context_window_size
        )
        translations: list[str] = []

        for i, text in enumerate(texts):
            context = None
            if window_size > 0 and i > 0:
                start = max(0, i - window_size)
                context = " ".join(translations[start:i])

            try:
                translated = self.translate(text, target_language, context)
                translations.append(translated)
            except Exception as e:
                logger.error(f"Failed to translate text {i}: {e}")
                translations.append(text)  # keep original on failure

        return translations
