"""
LLM client with OpenAI-compatible API and retry logic.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from settings import settings

logger = logging.getLogger("OfficeTranslator.LLMClient")


class LLMClient:
    """
    OpenAI-compatible LLM client with exponential backoff retry logic.
    Supports local LLM backends (LM Studio, AnythingLLM) and cloud providers.
    """
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        max_retries: int = None,
    ):
        """
        Initialize LLM client.
        
        Args:
            base_url: API base URL (defaults to settings)
            api_key: API key (defaults to settings)
            model: Model identifier (defaults to settings)
            max_retries: Max retry attempts (defaults to settings)
        """
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.max_retries = max_retries or settings.max_retries
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        self.connected_model_id = None
        
        # Load glossary if configured
        self.glossary: dict[str, str] = {}
        self._load_glossary()
        
        logger.info(f"LLM Client initialized: {self.base_url} using {self.model}")
    
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
    
    def check_connection(self) -> bool:
        """
        Check if LLM backend is reachable.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Use a short timeout for the check
            models = self.client.models.list(timeout=5.0)
            
            # Auto-detect context window if exposed by backend
            if models.data:
                # We check the first model or iterate to find the active one if we knew it
                # For simplicity, check the first one as local servers often return just one or consistent configs
                first_model = models.data[0]
                self.connected_model_id = first_model.id
                
                # Access dictionary representation
                model_data = first_model.model_dump() if hasattr(first_model, 'model_dump') else first_model.__dict__
                
                # Look for common context keys used by LM Studio/Ollama/LocalAI
                # LM Studio sometimes puts it in 'permission' or root
                candidates = ['context_window', 'n_ctx', 'max_context_length', 'max_tokens']
                
                for key in candidates:
                    if key in model_data and model_data[key]:
                        try:
                            new_window = int(model_data[key])
                            if new_window > 0 and new_window != settings.llm_context_window:
                                settings.llm_context_window = new_window
                                logger.info(f"Auto-detected context window: {new_window}")
                                break
                        except (ValueError, TypeError):
                            pass

            logger.debug(f"LLM connection OK. Available models: {[m.id for m in models.data]}")
            return True
        except Exception as e:
            logger.error(f"LLM connection failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def translate(
        self,
        text: str,
        target_language: str = None,
        context: Optional[str] = None,
        preserve_formatting: bool = False,
    ) -> str:
        """
        Translate text to target language using LLM.
        
        Args:
            text: Text to translate
            target_language: Target language (defaults to settings)
            context: Optional context from previous text
            preserve_formatting: Whether to preserve inline formatting tags
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return text
        
        target_lang = target_language or settings.target_language
        
        # Build system prompt
        system_prompt = self._build_system_prompt(target_lang, context, preserve_formatting)
        
        # Apply glossary pre-processing hints
        user_prompt = text
        if self.glossary:
            glossary_hints = self._get_relevant_glossary_hints(text)
            if glossary_hints:
                user_prompt = f"{text}\n\n[Glossary hints: {glossary_hints}]"
        
        logger.debug(f"Translating: {text[:50]}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=min(settings.llm_context_window // 2, len(text) * 3),  # Rough estimate for expansion, capped
            )
            
            translated = response.choices[0].message.content.strip()
            logger.debug(f"Translated to: {translated[:50]}...")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    def _build_system_prompt(
        self,
        target_language: str,
        context: Optional[str],
        preserve_formatting: bool,
    ) -> str:
        """Build the system prompt for translation."""
        prompt_parts = [
            f"You are a professional translator. Translate the following text to {target_language}.",
            "Maintain the original meaning, tone, and style.",
            "Preserve proper nouns, technical terms, and named entities in their original language.",
            "Only output the translation, nothing else.",
        ]
        
        if preserve_formatting:
            prompt_parts.append(
                "Preserve any formatting tags like <b>, </b>, <i>, </i> in their exact positions relative to the text."
            )
        
        if context:
            prompt_parts.append(
                f"\nContext from previous text (do not translate, use for tone and consistency): {context}"
            )
        
        return "\n".join(prompt_parts)
    
    def _get_relevant_glossary_hints(self, text: str) -> str:
        """Get glossary entries relevant to the input text."""
        hints = []
        text_lower = text.lower()
        
        for source, target in self.glossary.items():
            if source.lower() in text_lower:
                hints.append(f'"{source}" → "{target}"')
        
        return ", ".join(hints) if hints else ""
    
    def batch_translate(
        self,
        texts: list[str],
        target_language: str = None,
        context_window: int = None,
    ) -> list[str]:
        """
        Translate multiple texts with context window.
        
        Args:
            texts: List of texts to translate
            target_language: Target language
            context_window: Number of previous translations to use as context
            
        Returns:
            List of translated texts
        """
        window_size = context_window if context_window is not None else settings.context_window_size
        translations = []
        
        for i, text in enumerate(texts):
            # Build context from previous translations
            context = None
            if window_size > 0 and i > 0:
                context_start = max(0, i - window_size)
                context_texts = translations[context_start:i]
                context = " ".join(context_texts)
            
            try:
                translated = self.translate(text, target_language, context)
                translations.append(translated)
            except Exception as e:
                logger.error(f"Failed to translate text {i}: {e}")
                translations.append(text)  # Keep original on failure
        
        return translations

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def translate_image(
        self,
        base64_image: str,
        target_language: str = None,
    ) -> str:
        """
        Analyze and translate text in an image using a VLM.
        Returns a JSON string containing regions with bboxes and translated text.
        """
        target_lang = target_language or settings.target_language
        
        system_prompt = (
            f"You are a sophisticated OCR and translation engine. "
            f"Detect all text in the image and translate it to {target_lang}. "
            f"Preserve proper nouns, technical terms, and named entities in their original language. "
            f"Return a JSON object with a single key 'regions', which is a list of objects. "
            f"Each object must have: 'bbox' [x1, y1, x2, y2] (integers) and 'text' (the translated string). "
            f"Do not include the original text. Return ONLY valid JSON."
        )

        try:
            try:
                # Try to force JSON mode if supported
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "Locate and translate all text in this image."},
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        },
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
            except Exception as e:
                # Fallback for backends that don't support response_format or other API errors
                # We log it and try again without the strict json constraint
                logger.warning(f"VLM call with json_object failed ({e}), retrying without strict format...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "Locate and translate all text in this image."},
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        },
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                )
                
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"VLM Image translation failed: {e}")
            raise
