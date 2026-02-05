"""
High-level translation service orchestrating LLM and vision processing.
"""

import logging
from typing import Callable, Generator, Optional

from .llm_client import LLMClient
from .vision_engine import VisionEngine, TextRegion
from settings import settings

logger = logging.getLogger("OfficeTranslator.TranslationService")


class TranslationService:
    """
    High-level translation service that coordinates LLM and vision components.
    Provides progress callbacks for UI integration.
    """
    
    def __init__(
        self,
        llm_client: LLMClient = None,
        vision_engine: VisionEngine = None,
    ):
        """
        Initialize translation service.
        
        Args:
            llm_client: LLM client instance (creates new if None)
            vision_engine: Vision engine instance (creates new if None)
        """
        self._llm_client = llm_client
        self._vision_engine = vision_engine
        self._context_buffer: list[str] = []
        
        self.on_progress: Optional[Callable[[int, int, str], None]] = None
    
    @property
    def llm_client(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    @property
    def vision_engine(self) -> VisionEngine:
        """Lazy-load vision engine."""
        if self._vision_engine is None:
            self._vision_engine = VisionEngine()
        return self._vision_engine
    
    def check_ready(self) -> tuple[bool, str]:
        """
        Check if the service is ready (LLM connected).
        
        Returns:
            Tuple of (is_ready, status_message)
        """
        try:
            if self.llm_client.check_connection():
                model = self.llm_client.connected_model_id or settings.llm_model
                ctx = settings.llm_context_window
                return True, f"🟢 Ready: {model} ({ctx})"
            else:
                return False, "🔴 LLM Disconnected"
        except Exception as e:
            return False, f"🔴 Error: {str(e)}"
    
    def translate_text(
        self,
        text: str,
        use_context: bool = True,
        target_language: str = None,
    ) -> str:
        """
        Translate a single text string.
        
        Args:
            text: Text to translate
            use_context: Whether to use context from previous translations
            target_language: Target language
            
        Returns:
            Translated text
        """
        context = None
        if use_context and self._context_buffer:
            context_str = " ".join(self._context_buffer[-settings.context_window_size:])
            
            # Truncate context based on configured window size
            # Assume ~4 chars per token, use 20% of window for history context
            max_chars = int((settings.llm_context_window * 4) * 0.2)
            
            if len(context_str) > max_chars:
                context_str = "..." + context_str[-max_chars:]
            context = context_str
        
        translated = self.llm_client.translate(
            text,
            target_language=target_language,
            context=context,
            preserve_formatting=settings.preserve_formatting,
        )
        
        # Update context buffer
        if use_context:
            self._context_buffer.append(translated)
            # Keep buffer bounded
            if len(self._context_buffer) > settings.context_window_size * 2:
                self._context_buffer = self._context_buffer[-settings.context_window_size:]
        
        return translated
    
    def translate_texts(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] = None,
    ) -> Generator[tuple[int, str, str], None, None]:
        """
        Translate multiple texts with progress updates.
        
        Args:
            texts: List of texts to translate
            progress_callback: Callback(current, total) for progress updates
            
        Yields:
            Tuples of (index, original_text, translated_text)
        """
        total = len(texts)
        self.reset_context()
        
        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, total)
            
            try:
                translated = self.translate_text(text)
                yield i, text, translated
            except Exception as e:
                logger.error(f"Translation failed for item {i}: {e}")
                yield i, text, text  # Return original on failure
    
    def translate_image(
        self,
        image_data: bytes,
        smart_colors: bool = True,
        method: str = None,
    ) -> tuple[bytes, list[TextRegion]]:
        """
        Detect and translate text in an image.
        
        Args:
            image_data: Image bytes
            smart_colors: Use adaptive background colors
            method: Translation method ('ocr' or 'vlm'). Defaults to settings.
            
        Returns:
            Tuple of (processed image bytes, text regions)
        """
        if not settings.translate_images:
            logger.debug("Image translation disabled, returning original")
            return image_data, []
        
        current_method = method or settings.image_translation_method
        
        if current_method == "vlm":
            logger.info("Using Experimental VLM image translation")
            return self.vision_engine.process_image_vlm(
                image_data,
                translate_image_func=lambda b64: self.llm_client.translate_image(b64),
                smart_colors=smart_colors
            )
        else:
            return self.vision_engine.process_image(
                image_data,
                translate_func=lambda t: self.translate_text(t, use_context=False),
                smart_colors=smart_colors,
            )
    
    def reset_context(self) -> None:
        """Clear the translation context buffer."""
        self._context_buffer.clear()
        logger.debug("Context buffer cleared")
