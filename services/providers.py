"""
Translation providers for OfficeTranslator.
Supports multiple backends (LLM and Argos Translate).
"""

import abc
import logging
from typing import Optional

from utils.text_utils import get_iso_639_1

logger = logging.getLogger("OfficeTranslator.Providers")


class BaseTranslationProvider(abc.ABC):
    """Abstract base class for translation backends."""

    @abc.abstractmethod
    def translate(
        self,
        text: str,
        target_language: str,
        context: Optional[str] = None,
        preserve_formatting: bool = False,
    ) -> str:
        """Translate text to target language."""
        pass

    @abc.abstractmethod
    def translate_image(
        self,
        base64_image: str,
        target_language: str,
    ) -> str:
        """Translate image (VLM only, others return empty/raise)."""
        pass

    @abc.abstractmethod
    def check_connection(self) -> tuple[bool, str]:
        """Check if the provider is ready."""
        pass


class LLMProvider(BaseTranslationProvider):
    """Translation provider using LLMClient (OpenAI compatible)."""

    def __init__(self):
        from services.llm_client import LLMClient
        self.client = LLMClient()
        
    def translate(
        self,
        text: str,
        target_language: str,
        context: Optional[str] = None,
        preserve_formatting: bool = False,
    ) -> str:
        return self.client.translate(
            text=text,
            target_language=target_language,
            context=context,
            preserve_formatting=preserve_formatting,
        )

    def translate_image(
        self,
        base64_image: str,
        target_language: str,
    ) -> str:
        return self.client.translate_image(
            base64_image=base64_image,
            target_language=target_language,
        )

    def check_connection(self) -> tuple[bool, str]:
        from settings import settings
        is_ready = self.client.check_connection()
        if is_ready:
            model = self.client.connected_model_id or settings.llm_model
            ctx = self.client.context_window
            return True, f"🟢 Ready: {model} ({ctx})"
        return False, "🔴 LLM Disconnected"


class ArgosProvider(BaseTranslationProvider):
    """
    Translation provider using Argos Translate (OpenNMT).
    Fast, offline, but doesn't support context or HTML/XML formatting well.
    """

    def __init__(self):
        self._installed_models: set[tuple[str, str]] = set()
        self._is_ready = False
        try:
            import argostranslate.package
            import argostranslate.translate
            # Update package index on init
            argostranslate.package.update_package_index()
            self._is_ready = True
        except ImportError:
            logger.error("argostranslate is not installed.")

    def _ensure_model(self, source_code: str, target_code: str) -> bool:
        """Ensure the translation model for the given pair is installed."""
        if (source_code, target_code) in self._installed_models:
            return True

        import argostranslate.package
        import argostranslate.translate

        # Check if already installed
        installed = argostranslate.package.get_installed_packages()
        for pkg in installed:
            if pkg.from_code == source_code and pkg.to_code == target_code:
                self._installed_models.add((source_code, target_code))
                return True

        # Try to install
        logger.info(f"Downloading Argos model for {source_code} -> {target_code}...")
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            (pkg for pkg in available_packages 
             if pkg.from_code == source_code and pkg.to_code == target_code), 
            None
        )

        if package_to_install:
            argostranslate.package.install_from_path(package_to_install.download())
            self._installed_models.add((source_code, target_code))
            logger.info("Model installed successfully.")
            return True
        else:
            logger.error(f"No Argos model available for {source_code} -> {target_code}")
            return False

    def translate(
        self,
        text: str,
        target_language: str,
        context: Optional[str] = None,
        preserve_formatting: bool = False,
    ) -> str:
        if not text or not text.strip():
            return text

        if not self._is_ready:
            logger.warning("Argos Translate not available, returning original text.")
            return text

        import argostranslate.translate
        import langdetect

        # Detect source language
        try:
            source_lang_code = langdetect.detect(text)
        except Exception:
            source_lang_code = "en" # Fallback

        # Special case: argos maps chinese to 'zh'
        if source_lang_code in ["zh-cn", "zh-tw"]:
            source_lang_code = "zh"

        target_code = get_iso_639_1(target_language)
        
        if source_lang_code == target_code:
            return text

        # Argos requires exact pairs. If X->Y doesn't exist, try X->EN->Y
        installed = self._ensure_model(source_lang_code, target_code)
        
        try:
            if installed:
                return argostranslate.translate.translate(text, source_lang_code, target_code)
            else:
                # Pivot through English
                if source_lang_code != "en" and target_code != "en":
                    if self._ensure_model(source_lang_code, "en") and self._ensure_model("en", target_code):
                        english_text = argostranslate.translate.translate(text, source_lang_code, "en")
                        return argostranslate.translate.translate(english_text, "en", target_code)
        except Exception as e:
            logger.error(f"Argos translation failed: {e}")
            
        return text

    def translate_image(
        self,
        base64_image: str,
        target_language: str,
    ) -> str:
        """Argos cannot process images natively like a VLM."""
        raise NotImplementedError("ArgosProvider does not support direct image translation.")

    def check_connection(self) -> tuple[bool, str]:
        if self._is_ready:
            return True, "🟢 Ready: Argos Translate (Local NMT)"
        return False, "🔴 Argos Translate Not Installed"
