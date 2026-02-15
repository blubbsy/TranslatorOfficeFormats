"""
Standalone job runner logic for the global queue.
Executes translation tasks independently of the Streamlit session context.
"""

import logging
import tempfile
import traceback
import concurrent.futures
from pathlib import Path

from .queue_manager import Job
from .translation_service import TranslationService
from .llm_client import LLMClient
from .vision_engine import VisionEngine
from processors import get_processor, ContentType

logger = logging.getLogger("OfficeTranslator.JobRunner")

# Global instances for reuse (keeps connections / models alive)
_LLM_CLIENT = None
_VISION_ENGINE = None


def get_shared_services():
    """Return (or create) shared LLMClient and VisionEngine singletons."""
    global _LLM_CLIENT, _VISION_ENGINE
    if _LLM_CLIENT is None:
        _LLM_CLIENT = LLMClient()
    if _VISION_ENGINE is None:
        _VISION_ENGINE = VisionEngine()
    return _LLM_CLIENT, _VISION_ENGINE


def run_translation_job(job: Job):
    """
    Execute a translation job.
    Updates the *job* object in-place with progress and results.
    """
    job.status = "processing"
    job.status_msg = "Initialising..."
    job.progress = 0.0

    logger.info(f"Runner starting job {job.id}")

    # Write uploaded bytes to a temp file
    suffix = Path(job.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
        tmp_input.write(job.file_data)
        input_path = tmp_input.name

    output_path: str | None = None

    try:
        # --- Services ---
        llm_client, vision_engine = get_shared_services()

        # Pre-flight connectivity check (with one retry)
        connected = llm_client.check_connection()
        if not connected:
            import time

            logger.warning("LLM unreachable on first attempt, retrying in 3 s...")
            time.sleep(3)
            connected = llm_client.check_connection()
        if not connected:
            raise ConnectionError(
                "LLM Backend is unreachable. Please check your local server."
            )

        service = TranslationService(llm_client, vision_engine)

        # --- Processor ---
        processor = get_processor(input_path)
        processor.target_language = job.target_lang
        processor.load(input_path)

        # --- Extract content ---
        try:
            chunks = list(processor.extract_content_generator())
        except Exception as e:
            raise RuntimeError(f"Content extraction failed: {e}") from e

        total_chunks = len(chunks)

        if total_chunks == 0:
            job.status = "done"
            job.progress = 1.0
            job.status_msg = "Empty document"
            return

        text_chunks = [c for c in chunks if c.content_type != ContentType.IMAGE]
        image_chunks = [c for c in chunks if c.content_type == ContentType.IMAGE]

        # --- Restore already-translated chunks from a previous attempt ---
        processed_count = 0
        for chunk in chunks:
            if chunk.id in job.intermediate_results:
                try:
                    processor.apply_translation(
                        chunk.id, job.intermediate_results[chunk.id]
                    )
                except Exception as e:
                    logger.debug(f"Failed to restore chunk {chunk.id}: {e}")
                processed_count += 1

        def update_progress(msg: str):
            nonlocal processed_count
            job.status_msg = msg
            if total_chunks > 0:
                job.progress = min(processed_count / total_chunks, 1.0)

        # ---- 1. Translate text chunks (concurrent) ----
        if text_chunks:
            todo_text = [c for c in text_chunks if c.id not in job.intermediate_results]
            target_lang = job.target_lang

            def translate_chunk(chunk):
                try:
                    translated = service.translate_text(
                        chunk.text,
                        target_language=target_lang,
                        use_context=False,
                    )
                    return chunk.id, translated, None
                except Exception as e:
                    return chunk.id, chunk.text, e

            if todo_text:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_chunk = {
                        executor.submit(translate_chunk, c): c for c in todo_text
                    }
                    consecutive_errors = 0

                    for future in concurrent.futures.as_completed(future_to_chunk):
                        if _should_stop(job, executor):
                            return

                        processed_count += 1
                        update_progress(
                            f"Translating text ({processed_count}/{total_chunks})"
                        )

                        chunk_id, result_text, error = future.result()

                        if error:
                            logger.error(f"Failed to translate {chunk_id}: {error}")
                            consecutive_errors += 1
                            # Store original so we don't re-attempt on resume
                            job.intermediate_results[chunk_id] = result_text
                            processor.apply_translation(chunk_id, result_text)
                            if (
                                "connection" in str(error).lower()
                                or consecutive_errors > 5
                            ):
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise ConnectionError(
                                    f"Translation aborted after repeated errors: {error}"
                                )
                        else:
                            consecutive_errors = 0
                            job.intermediate_results[chunk_id] = result_text
                            processor.apply_translation(chunk_id, result_text)

        # ---- 2. Translate image chunks (concurrent, fewer workers) ----
        if image_chunks:
            todo_images = [
                c for c in image_chunks if c.id not in job.intermediate_results
            ]
            translate_images = job.settings.get("translate_images", True)
            img_method = job.settings.get("image_translation_method", "ocr")
            target_lang = job.target_lang

            def process_image_chunk(chunk):
                try:
                    if translate_images:
                        translated_image, _ = service.translate_image(
                            chunk.image_data,
                            method=img_method,
                            target_language=target_lang,
                        )
                        return chunk.id, translated_image, None
                    else:
                        return chunk.id, chunk.image_data, None
                except Exception as e:
                    return chunk.id, chunk.image_data, e

            if todo_images:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_img = {
                        executor.submit(process_image_chunk, c): c for c in todo_images
                    }

                    for future in concurrent.futures.as_completed(future_to_img):
                        if _should_stop(job, executor):
                            return

                        processed_count += 1
                        update_progress(
                            f"Processing images ({processed_count}/{total_chunks})"
                        )

                        chunk_id, result_data, error = future.result()

                        if error:
                            logger.error(f"Failed to process image {chunk_id}: {error}")
                        # Always store result (original or translated)
                        job.intermediate_results[chunk_id] = result_data
                        processor.apply_translation(chunk_id, result_data)

        # ---- 3. Save ----
        update_progress("Saving...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_output:
            output_path = tmp_output.name

        processor.save(output_path)

        with open(output_path, "rb") as f:
            job.result_data = f.read()

        job.result_name = f"translated_{job.filename}"
        job.status = "done"
        job.progress = 1.0
        job.status_msg = "Complete"

    except Exception as e:
        logger.error(f"Job {job.id} failed: {e}")
        logger.debug(traceback.format_exc())
        job.status = "error"
        job.error = str(e)
        job.status_msg = f"Error: {str(e)}"

    finally:
        # Clean up temp files
        for p in (input_path, output_path):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass


def _should_stop(job: Job, executor: concurrent.futures.ThreadPoolExecutor) -> bool:
    """Check cancel / pause flags and shut down the executor if needed."""
    if job.cancel_requested:
        executor.shutdown(wait=False, cancel_futures=True)
        job.status = "cancelled"
        job.status_msg = "Cancelled by user"
        return True
    if job.pause_requested:
        executor.shutdown(wait=False, cancel_futures=True)
        job.status = "paused"
        job.status_msg = "Paused by user"
        return True
    return False
