"""
Standalone job runner logic for the global queue.
Executes translation tasks independently of the Streamlit session context.
"""

import logging
import tempfile
import concurrent.futures
from pathlib import Path

from .queue_manager import Job
from .translation_service import TranslationService
from .llm_client import LLMClient
from .vision_engine import VisionEngine
from processors import get_processor, ContentType, ContentChunk

logger = logging.getLogger("OfficeTranslator.JobRunner")

# Global instances for reuse (to keep connections alive)
_LLM_CLIENT = None
_VISION_ENGINE = None

def get_shared_services():
    global _LLM_CLIENT, _VISION_ENGINE
    if _LLM_CLIENT is None:
        _LLM_CLIENT = LLMClient()
    if _VISION_ENGINE is None:
        _VISION_ENGINE = VisionEngine()
    return _LLM_CLIENT, _VISION_ENGINE

def run_translation_job(job: Job):
    """
    Execute a translation job.
    Updates the job object in-place with progress and results.
    """
    job.status = "processing"
    job.status_msg = "Initializing..."
    job.progress = 0.0
    
    logger.info(f"Runner starting job {job.id}")
    
    # Save file to temp
    suffix = Path(job.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_input:
        tmp_input.write(job.file_data)
        input_path = tmp_input.name
        
    try:
        # Get services
        llm_client, vision_engine = get_shared_services()
        
        # 1. Pre-flight check
        if not llm_client.check_connection():
            raise ConnectionError("LLM Backend is unreachable. Please check your local server.")
            
        service = TranslationService(llm_client, vision_engine)
        
        # Load processor
        processor = get_processor(input_path)
        processor.target_language = job.target_lang
        processor.load(input_path)
        
        # Extract content
        chunks = list(processor.extract_content_generator())
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            job.status = "done"
            job.progress = 1.0
            job.status_msg = "Empty document"
            return

        text_chunks = [c for c in chunks if c.content_type != ContentType.IMAGE]
        image_chunks = [c for c in chunks if c.content_type == ContentType.IMAGE]
        
        processed_count = 0
        
        # Restore already translated chunks from previous attempt
        for chunk in chunks:
            if chunk.id in job.intermediate_results:
                processor.apply_translation(chunk.id, job.intermediate_results[chunk.id])
                processed_count += 1

        def update_progress(msg):
            nonlocal processed_count
            job.status_msg = msg
            if total_chunks > 0:
                job.progress = processed_count / total_chunks
        
        # 1. Process Text Chunks (Concurrent)
        if text_chunks:
            # Filter out already done
            todo_text = [c for c in text_chunks if c.id not in job.intermediate_results]
            target_lang = job.target_lang
            
            def translate_chunk(chunk):
                try:
                    translated = service.translate_text(
                        chunk.text,
                        target_language=target_lang,
                        use_context=False
                    )
                    return chunk.id, translated, None
                except Exception as e:
                    return chunk.id, chunk.text, e

            if todo_text:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_chunk = {executor.submit(translate_chunk, c): c for c in todo_text}
                    
                    consecutive_errors = 0
                    
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        if job.cancel_requested or job.pause_requested:
                            executor.shutdown(wait=False, cancel_futures=True)
                            if job.pause_requested:
                                job.status = "paused"
                                job.status_msg = "Paused by user"
                            else:
                                job.status = "cancelled"
                                job.status_msg = "Cancelled by user"
                            return

                        processed_count += 1
                        update_progress(f"Translating text ({processed_count}/{total_chunks})")
                        
                        chunk_id, result_text, error = future.result()
                        
                        if error:
                            logger.error(f"Failed to translate {chunk_id}: {error}")
                            consecutive_errors += 1
                            if "connection" in str(error).lower() or consecutive_errors > 5:
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise ConnectionError(f"Translation aborted: {error}")
                        else:
                            consecutive_errors = 0
                            job.intermediate_results[chunk_id] = result_text
                            processor.apply_translation(chunk_id, result_text)

        # 2. Process Image Chunks (Concurrent)
        if image_chunks:
            todo_images = [c for c in image_chunks if c.id not in job.intermediate_results]
            translate_images = job.settings.get("translate_images", True)
            img_method = job.settings.get("image_translation_method", "ocr")
            target_lang = job.target_lang
            
            def process_image_chunk(chunk):
                try:
                    if translate_images:
                        translated_image, _ = service.translate_image(
                            chunk.image_data, 
                            method=img_method,
                            target_language=target_lang
                        )
                        return chunk.id, translated_image, None
                    else:
                        return chunk.id, chunk.image_data, None
                except Exception as e:
                    return chunk.id, chunk.image_data, e

            if todo_images:
                # Use fewer workers for images as they are more resource intensive (OCR/VLM)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_img = {executor.submit(process_image_chunk, c): c for c in todo_images}
                    
                    for future in concurrent.futures.as_completed(future_to_img):
                        if job.cancel_requested or job.pause_requested:
                            executor.shutdown(wait=False, cancel_futures=True)
                            if job.pause_requested:
                                job.status = "paused"
                                job.status_msg = "Paused by user"
                            else:
                                job.status = "cancelled"
                                job.status_msg = "Cancelled by user"
                            return

                        processed_count += 1
                        update_progress(f"Processing images ({processed_count}/{total_chunks})")
                        
                        chunk_id, result_data, error = future.result()
                        
                        if error:
                            logger.error(f"Failed to process image {chunk_id}: {error}")
                            # Images failing is often non-fatal, keep original
                            job.intermediate_results[chunk_id] = result_data
                            processor.apply_translation(chunk_id, result_data)
                        else:
                            job.intermediate_results[chunk_id] = result_data
                            processor.apply_translation(chunk_id, result_data)
        
        update_progress("Saving...")
        
        # Save result
        output_suffix = suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=output_suffix) as tmp_output:
            output_path = tmp_output.name
        
        processor.save(output_path)
        
        with open(output_path, "rb") as f:
            job.result_data = f.read()
            
        job.result_name = f"translated_{job.filename}"
        job.status = "done"
        job.progress = 1.0
        job.status_msg = "Complete"
        
        try:
            Path(output_path).unlink(missing_ok=True)
        except: pass
        
    except Exception as e:
        logger.error(f"Job {job.id} failed: {e}")
        job.status = "error"
        job.error = str(e)
        job.status_msg = f"Error: {str(e)}"
    
    finally:
        try:
            Path(input_path).unlink(missing_ok=True)
        except: pass
