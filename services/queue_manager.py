"""
Global Job Queue Manager for handling translation tasks sequentially across all users.
"""

import threading
import time
import uuid
import logging
import pickle
from pathlib import Path
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Callable
from streamlit.runtime.scriptrunner import add_script_run_ctx

logger = logging.getLogger("OfficeTranslator.Queue")

@dataclass
class Job:
    id: str
    filename: str
    file_data: bytes
    target_lang: str
    settings: Dict[str, Any]
    owner_session_id: str # To identify which user owns this job
    status: str = "pending" # pending, processing, done, error
    progress: float = 0.0
    status_msg: str = "Queued"
    result_data: Optional[bytes] = None
    result_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    cancel_requested: bool = False

class GlobalQueueManager:
    """Thread-safe singleton queue for managing translation jobs."""
    
    STATE_FILE = Path("queue_state.pkl")
    
    def __init__(self, processor_func: Callable[[Job], None]):
        self.queue = deque()
        self.jobs: Dict[str, Job] = {} # ID -> Job
        self.current_job_id: Optional[str] = None
        self.lock = threading.Lock()
        self.processor_func = processor_func
        self.processing_active = True
        
        self._load_state()
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        add_script_run_ctx(self.worker_thread)
        self.worker_thread.start()
        
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        add_script_run_ctx(self.cleanup_thread)
        self.cleanup_thread.start()
        logger.info("Global Queue Manager started")

    def _save_state(self):
        """Save current queue state to disk."""
        try:
            with open(self.STATE_FILE, "wb") as f:
                pickle.dump({"queue": self.queue, "jobs": self.jobs}, f)
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")

    def _load_state(self):
        """Load queue state from disk."""
        if self.STATE_FILE.exists() and self.STATE_FILE.stat().st_size > 0:
            try:
                with open(self.STATE_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.queue = data.get("queue", deque())
                    self.jobs = data.get("jobs", {})
                    
                    # Reset stuck jobs
                    for job_id, job in self.jobs.items():
                        if job.status == "processing":
                            job.status = "pending"
                            job.status_msg = "Resumed after restart"
                            if job_id not in self.queue:
                                self.queue.appendleft(job_id)
                    
                    logger.info(f"Restored {len(self.jobs)} jobs from disk")
            except (EOFError, pickle.UnpicklingError):
                logger.warning("Queue state file was empty or corrupted, starting fresh.")
            except Exception as e:
                logger.error(f"Failed to load queue state: {e}")

    def add_job(self, filename: str, file_data: bytes, target_lang: str, settings: Dict, session_id: str) -> str:
        """Add a new job to the queue."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            filename=filename,
            file_data=file_data,
            target_lang=target_lang,
            settings=settings,
            owner_session_id=session_id
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self.queue.append(job_id)
            position = len(self.queue)
            job.status_msg = f"Queued (Pos: {position})"
            self._save_state()
            
        logger.info(f"Job added: {job_id} for {filename}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job details by ID."""
        with self.lock:
            return self.jobs.get(job_id)

    def get_user_jobs(self, session_id: str) -> List[Job]:
        """Get all jobs belonging to a specific session."""
        with self.lock:
            # We want them sorted by creation time or queue position
            user_jobs = [j for j in self.jobs.values() if j.owner_session_id == session_id]
            # Simple sort by creation time
            return sorted(user_jobs, key=lambda j: j.created_at)

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get 1-based position in queue. Returns 0 if currently processing."""
        with self.lock:
            if self.current_job_id == job_id:
                return 0
            try:
                return self.queue.index(job_id) + 1
            except ValueError:
                return None

    def reorder_job(self, job_id: str, direction: str):
        """Move a job up or down in the queue."""
        with self.lock:
            if job_id not in self.queue:
                return
            
            try:
                idx = self.queue.index(job_id)
                if direction == "up" and idx > 0:
                    del self.queue[idx]
                    self.queue.insert(idx - 1, job_id)
                elif direction == "down" and idx < len(self.queue) - 1:
                    del self.queue[idx]
                    self.queue.insert(idx + 1, job_id)
                self._save_state()
            except ValueError:
                pass

    def cancel_job(self, job_id: str):
        """Cancel a job (remove from queue or stop processing)."""
        with self.lock:
            if job_id in self.queue:
                self.queue.remove(job_id)
            
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status == "processing":
                    job.cancel_requested = True
                    job.status_msg = "Cancelling..."
                else:
                    job.status = "cancelled"
                    job.status_msg = "Cancelled"
            
            self._save_state()

    def clear_user_jobs(self, session_id: str):
        """Clear all completed/cancelled jobs for a user."""
        with self.lock:
            to_remove = [
                jid for jid, job in self.jobs.items() 
                if job.owner_session_id == session_id and job.status in ["done", "error", "cancelled"]
            ]
            for jid in to_remove:
                del self.jobs[jid]
            self._save_state()

    def _worker_loop(self):
        """Background loop to process jobs."""
        while self.processing_active:
            job_id = None
            
            with self.lock:
                if self.queue:
                    job_id = self.queue.popleft()
                    self.current_job_id = job_id
                    self._save_state() # Update queue state (popped)
                    
                    # Update status for all others in queue
                    for idx, q_jid in enumerate(self.queue):
                        if q_jid in self.jobs:
                            self.jobs[q_jid].status_msg = f"Queued (Pos: {idx + 1})"
            
            if job_id:
                job = self.jobs.get(job_id)
                if job and job.status != "cancelled":
                    try:
                        logger.info(f"Processing job {job_id}")
                        self.processor_func(job)
                    except Exception as e:
                        logger.error(f"Worker error on job {job_id}: {e}")
                        job.status = "error"
                        job.error = str(e)
                        job.status_msg = "Internal Error"
                    finally:
                        self._save_state() # Save result/status
                
                self.current_job_id = None
            else:
                time.sleep(0.5)

    def _cleanup_loop(self):
        """Periodically remove old jobs."""
        while self.processing_active:
            time.sleep(600)  # Check every 10 mins
            now = time.time()
            max_age = 24 * 3600  # 24 hours
            
            with self.lock:
                to_remove = [
                    jid for jid, job in self.jobs.items() 
                    if now - job.created_at > max_age and job.status in ["done", "error", "cancelled"]
                ]
                for jid in to_remove:
                    del self.jobs[jid]
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old jobs")
                    self._save_state()

    def shutdown(self):
        """Gracefully stop the queue manager."""
        logger.info("Shutting down Global Queue Manager...")
        self.processing_active = False
        
        with self.lock:
            for job in self.jobs.values():
                if job.status == "processing":
                    job.cancel_requested = True
                    job.status_msg = "Server shutting down..."
            self._save_state()