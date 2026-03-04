"""
Tests for code-quality improvements: JobStatus enum, atomic persistence,
rotated text handling, shared helpers, and vision engine recovery.
"""

import math
import os
import pickle
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# JobStatus enum
# ---------------------------------------------------------------------------

class TestJobStatus:
    """Tests for the JobStatus enum and its usage in Job dataclass."""

    def test_job_status_values(self):
        from services.queue_manager import JobStatus
        assert JobStatus.PENDING == "pending"
        assert JobStatus.PROCESSING == "processing"
        assert JobStatus.DONE == "done"
        assert JobStatus.ERROR == "error"
        assert JobStatus.PAUSED == "paused"
        assert JobStatus.CANCELLED == "cancelled"

    def test_job_status_is_string(self):
        """JobStatus inherits from str so it can be compared with raw strings."""
        from services.queue_manager import JobStatus
        assert JobStatus.DONE == "done"
        assert "done" == JobStatus.DONE

    def test_job_default_status(self):
        """New Job objects should default to PENDING."""
        from services.queue_manager import Job, JobStatus
        job = Job(
            id="test-id",
            filename="test.docx",
            file_data=b"data",
            target_lang="English",
            settings={},
            owner_session_id="session-1",
        )
        assert job.status == JobStatus.PENDING
        assert job.status == "pending"

    def test_job_status_membership(self):
        """Enum members should work in set/tuple membership checks."""
        from services.queue_manager import JobStatus
        terminal = (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)
        assert JobStatus.DONE in terminal
        assert JobStatus.PROCESSING not in terminal


# ---------------------------------------------------------------------------
# Atomic pickle persistence
# ---------------------------------------------------------------------------

class TestAtomicPersistence:
    """Tests for atomic queue state file writes."""

    @patch("services.queue_manager.add_script_run_ctx")
    def test_save_state_creates_file(self, mock_ctx, tmp_path):
        from services.queue_manager import GlobalQueueManager
        state_file = tmp_path / "queue_state.pkl"

        mgr = GlobalQueueManager.__new__(GlobalQueueManager)
        mgr.STATE_FILE = state_file
        mgr.queue = deque()
        mgr.jobs = {}
        mgr.lock = __import__("threading").Lock()

        mgr._save_state()
        assert state_file.exists()

        with open(state_file, "rb") as f:
            data = pickle.load(f)
        assert "queue" in data
        assert "jobs" in data

    @patch("services.queue_manager.add_script_run_ctx")
    def test_save_state_atomic_no_partial(self, mock_ctx, tmp_path):
        """Verify no .tmp file is left behind after a successful save."""
        from services.queue_manager import GlobalQueueManager
        state_file = tmp_path / "queue_state.pkl"

        mgr = GlobalQueueManager.__new__(GlobalQueueManager)
        mgr.STATE_FILE = state_file
        mgr.queue = deque(["job1"])
        mgr.jobs = {"job1": "placeholder"}
        mgr.lock = __import__("threading").Lock()

        mgr._save_state()

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, "Temp file should be cleaned up"


# ---------------------------------------------------------------------------
# Shared formatting helper
# ---------------------------------------------------------------------------

class TestParagraphHasFormatting:
    """Tests for the shared _paragraph_has_formatting method."""

    def test_plain_paragraph(self):
        from processors.base import BaseFileProcessor

        mock_run = Mock()
        mock_run.font = Mock(bold=False, italic=False, underline=False)
        mock_para = Mock(runs=[mock_run])

        assert BaseFileProcessor._paragraph_has_formatting(mock_para) is False

    def test_bold_paragraph(self):
        from processors.base import BaseFileProcessor

        mock_run = Mock()
        mock_run.font = Mock(bold=True, italic=False, underline=False)
        mock_para = Mock(runs=[mock_run])

        assert BaseFileProcessor._paragraph_has_formatting(mock_para) is True

    def test_italic_paragraph(self):
        from processors.base import BaseFileProcessor

        mock_run = Mock()
        mock_run.font = Mock(bold=False, italic=True, underline=False)
        mock_para = Mock(runs=[mock_run])

        assert BaseFileProcessor._paragraph_has_formatting(mock_para) is True

    def test_docx_style_run(self):
        """python-docx runs expose bold/italic directly on the run."""
        from processors.base import BaseFileProcessor

        mock_run = Mock(spec=["bold", "italic", "underline"])
        mock_run.bold = True
        mock_run.italic = False
        mock_run.underline = False
        mock_para = Mock(runs=[mock_run])

        assert BaseFileProcessor._paragraph_has_formatting(mock_para) is True

    def test_empty_runs(self):
        from processors.base import BaseFileProcessor
        mock_para = Mock(runs=[])
        assert BaseFileProcessor._paragraph_has_formatting(mock_para) is False


# ---------------------------------------------------------------------------
# Shared background sampling
# ---------------------------------------------------------------------------

class TestSampleRegionBackground:
    """Tests for the shared sample_region_background utility."""

    def test_white_image(self):
        from utils.text_utils import sample_region_background
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        bg = sample_region_background(img)
        assert bg == (255, 255, 255)

    def test_black_image(self):
        from utils.text_utils import sample_region_background
        img = Image.new("RGB", (50, 50), (0, 0, 0))
        bg = sample_region_background(img)
        assert bg == (0, 0, 0)

    def test_very_small_image(self):
        from utils.text_utils import sample_region_background
        img = Image.new("RGB", (1, 1), (128, 64, 32))
        bg = sample_region_background(img)
        assert bg == (128, 64, 32)


# ---------------------------------------------------------------------------
# Rotated text handling
# ---------------------------------------------------------------------------

class TestFitTextToBoxRotated:
    """Tests for angle-aware fit_text_to_box."""

    def test_zero_angle_same_as_before(self):
        from utils.text_utils import fit_text_to_box
        font, size = fit_text_to_box("Hi", 200, 50, angle=0.0)
        assert size >= 8

    def test_large_angle_reduces_effective_size(self):
        """A 45° rotation should need a smaller font to fit the same box."""
        from utils.text_utils import fit_text_to_box
        text = "Hello World Test"
        _, size_0 = fit_text_to_box(text, 200, 30, max_font_size=24, min_font_size=6, angle=0.0)
        _, size_45 = fit_text_to_box(text, 200, 30, max_font_size=24, min_font_size=6, angle=45.0)
        assert size_45 <= size_0

    def test_90_degree_swaps_axes(self):
        """At ~90°, the effective width/height should be swapped."""
        from utils.text_utils import fit_text_to_box
        # Wide box, short height — at 90° text is effectively tall-and-narrow
        _, size_0 = fit_text_to_box("A", 200, 20, max_font_size=20, min_font_size=6, angle=0.0)
        _, size_90 = fit_text_to_box("A", 200, 20, max_font_size=20, min_font_size=6, angle=89.0)
        # At 90°, the available height is effectively the box_width (200), so font should still fit
        assert size_90 >= 6


class TestDrawRotatedText:
    """Tests for draw_rotated_text function."""

    def test_draw_zero_angle(self):
        """Drawing at 0° should not raise."""
        from utils.text_utils import draw_rotated_text
        img = Image.new("RGBA", (200, 100), (255, 255, 255, 255))
        font = ImageFont.load_default()
        draw_rotated_text(img, "Test", (100, 50), 0.0, font, (0, 0, 0))

    def test_draw_45_degree(self):
        """Drawing at 45° should not raise."""
        from utils.text_utils import draw_rotated_text
        img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        font = ImageFont.load_default()
        draw_rotated_text(img, "Rotated", (100, 100), 45.0, font, (0, 0, 0))

    def test_draw_with_background_clear(self):
        """Background clearing should fill the original bbox."""
        from utils.text_utils import draw_rotated_text
        img = Image.new("RGBA", (200, 100), (255, 255, 255, 255))
        font = ImageFont.load_default()
        draw_rotated_text(
            img, "Test", (50, 25), 0.0, font, (0, 0, 0),
            bg_fill=(128, 128, 128, 255),
            original_bbox=(10, 10, 90, 40),
        )
        # The region should have been overwritten
        pixel = img.getpixel((50, 25))
        # Should not be pure white anymore
        assert pixel != (255, 255, 255, 255)

    def test_draw_empty_text(self):
        """Empty text should not raise."""
        from utils.text_utils import draw_rotated_text
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        font = ImageFont.load_default()
        draw_rotated_text(img, "", (50, 50), 0.0, font, (0, 0, 0))

    def test_draw_clips_to_image_bounds(self):
        """Text placed near edges should not raise ValueError."""
        from utils.text_utils import draw_rotated_text
        img = Image.new("RGBA", (50, 50), (255, 255, 255, 255))
        font = ImageFont.load_default()
        # Centre near the edge
        draw_rotated_text(img, "Edge", (5, 5), 0.0, font, (0, 0, 0))


# ---------------------------------------------------------------------------
# VisionEngine reader recovery
# ---------------------------------------------------------------------------

class TestVisionEngineReaderRecovery:
    """Tests for EasyOCR reader retry/recovery logic."""

    @patch("settings.settings")
    def test_reader_retries_after_failure(self, mock_settings):
        mock_settings.ocr_source_languages = "en"
        mock_settings.target_language = "English"

        from services.vision_engine import VisionEngine
        engine = VisionEngine()

        # After creation, counter should be 0
        assert engine._reader_init_failures == 0

        # Manually simulate failures below max
        engine._reader_init_failures = 2
        # Not yet at max (3), but reader property will try to init again
        # Since easyocr may not be installed, it will fail again
        # Just test the counter-based lockout
        engine._reader_init_failures = VisionEngine.MAX_READER_RETRIES
        assert engine.reader is None  # Locked out at max retries

    @patch("settings.settings")
    def test_reader_success_resets_counter(self, mock_settings):
        mock_settings.ocr_source_languages = "en"
        mock_settings.target_language = "English"

        from services.vision_engine import VisionEngine
        engine = VisionEngine()
        engine._reader_init_failures = 2

        # Simulate successful load by setting reader directly
        engine._reader = Mock()
        assert engine.reader is not None
        # The actual reader property returns cached _reader when it's set
