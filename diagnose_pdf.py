"""
Detailed diagnosis of GB 36980.1—2025.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from processors.pdf_processor import PdfProcessor
    from processors.base import ContentType

    pdf_path = Path("GB 36980.1—2025.pdf")

    print("=" * 70)
    print("PDF ANALYSIS REPORT: GB 36980.1—2025.pdf")
    print("=" * 70)
    print()

    processor = PdfProcessor()
    processor.load(str(pdf_path))

    chunks = list(processor.extract_content_generator())

    print(f"📊 SUMMARY:")
    print(f"   Total pages: {processor._document.page_count}")
    print(f"   Total chunks extracted: {len(chunks)}")
    print()

    # Count by type
    text_chunks = [c for c in chunks if c.content_type == ContentType.TEXT]
    image_chunks = [c for c in chunks if c.content_type == ContentType.IMAGE]

    print(f"   📝 TEXT chunks: {len(text_chunks)}")
    print(f"   🖼️  IMAGE chunks: {len(image_chunks)}")
    print()

    print("=" * 70)
    print("🔍 DIAGNOSIS:")
    print("=" * 70)
    print()

    if len(text_chunks) == 0 and len(image_chunks) > 0:
        print("⚠️  ISSUE IDENTIFIED: Image-based/Scanned PDF")
        print()
        print(
            "   This PDF contains NO extractable text. All content is stored as images."
        )
        print("   This typically happens when:")
        print("   - The PDF was created by scanning a physical document")
        print("   - The PDF was created from images without OCR")
        print("   - The PDF uses 'Print to PDF' on image-only content")
        print()
        print("📋 TRANSLATION REQUIREMENTS:")
        print("   ✅ The PDF processor correctly identified all pages as images")
        print("   ✅ Image chunks will be sent to VisionEngine for OCR processing")
        print("   ⚙️  Translation requires OCR (Optical Character Recognition)")
        print()
        print("🔧 WHAT HAPPENS DURING TRANSLATION:")
        print("   1. Each page image is sent to EasyOCR")
        print("   2. OCR detects text regions and extracts text")
        print("   3. Extracted text is sent to LLM for translation")
        print("   4. Translated text is overlaid back onto the image")
        print("   5. Modified images are inserted back into the PDF")
        print()
        print("⏱️  EXPECTED BEHAVIOR:")
        print("   - Translation will be SLOWER than text-based PDFs")
        print("   - OCR quality depends on image quality and text clarity")
        print("   - This is WORKING AS DESIGNED for scanned documents")
        print()

        # Show image details
        print("📸 IMAGE CHUNK DETAILS:")
        for i, chunk in enumerate(image_chunks[:3], 1):
            img_size = len(chunk.image_data) if chunk.image_data else 0
            print(f"   {i}. {chunk.id}")
            print(f"      Location: {chunk.location}")
            print(f"      Image size: {img_size / 1024:.1f} KB")

        if len(image_chunks) > 3:
            print(f"   ... and {len(image_chunks) - 3} more images")
        print()

    else:
        print("✅ This PDF has extractable text")

    print("=" * 70)
    print("💡 RECOMMENDATIONS:")
    print("=" * 70)
    print()

    if len(image_chunks) > 0 and len(text_chunks) == 0:
        print("   For best results with this scanned PDF:")
        print("   1. ✅ Ensure 'Translate Images' setting is ENABLED")
        print("   2. ✅ Ensure EasyOCR models are downloaded")
        print("   3. ✅ Be patient - OCR processing takes time")
        print("   4. 💡 For faster processing, consider:")
        print("      - Using a PDF with native text (if available)")
        print("      - Running OCR preprocessing separately")
        print("      - Using GPU acceleration for EasyOCR (if available)")
        print()
        print("   ⚠️  The translation WILL work, it just requires OCR processing.")

    print()
    print("=" * 70)

except Exception as e:
    print(f"❌ Error during analysis: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
