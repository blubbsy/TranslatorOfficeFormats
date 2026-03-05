"""
Quick diagnostic script to analyze the GB 36980.1—2025.pdf file
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import fitz  # PyMuPDF
    from processors.pdf_processor import PdfProcessor

    pdf_path = Path("GB 36980.1—2025.pdf")

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(f"📄 Analyzing: {pdf_path}")
    print(f"   File size: {pdf_path.stat().st_size / 1024:.2f} KB")
    print()

    # Try opening with PyMuPDF directly
    print("🔍 Step 1: Opening with PyMuPDF...")
    try:
        doc = fitz.open(str(pdf_path))
        print(f"   ✅ Successfully opened")
        print(f"   📖 Pages: {len(doc)}")
        print(f"   🔒 Encrypted: {doc.is_encrypted}")
        print(f"   📝 Metadata: {doc.metadata}")
        print()

        # Check first page
        if len(doc) > 0:
            print("🔍 Step 2: Analyzing first page...")
            page = doc[0]
            print(f"   📐 Page size: {page.rect.width} x {page.rect.height}")

            # Try extracting text
            text_dict = page.get_text("dict")
            print(f"   📊 Text blocks found: {len(text_dict.get('blocks', []))}")

            # Try extracting images
            images = page.get_images()
            print(f"   🖼️  Images found: {len(images)}")
            print()

        doc.close()

    except Exception as e:
        print(f"   ❌ PyMuPDF Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print()

    # Try with our processor
    print("🔍 Step 3: Testing with PdfProcessor...")
    try:
        processor = PdfProcessor()
        processor.load(str(pdf_path))
        print(f"   ✅ Loaded successfully")
        print(
            f"   📖 Pages: {processor._document.page_count if processor._document else 'unknown'}"
        )

        print("\n🔍 Step 4: Extracting content...")
        chunks = list(processor.extract_content_generator())
        print(f"   ✅ Extracted {len(chunks)} content chunks")

        # Analyze chunk types
        from collections import Counter

        chunk_types = Counter(chunk.content_type for chunk in chunks)
        print(f"   📊 Chunk breakdown:")
        for ctype, count in chunk_types.items():
            print(f"      - {ctype}: {count}")

        # Show first few chunks
        print(f"\n   📝 First 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            text_preview = (
                chunk.text[:60] + "..." if len(chunk.text) > 60 else chunk.text
            )
            print(f"      {i+1}. [{chunk.content_type}] {text_preview}")

    except Exception as e:
        print(f"   ❌ Processor Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        print()

    print("\n✅ Analysis complete!")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Make sure you're running this from the activated virtual environment")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
