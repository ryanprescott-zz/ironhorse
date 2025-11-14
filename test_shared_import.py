#!/usr/bin/env python3
"""Test script to verify shared package structure."""
import sys

# Add components directory to path
sys.path.insert(0, '/Users/prescottrm/projects/ironhorse/components/shared')

try:
    from shared.schemas import Document, Chunk, APIResponse, ResponseStatus
    print("✓ Import successful!")
    print(f"  - Document: {Document}")
    print(f"  - Chunk: {Chunk}")
    print(f"  - APIResponse: {APIResponse}")
    print(f"  - ResponseStatus: {ResponseStatus}")

    # Test creating instances
    doc = Document(doc_id="test", content="test content")
    print(f"\n✓ Created Document instance: {doc.doc_id}")

    chunk = Chunk(chunk_id="test", text="test text")
    print(f"✓ Created Chunk instance: {chunk.chunk_id}")

    response = APIResponse.success(data={"test": "data"})
    print(f"✓ Created APIResponse instance: {response.status}")

    print("\n✓ All tests passed! New package structure is working correctly.")
    sys.exit(0)

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
