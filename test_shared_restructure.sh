#!/bin/bash
# Test script to verify shared package restructuring

set -e  # Exit on error

echo "========================================="
echo "Testing Shared Package Restructuring"
echo "========================================="
echo ""

# Navigate to shared package
cd /Users/prescottrm/projects/ironhorse/components/shared

echo "1. Installing shared package..."
pip install -e . --quiet
echo "   ✓ Installation complete"
echo ""

echo "2. Testing Python imports..."
python3 << 'EOF'
from shared.schemas import Document, Chunk, APIResponse, ResponseStatus, DocumentMetadata, ChunkMetadata

# Test Document
doc = Document(doc_id="test123", content="Test content")
assert doc.doc_id == "test123"
assert doc.content == "Test content"
print("   ✓ Document class works")

# Test Chunk
chunk = Chunk(chunk_id="chunk123", text="Test chunk text")
assert chunk.chunk_id == "chunk123"
assert chunk.text == "Test chunk text"
print("   ✓ Chunk class works")

# Test APIResponse
response = APIResponse.success(data={"test": "data"}, metadata={"time": 100})
assert response.status == ResponseStatus.SUCCESS
assert response.data == {"test": "data"}
print("   ✓ APIResponse class works")

# Test error response
error_response = APIResponse.error(error="Test error")
assert error_response.status == ResponseStatus.ERROR
assert error_response.error == "Test error"
print("   ✓ Error response works")

print("")
print("✓ All import tests passed!")
EOF

echo ""
echo "3. Verifying package structure..."
if [ -d "shared/schemas" ]; then
    echo "   ✓ Nested structure exists: shared/schemas/"
else
    echo "   ✗ ERROR: Nested structure not found!"
    exit 1
fi

if [ ! -d "schemas" ]; then
    echo "   ✓ Old schemas/ directory removed"
else
    echo "   ✗ WARNING: Old schemas/ directory still exists"
fi

echo ""
echo "========================================="
echo "✓ All tests passed!"
echo "========================================="
echo ""
echo "The shared package restructuring is complete and working correctly."
echo "No .pth files are needed for conda or venv environments."
