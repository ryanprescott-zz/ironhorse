#!/bin/bash
# Script to update component environments with restructured shared package

set -e

echo "========================================="
echo "Updating Component Environments"
echo "========================================="
echo ""

# Update docling-parser
echo "1. Updating docling-parser environment..."
cd /Users/prescottrm/projects/ironhorse/components/docling-parser

# Remove any old .pth files if they exist
if [ -f ".venv/lib/python3.13/site-packages/components_path.pth" ]; then
    echo "   - Removing old .pth file"
    rm .venv/lib/python3.13/site-packages/components_path.pth
fi

# Reinstall shared package
echo "   - Reinstalling shared package..."
.venv/bin/pip uninstall -y shared-schemas 2>/dev/null || true
.venv/bin/pip install -e ../shared --quiet
echo "   ✓ docling-parser environment updated"
echo ""

# Update langchain-splitter
echo "2. Updating langchain-splitter environment..."
cd /Users/prescottrm/projects/ironhorse/components/langchain-splitter

# Remove any old .pth files if they exist
if [ -f ".venv/lib/python3.13/site-packages/components_path.pth" ]; then
    echo "   - Removing old .pth file"
    rm .venv/lib/python3.13/site-packages/components_path.pth
fi

# Reinstall shared package
echo "   - Reinstalling shared package..."
.venv/bin/pip uninstall -y shared-schemas 2>/dev/null || true
.venv/bin/pip install -e ../shared --quiet
echo "   ✓ langchain-splitter environment updated"
echo ""

echo "========================================="
echo "✓ All component environments updated!"
echo "========================================="
echo ""
echo "Testing imports..."
echo ""

# Test docling-parser
echo "3. Testing docling-parser imports..."
cd /Users/prescottrm/projects/ironhorse/components/docling-parser
.venv/bin/python3 << 'EOF'
from shared.schemas import Document, APIResponse
print("   ✓ docling-parser can import shared.schemas")
EOF

# Test langchain-splitter
echo "4. Testing langchain-splitter imports..."
cd /Users/prescottrm/projects/ironhorse/components/langchain-splitter
.venv/bin/python3 << 'EOF'
from shared.schemas import Chunk, APIResponse
print("   ✓ langchain-splitter can import shared.schemas")
EOF

echo ""
echo "========================================="
echo "✓ All tests passed!"
echo "========================================="
