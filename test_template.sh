#!/bin/bash
# Test script to verify cookiecutter template works with new shared package structure

set -e

echo "========================================="
echo "Testing Cookiecutter Template"
echo "========================================="
echo ""

# Navigate to components directory
cd /Users/prescottrm/projects/ironhorse/components

# Remove any existing test component
if [ -d "template-test" ]; then
    echo "Removing old template-test component..."
    rm -rf template-test
fi

echo "1. Generating component from template..."
cookiecutter ../templates/component --no-input \
    component_name=template-test \
    component_description="Test component for template validation" \
    component_class_name=TemplateTest \
    component_port=8090 \
    python_version=3.11

echo "   ✓ Component generated"
echo ""

echo "2. Setting up component environment..."
cd template-test

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install shared package first
echo "   - Installing shared package..."
pip install -e ../shared --quiet

# Install component
echo "   - Installing component..."
pip install -e ".[dev]" --quiet

echo "   ✓ Environment setup complete"
echo ""

echo "3. Testing imports..."
python3 << 'EOF'
# Test that shared imports work
from shared.schemas import Document, Chunk, APIResponse, ResponseStatus

# Test that component imports work
from template_test.api.routes import router
from template_test.core import TemplateTest

print("   ✓ All imports successful")
EOF

echo ""

echo "4. Running tests..."
pytest tests/ -v --tb=short

echo ""
echo "========================================="
echo "✓ Template test passed!"
echo "========================================="
echo ""
echo "The cookiecutter template correctly generates components that:"
echo "  - Import from the shared package"
echo "  - Use the APIResponse wrapper"
echo "  - Have passing stub tests"
echo ""
echo "Cleaning up..."
cd ..
rm -rf template-test
echo "✓ Test component removed"
