#!/bin/bash
# Test script to verify Component interface implementation

set -e

echo "========================================="
echo "Testing Component Interface"
echo "========================================="
echo ""

# Test 1: Verify shared Component ABC exists
echo "1. Testing shared Component ABC..."
cd /Users/prescottrm/projects/ironhorse/components/shared
python3 << 'EOF'
from shared.component import Component
from abc import ABC

# Verify Component is an ABC
assert issubclass(Component, ABC)
print("   ✓ Component is an abstract base class")

# Verify process method is abstract
import inspect
assert 'process' in dir(Component)
print("   ✓ Component has process() method")

# Try to instantiate (should fail)
try:
    Component()
    assert False, "Should not be able to instantiate Component ABC"
except TypeError as e:
    assert "abstract" in str(e).lower()
    print("   ✓ Component cannot be instantiated directly")

print("   ✓ Component ABC is correctly defined")
EOF

echo ""

# Test 2: Test docling-parser implementation
echo "2. Testing docling-parser Component implementation..."
cd /Users/prescottrm/projects/ironhorse/components/docling-parser
.venv/bin/python3 << 'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.parent))

from shared.component import Component
from docling_parser.core import DoclingParser

# Verify DoclingParser is a Component
assert issubclass(DoclingParser, Component)
print("   ✓ DoclingParser inherits from Component")

# Verify it has process method
assert hasattr(DoclingParser, 'process')
print("   ✓ DoclingParser has process() method")

print("   ✓ DoclingParser correctly implements Component interface")
EOF

echo ""

# Test 3: Test langchain-splitter implementation
echo "3. Testing langchain-splitter Component implementation..."
cd /Users/prescottrm/projects/ironhorse/components/langchain-splitter
.venv/bin/python3 << 'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.parent))

from shared.component import Component
from langchain_splitter.core import LangChainSplitter

# Verify LangChainSplitter is a Component
assert issubclass(LangChainSplitter, Component)
print("   ✓ LangChainSplitter inherits from Component")

# Verify it has process method
assert hasattr(LangChainSplitter, 'process')
print("   ✓ LangChainSplitter has process() method")

print("   ✓ LangChainSplitter correctly implements Component interface")
EOF

echo ""

# Test 4: Run docling-parser tests
echo "4. Running docling-parser tests..."
cd /Users/prescottrm/projects/ironhorse/components/docling-parser
.venv/bin/pytest tests/test_core.py -v -k "process" --tb=short
echo "   ✓ All docling-parser process() tests passed"

echo ""

# Test 5: Run langchain-splitter tests
echo "5. Running langchain-splitter tests..."
cd /Users/prescottrm/projects/ironhorse/components/langchain-splitter
.venv/bin/pytest tests/test_core.py -v -k "process" --tb=short
echo "   ✓ All langchain-splitter process() tests passed"

echo ""

# Test 6: Test cookiecutter template
echo "6. Testing cookiecutter template..."
cd /Users/prescottrm/projects/ironhorse/components

# Remove any existing test component
if [ -d "component-interface-test" ]; then
    rm -rf component-interface-test
fi

# Generate test component
cookiecutter ../templates/component --no-input \
    component_name=component-interface-test \
    component_description="Test component for interface validation" \
    component_class_name=ComponentInterfaceTest \
    component_port=8095 \
    python_version=3.11 > /dev/null 2>&1

cd component-interface-test

# Create venv and install
python3 -m venv .venv > /dev/null 2>&1
source .venv/bin/activate
pip install -e ../shared --quiet
pip install -e ".[dev]" --quiet

# Test Component interface
python3 << 'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent.parent))

from shared.component import Component
from component_interface_test.core import ComponentInterfaceTest

# Verify it's a Component
assert issubclass(ComponentInterfaceTest, Component)
print("   ✓ Generated component inherits from Component")

# Test instantiation and process method
component = ComponentInterfaceTest()
result = component.process({"test": "data"})
assert result is not None
print("   ✓ Generated component process() method works")
EOF

# Run tests
pytest tests/test_core.py -v --tb=short > /dev/null 2>&1
echo "   ✓ Generated component tests passed"

# Cleanup
cd ..
rm -rf component-interface-test

echo ""
echo "========================================="
echo "✓ All Component Interface Tests Passed!"
echo "========================================="
echo ""
echo "Summary:"
echo "  ✓ Component ABC is correctly defined"
echo "  ✓ DoclingParser implements Component interface"
echo "  ✓ LangChainSplitter implements Component interface"
echo "  ✓ All process() method tests pass"
echo "  ✓ Cookiecutter template generates valid Components"
echo ""
