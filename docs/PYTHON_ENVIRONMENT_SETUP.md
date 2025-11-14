# Python Environment Setup for Component Development

Complete guide for setting up your Python environment to develop and test components.

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Running Tests](#running-tests)
5. [Working with Multiple Components](#working-with-multiple-components)
6. [Common Issues & Solutions](#common-issues--solutions)
7. [Using uv (Faster Alternative)](#using-uv-faster-alternative)
8. [Tips & Best Practices](#tips--best-practices)

---

## Quick Setup

For experienced developers who want to get started immediately:

```bash
# Navigate to component
cd components/docling-parser

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Install shared schemas
pip install -e ../shared/

# Install component with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

Done! Skip to [Running Tests](#running-tests) for more options.

---

## Prerequisites

### Required Software

#### 1. Python 3.11 or Higher

Check your Python version:

```bash
python3.11 --version
```

If not installed:

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

#### 2. pip (Usually comes with Python)

Verify pip is installed:

```bash
python3.11 -m pip --version
```

Upgrade pip:

```bash
python3.11 -m pip install --upgrade pip
```

---

## Step-by-Step Setup

### Step 1: Navigate to Component Directory

```bash
# From project root
cd components/docling-parser

# Or for langchain-splitter
cd components/langchain-splitter
```

Verify you're in the right location:

```bash
pwd
# Should show: .../components/docling-parser

ls
# Should show: core/ api/ config/ tests/ pyproject.toml README.md
```

### Step 2: Create Virtual Environment

A virtual environment isolates your component's dependencies from your system Python.

```bash
# Create virtual environment named .venv
python3.11 -m venv .venv
```

This creates a `.venv` directory containing Python and pip executables.

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Verify activation:**

Your prompt should now be prefixed with `(.venv)`:

```bash
(.venv) user@machine:~/components/docling-parser$
```

Check Python location:

```bash
which python    # macOS/Linux
where python    # Windows

# Should point to .venv/bin/python
```

### Step 4: Install Shared Schemas

All components depend on the shared schemas package:

```bash
pip install -e ../shared/
```

**What this does:**
- Installs the `shared-schemas` package in editable mode
- Allows you to import: `from shared.schemas import Document, Chunk, APIResponse`
- Any changes to shared/ are immediately available

**Verify installation:**

```bash
python -c "from shared.schemas import Document; print('✓ Shared schemas installed')"
```

Expected output:
```
✓ Shared schemas installed
```

### Step 5: Install Component Dependencies

Install the component itself with all development dependencies:

```bash
pip install -e ".[dev]"
```

**What this installs:**
- **Production dependencies**: pydantic, fastapi, uvicorn, component-specific packages
- **Development dependencies**: pytest, pytest-cov, pytest-asyncio, httpx
- **Component itself**: In editable mode, so changes are immediately available

**The `[dev]` extra** refers to the `optional-dependencies.dev` section in `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]
```

### Step 6: Verify Installation

Check that everything is installed correctly:

```bash
# List installed packages
pip list

# Should include:
# shared-schemas          0.1.0   /path/to/shared
# docling-parser          0.1.0   /path/to/components/docling-parser
# pytest                  7.x.x
# pytest-cov              4.x.x
# pydantic                2.x.x
# fastapi                 0.x.x
```

**Test imports:**

```bash
# Test shared schemas
python -c "from shared.schemas import Document, Chunk, APIResponse; print('✓ Shared schemas OK')"

# Test component
python -c "from docling_parser.core import DoclingParser; print('✓ Component OK')"
```

Expected output:
```
✓ Shared schemas OK
✓ Component OK
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output (shows each test name)
pytest -v

# Run with extra verbose output (shows test docstrings)
pytest -vv
```

### Test Output Options

```bash
# Show print statements (don't capture output)
pytest -s

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Stop after N failures
pytest --maxfail=3
```

### Running Specific Tests

```bash
# Run specific test file
pytest tests/test_core.py

# Run specific test class
pytest tests/test_core.py::TestDoclingParser

# Run specific test method
pytest tests/test_core.py::TestDoclingParser::test_initialization

# Run tests matching a pattern
pytest -k "test_init"

# Run tests that failed last time
pytest --lf

# Run tests in the order they failed last time
pytest --ff
```

### Coverage Reports

```bash
# Run with coverage
pytest --cov=docling_parser

# Run with coverage and HTML report
pytest --cov=docling_parser --cov-report=html

# Run with coverage and terminal report showing missing lines
pytest --cov=docling_parser --cov-report=term-missing

# Combined: HTML + terminal with missing lines
pytest --cov=docling_parser --cov-report=html --cov-report=term-missing
```

**View HTML coverage report:**

```bash
# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html

# Windows
start htmlcov/index.html
```

### Parallel Test Execution

For faster test runs, install pytest-xdist:

```bash
pip install pytest-xdist

# Run tests in parallel (auto-detect CPU cores)
pytest -n auto

# Run tests on 4 cores
pytest -n 4
```

### Watch Mode

Install pytest-watch for automatic test running on file changes:

```bash
pip install pytest-watch

# Watch and run tests on changes
ptw

# Watch with coverage
ptw -- --cov=docling_parser
```

---

## Working with Multiple Components

### Option 1: Separate Environments (Recommended)

Create a separate virtual environment for each component:

```bash
# Terminal 1 - Docling Parser
cd components/docling-parser
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ../shared/
pip install -e ".[dev]"
pytest

# Terminal 2 - LangChain Splitter
cd components/langchain-splitter
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ../shared/
pip install -e ".[dev]"
pytest
```

**Advantages:**
- Isolated dependencies
- No conflicts between components
- Cleaner environment

### Option 2: Shared Environment

Create a single environment at project root:

```bash
# From project root
python3.11 -m venv .venv
source .venv/bin/activate

# Install shared schemas
pip install -e components/shared/

# Install all components
pip install -e "components/docling-parser[dev]"
pip install -e "components/langchain-splitter[dev]"

# Run tests for all components
pytest components/

# Run tests for specific component
pytest components/docling-parser/
pytest components/langchain-splitter/
```

**Advantages:**
- Single environment to manage
- Can test cross-component integration
- Easier to run all tests

**Disadvantages:**
- Potential dependency conflicts
- Larger environment

### Running All Tests

From project root with shared environment:

```bash
# All tests with coverage
pytest --cov=components --cov-report=html

# Specific components
pytest components/docling-parser/ components/langchain-splitter/

# Parallel execution
pytest -n auto components/
```

---

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError for 'shared'

**Error:**
```
ModuleNotFoundError: No module named 'shared'
```

**Cause:** Shared schemas package not installed.

**Solution:**
```bash
# Install shared schemas
pip install -e ../shared/

# Verify
python -c "import shared; print(shared.__file__)"
```

### Issue 2: No Tests Collected

**Error:**
```
collected 0 items
```

**Causes & Solutions:**

1. **Wrong directory:**
   ```bash
   # Check location
   pwd
   # Should be in components/<component-name>

   # Navigate to correct location
   cd components/docling-parser
   ```

2. **Test files not named correctly:**
   - Test files must be named `test_*.py` or `*_test.py`
   - Test functions must start with `test_`
   - Test classes must start with `Test`

3. **Check test discovery:**
   ```bash
   pytest --collect-only
   ```

### Issue 3: Import Errors in Tests

**Error:**
```
ImportError: attempted relative import with no known parent package
```

**Cause:** Tests can't find the component or shared modules.

**Solution:** Ensure test files have proper sys.path setup:

```python
# At top of test file
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Document
from docling_parser.core import DoclingParser
```

### Issue 4: Wrong Python Version

**Error:** Virtual environment uses wrong Python version.

**Solution:**
```bash
# Remove old venv
rm -rf .venv

# Create with specific Python version
python3.11 -m venv .venv
source .venv/bin/activate

# Verify
python --version
# Should show: Python 3.11.x
```

### Issue 5: Virtual Environment Not Activated

**Symptom:** pip installs packages globally or tests fail.

**Check if activated:**
```bash
# Should show .venv path
which python

# Should be prefixed with (.venv)
echo $PS1
```

**Solution:**
```bash
# Activate it
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Issue 6: External Dependencies Not Installed

**Error:**
```
ImportError: No module named 'docling'
ImportError: No module named 'langchain_text_splitters'
```

**Cause:** Optional external dependencies not installed.

**Solution:**

For development/testing (uses mocks):
```bash
# Tests use mocks, should work without external deps
pytest
```

For actual functionality:
```bash
# Install external dependencies
pip install docling                    # for docling-parser
pip install langchain-text-splitters   # for langchain-splitter
```

### Issue 7: Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Never use sudo with pip in a virtual environment
# If you see this, you likely forgot to activate venv

# Activate venv first
source .venv/bin/activate

# Then install
pip install -e ".[dev]"
```

### Issue 8: pytest Not Found

**Error:**
```
command not found: pytest
```

**Cause:** pytest not installed or venv not activated.

**Solution:**
```bash
# Activate venv
source .venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Verify
which pytest
# Should show: .venv/bin/pytest
```

---

## Using uv (Faster Alternative)

uv is a faster Python package installer (10-100x faster than pip).

### Install uv

```bash
# Option 1: Using pip
pip install uv

# Option 2: Using curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Setup with uv

```bash
# Navigate to component
cd components/docling-parser

# Create virtual environment
uv venv

# Activate (same as before)
source .venv/bin/activate

# Install shared schemas (much faster!)
uv pip install -e ../shared/

# Install component with dev dependencies (much faster!)
uv pip install -e ".[dev]"

# Run tests (same as before)
pytest
```

### Speed Comparison

**Example: Installing docling-parser dependencies**

```bash
# With pip: ~45 seconds
time pip install -e ".[dev]"

# With uv: ~5 seconds
time uv pip install -e ".[dev]"
```

### Using uv Commands

```bash
# Install package
uv pip install package-name

# Install from requirements
uv pip install -r requirements.txt

# Install editable package
uv pip install -e .

# Sync dependencies (install/update all)
uv pip sync requirements.txt

# List installed packages
uv pip list

# Show package info
uv pip show package-name
```

---

## Tips & Best Practices

### 1. Always Activate Virtual Environment

Before any development work:

```bash
source .venv/bin/activate
```

Add to your shell profile for convenience:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias venv='source .venv/bin/activate'
```

Usage:
```bash
cd components/docling-parser
venv  # activates .venv
```

### 2. Keep Dependencies Updated

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade pytest

# Update all dev dependencies
pip install --upgrade pytest pytest-cov pytest-asyncio httpx
```

### 3. Use pytest.ini for Configuration

Create `pytest.ini` in component root for persistent settings:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=docling_parser --cov-report=html --cov-report=term-missing
```

Now just run:
```bash
pytest  # Uses settings from pytest.ini
```

### 4. Use VS Code with Python Extension

If using VS Code, configure it to use your virtual environment:

1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Select "Python: Select Interpreter"
3. Choose `.venv/bin/python`

The extension will automatically discover tests and show them in the Test Explorer.

### 5. Add .venv to .gitignore

Ensure `.gitignore` includes:

```gitignore
.venv/
venv/
*.pyc
__pycache__/
.pytest_cache/
htmlcov/
.coverage
*.egg-info/
```

### 6. Create a Setup Script

Create `setup_dev.sh` in component root:

```bash
#!/bin/bash
# setup_dev.sh - Quick development environment setup

set -e

echo "Setting up development environment..."

# Check Python version
python3.11 --version || { echo "Python 3.11 required"; exit 1; }

# Create venv
echo "Creating virtual environment..."
python3.11 -m venv .venv

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing shared schemas..."
pip install -e ../shared/

echo "Installing component with dev dependencies..."
pip install -e ".[dev]"

# Verify
echo "Verifying installation..."
python -c "from shared.schemas import Document; print('✓ Shared schemas OK')"
python -c "from docling_parser.core import DoclingParser; print('✓ Component OK')"

echo "✓ Setup complete! Run 'source .venv/bin/activate' to activate the environment."
```

Make it executable and run:

```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

### 7. Use Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
pip install pre-commit black ruff mypy

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
EOF

# Install hooks
pre-commit install

# Now tests run automatically before commit
```

### 8. Quick Test Feedback Loop

For rapid development:

```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and run tests automatically
ptw -- -v --cov=docling_parser

# In another terminal, edit your code
# Tests run automatically on save!
```

### 9. Debugging Tests

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on error (not assertion)
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb

# With ipdb (better debugger)
pip install ipdb
pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb
```

### 10. Environment Variables for Tests

Create `.env.test` file:

```bash
# .env.test
DOCLING_PARSER_LOG_LEVEL=DEBUG
LANGCHAIN_SPLITTER_CHUNK_SIZE=500
```

Load in tests:

```python
# conftest.py
import os
from dotenv import load_dotenv

load_dotenv('.env.test')
```

---

## Complete Example Workflow

Here's a complete workflow from scratch to running tests:

```bash
# 1. Navigate to project
cd ~/projects/ai-toolkit/components/docling-parser

# 2. Check Python version
python3.11 --version
# Python 3.11.x

# 3. Create virtual environment
python3.11 -m venv .venv

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install shared schemas
pip install -e ../shared/
# Processing .../shared
# Successfully installed shared-schemas-0.1.0

# 7. Install component with dev dependencies
pip install -e ".[dev]"
# Processing .../components/docling-parser
# Successfully installed docling-parser-0.1.0 pytest-7.4.0 ...

# 8. Verify installation
pip list | grep -E "(shared|docling)"
# docling-parser          0.1.0
# shared-schemas          0.1.0

# 9. Run tests
pytest -v
# ===== test session starts =====
# collected 15 items
#
# tests/test_core.py::TestDoclingParser::test_initialization PASSED
# tests/test_core.py::TestDoclingParser::test_parse_document PASSED
# ...
# ===== 15 passed in 2.34s =====

# 10. Run with coverage
pytest --cov=docling_parser --cov-report=html
# Coverage: 87%

# 11. View coverage report
open htmlcov/index.html

# 12. Develop and test
# Edit code in your editor...
pytest  # Run tests again

# 13. When done, deactivate
deactivate
```

---

## Summary

**Essential Commands:**

```bash
# Setup (one time)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ../shared/
pip install -e ".[dev]"

# Daily workflow
source .venv/bin/activate  # Always activate first
pytest                      # Run tests
pytest --cov=component     # With coverage
deactivate                 # When done
```

**Remember:**
1. Always activate virtual environment before working
2. Install shared schemas first
3. Install component with `[dev]` for test dependencies
4. Run `pytest` from component root directory
5. Keep your environment updated

---

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [uv Documentation](https://github.com/astral-sh/uv)
- [Component Development Guide](COMPONENT_DEVELOPMENT_GUIDE.md)
- [Developer Quick Reference](DEVELOPER_QUICK_REFERENCE.md)

---

**Need Help?**

If you encounter issues not covered here:
1. Check [Common Issues](#common-issues--solutions)
2. Verify virtual environment is activated
3. Ensure you're in the correct directory
4. Check that shared schemas are installed
5. Try recreating the virtual environment

For component-specific issues, see the component's README.
