#!/bin/bash
# run_all_tests.sh - Run all tests for Steam Jackalope on Linux

echo "Running all tests for Steam Jackalope..."

# Set PYTHONPATH to include current directory for module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest is not installed. Installing..."
    pip install pytest
fi

# Run pytest
echo ""
echo "Executing pytest..."
pytest tests/

if [ $? -ne 0 ]; then
    echo ""
    echo "Tests failed!"
    exit 1
fi

echo ""
echo "All tests passed!"
