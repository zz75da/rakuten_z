#!/usr/bin/env python3
"""
Convenience test runner for the Rakuten MLOps platform
=======================================================
Purpose : Programmatic entry-point for running the full test suite with
          coverage reporting, usable outside of a plain `pytest` invocation.

Usage   :
    python tests/run_tests.py          # run all tests with coverage
    py -3.12 -m pytest tests/unit/     # unit tests only (recommended)
    py -3.12 -m pytest tests/integration/  # integration tests only

Note    : Direct `pytest` invocation is preferred over this script.
          Coverage targets reference the Docker service packages; adjust
          --cov arguments if running against the local source tree.
"""

import pytest
import sys
import os

def run_tests():
    """Run all tests"""
    # Add the project root to Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run tests with coverage
    pytest_args = [
        "-v",
        "--cov=preprocess_api",
        "--cov=gate_api", 
        "--cov=train_api",
        "--cov=predict_api",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report",
        "tests/"
    ]
    
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests()