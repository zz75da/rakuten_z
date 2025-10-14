#!/usr/bin/env python3
"""Test runner for Rakuten MLOps platform"""

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