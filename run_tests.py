#!/usr/bin/env python3
"""
Test runner script for medAI MVP.
Runs all tests with proper configuration and reporting.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Main test runner."""
    print(" medAI MVP Test Runner")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Test commands
    test_commands = [
        {
            "cmd": ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            "description": "All Tests"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/test_stt.py", "-v"],
            "description": "STT Service Tests"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/test_ner.py", "-v"],
            "description": "NER Service Tests"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/test_llm.py", "-v"],
            "description": "LLM Service Tests"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/test_translation.py", "-v"],
            "description": "Translation Service Tests"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/test_clinical_agent.py", "-v"],
            "description": "Clinical Agent Tests"
        }
    ]
    
    # Run tests
    success_count = 0
    total_count = len(test_commands)
    
    for test_cmd in test_commands:
        if run_command(test_cmd["cmd"], test_cmd["description"]):
            success_count += 1
            print(f"[OK] {test_cmd['description']} - PASSED")
        else:
            print(f"[ERROR] {test_cmd['description']} - FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary: {success_count}/{total_count} test suites passed")
    print(f"{'='*60}")
    
    if success_count == total_count:
        print(" All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
