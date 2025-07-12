#!/usr/bin/env python3
"""
Step by step debugging
"""

import sys
import os

print("Step 1: Current working directory")
print(f"CWD: {os.getcwd()}")

print("\nStep 2: Check if Trinity_Fixed.py exists")
print(f"File exists: {os.path.exists('Trinity_Fixed.py')}")
print(f"File size: {os.path.getsize('Trinity_Fixed.py') if os.path.exists('Trinity_Fixed.py') else 'N/A'}")

print("\nStep 3: Try to read first few lines")
try:
    with open('Trinity_Fixed.py', 'r', encoding='utf-8') as f:
        first_lines = [f.readline().strip() for _ in range(5)]
    for i, line in enumerate(first_lines):
        print(f"Line {i+1}: {line}")
except Exception as e:
    print(f"Error reading file: {e}")

print("\nStep 4: Try to compile the file")
try:
    with open('Trinity_Fixed.py', 'r', encoding='utf-8') as f:
        content = f.read()
    compiled = compile(content, 'Trinity_Fixed.py', 'exec')
    print("✓ File compiled successfully")
except Exception as e:
    print(f"❌ Compilation error: {e}")
    sys.exit(1)

print("\nStep 5: Try to execute with explicit output")
try:
    print("Executing Trinity_Fixed.py content...")
    sys.stdout.flush()  # Force flush before execution
    exec(compiled)
    print("✓ Execution completed")
except Exception as e:
    print(f"❌ Execution error: {e}")
    import traceback
    traceback.print_exc()
