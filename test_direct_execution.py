#!/usr/bin/env python3
"""
Direct test of Trinity_Fixed.py execution
"""

print("Testing Trinity_Fixed.py direct execution...")

try:
    exec(open('Trinity_Fixed.py').read())
    print("✓ Trinity_Fixed.py executed successfully")
except Exception as e:
    print(f"❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
