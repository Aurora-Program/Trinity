#!/usr/bin/env python3
"""
Debug version to trace Trinity_Fixed.py execution
"""

print("=== DEBUGGING TRINITY_FIXED.PY ===")

# Add debug prints at key points
debug_code = '''
print("DEBUG: Starting Trinity_Fixed.py execution")

# Check if main guard works
print("DEBUG: About to check __name__")
print(f"DEBUG: __name__ = {__name__}")

if __name__ == "__main__":
    print("DEBUG: Inside main block")
    
    # Add debug at each step
    print("DEBUG: Creating KnowledgeBase...")
    kb = None
    try:
        exec(open('Trinity_Fixed.py').read().replace('if __name__ == "__main__":', 'if True:'))
    except Exception as e:
        print(f"DEBUG: Error during execution: {e}")
        import traceback
        traceback.print_exc()
'''

try:
    exec(debug_code)
except Exception as e:
    print(f"DEBUG: Exception in debug code: {e}")
    import traceback
    traceback.print_exc()
