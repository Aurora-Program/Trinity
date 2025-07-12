#!/usr/bin/env python3
"""
Minimal test to verify Trinity_Fixed.py classes are working
"""

print("Starting minimal Trinity test...")

try:
    # Test importing the classes from Trinity_Fixed
    import sys
    sys.path.append('.')
    
    print("Attempting to import Trinity_Fixed...")
    from Trinity_Fixed import Trigate, Transcender, KnowledgeBase, Evolver, Extender
    print("✓ Successfully imported all classes")
    
    # Test basic instantiation
    print("Testing basic instantiation...")
    kb = KnowledgeBase()
    print("✓ KnowledgeBase created")
    
    trans = Transcender()
    print("✓ Transcender created")
    
    evolver = Evolver(kb)
    print("✓ Evolver created")
    
    extender = Extender()
    print("✓ Extender created")
    
    # Test basic functionality
    print("Testing basic functionality...")
    kb.create_space("test_space", "Test space")
    print("✓ Space created successfully")
    
    # Test fractal vector creation
    fv_test = trans.generate_fractal_vector("test_concept", "test_space")
    print(f"✓ Fractal vector created: {type(fv_test)}")
    print(f"  Layer 1: {fv_test.get('layer1', 'Not found')}")
    
    print("\n" + "="*50)
    print("MINIMAL TEST COMPLETED SUCCESSFULLY")
    print("="*50)
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
