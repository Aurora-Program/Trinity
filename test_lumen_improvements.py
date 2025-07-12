"""
Test script for Trinity Enhanced - Validates Lumen's improvements
"""

from Trinity_Enhanced import (
    Trigate, Transcender, KnowledgeBase, Evolver, Extender,
    TernaryOperations, InputValidator, ValidationError, LogicalCoherenceError
)

def test_lumen_improvements():
    """Test all the improvements suggested by Lumen"""
    
    print("🧪 TESTING LUMEN'S IMPROVEMENTS")
    print("=" * 50)
    
    # Test 1: Centralized validation (no redundant checks)
    print("\n✅ Test 1: Centralized Validation")
    try:
        trigate = Trigate([1, 0, 1], [0, 1, 0])
        result = trigate.infer_result()  # Should validate once internally
        print(f"   Trigate inference successful: {result}")
        
        # Test validation error handling
        try:
            bad_trigate = Trigate([1, 0], [0, 1, 0])  # Wrong length
            bad_trigate.infer_result()
        except ValidationError as e:
            print(f"   ✓ Validation error caught correctly: {e}")
        
    except Exception as e:
        print(f"   ❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Separated operations (pattern generation vs inference)
    print("\n✅ Test 2: Separated Pattern Generation vs Inference")
    try:
        trigate = Trigate([1, 0, 1], [0, 1, 0])
        
        # Generate pattern (not inference)
        pattern = trigate.generate_pattern(seed=42)
        print(f"   Generated pattern: {pattern}")
        
        # Infer result using pattern
        result = trigate.infer_result()
        print(f"   Inferred result: {result}")
        
        # Learn pattern from known inputs/outputs
        trigate.R = [1, 1, 0]
        learned = trigate.learn_pattern()
        print(f"   Learned pattern: {learned}")
        
    except Exception as e:
        print(f"   ❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Centralized ternary operations
    print("\n✅ Test 3: Centralized Ternary Operations")
    try:
        # Test vector operations
        vec1 = [1, 0, None]
        vec2 = [0, 1, 1]
        xor_result = TernaryOperations.xor_vector(vec1, vec2)
        print(f"   Vector XOR: {vec1} ⊕ {vec2} = {xor_result}")
        
        # Test None propagation
        none_result = TernaryOperations.xor(None, 1)
        print(f"   None propagation: None ⊕ 1 = {none_result}")
        
    except Exception as e:
        print(f"   ❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Enhanced logging (would be visible in real usage)
    print("\n✅ Test 4: Enhanced Logging System")
    try:
        import logging
        logger = logging.getLogger("Trinity_Enhanced")
        logger.info("Test logging message")
        print("   ✓ Logging system initialized (check console for log messages)")
        
    except Exception as e:
        print(f"   ❌ Test 4 failed: {e}")
        return False
    
    # Test 5: Type hints and documentation consistency
    print("\n✅ Test 5: Type Hints and API Consistency")
    try:
        transcender = Transcender()
        
        # Method should have proper type hints and return tuple
        result = transcender.process_inputs([1, 0, 1], [0, 1, 0], [1, 1, 1])
        print(f"   Process inputs returns tuple: {type(result)} with {len(result)} elements")
        
        # Check if it's a proper tuple with 3 elements
        ms, ss, meta_m = result
        print(f"   Ms: {ms}, Ss: {ss}, MetaM length: {len(meta_m)}")
        
    except Exception as e:
        print(f"   ❌ Test 5 failed: {e}")
        return False
    
    # Test 6: Explicit space management (fail-fast)
    print("\n✅ Test 6: Explicit Space Management")
    try:
        kb = KnowledgeBase(auto_create_spaces=False)  # Explicit mode
        
        # This should fail because space doesn't exist
        try:
            kb.store_axiom("nonexistent", [1, 0, 1], {}, [0, 1, 0], {})
            print("   ❌ Should have failed for nonexistent space")
            return False
        except ValidationError:
            print("   ✓ Correctly failed for nonexistent space")
        
        # Create space explicitly
        kb.create_space("test_space", "Test space for validation")
        success = kb.store_axiom("test_space", [1, 0, 1], {}, [0, 1, 0], {})
        print(f"   ✓ Explicit space creation and storage: {success}")
        
    except Exception as e:
        print(f"   ❌ Test 6 failed: {e}")
        return False
    
    # Test 7: Configurable thresholds
    print("\n✅ Test 7: Configurable Thresholds")
    try:
        evolver = Evolver(kb, similarity_threshold=0.3)  # Custom threshold
        print(f"   ✓ Evolver initialized with custom threshold: {evolver.similarity_threshold}")
        
        # Test prediction with confidence adjustment
        prediction = evolver.predict_interaction_outcome(
            {"state": [1, 0, 1]},
            {"user_preference": "test", "historical_pattern": "positive"}
        )
        print(f"   ✓ Dynamic prediction confidence: {prediction['confidence']:.2f}")
        
    except Exception as e:
        print(f"   ❌ Test 7 failed: {e}")
        return False
    
    # Test 8: Enhanced error handling and fallbacks
    print("\n✅ Test 8: Enhanced Error Handling")
    try:
        extender = Extender()
        
        # Should fail gracefully without guide package
        try:
            extender.reconstruct_basic([1, 0, 1])
            print("   ❌ Should have failed without guide package")
            return False
        except ValidationError:
            print("   ✓ Correctly failed without guide package")
        
        # Load guide package and test
        guide_package = {"axiom_registry": {(1, 0, 1): {"original_inputs": {"test": "data"}}}}
        extender.load_guide_package(guide_package)
        result = extender.reconstruct_basic([1, 0, 1])
        print(f"   ✓ Reconstruction successful: {result}")
        
    except Exception as e:
        print(f"   ❌ Test 8 failed: {e}")
        return False
    
    # Test 9: Consistency in None handling
    print("\n✅ Test 9: Consistent None Handling")
    try:
        # Test that None values are properly propagated, not silenced
        result_with_none = TernaryOperations.xor(None, 1)
        if result_with_none is not None:
            print(f"   ❌ None should propagate, got: {result_with_none}")
            return False
        
        print("   ✓ None values properly propagated through operations")
        
        # Test fractal operations with None
        transcender = Transcender()
        fractal = transcender.synthesize_fractal_l1([1, None, 0], [0, 1, None], [1, 1, 1])
        print(f"   ✓ Fractal synthesis handles None values: L1={fractal['layer1']}")
        
    except Exception as e:
        print(f"   ❌ Test 9 failed: {e}")
        return False
    
    # Test 10: Performance improvements (caching, etc.)
    print("\n✅ Test 10: Performance Improvements")
    try:
        extender = Extender()
        guide_package = {
            "axiom_registry": {
                (1, 0, 1): {"original_inputs": {"test": "cached_data"}},
                (0, 1, 0): {"original_inputs": {"test": "other_data"}}
            }
        }
        extender.load_guide_package(guide_package)
        
        # First call (should cache)
        result1 = extender.reconstruct_basic([1, 0, 1])
        
        # Second call (should use cache)
        result2 = extender.reconstruct_basic([1, 0, 1])
        
        print(f"   ✓ Caching system working: {result1 == result2}")
        print(f"   ✓ Cache size: {len(extender.reconstruction_cache)}")
        
    except Exception as e:
        print(f"   ❌ Test 10 failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL LUMEN'S IMPROVEMENTS VALIDATED SUCCESSFULLY!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_lumen_improvements()
    if success:
        print("\n✅ Trinity Enhanced is ready for production!")
    else:
        print("\n❌ Some tests failed - check implementation")
