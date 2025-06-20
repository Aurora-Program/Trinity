#!/usr/bin/env python3
"""
TRINITY LIBRARY - PRACTICAL EXAMPLES
===================================
Real-world usage examples demonstrating Trinity's capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Trinity import Trigate, Transcender, KnowledgeBase, Evolver, Extender

def example_1_basic_inference():
    """Example 1: Basic ternary logic inference"""
    print("🔧 EXAMPLE 1: Basic Ternary Logic Inference")
    print("-" * 50)
    
    # Create a Trigate with some inputs
    trigate = Trigate([1, 0, 1], [0, 1, 0])
    
    # Perform inference
    result = trigate.inferir()
    print(f"Input A: {trigate.A}")
    print(f"Input B: {trigate.B}")
    print(f"Inferred result: {result}")
    
    # Now let's add a known result and see if it can learn
    trigate.R = [1, 1, 1]  # Set R before learning
    learned = trigate.aprender()
    print(f"Learning successful: {learned is not None}")
    print(f"Meta-information M: {trigate.M}")
    print()

def example_2_uncertainty_handling():
    """Example 2: Handling uncertainty with None values"""
    print("🔍 EXAMPLE 2: Uncertainty Handling")
    print("-" * 50)
    
    # Create inputs with uncertainty (None values)
    trigate = Trigate([1, None, 0], [0, 1, None])
    
    result = trigate.inferir()
    synthesis = trigate.sintesis_S()
    
    print(f"Input A (with uncertainty): {trigate.A}")
    print(f"Input B (with uncertainty): {trigate.B}")
    print(f"Inference result: {result}")
    print(f"Synthesis result: {synthesis}")
    
    # Try learning with uncertain inputs
    trigate.R = [1, None, 0]  # Set R with some uncertainty
    learned = trigate.aprender()
    print(f"Learning with uncertainty successful: {learned is not None}")
    print(f"Meta-information M: {trigate.M}")
    print("Note: None values propagate through operations")
    print()

def example_3_fractal_synthesis():
    """Example 3: Hierarchical fractal synthesis"""
    print("🌟 EXAMPLE 3: Fractal Synthesis")
    print("-" * 50)
    
    transcender = Transcender()
    
    # Process three inputs to create fractal structure
    inputs = ([1, 0, 1], [0, 1, 0], [1, 1, 1])
    fractal_vector = transcender.level1_synthesis(*inputs)
    
    print(f"Input vectors: {inputs}")
    print(f"Fractal Layer 1: {fractal_vector['layer1']}")
    print(f"Fractal Layer 2: {fractal_vector['layer2']}")
    print(f"Fractal Layer 3 (first 3): {fractal_vector['layer3'][:3]}...")
    print(f"Total layer 3 elements: {len(fractal_vector['layer3'])}")
    print()

def example_4_knowledge_management():
    """Example 4: Knowledge base management"""
    print("🧠 EXAMPLE 4: Knowledge Management")
    print("-" * 50)
    
    # Create knowledge base and multiple spaces
    kb = KnowledgeBase()
    kb.create_space("medical", "Medical diagnosis patterns")
    kb.create_space("robotics", "Robotics behavior patterns")
    
    evolver = Evolver(kb)
    transcender = Transcender()
    
    # Create and store different types of knowledge
    medical_pattern = transcender.level1_synthesis([1, 0, 1], [0, 1, 1], [1, 0, 0])
    robotics_pattern = transcender.level1_synthesis([0, 1, 0], [1, 0, 1], [0, 0, 1])
    
    # Store in appropriate spaces
    evolver.formalize_fractal_axiom(
        medical_pattern, 
        {"domain": "cardiology", "confidence": 0.85}, 
        "medical"
    )
    
    evolver.formalize_fractal_axiom(
        robotics_pattern,
        {"task": "navigation", "environment": "indoor"},
        "robotics"
    )
    
    # Query knowledge base
    print("Available spaces:", kb.list_spaces())
    print("Medical space stats:", kb.space_stats("medical"))
    print("Robotics space stats:", kb.space_stats("robotics"))
    
    # Retrieve specific axiom
    medical_axiom = kb.get_axiom_by_ms("medical", medical_pattern["layer1"])
    if medical_axiom:
        print(f"Retrieved medical axiom: {medical_axiom['original_inputs']}")
    print()

def example_5_pattern_detection():
    """Example 5: Pattern detection with Evolver"""
    print("🔍 EXAMPLE 5: Pattern Detection")
    print("-" * 50)
    
    kb = KnowledgeBase()
    evolver = Evolver(kb)
    
    # Test different vector patterns using detect_fractal_pattern
    patterns = {
        "unitary": [1, 1, 1],
        "uniform_zeros": [0, 0, 0],
        "uniform_none": [None, None, None], 
        "complex": [1, 0, 1],
        "mixed": [0, 1, 0]
    }
    
    for name, vector in patterns.items():
        try:
            pattern = evolver.detect_fractal_pattern(vector)
            print(f"{name:15} {vector} → {pattern}")
        except Exception as e:
            print(f"{name:15} {vector} → Error: {e}")
    print()

def example_6_reconstruction():
    """Example 6: Knowledge reconstruction"""
    print("🔄 EXAMPLE 6: Knowledge Reconstruction")
    print("-" * 50)
    
    # Setup complete pipeline
    kb = KnowledgeBase()
    evolver = Evolver(kb)
    transcender = Transcender()
    extender = Extender()
    
    # Create and store some knowledge
    original_inputs = ([1, 0, 1], [0, 1, 0], [1, 1, 1])
    fractal_vector = transcender.level1_synthesis(*original_inputs)
    
    evolver.formalize_fractal_axiom(
        fractal_vector, 
        {"experiment": "reconstruction_test", "inputs": original_inputs}
    )
    
    # Generate guide package and load into extender
    guide_package = evolver.generate_guide_package("default")
    extender.load_guide_package(guide_package)
      # Attempt reconstruction
    target_ms = fractal_vector["layer1"]
    reconstructed = extender.reconstruct(target_ms)
    
    print(f"Original inputs: {original_inputs}")
    print(f"Target Ms: {target_ms}")
    print(f"Reconstructed: {reconstructed}")
    
    # Check if reconstruction is successful
    if isinstance(reconstructed, dict) and "inputs" in reconstructed:
        reconstructed_inputs = reconstructed["inputs"]
        success = reconstructed_inputs == original_inputs
    elif isinstance(reconstructed, (tuple, list)) and len(reconstructed) == 3:
        success = tuple(reconstructed) == original_inputs
    else:
        success = False
        
    print(f"Reconstruction successful: {success}")
    print()

def example_7_medical_diagnosis():
    """Example 7: Medical diagnosis simulation"""
    print("🏥 EXAMPLE 7: Medical Diagnosis Simulation")
    print("-" * 50)
    
    kb = KnowledgeBase()
    kb.create_space("diagnosis", "Medical diagnostic patterns")
    evolver = Evolver(kb)
    transcender = Transcender()
    
    # Simulate patient symptoms as ternary vectors
    # [symptom1, symptom2, symptom3] where 1=present, 0=absent, None=uncertain
    patients = {
        "patient_1": {
            "symptoms": ([1, 0, 1], [0, 1, 0], [1, 1, 0]),  # Clear pattern
            "diagnosis": "condition_A"
        },
        "patient_2": {
            "symptoms": ([0, 1, 0], [1, 0, 1], [0, 0, 1]),  # Different pattern
            "diagnosis": "condition_B"
        },
        "patient_3": {
            "symptoms": ([1, None, 1], [None, 1, 0], [1, 1, None]),  # Uncertain
            "diagnosis": "uncertain"
        }
    }
    
    # Store diagnostic patterns
    for patient_id, data in patients.items():
        fractal_pattern = transcender.level1_synthesis(*data["symptoms"])
        context = {
            "patient_id": patient_id,
            "symptoms": data["symptoms"],
            "diagnosis": data["diagnosis"]
        }
        evolver.formalize_fractal_axiom(fractal_pattern, context, "diagnosis")
        print(f"Stored pattern for {patient_id}: {data['diagnosis']}")
    
    # Simulate new patient diagnosis
    new_symptoms = ([1, 0, 1], [0, 1, 0], [1, 1, 0])  # Similar to patient_1
    new_pattern = transcender.level1_synthesis(*new_symptoms)
    
    # Check if we have a matching diagnosis
    existing_axiom = kb.get_axiom_by_ms("diagnosis", new_pattern["layer1"])
    if existing_axiom:
        print(f"\nNew patient symptoms: {new_symptoms}")
        print(f"Matched diagnosis: {existing_axiom['original_inputs']['diagnosis']}")
    else:
        print(f"\nNew patient symptoms: {new_symptoms}")
        print("No matching diagnosis found - novel case")
    
    print(f"Total diagnostic patterns: {kb.space_stats('diagnosis')['axiom_count']}")
    print()

def example_8_scalability_demo():
    """Example 8: Scalability demonstration"""
    print("📈 EXAMPLE 8: Scalability Demonstration")
    print("-" * 50)
    
    import time
    
    kb = KnowledgeBase()
    evolver = Evolver(kb)
    transcender = Transcender()
    
    # Test storing many patterns with more varied inputs to avoid coherence conflicts
    num_patterns = 50  # Reduced to avoid too many conflicts
    start_time = time.time()
    
    stored_count = 0
    for i in range(num_patterns):
        # Generate more varied patterns to reduce coherence conflicts
        pattern_a = [i % 2, (i + 1) % 2, (i + 2) % 2]
        pattern_b = [(i + 2) % 2, (i + 1) % 2, i % 2]
        pattern_c = [(i + 1) % 2, (i + 2) % 2, (i + 3) % 2]
        
        fractal_vector = transcender.level1_synthesis(pattern_a, pattern_b, pattern_c)
        
        # Try to store, count successes
        success = evolver.formalize_fractal_axiom(
            fractal_vector, 
            {"index": i, "batch": "scalability_test", "patterns": [pattern_a, pattern_b, pattern_c]}
        )
        if success:
            stored_count += 1
    
    storage_time = time.time() - start_time
      # Test retrieval performance on stored patterns
    start_time = time.time()
    
    successful_retrievals = 0
    stored_ms_values = []
    
    # First, collect some actual Ms values that were stored
    for space_name, space in kb.spaces.items():
        if space_name == "default":
            for ms_key in list(space["axiom_registry"].keys())[:25]:  # Test up to 25
                stored_ms_values.append(list(ms_key))
    
    # Test retrieval with actual stored Ms values
    for test_ms in stored_ms_values:
        axiom = kb.get_axiom_by_ms("default", test_ms)
        if axiom:
            successful_retrievals += 1
    
    retrieval_time = time.time() - start_time
    
    print(f"Attempted to store {num_patterns} patterns")
    print(f"Successfully stored: {stored_count} patterns in {storage_time:.3f} seconds")
    if stored_count > 0:
        print(f"Average storage time: {storage_time/stored_count*1000:.2f} ms per successful pattern")
    print(f"Tested retrieval of 25 patterns in {retrieval_time:.3f} seconds")
    print(f"Successful retrievals: {successful_retrievals}")
    if successful_retrievals > 0:
        print(f"Average retrieval time: {retrieval_time/successful_retrievals*1000:.2f} ms per query")
    print(f"Total axioms in knowledge base: {kb.space_stats('default')['axiom_count']}")
    print(f"Storage efficiency: {stored_count/num_patterns*100:.1f}%")
    print()

def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 TRINITY LIBRARY - PRACTICAL EXAMPLES")
    print("=" * 60)
    print()
    
    examples = [
        example_1_basic_inference,
        example_2_uncertainty_handling,
        example_3_fractal_synthesis,
        example_4_knowledge_management,
        example_5_pattern_detection,
        example_6_reconstruction,
        example_7_medical_diagnosis,
        example_8_scalability_demo
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print()
    
    print("✅ All examples completed!")
    print()
    print("💡 Next steps:")
    print("- Explore API_DOCUMENTATION.md for detailed method reference")
    print("- Run benchmark_trinity.py for performance analysis")
    print("- Run test_trinity_complete.py for comprehensive testing")

if __name__ == "__main__":
    run_all_examples()
