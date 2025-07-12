#!/usr/bin/env python3
"""
AURORA ADVANCED DEMO
===================
Demonstrates complete Aurora architecture integration with Trinity library:
- Authentic fractal structure synthesis
- Ternary truth table operations
- Semantic relationship mapping (Relator)
- Conversational flow dynamics
- Enhanced reconstruction with MetaM utilization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Trinity import (
    Trigate, Transcender, KnowledgeBase, Evolver, 
    TernaryTruthTable, Relator, Dynamics, EnhancedExtender,
    LogicalCoherenceError, FractalStructureError
)

def demo_ternary_truth_table():
    """Demuestra operaciones de tabla de verdad ternaria completa"""
    print("🔢 DEMO: Tabla de Verdad Ternaria Aurora")
    print("=" * 50)
    
    tt = TernaryTruthTable()
    
    # Casos de prueba con estados: 0, 1, None
    test_cases = [
        (0, 0), (0, 1), (0, None),
        (1, 0), (1, 1), (1, None),
        (None, 0), (None, 1), (None, None)
    ]
    
    print("a\tb\tAND\tOR\tIMPL\tEQUIV")
    print("-" * 40)
    
    for a, b in test_cases:
        and_result = tt.ternary_and(a, b)
        or_result = tt.ternary_or(a, b)
        impl_result = tt.ternary_implication(a, b)
        equiv_result = tt.ternary_equivalence(a, b)
        
        print(f"{a}\t{b}\t{and_result}\t{or_result}\t{impl_result}\t{equiv_result}")
    
    # Operación NOT
    print(f"\nNOT operations:")
    for val in [0, 1, None]:
        not_result = tt.ternary_not(val)
        print(f"NOT {val} = {not_result}")
    
    print()

def demo_semantic_mapping():
    """Demuestra mapeo semántico con componente Relator"""
    print("🗺️ DEMO: Mapeo Semántico (Relator)")
    print("=" * 50)
    
    # Crear conceptos para análisis semántico
    concepts = [
        {"Ms": [1, 0, 1], "MetaM": [[0,1,1], [1,0,1], [0,0,1], [1,1,0]]},
        {"Ms": [1, 0, 0], "MetaM": [[0,1,1], [1,0,0], [0,0,1], [1,1,1]]},
        {"Ms": [0, 1, 1], "MetaM": [[1,0,1], [0,1,1], [1,1,0], [0,0,1]]},
        {"Ms": [1, 1, 1], "MetaM": [[1,1,1], [1,1,1], [1,1,1], [1,1,1]]},
    ]
    
    relator = Relator()
    space_name = "semantic_test"
    
    # Calcular distancias semánticas
    print("Distancias semánticas entre conceptos:")
    for i, concept_a in enumerate(concepts):
        for j, concept_b in enumerate(concepts[i+1:], i+1):
            distance = relator.compute_semantic_distance(concept_a, concept_b, space_name)
            print(f"Concepto {i} → Concepto {j}: distancia = {distance:.3f}")
    
    # Encontrar vecinos semánticos
    print(f"\nVecinos semánticos del concepto 0:")
    neighbors = relator.find_semantic_neighbors(concepts[0], space_name, max_distance=0.5)
    for neighbor_ms, distance in neighbors:
        print(f"  Vecino {neighbor_ms}: distancia = {distance:.3f}")
    
    print()

def demo_conversational_dynamics():
    """Demuestra flujo conversacional con componente Dynamics"""
    print("💬 DEMO: Dinámicas Conversacionales")
    print("=" * 50)
    
    dynamics = Dynamics()
    
    # Simular secuencia de interacciones
    interactions = [
        {
            "input": {"Ms": [1, 0, 1]},
            "output": {"Ms": [0, 1, 0], "MetaM": [[0,1,1], [1,0,1]]},
            "context": {"user_intent": "query", "domain": "physics"}
        },
        {
            "input": {"Ms": [0, 1, 0]},
            "output": {"Ms": [1, 1, 0], "MetaM": [[1,1,0], [0,1,1]]},
            "context": {"user_intent": "exploration", "domain": "physics"}
        },
        {
            "input": {"Ms": [1, 1, 0]},
            "output": {"Ms": [0, 0, 1], "MetaM": [[0,0,1], [1,0,0]]},
            "context": {"user_intent": "clarification", "domain": "physics"}
        }
    ]
    
    # Registrar interacciones
    for i, interaction in enumerate(interactions):
        interaction_id = dynamics.register_interaction(
            interaction["input"], 
            interaction["output"], 
            interaction["context"]
        )
        print(f"Interacción {i+1} registrada con ID {interaction_id}")
    
    # Predecir siguiente estado
    current_state = {"Ms": [0, 0, 1]}
    prediction = dynamics.predict_next_state(current_state)
    
    if prediction:
        print(f"\nPredicción para estado {current_state['Ms']}:")
        print(f"  Siguiente estado: {prediction['Ms']}")
        print(f"  Confianza: {prediction['confidence']}")
    
    # Mostrar flujo conversacional
    flow = dynamics.get_conversation_flow(window_size=3)
    print(f"\nFlujo conversacional reciente ({len(flow)} interacciones):")
    for i, interaction in enumerate(flow):
        print(f"  {i+1}. {interaction['input']['Ms']} → {interaction['output']['Ms']}")
    
    print()

def demo_enhanced_reconstruction():
    """Demuestra reconstrucción mejorada con MetaM"""
    print("🔄 DEMO: Reconstrucción Mejorada con MetaM")
    print("=" * 50)
    
    # Configurar sistema
    kb = KnowledgeBase()
    kb.create_space("reconstruction_test", "Espacio para pruebas de reconstrucción")
    
    evolver = Evolver(kb)
    transcender = Transcender()
    enhanced_extender = EnhancedExtender()
    
    # Crear y almacenar algunos axiomas
    test_cases = [
        {"A": [1, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]},
        {"A": [0, 1, 1], "B": [1, 0, 1], "C": [0, 0, 1]},
        {"A": [1, 1, 0], "B": [0, 0, 1], "C": [1, 0, 1]}
    ]
    
    stored_axioms = []
    for i, case in enumerate(test_cases):
        Ms, Ss, MetaM = transcender.procesar(case["A"], case["B"], case["C"])
        
        # Usar formalización con dinámicas
        evolver.formalize_with_dynamics(
            transcender.last_run_data,
            "reconstruction_test",
            {"case_id": i, "test_type": "reconstruction_demo"}
        )
        
        stored_axioms.append({"Ms": Ms, "MetaM": MetaM, "inputs": case})
        print(f"Caso {i+1}: Ms={Ms} almacenado")
    
    # Generar paquete de guías mejorado
    enhanced_guide = evolver.generate_enhanced_guide_package("reconstruction_test")
    enhanced_extender.load_guide_package(enhanced_guide)
    
    # Probar reconstrucción directa
    print(f"\n--- Reconstrucción Directa ---")
    target_ms = stored_axioms[0]["Ms"]
    reconstruction = enhanced_extender.reconstruct_with_metam(target_ms, "reconstruction_test")
    
    if reconstruction:
        print(f"Reconstrucción exitosa:")
        print(f"  Tipo: {reconstruction['reconstruction_type']}")
        print(f"  Confianza: {reconstruction['confidence']:.3f}")
        print(f"  Inputs: {reconstruction['inputs']}")
    
    # Probar reconstrucción por aproximación
    print(f"\n--- Reconstrucción por Aproximación ---")
    approximate_ms = [1, 0, 0]  # Similar a stored_axioms[0] pero no exacto
    reconstruction = enhanced_extender.reconstruct_with_metam(approximate_ms, "reconstruction_test")
    
    if reconstruction:
        print(f"Reconstrucción por aproximación:")
        print(f"  Tipo: {reconstruction['reconstruction_type']}")
        print(f"  Confianza: {reconstruction['confidence']:.3f}")
        print(f"  Distancia semántica: {reconstruction.get('semantic_distance', 'N/A')}")
    
    print()

def demo_complete_aurora_pipeline():
    """Demuestra pipeline completo Aurora con manejo de errores"""
    print("🌟 DEMO: Pipeline Completo Aurora")
    print("=" * 50)
    
    try:
        # Configurar sistema completo
        kb = KnowledgeBase()
        kb.create_space("aurora_complete", "Demostración completa Aurora")
        
        evolver = Evolver(kb)
        transcender = Transcender()
        
        # Crear vector fractal auténtico
        print("1. Creando vector fractal auténtico...")
        fv = transcender.level1_synthesis([1, 0, 1], [0, 1, 0], [1, 1, 1])
        
        print(f"   Layer 1 (3 trits): {fv['layer1']}")
        print(f"   Layer 2 (3 elementos): {len(fv['layer2'])} elementos")
        print(f"   Layer 3 (9 Transcenders): {len(fv['layer3'])} elementos")
        
        # Validar estructura fractal
        print("2. Validando coherencia fractal...")
        is_coherent = kb.validate_fractal_coherence("aurora_complete", fv, fv)
        print(f"   Coherencia: {'✅ Válida' if is_coherent else '❌ Inválida'}")
        
        # Análisis semántico
        print("3. Analizando relaciones semánticas...")
        axiom_list = [
            {"Ms": fv["layer1"], "MetaM": fv["layer3"]},
            {"Ms": [0, 1, 0], "MetaM": [[1,0,1], [0,1,0]] * 4},  # Axioma similar
            {"Ms": [1, 1, 1], "MetaM": [[1,1,1], [0,0,0]] * 4}   # Axioma diferente
        ]
        
        relationships = evolver.analyze_semantic_relationships(axiom_list, "aurora_complete")
        print(f"   Relaciones encontradas: {len(relationships)}")
        
        # Predicción de dinámicas
        print("4. Predicción de dinámicas...")
        current_state = {"Ms": fv["layer1"]}
        prediction = evolver.predict_interaction_outcome(current_state, {"domain": "demo"})
        
        if prediction:
            print(f"   Predicción: {prediction['Ms']} (confianza: {prediction['confidence']})")
        else:
            print("   No hay datos suficientes para predicción")
        
        print("✅ Pipeline Aurora completado exitosamente")
        
    except LogicalCoherenceError as e:
        print(f"❌ Error de coherencia lógica: {e}")
    except FractalStructureError as e:
        print(f"❌ Error de estructura fractal: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
    
    print()

def main():
    """Función principal que ejecuta todas las demostraciones"""
    print("🚀 SISTEMA AURORA - DEMOSTRACIÓN AVANZADA")
    print("=" * 60)
    print("Integración completa de arquitectura Aurora con Trinity")
    print("=" * 60)
    print()
    
    # Ejecutar todas las demostraciones
    demo_ternary_truth_table()
    demo_semantic_mapping()
    demo_conversational_dynamics()
    demo_enhanced_reconstruction()
    demo_complete_aurora_pipeline()
    
    print("🎉 DEMOSTRACIÓN COMPLETA FINALIZADA")
    print("=" * 60)
    print("Trinity v2.1 - Aurora Architecture Integration Complete")
    print("✅ Authentic fractal synthesis")
    print("✅ Ternary truth table operations")
    print("✅ Semantic relationship mapping")
    print("✅ Conversational flow dynamics")
    print("✅ Enhanced MetaM reconstruction")
    print("✅ Strict coherence validation")
    print("=" * 60)

if __name__ == "__main__":
    main()
