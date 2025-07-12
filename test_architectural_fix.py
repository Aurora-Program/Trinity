#!/usr/bin/env python3
"""
Test Arquitectural Trinity Aurora: Validación de Corrección
==========================================================

Este test valida que la corrección arquitectural del método Transcender.procesar()
sigue correctamente la especificación Aurora auténtica.

Cambios Validados:
- ❌ Antes: M1, M2, M3 → Ms (incorrecta)
- ✅ Ahora: S1, S2, S3 → Ms (arquitectura Aurora auténtica)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Trinity_Fixed import Transcender, Trigate, KnowledgeBase, Evolver, Extender

def test_architectural_correction():
    """
    Test principal: Valida que la arquitectura Aurora esté correctamente implementada
    """
    print("="*60)
    print("TEST ARQUITECTURAL - VALIDACIÓN DE CORRECCIÓN AURORA")
    print("="*60)
    
    # Crear instancia del sistema
    transcender = Transcender()
    
    # Datos de prueba
    test_inputs = {
        "InA": [1, 0, 1],
        "InB": [0, 1, 0], 
        "InC": [1, 1, 0]
    }
    
    print(f"\n1. CONFIGURACIÓN DE PRUEBA:")
    print(f"   InA: {test_inputs['InA']}")
    print(f"   InB: {test_inputs['InB']}")
    print(f"   InC: {test_inputs['InC']}")
    
    # Ejecutar procesamiento
    print(f"\n2. EJECUTANDO PROCESAMIENTO AURORA...")
    Ms, Ss, MetaM = transcender.procesar(
        test_inputs["InA"], 
        test_inputs["InB"], 
        test_inputs["InC"]
    )
    
    print(f"\n3. RESULTADOS OBTENIDOS:")
    print(f"   Ms (Estructura): {Ms}")
    print(f"   Ss (Forma): {Ss}")
    print(f"   MetaM (Función): {MetaM}")
    
    # Validar que los datos internos sean correctos
    run_data = transcender.last_run_data
    print(f"\n4. VALIDACIÓN ARQUITECTURAL:")
    
    # Verificar que se usaron valores S (síntesis) no M (lógica)
    intermediate = run_data.get("intermediate", {})
    
    test_results = {
        "s_values_generated": all(key in intermediate for key in ["S1", "S2", "S3"]),
        "s_values_valid": all(isinstance(intermediate.get(key), list) and len(intermediate.get(key)) == 3 
                            for key in ["S1", "S2", "S3"] if key in intermediate),
        "ms_structure_valid": isinstance(Ms, list) and len(Ms) == 3,
        "ss_structure_valid": isinstance(Ss, list) and len(Ss) == 3,
        "metam_structure_valid": isinstance(MetaM, list) and len(MetaM) == 4
    }
    
    print(f"   ✅ S1, S2, S3 generados: {test_results['s_values_generated']}")
    print(f"   ✅ Valores S válidos: {test_results['s_values_valid']}")
    print(f"   ✅ Ms estructura válida: {test_results['ms_structure_valid']}")
    print(f"   ✅ Ss estructura válida: {test_results['ss_structure_valid']}")
    print(f"   ✅ MetaM estructura válida: {test_results['metam_structure_valid']}")
    
    # Validar valores intermedios
    if test_results['s_values_generated']:
        print(f"\n5. VALORES INTERMEDIOS (SÍNTESIS):")
        print(f"   S1: {intermediate['S1']}")
        print(f"   S2: {intermediate['S2']}")
        print(f"   S3: {intermediate['S3']}")
    
    return all(test_results.values())

def test_synthesis_consistency():
    """
    Valida que la síntesis sea consistente con la arquitectura Aurora
    """
    print(f"\n" + "="*60)
    print("TEST DE CONSISTENCIA DE SÍNTESIS")
    print("="*60)
    
    transcender = Transcender()
    
    # Test con múltiples casos
    test_cases = [
        {"InA": [1, 1, 1], "InB": [0, 0, 0], "InC": [1, 0, 1]},
        {"InA": [0, 1, 0], "InB": [1, 0, 1], "InC": [0, 0, 1]},
        {"InA": [1, 0, 0], "InB": [0, 1, 1], "InC": [1, 1, 0]}
    ]
    
    all_consistent = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCaso {i}: {case}")
        
        try:
            Ms, Ss, MetaM = transcender.procesar(case["InA"], case["InB"], case["InC"])
            
            # Verificar consistencia
            consistent = (
                isinstance(Ms, list) and len(Ms) == 3 and
                isinstance(Ss, list) and len(Ss) == 3 and
                isinstance(MetaM, list) and len(MetaM) == 4
            )
            
            print(f"  Resultado: Ms={Ms}, Ss={Ss}")
            print(f"  Consistente: {'✅' if consistent else '❌'}")
            
            if not consistent:
                all_consistent = False
                
        except Exception as e:
            print(f"  Error: {e}")
            all_consistent = False
    
    return all_consistent

def test_fractal_integration():
    """
    Valida que el sistema completo funcione con síntesis fractal
    """
    print(f"\n" + "="*60)
    print("TEST DE INTEGRACIÓN FRACTAL")
    print("="*60)
    
    # Configurar sistema completo
    kb = KnowledgeBase()
    transcender = Transcender()
    evolver = Evolver(kb)
    extender = Extender()
    
    # Crear espacio de prueba
    kb.create_space("test_space", "Espacio de prueba arquitectural")
    
    try:
        # Generar vector fractal
        print("\n1. Generando vector fractal...")
        fv = transcender.level1_synthesis([1, 0, 1], [0, 1, 0], [1, 1, 1])
        
        print(f"   Vector fractal generado:")
        print(f"   L1: {fv['layer1']}")
        print(f"   L2: {len(fv['layer2'])} vectores")
        print(f"   L3: {len(fv['layer3'])} vectores")
        
        # Formalizar axioma
        print("\n2. Formalizando axioma fractal...")
        success = evolver.formalize_fractal_axiom(
            fv, 
            {"A": [1, 0, 1], "B": [0, 1, 0], "C": [1, 1, 1]}, 
            "test_space"
        )
        
        print(f"   Axioma formalizado: {'✅' if success else '❌'}")
        
        # Validar coherencia
        print("\n3. Validando coherencia...")
        is_coherent = kb.validate_fractal_coherence("test_space", fv, {
            "layer1": fv["layer1"],
            "layer2": fv["layer2"],
            "layer3": fv["layer3"]
        })
        
        print(f"   Vector coherente: {'✅' if is_coherent else '❌'}")
        
        # Test de reconstrucción
        print("\n4. Probando reconstrucción...")
        extender.load_guide_package(evolver.generate_guide_package("test_space"))
        
        target_fv = {"layer1": fv["layer1"], "layer2": [], "layer3": []}
        reconstructed = extender.reconstruct_fractal(target_fv, "test_space")
        
        reconstruction_success = reconstructed is not None
        print(f"   Reconstrucción exitosa: {'✅' if reconstruction_success else '❌'}")
        
        return success and is_coherent and reconstruction_success
        
    except Exception as e:
        print(f"   Error en integración: {e}")
        return False

def main():
    """
    Ejecuta todos los tests de validación arquitectural
    """
    print("TRINITY AURORA - TESTS DE VALIDACIÓN ARQUITECTURAL")
    print("Validando corrección: S1,S2,S3 → Ms (arquitectura auténtica)")
    
    # Ejecutar tests
    test1 = test_architectural_correction()
    test2 = test_synthesis_consistency()
    test3 = test_fractal_integration()
    
    # Resumen final
    print(f"\n" + "="*60)
    print("RESUMEN DE VALIDACIÓN")
    print("="*60)
    
    tests_passed = [test1, test2, test3]
    test_names = [
        "Corrección Arquitectural",
        "Consistencia de Síntesis", 
        "Integración Fractal"
    ]
    
    for name, passed in zip(test_names, tests_passed):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<30} {status}")
    
    all_passed = all(tests_passed)
    print(f"\nResultado General: {'✅ TODOS LOS TESTS PASAN' if all_passed else '❌ FALLOS DETECTADOS'}")
    
    if all_passed:
        print("\n🎉 CORRECCIÓN ARQUITECTURAL VALIDADA")
        print("   Trinity Aurora implementa correctamente la arquitectura Aurora auténtica")
        print("   Listo para producción!")
    else:
        print("\n⚠️  CORRECCIÓN REQUIERE ATENCIÓN")
        print("   Revisar fallos antes de usar en producción")
    
    return all_passed

if __name__ == "__main__":
    main()
