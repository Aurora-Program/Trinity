from knowledge import KnowledgeBase
from transcender import Transcender
from evolver import Evolver
from extender import Extender


# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL: DEMO COMPLETA DEL SISTEMA
# ==============================================================================
if __name__ == "__main__":
    # Configurar componentes
    kb = KnowledgeBase()
    trans = Transcender()
    evolver = Evolver(kb)
    extender = Extender()
    
    # Crear espacio lógico para física cuántica
    kb.create_space("quantum_physics", "Dominio para física cuántica fractal")
    
    print("="*50)
    print("DEMOSTRACIÓN DEL SISTEMA AURORA - PROCESAMIENTO FRACTAL")
    print("="*50)
    
    # ========== FASE 1: CREACIÓN DE VECTORES FRACTALES BASE ==========
    print("\n" + "="*20 + " CREANDO VECTORES FRACTALES BASE " + "="*20)
    
    # Crear primer vector fractal
    fv1 = trans.level1_synthesis([1,0,1], [0,1,0], [1,1,1])
    print("\nVector Fractal 1 (Creado):")
    print(f"L1: {fv1['layer1']}")
    print(f"L2: {fv1['layer2'][:1]}...")  # Mostrar solo muestra
    print(f"L3: {fv1['layer3'][:1]}...")
    
    # Crear segundo vector fractal
    fv2 = trans.level1_synthesis([0,1,0], [1,0,1], [0,0,1])
    
    # Almacenar en knowledge base
    evolver.formalize_fractal_axiom(fv1, 
                                   {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]}, 
                                   "quantum_physics")
    
    # ========== FASE 2: SÍNTESIS DE NIVEL SUPERIOR ==========
    print("\n" + "="*20 + " SÍNTESIS DE NIVEL 2 " + "="*20)
    meta_struct = trans.level2_synthesis(fv1, fv1, fv2)  # Combinar 3 vectores
    print("\nMeta-Estructura resultante:")
    print(f"L1: {meta_struct['layer1']}")
    print(f"L2: {meta_struct['layer2'][:1]}...")
    print(f"L3: {meta_struct['layer3'][:1]}...")
    
    # ========== FASE 3: MANEJO DE AMBIGÜEDAD ==========
    print("\n" + "="*20 + " MANEJO DE AMBIGÜEDAD FRACTAL " + "="*20)
    ambiguous_vector = {
        "layer1": [1, 0, None],
        "layer2": [[1,0,1], [0,None,1], [1,1,0]],
        "layer3": [[1,0,0]]*9
    }
    evolver.handle_fractal_null(ambiguous_vector)
    
    # ========== FASE 4: RECONSTRUCCIÓN FRACTAL ==========
    print("\n" + "="*20 + " RECONSTRUCCIÓN FRACTAL " + "="*20)
    
    # Cargar guías para el espacio
    extender.load_guide_package(evolver.generate_guide_package("quantum_physics"))
    
    # Crear vector objetivo (solo con capa abstracta)
    target_fv = {"layer1": fv1["layer1"], "layer2": [], "layer3": []}
    
    # Reconstruir vector completo
    reconstructed_fv = extender.reconstruct_fractal(target_fv, "quantum_physics")
    print("\nVector Fractal Reconstruido:")
    print(f"L1: {reconstructed_fv['layer1']}")
    print(f"L2: {reconstructed_fv['layer2'][:1]}...")
    print(f"L3: {reconstructed_fv['layer3'][:1]}...")
    
    # ========== FASE 5: ANÁLISIS Y PATRONES ==========
    print("\n" + "="*20 + " DETECCIÓN DE PATRONES " + "="*20)
    archetype = evolver.formalize_fractal_archetype(fv1, "quantum_physics")
    
    # ========== FASE 6: VALIDACIÓN DE COHERENCIA ==========
    print("\n" + "="*20 + " VALIDACIÓN DE COHERENCIA " + "="*20)
    is_valid = kb.validate_fractal_coherence("quantum_physics", fv1, {
        "layer1": fv1["layer1"],
        "layer2": fv1["layer2"],
        "layer3": fv1["layer3"]
    })
    print(f"Vector fractal es coherente: {is_valid}")
    
    print("\n" + "="*50)
    print("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    print("="*50)