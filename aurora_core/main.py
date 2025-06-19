from fractalVector import FractalProcessor
from knowledge import KnowledgeBase
from evolver import Evolver
from extender import Extender       


if __name__ == "__main__":
    # Configurar componentes
    fp = FractalProcessor()
    kb = KnowledgeBase()
    
    # Crear espacio lógico para conceptos fractales
    kb.create_space("fractal_concepts", "Dominio para conocimiento fractal")
    
    print("="*20 + " SÍNTESIS NIVEL 1: CREANDO VECTORES BASE " + "="*20)
    # Crear vectores base (documentación: 4.2)
    fv_base1 = fp.level1_synthesis([1,0,1], [0,1,0], [1,1,1])
    fv_base2 = fp.level1_synthesis([0,1,0], [1,0,1], [0,0,1])
    fv_base3 = fp.level1_synthesis([1,1,0], [0,0,1], [1,0,0])
    
    print("\nVector Fractal 1:")
    print(fv_base1)
    
    # Almacenar en base de conocimiento
    # Guardar Ms, Ss y MetaM reales del vector fractal base
    Ms = fv_base1.layer1
    Ss = fv_base1.layer2  # O puedes definir una capa específica como forma
    MetaM = fv_base1.layer3  # O puedes definir una estructura más abstracta

    # Formalizar el axioma con los datos correctos
    # Si prefieres usar solo la capa 1 como Ms y las otras como Ss y MetaM, ajusta aquí
    # Aquí se usa layer1 como Ms, layer2 como Ss, layer3 como MetaM

    evolver = Evolver(kb)
    evolver.formalize_axiom({
        "inputs": {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]},
        "outputs": {
            "Ms": Ms,
            "Ss": Ss,
            "MetaM": MetaM
        }
    }, "fractal_concepts")
    
    print("\n" + "="*20 + " SÍNTESIS NIVEL 2: COMBINANDO VECTORES " + "="*20)
    # Crear meta-estructura (documentación: 4.3)
    meta_struct = fp.level2_synthesis(fv_base1, fv_base2, fv_base3)
    print("\nMeta-Estructura resultante:")
    print(f"L1: {meta_struct['layer1']}")
    print(f"L2: {meta_struct['layer2']}")
    print(f"L3: {meta_struct['layer3'][:2]}...")  # Mostrar solo muestra por brevedad
    
    print("\n" + "="*20 + " SÍNTESIS NIVEL 3: SALTO RECURSIVO " + "="*20)
    # Crear meta-estructuras adicionales
    meta_struct2 = fp.level2_synthesis(fv_base2, fv_base3, fv_base1)
    meta_struct3 = fp.level2_synthesis(fv_base3, fv_base1, fv_base2)
    
    # Crear nuevo vector fractal de alto nivel (documentación: 4.4)
    fv_high_level = fp.level3_synthesis(meta_struct, meta_struct2, meta_struct3)
    print("\nVector Fractal de Alto Nivel:")
    print(fv_high_level)
    
    print("\n" + "="*20 + " ANÁLISIS FRACTAL " + "="*20)
    # Comparar vectores (documentación: 4.5)
    print("\nComparando vectores base:")
    similarity = fp.analyze_fractal(fv_base1, fv_base2)
    
    print("\nComparando vector base con vector de alto nivel:")
    fp.analyze_fractal(fv_base1, fv_high_level)
    
    print("\n" + "="*20 + " RECONSTRUCCIÓN DESDE MEMORIA " + "="*20)
    # Reconstrucción usando el Extender
    extender = Extender()
    extender.load_guide_package(evolver.generate_guide_package("fractal_concepts"))
    
    target_ms = fv_base1.layer1
    reconstructed = extender.reconstruct(target_ms)
    print(f"\nReconstrucción para Ms={target_ms}:")
    print(f"Entradas originales: {reconstructed}")