import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from aurora_core.Transcender import Transcender
from aurora_core.evolver import Evolver
from itertools import product

def ejemplo_dinamica_dialogo():
    # Definir 3 tensores fractales (pregunta, respuesta, análisis)
    tensores = [
        [ (1,0,1), (0,1,0), (1,1,0) ],   # Pregunta
        [ (0,1,1), (1,0,0), (0,0,1) ],   # Respuesta
        [ (1,1,1), (0,0,0), (1,0,0) ]    # Análisis
    ]

    # Procesar cada tensor con un Transcender
    metaMs = []
    for idx, tensor in enumerate(tensores):
        t = Transcender()
        # Usamos un M_emergent objetivo arbitrario para cada uno
        M_emergent_obj = (1,0,1) if idx == 0 else (0,1,1) if idx == 1 else (1,1,0)
        resultado = t.deep_learning(tensor[0], tensor[1], tensor[2], M_emergent_obj)
        metaMs.append(resultado['MetaM'])
        print(f"Transcender {idx+1} ({'Pregunta' if idx==0 else 'Respuesta' if idx==1 else 'Análisis'}):")
        print(f"  Vectores: {tensor}")
        print(f"  M_emergent objetivo: {M_emergent_obj}")
        print(f"  MetaM: {resultado['MetaM']}")
        print(f"  R_hipotesis: {resultado['R_hipotesis']}")
        print()

    # Analizar la dinámica: cómo cambia MetaM de 1→2→3
    evolver = Evolver()
    print("=== Dinámica entre MetaMs ===")
    print(f"MetaMs: {metaMs}")
    dinamica = evolver.analyze_metaMs(metaMs)
    print(f"Patrón dinámico detectado: {dinamica}")

    # Formalizar axioma/arquetipo
    axiom = evolver.formalize_axiom(dinamica, metaMs)
    print(f"Axioma/arquetipo aprendido: {axiom}")


    # Usar Evolver para resolver ambigüedad y evaluar axiomas
    print("\n=== Resolución de ambigüedad y evaluación de axiomas ===")
    metaM_target = metaMs[2] if len(metaMs) > 2 else None
    resolucion = evolver.resolve_ambiguity(metaMs, dinamica, metaM_target=metaM_target)
    print(f"Resultado de resolución: {resolucion}")

if __name__ == "__main__":
    ejemplo_dinamica_dialogo()

