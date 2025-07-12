import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aurora_core.Transcender import Transcender
from aurora_core.knowledgeBase import KnowledgeBase
import itertools

def all_3bit_vectors():
    return list(itertools.product([0,1], repeat=3))

def main():
    kb = KnowledgeBase()
    transcender = Transcender()
    total = 0
    resueltos = 0
    ejemplos = []
    for A in all_3bit_vectors():
        for B in all_3bit_vectors():
            for C in all_3bit_vectors():
                for M_emergent in all_3bit_vectors():
                    total += 1
                    resultado = transcender.deep_learning(A, B, C, M_emergent)
                    R_hipotesis = resultado['R_hipotesis']
                    MetaM = resultado['MetaM']
                    # Consideramos resuelto si no hay None en R_hipotesis
                    if all(r is not None for r in R_hipotesis):
                        resueltos += 1
                        kb.add_entry(A, B, C, M_emergent, MetaM, R_hipotesis, transcender_id='T1')
                        if len(ejemplos) < 5:
                            ejemplos.append({
                                'A': A, 'B': B, 'C': C, 'M_emergent': M_emergent,
                                'MetaM': MetaM, 'R_hipotesis': R_hipotesis
                            })

    print(f"Total de combinaciones: {total}")
    print(f"Casos resueltos (R_hipotesis sin None): {resueltos}")
    print("Ejemplos de casos resueltos:")
    for ej in ejemplos:
        print(ej)

    print(f"Entradas almacenadas en KnowledgeBase: {len(kb.all_entries())}")

if __name__ == "__main__":
    main()
