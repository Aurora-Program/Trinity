import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aurora_core.Transcender import Transcender

def ejemplo_fractal():

    from itertools import product

    def proceso_fractal(tensores, M_emergent_obj=(1,0,1)):
        transcenders = []
        M_emergents = []
        R_hipotesis_list = []
        for idx, tensor in enumerate(tensores):
            t = Transcender()
            resultado = t.deep_learning(tensor[0], tensor[1], tensor[2], M_emergent_obj)
            transcenders.append(t)
            M_emergents.append(resultado['MetaM'])
            R_hipotesis_list.append(resultado['R_hipotesis'])
            print(f"Transcender {idx+1}:")
            print(f"  Vectores: {tensor}")
            print(f"  M_emergent objetivo: {M_emergent_obj}")
            print(f"  MetaM (salida intermedia): {resultado['MetaM']}")
            print(f"  R_hipotesis: {resultado['R_hipotesis']}")
            print()
        # Transcender superior
        found = False
        MetaM_global = None
        for M_emergent_sup_obj in product([0,1], repeat=3):
            transcender_sup = Transcender()
            resultado_sup = transcender_sup.deep_learning(M_emergents[0], M_emergents[1], M_emergents[2], M_emergent_sup_obj)
            if all(r is not None for r in resultado_sup['R_hipotesis']):
                print("Transcender Superior (¡Transcendencia encontrada!):")
                print(f"  M_emergents intermedios: {M_emergents}")
                print(f"  M_emergent objetivo superior: {M_emergent_sup_obj}")
                print(f"  MetaM global: {resultado_sup['MetaM']}")
                print(f"  R_hipotesis superior: {resultado_sup['R_hipotesis']}")
                MetaM_global = resultado_sup['MetaM']
                found = True
                break
        if not found:
            print("No se encontró ninguna combinación de M_emergent objetivo superior que permita trascender esta estructura.")
        print("-"*60)
        return MetaM_global

    # Ejecutar el proceso fractal tres veces con diferentes tensores
    tensores_list = [
        [ [(1,0,1), (0,1,0), (1,1,0)], [(0,1,1), (1,0,0), (0,0,1)], [(1,1,1), (0,0,0), (1,0,0)] ],
        [ [(0,0,1), (1,1,0), (1,0,1)], [(1,1,1), (0,1,0), (0,0,1)], [(1,0,0), (1,1,1), (0,1,1)] ],
        [ [(1,1,0), (0,0,1), (1,0,1)], [(0,1,0), (1,0,1), (1,1,1)], [(0,1,1), (1,0,0), (0,0,0)] ]
    ]
    MetaMs_globales = []
    for i, tensores in enumerate(tensores_list):
        print(f"=== Proceso fractal {i+1} ===")
        MetaM_global = proceso_fractal(tensores)
        MetaMs_globales.append(MetaM_global)


    # Transcender supra-superior
    print("=== Transcender Supra-Superior ===")
    found = False
    for M_emergent_sup_obj in product([0,1], repeat=3):
        transcender_sup = Transcender()
        if None in MetaMs_globales:
            print("No se puede trascender: alguna estructura fractal previa no tiene MetaM global.")
            break
        resultado_sup = transcender_sup.deep_learning(MetaMs_globales[0], MetaMs_globales[1], MetaMs_globales[2], M_emergent_sup_obj)
        if all(r is not None for r in resultado_sup['R_hipotesis']):
            print("¡Transcendencia SUPRA encontrada!")
            print(f"  MetaMs globales: {MetaMs_globales}")
            print(f"  M_emergent objetivo SUPRA: {M_emergent_sup_obj}")
            print(f"  MetaM SUPRA: {resultado_sup['MetaM']}" )
            print(f"  R_hipotesis SUPRA: {resultado_sup['R_hipotesis']}" )
            found = True
            break
    if not found and None not in MetaMs_globales:
        print("No se encontró ninguna combinación SUPRA que permita trascender la estructura fractal completa.")

    # Análisis de patrones entre MetaMs globales
    print("\n=== Análisis de patrones entre MetaMs globales ===")
    print(f"MetaMs globales: {MetaMs_globales}")
    if None in MetaMs_globales:
        print("No se puede analizar patrones: falta algún MetaM global.")
        return
    # Ejemplo de análisis: igualdad, XOR, AND, OR
    iguales = all(m == MetaMs_globales[0] for m in MetaMs_globales)
    if iguales:
        print("Todos los MetaM globales son iguales.")
    else:
        print("MetaM globales diferentes.")
    # XOR bit a bit
    xor = [MetaMs_globales[0][i] ^ MetaMs_globales[1][i] ^ MetaMs_globales[2][i] for i in range(3)]
    print(f"XOR bit a bit de los MetaM globales: {xor}")
    # AND bit a bit
    anded = [MetaMs_globales[0][i] & MetaMs_globales[1][i] & MetaMs_globales[2][i] for i in range(3)]
    print(f"AND bit a bit de los MetaM globales: {anded}")
    # OR bit a bit
    ored = [MetaMs_globales[0][i] | MetaMs_globales[1][i] | MetaMs_globales[2][i] for i in range(3)]
    print(f"OR bit a bit de los MetaM globales: {ored}")

if __name__ == "__main__":
    ejemplo_fractal()
