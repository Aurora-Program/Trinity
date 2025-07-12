import sys
import os
import random

# Asegúrate de que la ruta sea la correcta para tu estructura de proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allcode import FractalTensor, Transcender, Evolver # Asume que el código está en allcode.py

# --- Test Mejorado ---

# 1. Crear un tensor fractal aleatorio
fractal = FractalTensor.random()
print('Tensor fractal generado:')
print(fractal)

# 2. Procesar el tensor de una manera que genere resultados más diversos
trans = Transcender()
resultados_diversos = {'nivel_3': [], 'nivel_9': [], 'nivel_27': []}

print('\nResultados Transcender (modo diverso):')
for nivel_nombre, vectores in fractal.as_dict().items():
    print(f'Procesando Nivel {nivel_nombre}:')
    # Para cada vector, lo combinamos con otros dos vectores aleatorios del mismo nivel
    # Esto creará resultados MetaM mucho más variados.
    for i in range(len(vectores)):
        A = vectores[i]
        B = random.choice(vectores)
        C = random.choice(vectores)
        res = trans.compute(A, B, C)
        resultados_diversos[nivel_nombre].append(res)
        print(f'  [{i}] A={A}, B={B}, C={C} -> MetaM={res["MetaM"]}')


# 3. Analizar la dinámica de los resultados de nivel 9 (ahora más interesantes)
nivel9_metaM = [r['MetaM'] for r in resultados_diversos['nivel_9'] if r and 'MetaM' in r and None not in r['MetaM']]
if len(nivel9_metaM) < 2:
    print("\nNo hay suficientes MetaMs válidos en el nivel 9 para analizar la dinámica.")
else:
    evolver = Evolver()
    dyn = evolver.analyze_dynamics_adaptive(nivel9_metaM)
    print('\nDinámica adaptativa sobre MetaM de nivel 9 (diverso):')
    print(dyn)

# 4. Relator adaptativo sobre los resultados de nivel 3 (ahora más interesantes)
nivel3_metaM = [r['MetaM'] for r in resultados_diversos['nivel_3'] if r and 'MetaM' in r and None not in r['MetaM']]
if len(nivel3_metaM) < 2:
    print("\nNo hay suficientes MetaMs válidos en el nivel 3 para analizar relaciones.")
else:
    if 'evolver' not in locals(): evolver = Evolver()
    rel = evolver.relate_vectors_adaptive(nivel3_metaM)
    print('\nRelator adaptativo sobre MetaM de nivel 3 (diverso):')
    print(rel)