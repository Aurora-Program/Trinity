import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allcode import FractalTensor, Evolver, KnowledgeBase
import random


# 1. Generar tensores fractales completamente informados (sin None)

# Test largo: muchos tensores y vectores informados aleatorios
def tensor_full_random():
    return FractalTensor(
        nivel_3=[[random.choice([0,1]) for _ in range(3)] for _ in range(3)],
        nivel_9=[[random.choice([0,1]) for _ in range(3)] for _ in range(9)],
        nivel_27=[[random.choice([0,1]) for _ in range(3)] for _ in range(27)]
    )

random.seed(123)
N_TENSORES = 20
tensores = [tensor_full_random() for _ in range(N_TENSORES)]

# 2. Inicializar Evolver y KnowledgeBase
evolver = Evolver()
kb = KnowledgeBase()

# 3. Sintetizar y analizar cada tensor
todos_metaMs = []
print('--- SÍNTESIS Y META-MS ---')
for idx, tensor in enumerate(tensores):
    resultados = evolver.base_transcender.compute_fractal(tensor)
    metaMs = [res['MetaM'] for res in resultados['nivel_3'] if 'MetaM' in res]
    todos_metaMs.append(metaMs)
    print(f"Tensor {idx} nivel_3 MetaMs:", metaMs)

# 4. Analizar relaciones entre MetaMs y emergencia de vector superior
print('\n--- RELACIONES ENTRE META-MS ---')
for idx, metaMs in enumerate(todos_metaMs):
    rel = evolver.relate_vectors(metaMs)
    print(f"Relaciones tensor {idx}:", rel)
    if rel['emergent_vector']:
        print(f"  -> Vector superior emergente: {rel['emergent_vector']}")

# 5. Analizar dinámica adaptativa global
print('\n--- DINÁMICA GLOBAL ---')
metaMs_finales = [ms[-1] for ms in todos_metaMs if ms]
dinamica = evolver.analyze_dynamics_adaptive(metaMs_finales)
print("Dinámica adaptativa entre MetaMs finales:", dinamica)

# 6. Descubrir arquetipo global
print('\n--- ARQUETIPO GLOBAL ---')
arquetipo = evolver.compute_archetypes(*tensores, nivel='nivel_3')
print("Arquetipo global descubierto:", arquetipo)

# 7. Guardar conocimiento en la base y validar coherencia
print('\n--- BASE DE CONOCIMIENTO ---')
for idx, metaMs in enumerate(todos_metaMs):
    if len(metaMs) >= 3:
        try:
            kb.add_entry([0,1,1], [1,0,1], [1,1,0], metaMs[0], metaMs[1], [metaMs[2]], transcender_id=idx)
            print(f"Entrada {idx} añadida a la base.")
        except Exception as e:
            print(f"Error añadiendo entrada {idx}: {e}")
print("Entradas almacenadas:", kb.all_entries())

# 8. Prueba de extensión (reconstrucción)
print('\n--- EXTENSIÓN (RECONSTRUCCIÓN) ---')
from allcode import Extender
extender = Extender(knowledge_base=kb, evolver=evolver)
if metaMs_finales:
    reconstruccion = extender.extend(metaMs_finales[:3])
    print("Reconstrucción a partir de MetaMs finales:", reconstruccion)
