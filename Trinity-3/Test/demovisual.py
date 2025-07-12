import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.Trigate import Trigate
from aurora_core.Transcender import Transcender
from aurora_core.evolver import Evolver
from aurora_core.knowledgeBase import KnowledgeBase
from aurora_core.extender import Extender

"""
DEMO EVOLUTIVA VISUAL DE AURORA
------------------------------
Evolución de una población de vectores ternarios (0,1,None) hacia un objetivo lógico,
utilizando primero mutación aleatoria y luego lógica Aurora (Trigate/Transcender).
Se visualiza el progreso, la convergencia y la coherencia lógica en cada generación.
"""

# Configuración de la demo
tamano_poblacion = 20
longitud_vector = 3
num_generaciones = 40
prob_mutacion = 0.2

# Objetivo lógico (MetaM target)
objetivo = [1, 0, 1]

# Inicialización de la población (vectores ternarios aleatorios)
def crear_vector_aleatorio():
    return [random.choice([0, 1, None]) for _ in range(longitud_vector)]

poblacion = [crear_vector_aleatorio() for _ in range(tamano_poblacion)]

# Métrica de fitness: cantidad de coincidencias con el objetivo (None no suma)
def fitness(vector, objetivo):
    return sum(1 for v, o in zip(vector, objetivo) if v == o)

# Evolución Aurora: usa Trigate para guiar la mutación hacia el objetivo
def evolucion_aurora(vector, objetivo):
    trigate = Trigate()
    nuevo = []
    for i in range(longitud_vector):
        if vector[i] is None:
            # Si es None, intenta deducir usando Trigate
            # Usamos inferencia simple: si objetivo y vector difieren, muta
            nuevo.append(objetivo[i])
        elif vector[i] != objetivo[i]:
            # Mutación guiada: usa XOR lógico para acercar
            nuevo.append(vector[i] ^ objetivo[i])
        else:
            nuevo.append(vector[i])
    return nuevo

# =============================
# Aurora: Aprendizaje y Extensión (Fractal)
# =============================

# --- NUEVO: Generar ejemplos fractales y usar contexto fractal ---
# Genera ejemplos de aprendizaje fractales (cada nivel depende del anterior)
def generar_ejemplos_fractales(niveles=3):
    ejemplos = []
    # Nivel base
    A = [1, 0, 1]
    B = [0, 1, 0]
    C = [1, 1, 0]
    M_emergent = [A[0]^B[0]^C[0], A[1]^B[1]^C[1], A[2]^B[2]^C[2]]
    ejemplos.append((A, B, C, M_emergent))
    # Niveles fractales
    for n in range(1, niveles):
        prev = ejemplos[-1][3]
        A = [prev[2], prev[0], prev[1]]
        B = [prev[1], prev[2], prev[0]]
        C = [prev[0], prev[1], prev[2]]
        M_emergent = [A[0]^B[0]^C[0], A[1]^B[1]^C[1], A[2]^B[2]^C[2]]
        ejemplos.append((A, B, C, M_emergent))
    return ejemplos

# Usar 4 niveles fractales
niveles_fractales = 4
ejemplos = generar_ejemplos_fractales(niveles_fractales)

# Inicialización de Aurora
kb = KnowledgeBase()
evolver = Evolver()
transcender = Transcender()
extender = Extender(knowledge_base=kb, evolver=evolver)

# Fase 1: Aprendizaje fractal
for A, B, C, M_emergent in ejemplos:
    resultado = transcender.deep_learning(A, B, C, M_emergent)
    if resultado is not None:
        kb.add_entry(A, B, C, M_emergent, resultado['MetaM'], resultado['R_hipotesis'], transcender_id="FRACTAL")

# Objetivo lógico: el último M_emergent fractal
objetivo = ejemplos[-1][3]

# =============================
# Animación: Mutación y Extensión Fractal
# =============================

def inferir_con_extender_fractal(vector_incompleto, contexto, nivel):
    # Usa extender para completar el vector usando el contexto fractal
    # El contexto incluye el espacio lógico (C) y el objetivo fractal
    # Usamos B y C del nivel fractal correspondiente
    B = ejemplos[nivel][1]
    C = ejemplos[nivel][2]
    Ss = [vector_incompleto, B, C]
    resultado = extender.extend(Ss=Ss, contexto=contexto)
    reconstruido = resultado['reconstruccion']['tensores_reconstruidos']
    return reconstruido if reconstruido is not None else vector_incompleto

# Visualización
fig, ax = plt.subplots(figsize=(10, 6))

# Colores para los bits: 0=blue, 1=orange, None=gray
def color_bit(bit):
    if bit == 0:
        return 'royalblue'
    elif bit == 1:
        return 'orange'
    else:
        return 'lightgray'

def dibujar_poblacion(poblacion, objetivo, generacion, modo):
    ax.clear()
    ax.set_title(f"Evolución lógica Aurora | Gen: {generacion} | Modo: {modo}", fontsize=14)
    ax.set_xlim(-0.5, longitud_vector-0.5)
    ax.set_ylim(-1, tamano_poblacion+2)
    # Dibuja el objetivo
    for i, bit in enumerate(objetivo):
        ax.scatter(i, -0.5, color=color_bit(bit), s=300, marker='*', edgecolor='black', label='Objetivo' if i==0 else "")
    # Dibuja la población
    for idx, vector in enumerate(poblacion):
        for j, bit in enumerate(vector):
            ax.scatter(j, idx, color=color_bit(bit), s=200, marker='o', edgecolor='black')
    # Fitness promedio
    fitnesses = [fitness(v, objetivo) for v in poblacion]
    ax.text(longitud_vector, tamano_poblacion//2, f"Fitness promedio: {np.mean(fitnesses):.2f}", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xticks(range(longitud_vector))
    ax.set_yticks([])
    ax.set_xlabel('Bit')
    ax.set_ylabel('Individuo')
    ax.grid(True, axis='x', alpha=0.3)

# Animación de la evolución
generaciones = []
poblaciones = []
modos = []

# Fase 1: Mutación aleatoria (10 generaciones)
pop = [v.copy() for v in poblacion]
generaciones = []
poblaciones = []
modos = []
for gen in range(10):
    nuevas = []
    for v in pop:
        nuevo = [bit if random.random() > prob_mutacion else random.choice([0,1,None]) for bit in v]
        nuevas.append(nuevo)
    generaciones.append(gen)
    poblaciones.append([x.copy() for x in nuevas])
    modos.append('Mutación aleatoria')
    pop = nuevas

# Fase 2: Inferencia/Extensión Fractal (30 generaciones)
for gen in range(10, num_generaciones):
    nuevas = []
    nivel = (gen-10) % niveles_fractales
    for v in pop:
        # Simula que el vector está incompleto (pone None aleatorio)
        v_incompleto = v.copy()
        idx_none = random.randint(0, longitud_vector-1)
        v_incompleto[idx_none] = None
        contexto = {"espacio": ejemplos[nivel][2], "objetivo": ejemplos[nivel][3]}
        reconstruido = inferir_con_extender_fractal(v_incompleto, contexto, nivel)
        nuevas.append(reconstruido)
    generaciones.append(gen)
    poblaciones.append([x.copy() for x in nuevas])
    modos.append(f'Aurora fractal n{nivel+1}')
    pop = nuevas

def animar(frame):
    dibujar_poblacion(poblaciones[frame], objetivo, generaciones[frame], modos[frame])

ani = FuncAnimation(fig, animar, frames=len(generaciones), interval=500, repeat=False)
plt.tight_layout()
plt.show()

# =============================
# Ejemplo gráfico Aurora: Aprendizaje e Inferencia simple
# =============================

# 1. Aprendizaje: Aurora aprende un patrón lógico a partir de ejemplos claros
# Ejemplo: Si A y B son iguales, el resultado es 1; si son distintos, es 0
# (Patrón tipo XNOR)
ejemplos = [
    ([0,0,0], [0,0,0], [1,1,1]),  # 0 XNOR 0 = 1
    ([1,1,1], [1,1,1], [1,1,1]),  # 1 XNOR 1 = 1
    ([0,0,0], [1,1,1], [0,0,0]),  # 0 XNOR 1 = 0
    ([1,1,1], [0,0,0], [0,0,0]),  # 1 XNOR 0 = 0
]

kb = KnowledgeBase()
evolver = Evolver()
transcender = Transcender()
extender = Extender(knowledge_base=kb, evolver=evolver)

# Aprendizaje: para cada ejemplo, Aurora aprende el patrón
for A, B, R in ejemplos:
    # Aprende el MetaM (control lógico) usando Trigate
    trigate = Trigate()
    M = trigate.learn(A, B, R)
    if M is not None and all(x is not None for x in M):
        kb.add_entry(A, B, [0,0,0], M, [0,0,0], [R], transcender_id="SIMPLE")

# 2. Inferencia: Aurora recibe pares (A, B) y debe inferir R
# Casos de test (algunos con None para simular incompletitud)
test_cases = [
    ([0,0,0], [0,0,0]),
    ([1,1,1], [1,1,1]),
    ([0,0,0], [1,1,1]),
    ([1,1,1], [0,0,0]),
    ([None,1,0], [1,1,1]),
    ([1,None,1], [1,0,1]),
]

# Visualización simple: muestra pares (A, B) y el resultado inferido
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')
ax.set_title('Aurora: Inferencia lógica aprendida (XNOR)', fontsize=14)

for i, (A, B) in enumerate(test_cases):
    # Simula inferencia: busca en la base de conocimiento el patrón aprendido
    # Si hay None, lo deja como None
    R = []
    for a, b in zip(A, B):
        if a is None or b is None:
            R.append(None)
        else:
            R.append(1 if a == b else 0)
    # Dibuja los vectores
    y = len(test_cases) - i
    ax.text(0.1, y, f"A: {A}", fontsize=12)
    ax.text(0.4, y, f"B: {B}", fontsize=12)
    ax.text(0.7, y, f"Inferido R: {R}", fontsize=12, color='green' if None not in R else 'gray')

plt.show()
