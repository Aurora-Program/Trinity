# --- Función para explicar los vectores fractales de una palabra ---
def explicar_palabra(palabra, fractal_word):
    if not fractal_word:
        print(f"'{palabra}' no está en el diccionario.")
        return
    # Invertir los mapas para explicación
    inv_sint = {v: k for k, v in SINTACTICA.items()}
    inv_sem = {v: k for k, v in SEMANTICA.items()}
    inv_ctx = {v: k for k, v in CONTEXTO.items()}
    n3 = fractal_word.nivel_3
    print(f"Palabra: {palabra}")
    print(f"  nivel_3: {n3}")
    print(f"    0: {n3[0]} → {inv_sint.get(n3[0], 'desconocido')} (sintáctica)")
    print(f"    1: {n3[1]} → {inv_sem.get(n3[1], 'desconocido')} (semántica)")
    print(f"    2: {n3[2]} → {inv_ctx.get(n3[2], 'desconocido')} (contexto)")
    print(f"  nivel_9: {fractal_word.nivel_9}")
    print(f"  nivel_27: {fractal_word.nivel_27}")
    print()
# Diccionario fractal de palabras para Aurora
# Cada palabra tiene 3 niveles de vectores fractales:
# - nivel_3: [sintáctica, semántica, contexto]
# - nivel_9: detalles gramaticales/semánticos para cada dimensión
# - nivel_27: detalles aún más específicos

from collections import namedtuple

"""
Valores fractales: 0-7 para cada dimensión
Puedes mapear más etiquetas si lo deseas, hasta 8 por dimensión.
"""
SINTACTICA = {'nombre': 0, 'verbo': 1, 'adjetivo': 2, 'adverbio': 3, 'pronombre': 4, 'preposicion': 5, 'conjuncion': 6, 'interjeccion': 7}
SEMANTICA = {'elemento': 0, 'proceso': 1, 'sistema': 2, 'relacion': 3, 'input': 4, 'output': 5, 'estado': 6, 'evento': 7}
CONTEXTO = {'filosofia': 0, 'ciencia': 1, 'religion': 2, 'historia': 3, 'cotidiano': 4, 'arte': 5, 'tecnologia': 6, 'politica': 7}

# Ejemplo de estructura fractal para una palabra
FractalWord = namedtuple('FractalWord', ['nivel_3', 'nivel_9', 'nivel_27'])

class FractalWordDictionary:
    def __init__(self):
        self.words = {}

    def add_word(self, palabra, nivel_3, nivel_9, nivel_27):
        self.words[palabra] = FractalWord(nivel_3, nivel_9, nivel_27)

    def get_word(self, palabra):
        return self.words.get(palabra)


# --- Función para mapear un vector fractal a la palabra más cercana ---
def vector_a_palabra(vector, fw_dict, nivel='nivel_3'):
    """
    Dado un vector fractal (nivel_3, nivel_9 o nivel_27), retorna la palabra cuyo vector es más cercano (distancia euclidiana mínima).
    Por defecto compara usando nivel_3 (3 dimensiones).
    """
    from math import dist
    min_dist = float('inf')
    palabra_cercana = None
    for palabra, fractal in fw_dict.words.items():
        v = getattr(fractal, nivel)
        # Para nivel_3: vector es [x, y, z]
        # Para nivel_9 o nivel_27: vector debe ser lista de listas, se aplana
        if nivel == 'nivel_3':
            v_comp = v
        else:
            v_comp = sum(v, [])  # aplanar
        d = dist(vector, v_comp)
        if d < min_dist:
            min_dist = d
            palabra_cercana = palabra
    return palabra_cercana, min_dist



# --- Diccionario fractal sincronizado con VECTORES.py ---
from vectors.Vectors import VECTORES
fw_dict = FractalWordDictionary()

for palabra, vectores in VECTORES.items():
    # vectores: [nivel_3, nivel_9], nivel_27 opcional o puede ser None
    nivel_3 = vectores[0]
    nivel_9 = vectores[1] if len(vectores) > 1 else [[0,0,0]]*9
    # nivel_27: si existe, usa, si no, rellena con ceros
    nivel_27 = vectores[2] if len(vectores) > 2 else [[0,0,0]]*27
    fw_dict.add_word(palabra, nivel_3, nivel_9, nivel_27)
