"""
Script para extraer palabras únicas de un texto y generar bloques de vectores fractales para añadir a vectors/Vectors.py.

Uso:
  python extract_and_generate_vectors.py archivo_texto.txt > vectors_block.txt

Cada línea de la salida tiene el formato:
  "palabra": [[g,k,s], [[..],[..],[..]], [[[0,0,0],[0,0,0],[0,0,0]], [[...]], [[...]]],

Donde [g,k,s] es el vector principal y las demás plantillas pueden ajustarse según docvectors.txt.
Por defecto se usan ceros. Debes editar después vectors_block.txt para asignar valores reales.
"""
import sys, re, json

if len(sys.argv) < 2:
    print("Uso: python extract_and_generate_vectors.py texto.txt > vectors_block.txt", file=sys.stderr)
    sys.exit(1)

# Leer y tokenizar
text = open(sys.argv[1], encoding='utf-8').read()
# Extraer palabras alfanuméricas
words = set(re.findall(r"\b[\wáéíóúñÁÉÍÓÚÑüÜ]+\b", text.lower(), flags=re.UNICODE))

# Definir plantillas
default_nivel3 = [0, 0, 0]
default_nivel9 = [[0, 0, 0] for _ in range(3)]
default_nivel27 = [[[0, 0, 0] for _ in range(3)] for _ in range(3)]

# Generar
for w in sorted(words):
    entry = [default_nivel3, default_nivel9, default_nivel27]
    print(f'    "{w}": {json.dumps(entry, ensure_ascii=False)},')
