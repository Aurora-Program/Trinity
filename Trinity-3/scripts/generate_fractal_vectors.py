"""
Script para generar automáticamente entradas fractales para VECTORES siguiendo las reglas de Docs/docvectors.txt.

Uso:
  1) Crear un CSV o TXT con líneas: palabra,gram_code,know_code,sys_code
     Ejemplo (sin cabecera):
       casa,1,1,2
       correr,2,1,3
  2) Ejecutar: python generate_fractal_vectors.py palabras.csv > vectors_block.txt
  3) Pegar el contenido de vectors_block.txt en vectors/Vectors.py

Categorías definidas en docvectors:
  - Gramática (1-7)
  - Conocimiento (1-7)
  - Sistémico (1-7)

Cada código usa tablas de subdimensiones, definidas a continuación.
"""
import csv
import sys
import json

# Definiciones de subdimensiones según docvectors.txt
SUBDIM_GRAM = {
    1: [1, 1, 2],  # Nombre: concreto, singular, femenino (ejemplo genérico)
    2: [3, 2, 1],  # Verbo: gerundio, intransitivo, acción física
    # Otras categorías (3-7) pueden agregarse aquí siguiendo docvectors.txt
}
# Similarmente SUBDIM_KNOW y SUBDIM_SYS deben definirse con valores adecuados
SUBDIM_KNOW = {1: [4,1,1], 2: [5,5,3], 3: [2,1,4], 4: [5,3,7], 5: [5,5,3], 6: [3,3,7], 7: [6,2,5]}
SUBDIM_SYS = {1: [1,4,1], 2: [4,4,4], 3: [2,3,2], 4: [4,5,1], 5: [2,3,2], 6: [1,2,1], 7: [3,4,2]}

if len(sys.argv) < 2:
    print("Uso: python generate_fractal_vectors.py palabras.csv > vectors_block.txt", file=sys.stderr)
    sys.exit(1)

with open(sys.argv[1], encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        palabra, g, k, s = row
        try:
            g, k, s = int(g), int(k), int(s)
        except ValueError:
            continue
        nivel_3 = [g, k, s]
        sub1 = SUBDIM_GRAM.get(g, [0,0,0])
        sub2 = SUBDIM_KNOW.get(k, [0,0,0])
        sub3 = SUBDIM_SYS.get(s, [0,0,0])
        nivel_9 = [sub1, sub2, sub3]
        nivel_27 = [[[0,0,0] for _ in range(3)] for _ in range(3)]
        entry = f'    "{palabra}": {json.dumps([nivel_3, nivel_9, nivel_27], ensure_ascii=False)},'
        print(entry)
