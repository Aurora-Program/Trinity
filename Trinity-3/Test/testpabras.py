def a_binario3(n):
    """Convierte un entero (0-7) en un vector binario de 3 bits."""
    if isinstance(n, int):
        return [int(x) for x in f"{n:03b}"]
    return n

def lista_a_binario3(lst):
    """Convierte una lista de enteros a lista de vectores binarios de 3 bits."""
    return [a_binario3(x) if isinstance(x, int) else lista_a_binario3(x) if isinstance(x, list) and x and isinstance(x[0], int) == False else x for x in lst]




# Ajuste de path para imports relativos al proyecto
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vectors.Vectors import VECTORES
from allcode import KnowledgeBase


import random

# Importar el diccionario fractal y la función de mapeo vector->palabra
from palabras import fw_dict, vector_a_palabra, explicar_palabra

def generar_dialogo_y_aprendizaje(vectores):
    kb = KnowledgeBase()
    dialogo = []
    palabras = list(vectores.keys())
    random.shuffle(palabras)
    arquetipos = ['Buscador', 'Escéptico', 'Narrador', 'Evolucionador', 'Aprendiz']
    relatores = ['Sabio', 'Crítico', 'Narrador', 'Historiador', 'Futurista']
    dinamicas = ['Exploración', 'Debate', 'Profundización', 'Evolución', 'Proyección']

    # 1. Fase de aprendizaje: Aurora "escucha" preguntas y respuestas
    for i in range(5):
        palabra = palabras[i]
        data = vectores[palabra]
        MetaM = [a_binario3(x) for x in data[0]]
        Ms = [ [a_binario3(bit) for bit in m] for m in data[1] ]
        if len(Ms) >= 3:
            A, B, C = Ms[0], Ms[1], Ms[2]
        else:
            A = B = C = [[0,0,0]]*3
        try:
            kb.add_entry(A, B, C, MetaM, MetaM, [1], transcender_id=palabra, Ms=Ms)
            coherencia = kb.check_coherence(Ms, MetaM)
        except Exception as e:
            coherencia = str(e)
        pregunta = f"¿Qué significa '{palabra}' en el contexto de la ética y la existencia?"
        respuesta = f"'{palabra}' representa un concepto fundamental que se puede analizar desde los vectores binarios: MetaM={MetaM}, Ms={Ms}."
        dialogo.append(f"[{arquetipos[i]} - {relatores[i]} - {dinamicas[i]}]")
        dialogo.append(f"Pregunta: {pregunta}")
        dialogo.append(f"Aurora responde: {respuesta}")
        dialogo.append(f"Coherencia Ms/MetaM: {coherencia}")
        dialogo.append("")

    # 2. Fase de evolución: preguntas encadenadas y contra-preguntas
    for i in range(5, 8):
        palabra = palabras[i]
        data = vectores[palabra]
        MetaM = [a_binario3(x) for x in data[0]]
        Ms = [ [a_binario3(bit) for bit in m] for m in data[1] ]
        pregunta = f"¿Cómo influye '{palabra}' en la dinámica social?"
        respuesta = f"'{palabra}' afecta la dinámica según sus vectores binarios: MetaM={MetaM}, Ms={Ms}."
        contrapregunta = f"¿Y si '{palabra}' cambiara su significado, cómo afectaría a los arquetipos?"
        dialogo.append(f"[Evolución - Narrador - Dinámica]")
        dialogo.append(f"Pregunta: {pregunta}")
        dialogo.append(f"Aurora responde: {respuesta}")
        dialogo.append(f"Contra-pregunta: {contrapregunta}")
        dialogo.append("")

    # 3. Fase de prueba: Aurora responde a frases nuevas
    pruebas = [
        "¿Qué es la libertad?",
        "¿Cómo se relaciona el amor y la ley?",
        "¿Qué papel tiene la familia en la sociedad?"
    ]
    for frase in pruebas:
        palabras_frase = [p for p in palabras if p in frase]
        if palabras_frase:
            p = palabras_frase[0]
            data = vectores[p]
            MetaM = [a_binario3(x) for x in data[0]]
            Ms = [ [a_binario3(bit) for bit in m] for m in data[1] ]
            respuesta = f"Aurora analiza '{p}' con MetaM={MetaM} y Ms={Ms}."
        else:
            respuesta = "Aurora no encuentra un vector asociado, pero intenta razonar filosóficamente."
        dialogo.append(f"[Usuario - Prueba]")
        dialogo.append(f"Frase: {frase}")
        dialogo.append(f"Aurora responde: {respuesta}")
        dialogo.append("")

    return '\n'.join(dialogo)


def ejemplo_sintesis_y_vector_a_palabra():
    # Ejemplo: Aurora "sintetiza" un nuevo vector (nivel_3)
    vector_sintetizado = [3, 1, 5]
    palabra_resultado, distancia = vector_a_palabra(vector_sintetizado, fw_dict, nivel='nivel_3')
    resultado = f"\nAurora sintetiza el vector {vector_sintetizado} → palabra más cercana: '{palabra_resultado}' (distancia {distancia:.2f})\n"
    # Explicación de la palabra encontrada
    fractal = fw_dict.get_word(palabra_resultado)
    # Usar la función de explicación si está disponible
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        explicar_palabra(palabra_resultado, fractal)
    resultado += buf.getvalue()
    return resultado


# --- Nueva función: dinámica y extensión de un vector ---
def ejemplo_dinamica_extension():
    # Selecciona una palabra base y su vector
    palabra_base = 'amor'
    fractal = fw_dict.get_word(palabra_base)
    vector_base = list(fractal.nivel_3)
    # Dinámica: suma 1 a cada componente módulo 8 (simula evolución conceptual)
    vector_dinamico = [(x + 1) % 8 for x in vector_base]
    # Extensión: permuta los valores (simula cambio de perspectiva)
    vector_extendido = [vector_base[2], vector_base[0], vector_base[1]]

    # Buscar palabras más cercanas y transformar los nuevos vectores en palabras
    palabra_dinamica, dist_din = vector_a_palabra(vector_dinamico, fw_dict, nivel='nivel_3')
    fractal_dinamica = fw_dict.get_word(palabra_dinamica)
    # Aplicar dinámica de nuevo sobre el vector_dinamico
    vector_dinamico2 = [(x + 1) % 8 for x in vector_dinamico]
    palabra_dinamica2, dist_din2 = vector_a_palabra(vector_dinamico2, fw_dict, nivel='nivel_3')
    fractal_dinamica2 = fw_dict.get_word(palabra_dinamica2)

    palabra_extendida, dist_ext = vector_a_palabra(vector_extendido, fw_dict, nivel='nivel_3')
    fractal_extendida = fw_dict.get_word(palabra_extendida)
    # Aplicar extensión de nuevo sobre el vector_extendido
    vector_extendido2 = [vector_extendido[2], vector_extendido[0], vector_extendido[1]]
    palabra_extendida2, dist_ext2 = vector_a_palabra(vector_extendido2, fw_dict, nivel='nivel_3')
    fractal_extendida2 = fw_dict.get_word(palabra_extendida2)

    # Respuestas simples: frase original y resultado de la dinámica/extensión
    respuesta = []
    respuesta.append(f"Frase original: '{palabra_base}' → vector: {vector_base}")
    respuesta.append(f"Dinámica (+1 mod 8): vector {vector_dinamico} → palabra: '{palabra_dinamica}'")
    respuesta.append(f"Dinámica aplicada de nuevo: vector {vector_dinamico2} → palabra: '{palabra_dinamica2}'")
    respuesta.append(f"Extensión (permuta): vector {vector_extendido} → palabra: '{palabra_extendida}'")
    respuesta.append(f"Extensión aplicada de nuevo: vector {vector_extendido2} → palabra: '{palabra_extendida2}'")
    return '\n'.join(respuesta)

# --- Nueva función: aprendizaje y respuesta basada en frases ---
def ejemplo_aprendizaje_y_respuesta():
    ejemplos = [
        ("¿Qué es una casa?", "Una casa es donde viven las personas"),
        ("¿Qué es el amor?", "El amor es un sentimiento profundo"),
        ("¿Qué es la ley?", "La ley es un conjunto de normas"),
        ("¿Qué es la familia?", "La familia es el núcleo de la sociedad"),
        ("¿Qué es la libertad?", "La libertad es la capacidad de elegir"),
        ("¿Qué es el fuego?", "El fuego es energía en combustión"),
        ("¿Qué es el agua?", "El agua es un líquido vital"),
        ("¿Qué es el sol?", "El sol es una estrella que da luz"),
        ("¿Qué es la montaña?", "La montaña es una elevación natural del terreno"),
        ("¿Qué es el perro?", "El perro es un animal doméstico y fiel"),
        ("¿Qué es el libro?", "El libro es un conjunto de páginas con conocimiento"),
        ("¿Qué es la música?", "La música es el arte de combinar sonidos"),
        ("¿Qué es la paz?", "La paz es la ausencia de conflicto"),
        ("¿Qué es la muerte?", "La muerte es el fin de la vida"),
        ("¿Qué es la vida?", "La vida es el estado de los seres vivos"),
        ("¿Qué es el arte?", "El arte es la expresión creativa humana"),
        ("¿Qué es la ciencia?", "La ciencia es el estudio sistemático de la naturaleza"),
        ("¿Qué es la política?", "La política es la gestión del poder en la sociedad"),
        ("¿Qué es la historia?", "La historia es el relato de los hechos pasados"),
    ]
    # Simular aprendizaje: extraer palabras clave y asociar vectores
    aprendizaje = []
    for pregunta, respuesta in ejemplos:
        # Tomar la palabra clave de la pregunta (última palabra antes de '?')
        palabra = pregunta.split()[-1].replace('?', '').lower()
        if palabra in fw_dict.words:
            vector = fw_dict.get_word(palabra).nivel_3
            aprendizaje.append((palabra, vector, respuesta))
    # Mostrar "aprendizaje"
    salida = []
    salida.append("--- Fase de aprendizaje ---")
    for palabra, vector, respuesta in aprendizaje:
        fractal = fw_dict.get_word(palabra)
        nivel_3 = getattr(fractal, 'nivel_3', None)
        nivel_9 = getattr(fractal, 'nivel_9', None)
        nivel_27 = getattr(fractal, 'nivel_27', None)
        salida.append(f"I: ¿Qué es {palabra}?\nR: {respuesta}\nTensor: nivel_3={nivel_3}, nivel_9={nivel_9}, nivel_27={nivel_27}")

    # Fase de síntesis: varias preguntas nuevas
    nuevas_preguntas = [
        "¿Qué es la sociedad?",
        "¿Qué es el mar?",
        "¿Qué es la estrella?",
        "¿Qué es el gato?",
        "¿Qué es la guerra?",
        "¿Qué es el arte?",
        "¿Qué es la ciencia?",
        "¿Qué es la política?",
        "¿Qué es la historia?",
    ]
    salida.append("--- Fase de respuesta ---")
    # Mecanismos de arquetipos, relatores y dinámicas
    arquetipos = ['Buscador', 'Escéptico', 'Narrador', 'Evolucionador', 'Aprendiz']
    relatores = ['Sabio', 'Crítico', 'Narrador', 'Historiador', 'Futurista']
    dinamicas = ['Exploración', 'Debate', 'Profundización', 'Evolución', 'Proyección']
    for i, nueva_pregunta in enumerate(nuevas_preguntas):
        palabra_nueva = nueva_pregunta.split()[-1].replace('?', '').lower()
        arquetipo = arquetipos[i % len(arquetipos)]
        relator = relatores[i % len(relatores)]
        dinamica = dinamicas[i % len(dinamicas)]
        salida.append(f"[{arquetipo} - {relator} - {dinamica}]")
        salida.append(f"Pregunta: {nueva_pregunta}")
        if palabra_nueva in fw_dict.words:
            vector_nuevo = fw_dict.get_word(palabra_nueva).nivel_3
            from math import dist
            distancias = []
            for palabra, fractal in fw_dict.words.items():
                v = getattr(fractal, 'nivel_3')
                d = dist(vector_nuevo, v)
                distancias.append((d, palabra))
            distancias.sort()
            cluster_cercano = [pal for _, pal in distancias[:3]]
            cluster_extendido = [pal for _, pal in distancias[-3:]]

            def tensor_str(fractal):
                # Devuelve string con los tres niveles del tensor
                n3 = getattr(fractal, 'nivel_3', None)
                n9 = getattr(fractal, 'nivel_9', None)
                n27 = getattr(fractal, 'nivel_27', None)
                return f"nivel_3: {n3}\nnivel_9: {n9}\nnivel_27: {n27}"

            salida.append("Aurora responde: Cluster de síntesis (más cercano):")
            for pal in cluster_cercano:
                fractal = fw_dict.get_word(pal)
                salida.append(f"  - {pal}:\n{tensor_str(fractal)}")
            salida.append("")
            salida.append("Aurora responde: Cluster de extensión (más lejano):")
            for pal in cluster_extendido:
                fractal = fw_dict.get_word(pal)
                salida.append(f"  - {pal}:\n{tensor_str(fractal)}")
            salida.append("")
        else:
            salida.append(f"Aurora responde: No se encontró vector para '{palabra_nueva}'\n")
    return '\n'.join(salida)

if __name__ == "__main__":
    print(generar_dialogo_y_aprendizaje(VECTORES))
    print("\n--- Ejemplo de síntesis y vector→palabra ---")
    print(ejemplo_sintesis_y_vector_a_palabra())
    print("\n--- Ejemplo de dinámica y extensión de vector ---")
    print(ejemplo_dinamica_extension())
    print("\n--- Ejemplo de aprendizaje y respuesta ---")
    print(ejemplo_aprendizaje_y_respuesta())
