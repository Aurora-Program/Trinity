import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from aurora_core.Lexer import Lexer # Importar el nuevo Lexer
from vectors.Vectors import VECTORES # Usar el diccionario de vectores directamente
from vectors.Lexemes import LEXEMES # Importar el diccionario de lexemas

# Instanciar el Lexer una vez para reutilizarlo
lexer = Lexer()

def extraer_palabras_relevantes(frase):
    """
    Extrae todas las palabras relevantes de la frase, ignorando stopwords.
    """
    frase = frase.strip().lower()
    frase = re.sub(r'[¿?¡!.,;:]', '', frase)
    palabras = frase.split()
    stopwords = {'el','la','los','las','un','una','unos','unas','de','del','en','y','o','a','por','para','con','sin','al'}
    palabras_filtradas = [p for p in palabras if p not in stopwords]
    return palabras_filtradas if palabras_filtradas else palabras

def sum_recursive(a, b):
    """
    Suma recursivamente dos valores. Si uno es lista y el otro un escalar,
    realiza la suma elemento a elemento.
    """
    if isinstance(a, list) and isinstance(b, list):
        return [sum_recursive(x, y) for x, y in zip(a, b)]
    elif isinstance(a, list):
        return [sum_recursive(x, b) for x in a]
    elif isinstance(b, list):
        return [sum_recursive(a, y) for y in b]
    else:
        return a + b

def average_recursive(val, n):
    if isinstance(val, list):
        return [average_recursive(x, n) for x in val]
    else:
        return val / n

def combinar_vectores(vectores):
    """
    Combina una lista de FractalVectors en uno solo promediando sus componentes.
    Realiza la suma elemento a elemento, manejando componentes que pueden ser listas.
    """
    if not vectores:
        return None
    
    from aurora_core.FractalVector import FractalVector
    
    n = len(vectores)
    # Inicializar suma con ceros para 27 componentes.
    suma = [0] * 27
    for v in vectores:
        v_list = v.to_list()
        for i, valor in enumerate(v_list):
            # Sumar recursivamente, sea cual sea la estructura
            suma[i] = sum_recursive(suma[i], valor)
    
    # Calcular el promedio para cada componente usando average_recursive
    promedio = [average_recursive(s, n) for s in suma]
            
    return FractalVector.from_list(promedio)

def vector_para_frase(frase):
    """
    Construye un FractalVector para una frase completa.
    1. Analiza cada palabra para obtener su lexema y morfemas.
    2. Busca el vector para el lexema en LEXEMES.
    3. Busca vectores para los morfemas (si existen).
    4. Combina los vectores (lexema + morfemas) para obtener el vector de la palabra.
    5. Promedia los vectores de todas las palabras para obtener el vector de la frase.
    """
    from aurora_core.FractalVector import FractalVector

    palabras = extraer_palabras_relevantes(frase)
    if not palabras:
        return "frase_vacia", None, "No se extrajeron palabras de la frase."

    vectores_de_palabras = []
    palabras_procesadas = []
    log_detalle = []

    for palabra in palabras:
        analisis = lexer.analizar_palabra(palabra)
        lexema = analisis['lexema']
        morfemas = analisis['morfemas']
        
        palabras_procesadas.append(f"{palabra}({lexema})")
        log_palabra = f"Palabra: '{palabra}' -> Lexema: '{lexema}', Morfemas: {morfemas}. "

        # 1. Buscar vector para el lexema o la palabra original
        vector_base_data = LEXEMES.get(lexema) or VECTORES.get(palabra)

        if not vector_base_data:
            log_detalle.append(log_palabra + "Estado: No se encontró vector base. Omitiendo.")
            continue

        # Si vector_base_data ya es un FractalVector, crear una copia usando to_list(),
        # de lo contrario, convertirlo directamente.
        if isinstance(vector_base_data, FractalVector):
            vector_palabra = FractalVector.from_list(vector_base_data.to_list())
        else:
            vector_palabra = FractalVector.from_list(vector_base_data)

        log_palabra += f"Vector base ('{lexema}' o '{palabra}') encontrado. "
       
        log_palabra += f"Vector base ('{lexema}' o '{palabra}') encontrado. "

        # 2. Buscar y combinar vectores de morfemas (lógica futura)
        # Por ahora, solo usamos el vector del lexema/palabra.
        # Aquí se añadiría la lógica para sumar los vectores de los morfemas.

        vectores_de_palabras.append(vector_palabra)
        log_detalle.append(log_palabra + "Estado: Vector de palabra añadido.")

    if not vectores_de_palabras:
        return "sin_vectores", None, " ".join(log_detalle)
    
    # 5. Promediar los vectores de todas las palabras válidas
    vector_frase = combinar_vectores(vectores_de_palabras)
    
    clave_frase = " ".join(palabras_procesadas)
    
    return clave_frase, vector_frase, " ".join(log_detalle)

def procesar_frase_individual(frase):
    """
    Wrapper para vector_para_frase que devuelve un diccionario compatible
    con el script de benchmark y otros módulos que esperan este formato.
    """
    clave, vector, log = vector_para_frase(frase)
    
    if vector:
        return {
            "clave_frase": clave,
            "vector_resultante": vector.to_list(),
            "log": log
        }
    else:
        # Si no se pudo generar un vector (p.ej., frase vacía o sin palabras conocidas),
        # devuelve un vector nulo para evitar errores en los cálculos posteriores.
        return {
            "clave_frase": clave,
            "vector_resultante": [0] * 27,
            "log": log
        }
