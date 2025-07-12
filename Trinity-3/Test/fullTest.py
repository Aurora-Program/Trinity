import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aurora_core.Transcender import Transcender
from aurora_core.knowledgeBase import KnowledgeBase
from aurora_core.evolver import Evolver
from aurora_core.extender import Extender
import random

def generar_fractal_logico(base_vector, regla):
    """
    Genera una secuencia fractal de 3 vectores de 3 bits siguiendo una regla lógica conocida.
    regla: función que toma dos vectores y devuelve el tercero (por ejemplo, XOR, AND, OR, etc.)
    """
    v1 = base_vector
    v2 = tuple(random.randint(0,1) for _ in range(3))
    v3 = regla(v1, v2)
    return [v1, v2, v3]

def xor_vec(a, b):
    return tuple(x ^ y for x, y in zip(a, b))

def and_vec(a, b):
    return tuple(x & y for x, y in zip(a, b))

def or_vec(a, b):
    return tuple(x | y for x, y in zip(a, b))

def suma_digitos(n):
    return sum(int(d) for d in str(n))

def numero_a_vectores(n, base=2, largo=27):
    # Convierte un número en un vector binario de longitud 'largo'
    binario = bin(n)[2:].zfill(largo)
    return [int(b) for b in binario]

def fractalizar_vector(vec, nivel=3):
    # Divide un vector largo en sub-vectores de tamaño 3 (nivel más bajo)
    return [vec[i:i+3] for i in range(0, len(vec), 3)]

def test_aurora_full():
    # 1. Fase de aprendizaje
    kb = KnowledgeBase()
    evolver = Evolver()
    transcender = Transcender()
    extender = Extender(knowledge_base=kb, evolver=evolver)
    reglas = [xor_vec, and_vec, or_vec]
    nombres_reglas = ['XOR', 'AND', 'OR']
    patrones_aprendidos = 0
    total_presentados = 0
    print("=== Fase de aprendizaje ===")
    # Ejemplos reales del universo (lógica binaria simple, por ejemplo, bits de sensores, compuertas lógicas, etc.)
    ejemplos_reales = [
        # XOR
        ((0,0,0), (0,0,1), xor_vec),
        ((1,0,0), (0,1,1), xor_vec),
        ((1,1,1), (1,0,0), xor_vec),
        # AND
        ((1,1,1), (1,1,0), and_vec),
        ((0,1,0), (1,0,1), and_vec),
        ((1,0,1), (1,1,1), and_vec),
        # OR
        ((0,0,1), (1,0,0), or_vec),
        ((1,1,0), (0,1,1), or_vec),
        ((0,1,1), (1,1,0), or_vec),
        # Más ejemplos representando combinaciones típicas de compuertas lógicas
        ((0,0,0), (1,1,1), xor_vec),
        ((1,1,1), (0,0,0), and_vec),
        ((0,0,1), (0,1,0), or_vec),
        ((1,0,1), (0,1,0), and_vec),
        ((1,1,0), (1,0,1), xor_vec),
        ((0,1,0), (1,1,1), or_vec),
        ((1,0,0), (0,0,1), and_vec),
        ((0,1,1), (1,0,0), xor_vec),
        ((1,1,1), (1,1,1), and_vec),
        ((0,0,0), (0,0,0), or_vec),
    ]
    for base, otro, regla in ejemplos_reales:
        fractal = generar_fractal_logico(base, regla)
        fractal[1] = otro  # Forzar el segundo vector a ser el ejemplo real
        fractal[2] = regla(fractal[0], fractal[1])
        nombre_regla = 'XOR' if regla == xor_vec else ('AND' if regla == and_vec else 'OR')
        resultado = transcender.deep_learning(fractal[0], fractal[1], fractal[2], fractal[2])
        kb.add_entry(fractal[0], fractal[1], fractal[2], fractal[2], resultado['MetaM'], resultado['R_hipotesis'], transcender_id=nombre_regla)
        patrones_aprendidos += 1
        total_presentados += 1
        print(f"Aprendido patrón {nombre_regla}: {fractal}")

    print(f"Total patrones presentados: {total_presentados}")
    print(f"Total patrones aprendidos: {patrones_aprendidos}")

    # 2. Fase de validación y test de respuesta inteligente
    print("\n=== Fase de validación y test de respuesta ===")
    aciertos = 0
    pruebas = 20
    for i, (base, otro, regla) in enumerate(ejemplos_reales):
        fractal = generar_fractal_logico(base, regla)
        fractal[1] = otro
        fractal[2] = regla(fractal[0], fractal[1])
        nombre_regla = 'XOR' if regla == xor_vec else ('AND' if regla == and_vec else 'OR')
        Ss = [fractal[0], fractal[1], fractal[2]]
        contexto = {'objetivo': fractal[2], 'M_emergent': fractal[2]}
        respuesta = extender.extend(Ss, contexto=contexto)
        reconstruido = respuesta['reconstruccion']['tensores_reconstruidos']
        print(f"Test {i+1}: {nombre_regla} | Entrada: {fractal[0]}, {fractal[1]} | Esperado: {fractal[2]} | Reconstruido: {reconstruido}")
        # Compara solo contra el valor esperado (C)
        if reconstruido == list(fractal[2]):
            aciertos += 1
    print(f"\nAciertos en test de respuesta: {aciertos}/{pruebas}")
    print(f"Cobertura: {aciertos/pruebas*100:.1f}%")

def test_fractal_suma_digitos():
    print("\n=== Test fractal: suma de dígitos ===")
    kb = KnowledgeBase()
    evolver = Evolver()
    transcender = Transcender()
    extender = Extender(knowledge_base=kb, evolver=evolver)
    # 1. Generar números y sus sumas de dígitos
    numeros = list(range(100, 110))  # 10 números para ejemplo
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)  # 9 sub-vectores de 3 bits
        # Para cada subvector, calculamos la suma de bits (como "resultado real")
        for i, subv in enumerate(subvecs):
            suma = sum(subv)
            # Creamos A, B, C como los 3 bits, y M_emergent como la suma (en binario, 2 bits)
            # Para forzar la fractalidad, usamos los sub-vectores como entradas a Transcender
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [suma%2, (suma>>1)%2, (suma>>2)%2]  # 3 bits para la suma
            resultado = transcender.deep_learning(A, B, C, M_emergent)
            kb.add_entry(A, B, C, M_emergent, resultado['MetaM'], resultado['R_hipotesis'], transcender_id='SUMA')
    # 2. Aprender arquetipos/dinámicas
    metaMs = [e['MetaM'] for e in kb.all_entries()]
    patrones = evolver.analyze_metaMs(metaMs)
    axiom = evolver.formalize_axiom(patrones, metaMs)
    print(f"Axioma aprendido: {axiom}")
    # 3. Validar reconstrucción
    aciertos = 0
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            suma = sum(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [suma%2, (suma>>1)%2, (suma>>2)%2]
            Ss = [A, B, C]
            contexto = {'objetivo': C, 'M_emergent': M_emergent}
            respuesta = extender.extend(Ss, contexto=contexto)
            reconstruido = respuesta['reconstruccion']['tensores_reconstruidos']
            print(f"N={n} | Entrada: {A} | Esperado: {C} | Reconstruido: {reconstruido}")
            if reconstruido == C:
                aciertos += 1
    print(f"Aciertos fractales: {aciertos}/" + str(len(numeros)*9))
    print(f"Cobertura fractal: {aciertos/(len(numeros)*9)*100:.1f}%")

def mayoria_bits(subv):
    # Devuelve 1 si la mayoría de los bits es 1, sino 0
    return 1 if sum(subv) >= 2 else 0

def producto_digitos(n):
    prod = 1
    for d in str(n):
        prod *= int(d)
    return prod

def test_fractal_mayoria_y_producto():
    print("\n=== Test fractal: mayoría de bits y producto de dígitos ===")
    kb = KnowledgeBase()
    evolver = Evolver()
    transcender = Transcender()
    extender = Extender(knowledge_base=kb, evolver=evolver)
    numeros = list(range(200, 210))  # 10 números para ejemplo
    # 1. Aprendizaje con mayoría de bits
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            maj = mayoria_bits(subv)
            # A, B, C: variantes del subvector
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [maj, maj, maj]  # 3 bits iguales para mayoría
            resultado = transcender.deep_learning(A, B, C, M_emergent)
            kb.add_entry(A, B, C, M_emergent, resultado['MetaM'], resultado['R_hipotesis'], transcender_id='MAYORIA')
    # 2. Aprendizaje con producto de dígitos
    for n in numeros:
        prod = producto_digitos(n)
        vec = numero_a_vectores(prod, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            suma = sum(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [suma%2, (suma>>1)%2, (suma>>2)%2]
            resultado = transcender.deep_learning(A, B, C, M_emergent)
            kb.add_entry(A, B, C, M_emergent, resultado['MetaM'], resultado['R_hipotesis'], transcender_id='PRODUCTO')
    # 3. Aprender arquetipos/dinámicas
    metaMs = [e['MetaM'] for e in kb.all_entries()]
    patrones = evolver.analyze_metaMs(metaMs)
    axiom = evolver.formalize_axiom(patrones, metaMs)
    print(f"Axioma aprendido: {axiom}")
    # 4. Validar reconstrucción
    aciertos = 0
    total = 0
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            maj = mayoria_bits(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [maj, maj, maj]
            Ss = [A, B, C]
            contexto = {'objetivo': C, 'M_emergent': M_emergent}
            respuesta = extender.extend(Ss, contexto=contexto)
            reconstruido = respuesta['reconstruccion']['tensores_reconstruidos']
            print(f"N={n} | MAYORIA | Entrada: {A} | Esperado: {C} | Reconstruido: {reconstruido}")
            if reconstruido == C:
                aciertos += 1
            total += 1
    for n in numeros:
        prod = producto_digitos(n)
        vec = numero_a_vectores(prod, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            suma = sum(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [suma%2, (suma>>1)%2, (suma>>2)%2]
            Ss = [A, B, C]
            contexto = {'objetivo': C, 'M_emergent': M_emergent}
            respuesta = extender.extend(Ss, contexto=contexto)
            reconstruido = respuesta['reconstruccion']['tensores_reconstruidos']
            print(f"N={n} | PRODUCTO | Entrada: {A} | Esperado: {C} | Reconstruido: {reconstruido}")
            if reconstruido == C:
                aciertos += 1
            total += 1
    print(f"Aciertos fractales complejos: {aciertos}/" + str(total))
    print(f"Cobertura fractal compleja: {aciertos/total*100:.1f}%")

def paridad_alterna(subv):
    # Paridad alterna: XOR de los bits en posiciones pares menos XOR de impares
    pares = [subv[i] for i in range(len(subv)) if i % 2 == 0]
    impares = [subv[i] for i in range(len(subv)) if i % 2 == 1]
    xor_pares = 0
    for b in pares:
        xor_pares ^= b
    xor_impares = 0
    for b in impares:
        xor_impares ^= b
    return (xor_pares ^ xor_impares)

def suma_condicional(subv):
    # Suma los bits solo si el primer bit es 1, si no devuelve 0
    return sum(subv) if subv[0] == 1 else 0

def test_fractal_reglas_complejas():
    print("\n=== Test fractal: reglas complejas (paridad alterna y suma condicional) ===")
    kb = KnowledgeBase()
    evolver = Evolver()
    transcender = Transcender()
    extender = Extender(knowledge_base=kb, evolver=evolver)
    numeros = list(range(300, 310))
    # 1. Aprendizaje con paridad alterna
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            par = paridad_alterna(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [par, par, par]
            resultado = transcender.deep_learning(A, B, C, M_emergent)
            kb.add_entry(A, B, C, M_emergent, resultado['MetaM'], resultado['R_hipotesis'], transcender_id='PARIDAD_ALT')
    # 2. Aprendizaje con suma condicional
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            scond = suma_condicional(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [scond%2, (scond>>1)%2, (scond>>2)%2]
            resultado = transcender.deep_learning(A, B, C, M_emergent)
            kb.add_entry(A, B, C, M_emergent, resultado['MetaM'], resultado['R_hipotesis'], transcender_id='SUMA_COND')
    # 3. Aprender arquetipos/dinámicas
    metaMs = [e['MetaM'] for e in kb.all_entries()]
    patrones = evolver.analyze_metaMs(metaMs)
    axiom = evolver.formalize_axiom(patrones, metaMs)
    print(f"Axioma aprendido: {axiom}")
    # 4. Validar reconstrucción
    aciertos = 0
    total = 0
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            par = paridad_alterna(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [par, par, par]
            Ss = [A, B, C]
            contexto = {'objetivo': C, 'M_emergent': M_emergent}
            respuesta = extender.extend(Ss, contexto=contexto)
            reconstruido = respuesta['reconstruccion']['tensores_reconstruidos']
            print(f"N={n} | PARIDAD_ALT | Entrada: {A} | Esperado: {C} | Reconstruido: {reconstruido}")
            if reconstruido == C:
                aciertos += 1
            total += 1
    for n in numeros:
        vec = numero_a_vectores(n, largo=27)
        subvecs = fractalizar_vector(vec, nivel=3)
        for i, subv in enumerate(subvecs):
            scond = suma_condicional(subv)
            A, B, C = subv, [((b+1)%2) for b in subv], [((b+1)%2) for b in subv[::-1]]
            M_emergent = [scond%2, (scond>>1)%2, (scond>>2)%2]
            Ss = [A, B, C]
            contexto = {'objetivo': C, 'M_emergent': M_emergent}
            respuesta = extender.extend(Ss, contexto=contexto)
            reconstruido = respuesta['reconstruccion']['tensores_reconstruidos']
            print(f"N={n} | SUMA_COND | Entrada: {A} | Esperado: {C} | Reconstruido: {reconstruido}")
            if reconstruido == C:
                aciertos += 1
            total += 1
    print(f"Aciertos fractales ultra: {aciertos}/" + str(total))
    print(f"Cobertura fractal ultra: {aciertos/total*100:.1f}%")

if __name__ == "__main__":
    test_aurora_full()
    test_fractal_suma_digitos()
    test_fractal_mayoria_y_producto()
    test_fractal_reglas_complejas()
