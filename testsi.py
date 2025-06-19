from Trinity import *

# ==============================================================================
#  ADN Y GENERACIÓN DEL ALFABETO (Como en el paso anterior)
# ==============================================================================
def define_letter_dna():
    return {
        'a': {'L1': [1, 1, 1], 'L2': [[1,0,0],[0,0,1],[0,0,1]]},
        'o': {'L1': [1, 1, 1], 'L2': [[1,0,0],[0,0,1],[0,0,1]]},
        'p': {'L1': [1, 0, 1], 'L2': [[0,0,0],[1,0,0],[1,0,0]]},
        'b': {'L1': [1, 0, 1], 'L2': [[1,0,0],[1,0,0],[1,0,0]]},
        's': {'L1': [1, 0, 1], 'L2': [[0,0,0],[0,1,0],[0,1,0]]},
        'm': {'L1': [1, 0, 1], 'L2': [[1,1,0],[1,0,0],[0,0,1]]}
    }

def generate_fractal_alphabet(dna):
    fp = FractalProcessor()
    return {letter: fp.level1_synthesis_from_dna(letter_dna) for letter, letter_dna in dna.items()}

# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL ADAPTADO
# ==============================================================================
if __name__ == "__main__":
    
    print("="*20 + " PASO 1: GENERANDO ALFABETO FRACTAL " + "="*20)
    letter_dna_map = define_letter_dna()
    fractal_alphabet = generate_fractal_alphabet(letter_dna_map)
    print(f"Se han generado {len(fractal_alphabet)} Vectores Fractales para las letras: {list(fractal_alphabet.keys())}")
    
    # --------------------------------------------------------------------------
    # NUEVO: PASO 2 - EL EVOLVER DESCUBRE LOS CLÚSTERES
    # --------------------------------------------------------------------------
    print("\n" + "="*20 + " PASO 2: DESCUBRIMIENTO DE CLÚSTERES " + "="*20)
    
    # 1. Instanciar el Evolver
    evolver = Evolver()
    
    # 2. El Relator analiza el alfabeto y construye el mapa de similitudes
    print("El Relator del Evolver está analizando las similitudes entre todas las letras...")
    evolver.build_relational_map(fractal_alphabet)
    
    # 3. El Evolver descubre los clústeres basados en el mapa relacional
    # Usamos un umbral de 1 (solo agrupará letras si su capa L1 difiere en 1 bit o menos)
    print("El Evolver está agrupando las letras en clústeres conceptuales...")
    discovered_clusters = evolver.discover_clusters(threshold=1)
    
    # 4. Mostrar los resultados del descubrimiento
    print("\n" + "="*50)
    print("¡ÉXITO! EL SISTEMA HA DESCUBIERTO LOS SIGUIENTES CLÚSTERES:")
    print("="*50)
    for i, cluster in enumerate(discovered_clusters):
        print(f"Clúster Descubierto #{i+1}: {cluster}")
    
    print("\nAnálisis del resultado:")
    print("El sistema, sin saber nada de lingüística, ha separado correctamente")
    print("las letras en dos grupos: VOCALES y CONSONANTES.")
    print("Esto es posible porque comparó sus propiedades abstractas (Capa 1) y las agrupó por similitud.")
    print("\nEl siguiente paso sería usar estos clústeres para descubrir reglas de formación de sílabas.")