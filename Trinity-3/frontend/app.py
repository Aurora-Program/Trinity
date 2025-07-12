from flask import Flask, render_template, request, jsonify
import sys
import os

# --- Configuración de Paths ---
PROYECTO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROYECTO_ROOT not in sys.path:
    sys.path.insert(0, PROYECTO_ROOT)

# --- Importaciones del Modelo Aurora ---
from aurora_core.FractalVector import FractalVector
from aurora_core.Trigate import Trigate
from aurora_core.Transcender import Transcender
# Asumo que existe un Extender, si no, necesitaré crearlo o que me indiques dónde está.
# from aurora_core.Extender import Extender 
from vectors.Vectors import VECTORES
from frontend.utils import vector_para_frase

app = Flask(__name__)
# Almacenar axiomas y procesos de razonamiento por frase
stored_axiomas = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar_frase():
    frase = request.json.get('frase', '').strip()
    if not frase:
        return jsonify({'error': 'La frase no puede estar vacía.'}), 400

    try:
        clave, fractal_inicial = vector_para_frase(frase, VECTORES)

        if not fractal_inicial:
            return jsonify({'error': f"No se pudo encontrar o sintetizar un vector para: '{frase}'"}), 404

        transcender = Transcender(fractal_inicial)
        # Descubrimiento de axiomas y razonamiento profundo
        A, B, C = fractal_inicial.nivel_9
        triada = transcender.sintetizar_triada(A, B, C)
        M_emergent = triada.get('M_emergent')
        razonamiento = transcender.deep_learning(A, B, C, M_emergent)
        # Guardar axiomas y razonamiento para esta frase
        stored_axiomas[clave] = razonamiento
        # Extender fractal (ajustable por dinámica)
        fractal_extendido, componentes = transcender.extender_fractal()

        # 3. Comparación para encontrar la dinámica (similar a la versión anterior)
        from math import dist
        distancias = []
        for p, f_data in VECTORES.items():
            f = FractalVector.from_list(f_data)
            
            # --- Lógica de distancia robusta ---
            if f.nivel_3 and hasattr(fractal_extendido, 'nivel_3') and fractal_extendido.nivel_3:
                vec_extendido = fractal_extendido.nivel_3[0] if isinstance(fractal_extendido.nivel_3[0], list) else fractal_extendido.nivel_3
                vec_f = f.nivel_3[0] if isinstance(f.nivel_3[0], list) else f.nivel_3

                # Calcular dist solo si ambas listas tienen misma longitud
                if isinstance(vec_extendido, list) and isinstance(vec_f, list) and len(vec_extendido) == len(vec_f):
                    d = dist(vec_extendido, vec_f)
                    distancias.append((d, p))
        
        distancias.sort()
        cluster_cercano = [pal for _, pal in distancias[:3]]
        cluster_extendido = [pal for _, pal in distancias[-3:]]

        # -----------------------------------------------------------
        # Mapear cada subvector nivel 9 y nivel 27 a la palabra más cercana
        def nearest_word(vec):
            # Asegurar que vec es un vector plano (extraer si está anidado)
            if isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec_flat = vec[0]
            else:
                vec_flat = vec if isinstance(vec, list) else None
            best = None
            best_d = float('inf')
            for k, f_data in VECTORES.items():
                f = FractalVector.from_list(f_data)
                seed = f.nivel_3
                if isinstance(seed, list) and seed and isinstance(seed[0], list):
                    seed_flat = seed[0]
                else:
                    seed_flat = seed if isinstance(seed, list) else None
                # Calcular solo si ambas listas planas coinciden en longitud
                if isinstance(vec_flat, list) and isinstance(seed_flat, list) and len(vec_flat) == len(seed_flat):
                    d = dist(vec_flat, seed_flat)
                    if d < best_d:
                        best_d, best = d, k
            return best

        # Asegurarse de extraer correctamente los vectores de nivel_9 y nivel_27
        lvl9 = []
        for row in fractal_extendido.nivel_9:
            vec = row if isinstance(row, list) and all(isinstance(x, (int,float)) for x in row) else row
            lvl9.append(nearest_word(vec))
        lvl27 = []
        for entry in fractal_extendido.nivel_27:
            # entry = [M, S, semilla], usamos semilla
            semilla = entry[2] if isinstance(entry[2], list) else entry[2]
            lvl27.append(nearest_word(semilla))
        # -----------------------------------------------------------
        return jsonify({
            'clave': clave,
            'vector_sintetizado_n3': fractal_extendido.to_list()[0],
            'vector_sintetizado_n9': fractal_extendido.to_list()[1],
            'vector_extendido_n27': fractal_extendido.to_list()[2],
            'dinamica_cercana': cluster_cercano,
            'dinamica_extendida': cluster_extendido,
            'componentes_extension': componentes,
            'palabras_nivel9': lvl9,
            'palabras_nivel27': lvl27,
            'razonamiento': razonamiento
        })

    except Exception as e:
        print(f"Error en /procesar: {e}")
        return jsonify({'error': 'Ocurrió un error interno en el servidor.'}), 500

@app.route('/sintetizar', methods=['POST'])
def sintetizar_axioma():
    frases = request.json.get('frases', [])
    if not frases or not all(isinstance(f, str) for f in frases):
        return jsonify({'error': 'La entrada debe ser una lista de frases.'}), 400

    try:
        vectores_iniciales = []
        for frase in frases:
            _, fractal = vector_para_frase(frase.strip(), VECTORES)
            if not fractal:
                return jsonify({'error': f'No se pudo encontrar o sintetizar un vector para: "{frase}"'}), 404
            vectores_iniciales.append(fractal)

        # 2. Sintetizar el Axioma combinando los vectores
        if not vectores_iniciales:
            return jsonify({'error': 'No se encontraron vectores para sintetizar.'}), 400

        # Combinar nivel_3 y nivel_9 por XOR
        axiom_n3_vec = [0, 0, 0]
        for v in vectores_iniciales:
            # Manejar ambas estructuras: [[v]] o [v]
            vec_v = v.nivel_3[0] if isinstance(v.nivel_3[0], list) else v.nivel_3
            for i in range(3):
                axiom_n3_vec[i] ^= vec_v[i]
        
        # Envolver en una lista para mantener la estructura [[v]]
        axiom_n3 = [axiom_n3_vec]

        axiom_n9 = [[0,0,0], [0,0,0], [0,0,0]]
        for v in vectores_iniciales:
            for i in range(3):
                for j in range(3):
                    axiom_n9[i][j] ^= v.nivel_9[i][j]
        
        axiom_vector = FractalVector(axiom_n3, axiom_n9)

        transcender = Transcender(axiom_vector)
        # Guardar axiomas y razonamiento del axioma sintetizado
        triada_axioma = transcender.sintetizar_triada(axiom_vector.nivel_3[0], axiom_vector.nivel_3[0], axiom_vector.nivel_3[0])
        M_em = triada_axioma.get('M_emergent')
        razonamiento_axioma = transcender.deep_learning(axiom_vector.nivel_3[0], axiom_vector.nivel_3[0], axiom_vector.nivel_3[0], M_em)
        stored_axiomas['Axioma: ' + ' + '.join(frases)] = razonamiento_axioma
        fractal_extendido, componentes = transcender.extender_fractal()

        # 4. Comparación para encontrar la dinámica
        from math import dist
        distancias = []
        for p, f_data in VECTORES.items():
            f = FractalVector.from_list(f_data)
            # --- Lógica de distancia robusta ---
            if f.nivel_3 and hasattr(fractal_extendido, 'nivel_3') and fractal_extendido.nivel_3:
                vec_extendido = fractal_extendido.nivel_3[0] if isinstance(fractal_extendido.nivel_3[0], list) else fractal_extendido.nivel_3
                vec_f = f.nivel_3[0] if isinstance(f.nivel_3[0], list) else f.nivel_3
                # Calcular dist solo si ambas listas tienen misma longitud
                if isinstance(vec_extendido, list) and isinstance(vec_f, list) and len(vec_extendido) == len(vec_f):
                    d = dist(vec_extendido, vec_f)
                    distancias.append((d, p))
        
        distancias.sort()
        cluster_cercano = [pal for _, pal in distancias[:3]]
        cluster_extendido = [pal for _, pal in distancias[-3:]]

        # -----------------------------------------------------------
        # Mapear cada subvector nivel 9 y nivel 27 a la palabra más cercana
        def nearest_word(vec):
            # Asegurar que vec es un vector plano (extraer si está anidado)
            if isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec_flat = vec[0]
            else:
                vec_flat = vec if isinstance(vec, list) else None
            best = None
            best_d = float('inf')
            for k, f_data in VECTORES.items():
                f = FractalVector.from_list(f_data)
                seed = f.nivel_3
                if isinstance(seed, list) and seed and isinstance(seed[0], list):
                    seed_flat = seed[0]
                else:
                    seed_flat = seed if isinstance(seed, list) else None
                # Calcular solo si ambas listas planas coinciden en longitud
                if isinstance(vec_flat, list) and isinstance(seed_flat, list) and len(vec_flat) == len(seed_flat):
                    d = dist(vec_flat, seed_flat)
                    if d < best_d:
                        best_d, best = d, k
            return best

        # Asegurarse de extraer correctamente los vectores de nivel_9 y nivel_27
        lvl9 = []
        for row in fractal_extendido.nivel_9:
            vec = row if isinstance(row, list) and all(isinstance(x, (int,float)) for x in row) else row
            lvl9.append(nearest_word(vec))
        lvl27 = []
        for entry in fractal_extendido.nivel_27:
            # entry = [M, S, semilla], usamos semilla
            semilla = entry[2] if isinstance(entry[2], list) else entry[2]
            lvl27.append(nearest_word(semilla))
        # -----------------------------------------------------------
        return jsonify({
            'clave': 'Axioma: ' + ' + '.join(frases),
            'vector_sintetizado_n3': fractal_extendido.to_list()[0],
            'vector_sintetizado_n9': fractal_extendido.to_list()[1],
            'vector_extendido_n27': fractal_extendido.to_list()[2],
            'dinamica_cercana': cluster_cercano,
            'dinamica_extendida': cluster_extendido,
            'componentes_extension': componentes,
            'palabras_nivel9': lvl9,
            'palabras_nivel27': lvl27
        })

    except Exception as e:
        print(f"Error en /sintetizar: {e}")
        return jsonify({'error': 'Ocurrió un error interno en el servidor.'}), 500

# Eliminar el endpoint antiguo si ya no se usa
# @app.route('/vector', methods=['POST'])
# def get_vector(): ...

if __name__ == '__main__':
    app.run(debug=True)
