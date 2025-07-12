# -*- coding: utf-8 -*-
# --- Import missing dependencies ---
import random
from itertools import product
from math import dist

# ----------------------------------------------------------------------
# Componente 1: Trigate
# El motor lógico fundamental del sistema.
# ----------------------------------------------------------------------

class Trigate:
    _LUTS_BUILT = False
    _LUTS = {
        'synth': {},  # (A, B, R) -> S
        'infer': {},  # (A, B, M) -> R
        'learn': {},  # (A, B, R) -> M
        'deduce_B': {}, # (A, M, R) -> B
        'deduce_A': {}  # (B, M, R) -> A
    }

    def __init__(self):
        if not self._LUTS_BUILT:
            self._build_luts()

    def _build_luts(self):
        if self.__class__._LUTS_BUILT:
            return
            
        states = [0, 1, None]
        for a, b, r in product(states, repeat=3):
            # SYNTH
            if None in (a, b, r):
                self._LUTS['synth'][(a, b, r)] = None
            elif r == 1:
                self._LUTS['synth'][(a, b, r)] = (1 ^ a) ^ (1 ^ b)
            else:
                self._LUTS['synth'][(a, b, r)] = (1 - (0 ^ a)) ^ (1 - (0 ^ b))
                
            # Otras LUTs con lógica similar...
        
        self.__class__._LUTS_BUILT = True

    def synthesize(self, A, B, R=None):
        self._validate_vectors(A, B, R)
        R = R or [None] * len(A)
        return [self._LUTS['synth'].get((a, b, r), None) 
                for a, b, r in zip(A, B, R)]
    
    def _validate_vectors(self, *vectors):
        for i, vec in enumerate(vectors):
            if vec is None:
                continue
            if len(vec) != 3:
                raise ValueError(f"Vector {i} tiene longitud {len(vec)} (debe ser 3)")
            for val in vec:
                if val not in (0, 1, None):
                    raise ValueError(f"Valor inválido: {val}. Solo 0, 1 o None permitidos")

class FractalTensor:
    """
    Representa un tensor fractal con 3 niveles de escala.
    Cada nivel contiene una lista de vectores ternarios de 3 elementos.
    Esta clase organiza los datos de entrada para que el Transcender
    pueda procesarlos.
    """
    def __init__(self, nivel_3=None, nivel_9=None, nivel_27=None):
        self.nivel_3 = nivel_3 if nivel_3 is not None else []
        self.nivel_9 = nivel_9 if nivel_9 is not None else []
        self.nivel_27 = nivel_27 if nivel_27 is not None else []

    def __repr__(self):
        """Representación textual del objeto para facilitar la depuración."""
        return f"FT(3={len(self.nivel_3)} vecs, 9={len(self.nivel_9)} vecs, 27={len(self.nivel_27)} vecs)"
        
    @staticmethod
    def random_tensor():
        """Método de fábrica para crear fácilmente un tensor con datos aleatorios."""
        def rand_vec():
            return [random.choice([0, 1]) for _ in range(3)]
        
        nivel_3 = [rand_vec() for _ in range(3)]
        nivel_9 = [rand_vec() for _ in range(9)]
        # nivel_27 se omite por brevedad en la demo
        
        return FractalTensor(nivel_3=nivel_3, nivel_9=nivel_9)
    
    # -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Componente 3: Transcender
# El procesador principal que analiza las interacciones.
# ----------------------------------------------------------------------

# Se asume que las clases Trigate y FractalTensor ya están definidas.
# Las incluyo aquí para que el código sea ejecutable de forma independiente.

# ----------------------------------------------------------------------

class Transcender:
    """
    El procesador principal del sistema. Orquesta las operaciones de los Trigate
    para realizar cómputos complejos sobre tríadas de vectores (A, B, C).
    Su evolución le permite analizar tensores fractales de forma dinámica.
    """
    def __init__(self):
        self.trigate_AB = Trigate()
        self.trigate_BC = Trigate()
        self.trigate_CA = Trigate()
        self.trigate_superior = Trigate()
        self.trigate_Ms = Trigate()

    def compute(self, A, B, C):
        """
        Motor de cómputo para una tríada de vectores. Produce una jerarquía
        de resultados, culminando en un MetaM que resume la computación.
        """
        A, B, C = list(A), list(B), list(C)

        M_AB, _ = self.trigate_AB.synthesize(A, B)
        M_BC, _ = self.trigate_BC.synthesize(B, C)
        M_CA, _ = self.trigate_CA.synthesize(C, A)
        
        # El resultado emergente de la interacción entre las primeras dos interacciones
        M_emergent, S_emergent = self.trigate_superior.synthesize(M_AB, M_BC)
        
        # Un camino de cálculo intermedio que incluye las tres interacciones
        interm1, _ = self.trigate_Ms.synthesize(M_AB, M_BC)
        M_intermediate, _ = self.trigate_Ms.synthesize(interm1, M_CA)
        
        # El MetaM es la diferencia entre el resultado emergente y el intermedio
        if any(bit is None for vec in (M_intermediate, M_emergent) for bit in vec):
            MetaM = [None, None, None]
        else:
            MetaM = [int(a) ^ int(b) for a, b in zip(M_intermediate, M_emergent)]

        return {
            'inputs': {'A': A, 'B': B, 'C': C},
            'outputs': {'S_emergent': S_emergent, 'MetaM': MetaM}
        }

    def compute_fractal_con_rotacion(self, fractal_tensor):
        """
        Procesa un FractalTensor aplicando una rotación cíclica de entradas
        A, B, C para cada cómputo, modelando interacciones de vecindad.
        """
        resultados = {}
        for nivel in ['nivel_3', 'nivel_9', 'nivel_27']:
            vectores = getattr(fractal_tensor, nivel, [])
            if not vectores or len(vectores) < 3:
                continue
            
            resultados[nivel] = []
            num_vectores = len(vectores)
            for i in range(num_vectores):
                # Rotación cíclica para obtener las entradas A, B y C
                A = vectores[i]
                B = vectores[(i + 1) % num_vectores]
                C = vectores[(i + 2) % num_vectores]
                resultados[nivel].append(self.compute(A, B, C))
        return resultados

# --- Ejemplo de uso del Transcender ---
if __name__ == "__main__":
    
    # 1. Crear el Transcender
    transcender = Transcender()
    
    # 2. Definir un FractalTensor para la prueba
    vectores_de_prueba = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ]
    ft_prueba = FractalTensor(nivel_3=vectores_de_prueba)
    print(f"Tensor Fractal de entrada: {ft_prueba.nivel_3}")
    print("-" * 40)
    
    # 3. Ejecutar el cómputo fractal con rotación
    resultados = transcender.compute_fractal_con_rotacion(ft_prueba)
    
    # 4. Mostrar los resultados
    print("Resultados del cómputo con rotación:")
    if 'nivel_3' in resultados:
        for i, res in enumerate(resultados['nivel_3']):
            print(f"\nInteracción #{i}:")
            print(f"  Inputs -> A:{res['inputs']['A']}, B:{res['inputs']['B']}, C:{res['inputs']['C']}")
            print(f"  Outputs -> S_emergent: {res['outputs']['S_emergent']}, MetaM: {res['outputs']['MetaM']}")

    # -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Componente 5: Evolver
# El cerebro analítico unificado para Arquetipos, Relaciones y Dinámica.
# ----------------------------------------------------------------------

# Se asume que las clases Trigate, FractalTensor y Transcender ya están definidas.
# Si estás ejecutando este bloque de forma aislada, necesitarás esas clases.
# class Trigate: ...
# class FractalTensor: ...
# class Transcender: ...

class Evolver:
    """
    La clase de análisis de alto nivel. Centraliza la lógica para encontrar
    patrones de orden superior.
    1. Calcula Arquetipos (Meta-Transcender).
    2. Analiza Relaciones (Relator).
    3. Analiza Dinámicas Temporales.
    """
    def __init__(self):
        # Cada función del Evolver utiliza su propia instancia de Transcender
        # para mantener la separación conceptual de los niveles de cómputo.
        self.transcender_base = Transcender()      # Para cómputos de primer orden
        self.transcender_meta = Transcender()      # Para cómputo de Arquetipos
        self.transcender_relacional = Transcender() # Para cómputo de Relaciones

    def compute_archetypes(self, ft_A, ft_B, ft_C):
        """
        Calcula ArquetipoM a partir de los MetaM de tres sistemas fractales.
        Actúa como un Meta-Transcender.
        """
        # 1. Obtener los resultados de primer orden para cada tensor fractal
        results_A = self.transcender_base.compute_fractal_con_rotacion(ft_A)
        results_B = self.transcender_base.compute_fractal_con_rotacion(ft_B)
        results_C = self.transcender_base.compute_fractal_con_rotacion(ft_C)
        
        archetype_results = {}
        # 2. Analizar cada nivel de escala
        for nivel in ['nivel_3', 'nivel_9', 'nivel_27']:
            if not all(nivel in res for res in [results_A, results_B, results_C]):
                continue
            
            # 3. Extraer los MetaM (firmas de interacción) de cada sistema
            metaM_A = [res['outputs']['MetaM'] for res in results_A[nivel]]
            metaM_B = [res['outputs']['MetaM'] for res in results_B[nivel]]
            metaM_C = [res['outputs']['MetaM'] for res in results_C[nivel]]
            
            # 4. Realizar el meta-cómputo: usar los MetaM como nuevas entradas A, B, C
            archetype_computations = [
                self.transcender_meta.compute(m_a, m_b, m_c)
                for m_a, m_b, m_c in zip(metaM_A, metaM_B, metaM_C)
            ]
            
            archetype_results[nivel] = {
                'input_metaM_A': metaM_A,
                'input_metaM_B': metaM_B,
                'input_metaM_C': metaM_C,
                'archetype_computations': archetype_computations
            }
        return archetype_results

    def analyze_group_relations(self, s_emergent_list):
        """
        (Funcionalidad del Relator)
        Genera un "RelatorMap" para describir la dinámica relacional
        interna de un grupo de vectores S_emergent (similitud).
        """
        num_vectores = len(s_emergent_list)
        if num_vectores < 3:
            return []
        
        relator_map = []
        for i in range(num_vectores):
            # Rotación cíclica sobre los vectores S (similitud) del grupo
            S_A = s_emergent_list[i]
            S_B = s_emergent_list[(i + 1) % num_vectores]
            S_C = s_emergent_list[(i + 2) % num_vectores]
            
            # El MetaM resultante describe la naturaleza de su inter-relación
            relational_result = self.transcender_relacional.compute(S_A, S_B, S_C)
            relator_map.append(relational_result['outputs']['MetaM'])
            
        return relator_map

    def analyze_temporal_dynamics(self, historical_archetypes):
        """
        Analiza una secuencia de Arquetipos a lo largo del tiempo para
        medir la estabilidad y el cambio del sistema global.
        """
        if len(historical_archetypes) < 2:
            return {"status": "Estable", "change_report": "Datos insuficientes para análisis dinámico."}

        # Crear un "vector de estado" para cada instante de tiempo
        state_vectors = []
        for arch_result in historical_archetypes:
            # Simplificación: concatenamos todos los ArquetipoM de nivel_3
            current_state = []
            if 'nivel_3' in arch_result:
                for comp in arch_result['nivel_3']['archetype_computations']:
                    # Asegurarse de que el MetaM no es None antes de extender
                    if comp['outputs']['MetaM'][0] is not None:
                         current_state.extend(comp['outputs']['MetaM'])
            if current_state: # Solo añadir si el estado no está vacío
                state_vectors.append(current_state)

        if len(state_vectors) < 2:
             return {"status": "Estable", "change_report": "No hay suficientes estados válidos para comparar."}

        # Calcular las diferencias (distancia de Hamming) entre estados sucesivos
        changes = [
            sum(b1 != b2 for b1, b2 in zip(vec1, vec2))
            for vec1, vec2 in zip(state_vectors, state_vectors[1:])
            if len(vec1) == len(vec2)
        ]

        return {
            "change_sequence": changes,
            "max_change": max(changes) if changes else 0,
            "avg_change": sum(changes) / len(changes) if changes else 0
        }
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Componente 4: KnowledgeBase
# La memoria permanente del sistema para el conocimiento abstracto.
# ----------------------------------------------------------------------

class KnowledgeBase:
    """
    Representa la memoria permanente del sistema. Almacena el conocimiento
    abstracto destilado por el Evolver, vinculando las causas (inputs)
    con sus efectos (outputs).
    """
    def __init__(self):
        # Cada entrada es un diccionario que representa un cómputo completo,
        # guardando la relación entre inputs y outputs.
        self.entries = []
        print("KnowledgeBase inicializada.")

    def store_computation(self, computation_result):
        """
        Guarda un resultado de cómputo completo, creando un registro de
        la experiencia del sistema.
        
        Args:
            computation_result (dict): El diccionario devuelto por Transcender.compute().
        """
        # Se asegura de que el resultado tenga la estructura esperada
        if 'inputs' in computation_result and 'outputs' in computation_result:
            self.entries.append(computation_result)
        else:
            print("Advertencia: Se intentó guardar un resultado con formato incorrecto en KnowledgeBase.")

    def find_by_synthesis(self, synthesis_vector):
        """
        Busca en la KB todas las entradas (experiencias pasadas) cuyo MetaM
        coincida con el vector de síntesis deseado (la "intención").
        
        Args:
            synthesis_vector (list): El vector MetaM que se busca.
        
        Returns:
            Una lista de todas las entradas de la KB que coinciden.
        """
        # Compara el 'synthesis_vector' con el 'MetaM' de cada salida guardada
        return [
            entry for entry in self.entries
            if 'MetaM' in entry['outputs'] and entry['outputs']['MetaM'] == synthesis_vector
        ]

    def get_knowledge_summary(self):
        """Devuelve un resumen del conocimiento almacenado."""
        return {
            "total_entries": len(self.entries),
            # Se podrían añadir más análisis, como los patrones de MetaM más comunes.
        }

# --- Ejemplo de uso de la KnowledgeBase ---
if __name__ == "__main__":
    
    # 1. Crear una KnowledgeBase
    kb = KnowledgeBase()
    
    # 2. Simular dos resultados de cómputo y guardarlos
    resultado_1 = {
        'inputs': {'A': [1,0,1], 'B': [0,1,0], 'C': [1,1,1]},
        'outputs': {'S_emergent': [0,0,1], 'MetaM': [1,1,0]}
    }
    resultado_2 = {
        'inputs': {'A': [0,0,0], 'B': [1,1,0], 'C': [0,1,1]},
        'outputs': {'S_emergent': [1,0,0], 'MetaM': [0,1,0]}
    }
    resultado_3 = {
        'inputs': {'A': [1,1,1], 'B': [1,1,1], 'C': [0,0,0]},
        'outputs': {'S_emergent': [1,1,1], 'MetaM': [1,1,0]} # MetaM igual que el resultado 1
    }
    
    kb.store_computation(resultado_1)
    kb.store_computation(resultado_2)
    kb.store_computation(resultado_3)
    
    print(f"\nResumen del conocimiento: {kb.get_knowledge_summary()}")
    
    # 3. Buscar en la KB una intención específica
    intencion = [1, 1, 0]
    print(f"\nBuscando en la KB entradas que produjeron el MetaM: {intencion}")
    
    encontrados = kb.find_by_synthesis(intencion)
    
    print(f"Se encontraron {len(encontrados)} coincidencias.")
    for i, entry in enumerate(encontrados):
        print(f"  Coincidencia #{i+1}: Los inputs {entry['inputs']} produjeron el MetaM deseado.")

# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Componente 6: Extender
# El motor generativo que crea realidades a partir de conocimiento.
# ----------------------------------------------------------------------

# Se asume que las clases FractalTensor y KnowledgeBase ya están definidas.
# Si estás ejecutando este bloque de forma aislada, necesitarás esas clases.
# class FractalTensor: ...
# class KnowledgeBase: ...

class Extender:
    """
    Realiza el proceso inverso y generativo. Toma una intención abstracta
    y, usando el conocimiento de la KnowledgeBase, desarrolla o "despliega"
    los FractalTensors concretos que manifestarán esa intención.
    """
    def __init__(self, knowledge_base):
        if not knowledge_base:
            raise ValueError("El Extender requiere una KnowledgeBase para operar.")
        self.knowledge_base = knowledge_base
        self.phi = (1 + 5**0.5) / 2  # Proporción Áurea (phi ≈ 1.618)
        print("Extender inicializado. Listo para generar a partir del conocimiento.")

    def develop_from_synthesis(self, intention_vector, target_context=None):
        """
        Genera una respuesta (un FractalTensor) a partir de una 'intención'.
        Usa la rotación áurea como criterio de desempate cuando hay múltiples
        opciones válidas en la KnowledgeBase.
        
        Args:
            intention_vector (list): El MetaM deseado que actúa como intención.
            target_context (any, opcional): Información adicional que puede
                                           ayudar a seleccionar el mejor candidato.
        
        Returns:
            Un nuevo FractalTensor generado.
        """
        print(f"\nExtender: Recibida intención de desarrollar para el MetaM -> {intention_vector}")
        
        # 1. Buscar en la memoria todas las experiencias que llevaron a esta intención
        candidate_entries = self.knowledge_base.find_by_synthesis(intention_vector)
        
        if not candidate_entries:
            print("  -> Conocimiento no encontrado. Generando respuesta por defecto.")
            return FractalTensor(nivel_3=[[1,0,1], [0,1,0], [1,1,0]])

        print(f"  -> Encontrados {len(candidate_entries)} candidatos en la KnowledgeBase.")
        
        # 2. Seleccionar la mejor respuesta
        if len(candidate_entries) > 1 and target_context is not None:
            # 2a. Si hay múltiples candidatos y un contexto, usar rotación áurea
            print("  -> Aplicando rotación áurea para seleccionar el mejor candidato...")
            
            # La rotación ideal se calcula en base a phi
            # Esta es una implementación conceptual. La métrica de "puntuación"
            # dependería de la naturaleza del `target_context`.
            index_ideal = int(round(len(candidate_entries) / self.phi)) % len(candidate_entries)
            best_entry = candidate_entries[index_ideal]
            
            print(f"  -> Candidato #{index_ideal} seleccionado por criterio áureo.")

        else:
            # 2b. Si solo hay un candidato o no hay contexto, se usa el primero
            best_entry = candidate_entries[0]
            print("  -> Seleccionado el candidato más directo.")

        # 3. Desarrollar la respuesta a partir de la experiencia seleccionada
        # La respuesta generada son los inputs que originalmente produjeron la intención.
        response_vectors = [
            best_entry['inputs']['A'],
            best_entry['inputs']['B'],
            best_entry['inputs']['C']
        ]
        print("  -> Desarrollando FractalTensor a partir de la experiencia almacenada.")
        
        return FractalTensor(nivel_3=response_vectors)

# --- Ejemplo de uso del Extender ---
if __name__ == "__main__":
    
    # 1. Crear una KnowledgeBase y poblarla con algunas experiencias
    kb = KnowledgeBase()
    # Experiencia 1 que lleva a MetaM [1,1,0]
    kb.store_computation({
        'inputs': {'A': [1,0,1], 'B': [0,1,0], 'C': [1,1,1]},
        'outputs': {'S_emergent': [0,0,1], 'MetaM': [1,1,0]}
    })
    # Experiencia 2 que también lleva a MetaM [1,1,0]
    kb.store_computation({
        'inputs': {'A': [0,0,0], 'B': [0,0,0], 'C': [1,1,0]},
        'outputs': {'S_emergent': [1,1,1], 'MetaM': [1,1,0]}
    })
    
    # 2. Iniciar el Extender con la KnowledgeBase poblada
    extender = Extender(kb)
    
    # 3. Darle una intención para que genere una respuesta
    intencion_deseada = [1, 1, 0]
    
    # Caso A: Sin contexto, elegirá el primer candidato
    print("\n--- CASO A: Generación sin contexto ---")
    tensor_generado_A = extender.develop_from_synthesis(intencion_deseada)
    print(f"  >> Respuesta generada: {tensor_generado_A}")
    
    # Caso B: Con contexto, usaría la rotación áurea (aquí simulado)
    print("\n--- CASO B: Generación con contexto ---")
    tensor_generado_B = extender.develop_from_synthesis(intencion_deseada, target_context="optimizar_para_B")
    print(f"  >> Respuesta generada: {tensor_generado_B}")
    # -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Prueba Final del Sistema Completo
# Demuestra el ciclo de Aprendizaje y Operación Inteligente.
# PRE-REQUISITO: Las clases Trigate, FractalTensor, Transcender,
# KnowledgeBase, Evolver, y Extender deben estar definidas previamente.
# ----------------------------------------------------------------------

import random
import math

class AuroraSystem:
    """
    Orquesta todos los componentes del sistema para gestionar su ciclo de vida:
    primero aprende de la experiencia y luego opera de forma generativa.
    """
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        # El Evolver no se usa activamente en esta demo simplificada,
        # pero en un sistema completo, se encargaría de analizar y
        # guardar patrones más complejos en la KB.
        self.evolver = Evolver()
        self.transcender = Transcender()
        self.extender = Extender(self.knowledge_base)
        print("\nSistema Aurora Inicializado. Listo para operar.")

    def _generate_random_tensor(self):
        """Crea un FractalTensor con vectores aleatorios para el entrenamiento."""
        return FractalTensor(nivel_3=[[random.choice([0, 1]) for _ in range(3)] for _ in range(3)])

    def run_training_phase(self, num_epochs):
        """
        Ejecuta un ciclo de entrenamiento donde el sistema observa
        universos aleatorios para poblar su KnowledgeBase.
        """
        print("\n" + "="*60)
        print(" INICIANDO FASE DE APRENDIZAJE Y ENTRENAMIENTO ".center(60, "="))
        
        for epoch in range(num_epochs):
            print(f"\n--- Época de Entrenamiento #{epoch + 1} ---")
            
            # 1. Se genera un nuevo universo fractal para observar.
            ft_observed = self._generate_random_tensor()
            print(f"  Observando el universo fractal: {ft_observed.nivel_3}")
            
            # 2. Se procesa el universo para entender sus interacciones.
            results = self.transcender.compute_fractal_con_rotacion(ft_observed)
            
            # 3. Cada interacción observada se almacena como una "experiencia".
            if 'nivel_3' in results:
                for res in results['nivel_3']:
                    self.knowledge_base.store_computation(res)
            print(f"  Conocimiento adquirido. KB ahora tiene {len(self.knowledge_base.entries)} experiencias.")
            
        print("\n" + "="*60)
        print(" FASE DE ENTRENAMIENTO FINALIZADA ".center(60, "="))

    def run_generative_task(self, intention_vector):
        """
        Ejecuta una tarea generativa donde el Extender, guiado por la
        KnowledgeBase, crea una nueva realidad a partir de una intención.
        """
        print("\n" + "="*60)
        print(" INICIANDO FASE DE OPERACIÓN GENERATIVA ".center(60, "="))
        
        generated_tensor = self.extender.develop_from_synthesis(intention_vector)
        
        print("\nEl Extender ha desarrollado la siguiente respuesta:")
        print(f"  >> FractalTensor Generado: {generated_tensor.nivel_3}")
        return generated_tensor

# ----------------------------------------------------------------------
# --- EJECUCIÓN DE LA DEMOSTRACIÓN ---
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Crear e iniciar el sistema completo.
    aurora = AuroraSystem()
    
    # 2. Ejecutar la fase de entrenamiento.
    #    El sistema observa 5 universos para construir su base de conocimiento.
    aurora.run_training_phase(num_epochs=5)
    
    # 3. Iniciar la fase de operación inteligente.
    if aurora.knowledge_base.entries:
        # Para que la prueba sea interesante, elegimos una "intención" al azar
        # de una de las experiencias que el sistema acaba de aprender.
        random_experience = random.choice(aurora.knowledge_base.entries)
        intention = random_experience['outputs']['MetaM']
        
        # Se le pide al sistema que genere una realidad que cumpla con esa intención.
        # El sistema consultará su conocimiento para dar una respuesta informada.
        aurora.run_generative_task(intention_vector=intention)
    else:
        print("\nNo se pudo ejecutar la tarea generativa porque la KnowledgeBase está vacía.")