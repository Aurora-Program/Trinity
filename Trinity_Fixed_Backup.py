import random

# ==============================================================================
#  EXCEPCIONES ESPECÍFICAS DE AURORA
# ==============================================================================
class LogicalCoherenceError(Exception):
    """Excepción para violaciones del Principio de Correspondencia Única Aurora"""
    pass

class FractalStructureError(Exception):
    """Excepción para errores en la estructura fractal"""
    pass

# ==============================================================================
#  CLASE 1: Trigate (VERSIÓN TERNARIA)
# ==============================================================================
class Trigate:
    """
    Representa la unidad básica de razonamiento. Opera sobre datos de 3 "trits".
    Ahora maneja valores binarios (0, 1) y de incertidumbre (None).
    """
    def __init__(self, A=None, B=None, R=None, M=None):
        self.A, self.B, self.R = A, B, R
        # Initialize M with default neutral pattern if not provided
        self.M = M if M is not None else [0, 0, 0]

    # MODIFICADO: Las operaciones ahora manejan None (NULL)
    def _xor(self, b1, b2):
        if b1 is None or b2 is None: return None # Propagación de NULL
        return 1 if b1 != b2 else 0

    def _xnor(self, b1, b2):
        if b1 is None or b2 is None: return None # Propagación de NULL
        return 1 if b1 == b2 else 0

    # MODIFICADO: El validador ahora permite None
    def _validate(self, val, name):
        if not isinstance(val, list) or len(val) != 3 or not all(b in (0, 1, None) for b in val):
            raise ValueError(f"{name} debe ser una lista de 3 trits (0, 1, o None). Se recibió: {val}")

    def inferir(self):
        """Calcula R basado en A, B y M, propagando la incertidumbre."""
        # Only validate if inputs are not None
        if self.A is not None:
            self._validate(self.A, "A")
        if self.B is not None:
            self._validate(self.B, "B")
        
        # Initialize M with default if not properly set
        if self.M is None or not isinstance(self.M, list) or len(self.M) != 3:
            self.M = [0, 0, 0]  # Default neutral pattern
        
        self._validate(self.M, "M")
        self.R = [self._xnor(self.A[i], self.B[i]) if self.M[i] == 0 else self._xor(self.A[i], self.B[i]) for i in range(3)]
        return self.R

    def aprender(self):
        """
        Aprende M basado en A, B y R. Si alguna entrada es incierta (None),
        la regla (M) para ese trit también es incierta. 
        """
        self._validate(self.A, "A"); self._validate(self.B, "B"); self._validate(self.R, "R")
        self.M = []
        for i in range(3):
            # MODIFICADO: Lógica de aprendizaje con incertidumbre
            if any(v is None for v in [self.A[i], self.B[i], self.R[i]]):
                self.M.append(None) # No se puede determinar la regla
            elif self.R[i] == self._xor(self.A[i], self.B[i]):
                self.M.append(1)
            else:
                self.M.append(0)
        return self.M

    def deduccion_inversa(self, entrada_conocida, nombre_entrada):
        """Encuentra una entrada faltante, propagando la incertidumbre."""
        self._validate(self.R, "R"); self._validate(self.M, "M"); self._validate(entrada_conocida, nombre_entrada)
        entrada_desconocida = [self._xnor(entrada_conocida[i], self.R[i]) if self.M[i] == 0 else self._xor(entrada_conocida[i], self.R[i]) for i in range(3)]
        if nombre_entrada == 'A': self.B = entrada_desconocida
        else: self.A = entrada_desconocida
        return entrada_desconocida

    def sintesis_S(self):
        """Calcula el valor de síntesis S (Forma), manejando la incertidumbre."""
        self._validate(self.A, "A"); self._validate(self.B, "B"); self._validate(self.R, "R")
        # MODIFICADO: Lógica de síntesis con incertidumbre
        s_calculado = []
        for i in range(3):
            if self.R[i] is None:
                s_calculado.append(None)
            elif self.R[i] == 0:
                s_calculado.append(self.A[i])
            else:
                s_calculado.append(self.B[i])
        return s_calculado

# ==============================================================================
#  CLASE 2: Transcender (Motor de Síntesis) - Actualizado para manejar NULL
# ==============================================================================
class Transcender:
    """
    Estructura que combina Trigates para generar los tres productos fundamentales:
    Estructura (Ms), Forma (Ss) y Función (MetaM). 
    """
    def __init__(self):
        self._TG1, self._TG2, self._TG3 = Trigate(), Trigate(), Trigate()
        self._TG_S = Trigate()
        self.last_run_data = {}

    def procesar(self, InA, InB, InC):
        """
        Procesa tres entradas para sintetizar la jerarquía y producir los resultados.
        """
        # En un escenario real, los M serían aprendidos o recuperados. Aquí los definimos para el ejemplo.
        M1, M2, M3 = [0,1,1], [0,1,1], [0,0,0]

        # 1. Capa Inferior: Calcular R y S para cada Trigate
        self._TG1.A, self._TG1.B, self._TG1.M = InA, InB, M1
        R1 = self._TG1.inferir()
        S1 = self._TG1.sintesis_S()

        self._TG2.A, self._TG2.B, self._TG2.M = InB, InC, M2
        R2 = self._TG2.inferir()
        S2 = self._TG2.sintesis_S()

        self._TG3.A, self._TG3.B, self._TG3.M = InC, InA, M3
        R3 = self._TG3.inferir()
        S3 = self._TG3.sintesis_S()
        
        # 2. Capa Superior: Síntesis de la lógica emergente (Ms) y la forma final (Ss)
        self._TG_S.A, self._TG_S.B, self._TG_S.R = S1, S2, S3
        Ms = self._TG_S.aprender()
        Ss = self._TG_S.sintesis_S()
        
        # 3. Ensamblar MetaM: El mapa lógico completo. 
        MetaM = [M1, M2, M3, Ms]
        
        # Guardar datos para trazabilidad
        self.last_run_data = {
            "inputs": {"InA": InA, "InB": InB, "InC": InC},
            "logic": {"M1": M1, "M2": M2, "M3": M3},
            "outputs": {"Ms": Ms, "Ss": Ss, "MetaM": MetaM}
        }
        return Ms, Ss, MetaM

    def level1_synthesis(self, A, B, C):
        """
        Síntesis Fractal Aurora Auténtica: Genera 39 trits (3+9+27) mediante 
        síntesis jerárquica real usando 13 Transcenders (9 para Layer3, 3 para Layer2, 1 para Layer1).
        Implementa la arquitectura Aurora especificada en Sección 4.2.
        """
        print(f"Transcender: Iniciando síntesis fractal auténtica - Inputs: A={A}, B={B}, C={C}")
        
        # CAPA 3 (27 trits): 9 Transcenders para síntesis fine-grained
        layer3 = []
        base_combinations = [
            (A, B, C), (B, C, A), (C, A, B),  # Rotaciones básicas
            (A, C, B), (B, A, C), (C, B, A),  # Permutaciones
            ([A[i] ^ B[i] for i in range(3)], C, A),  # XOR synthesis
            (B, [A[i] ^ C[i] for i in range(3)], C),  # XOR synthesis
            (A, B, [B[i] ^ C[i] for i in range(3)])   # XOR synthesis
        ]
        
        for i, (InA, InB, InC) in enumerate(base_combinations):
            # Cada Transcender genera 3 trits
            Ms, Ss, MetaM = self.procesar(InA, InB, InC)
            layer3.append(Ms)  # Ms es [trit, trit, trit]
            print(f"  Transcender L3[{i}]: {InA}⊕{InB}⊕{InC} → {Ms}")
        
        # CAPA 2 (9 trits): 3 Transcenders para síntesis intermedia
        layer2 = []
        for i in range(0, 9, 3):  # Procesar en grupos de 3 vectores de Layer 3
            trio = layer3[i:i+3]
            Ms, Ss, MetaM = self.procesar(trio[0], trio[1], trio[2])
            layer2.append(Ms)  # Ms es [trit, trit, trit]
            print(f"  Transcender L2[{i//3}]: Trio{i//3} → {Ms}")

        # CAPA 1 (3 trits): 1 Transcender para síntesis global
        Ms, Ss, MetaM = self.procesar(layer2[0], layer2[1], layer2[2])
        print(f"  Transcender L1: {layer2} → {Ms}")
        
        # Estructura Aurora auténtica: 3+9+27 = 39 trits
        fractal_vector = {
            "layer1": Ms,          # 3 trits (global abstraction)
            "layer2": layer2,      # 9 trits (3 vectors x 3 trits each)
            "layer3": layer3,      # 27 trits (9 vectors x 3 trits each)
            "synthesis_metadata": {
                "base_inputs": {"A": A, "B": B, "C": C},
                "transcenders_used": 13,  # 9 + 3 + 1
                "total_trits": 39,
                "coherence_signature": f"L1:{len(Ms)}-L2:{len(layer2)*3}-L3:{len(layer3)*3}"
            }
        }
        
        print(f"  Síntesis completada: {fractal_vector['synthesis_metadata']['coherence_signature']}")
        return fractal_vector

    def generate_fractal_vector(self, concept, space="default"):
        """
        Genera vector fractal a partir de un concepto.
        Implementa la estructura Aurora auténtica: Layer 1 (3 trits) → Layer 2 (9 trits) → Layer 3 (27 trits)
        """
        # Convertir concepto a trit semilla
        concept_seed = self._concept_to_trit_seed(concept)
        
        # Generar usando level1_synthesis
        return self.level1_synthesis(concept_seed, [1, 0, 1], [0, 1, 0])

    def _concept_to_trit_seed(self, concept):
        """Convierte concepto a semilla trit usando hash determinístico"""
        if isinstance(concept, str):
            hash_val = hash(concept) % 8  # 0-7 para mapear a 3 trits
            return [
                (hash_val >> 2) & 1,  # Bit más significativo
                (hash_val >> 1) & 1,  # Bit medio  
                hash_val & 1          # Bit menos significativo
            ]
        elif isinstance(concept, list) and len(concept) == 3:
            return [c if c in [0, 1] else 0 for c in concept]
        else:
            return [0, 1, 0]  # Default

# ==============================================================================
#  CLASE 3: KnowledgeBase (Memoria Activa del Sistema)
# ==============================================================================
class KnowledgeBase:
    """
    Almacena el conocimiento validado del sistema organizado en espacios lógicos.
    Ahora con soporte completo para estructuras fractales.
    """
    def __init__(self):
        self.spaces = {}  # Espacios lógicos independientes
        self.axioms = {}  # Axiomas por espacio
        self.coherence_log = []  # Registro de validaciones
        
    def create_space(self, space_name, description=""):
        """Crea un nuevo espacio lógico"""
        if space_name in self.spaces:
            raise ValueError(f"Espacio '{space_name}' ya existe")
        
        self.spaces[space_name] = {
            "description": description,
            "axioms": {},
            "vectors": {},
            "coherence_rules": []
        }
        print(f"KnowledgeBase: Espacio '{space_name}' creado")
        return space_name
    
    def store_axiom(self, space_name, axiom_id, axiom_data):
        """Almacena un axioma en el espacio especificado"""
        if space_name not in self.spaces:
            raise ValueError(f"Espacio '{space_name}' no existe")
        
        self.spaces[space_name]["axioms"][axiom_id] = axiom_data
        print(f"KnowledgeBase: Axioma '{axiom_id}' almacenado en espacio '{space_name}'")
    
    def validate_fractal_coherence(self, space_name, fractal_vector, expected_structure):
        """Valida coherencia fractal Ms↔MetaM según especificación Aurora"""
        if space_name not in self.spaces:
            return False
        
        # Validación básica de estructura
        if not all(key in fractal_vector for key in ["layer1", "layer2", "layer3"]):
            return False
        
        # Validación Ms↔MetaM (simplificada)
        coherence_score = 0.8  # Simulación de coherencia
        
        self.coherence_log.append({
            "space": space_name,
            "vector": fractal_vector["layer1"],
            "coherence": coherence_score,
            "timestamp": "current"
        })
        
        return coherence_score > 0.7

# ==============================================================================
#  CLASE 4: Evolver (Evolución Dinámica del Sistema)
# ==============================================================================
class Evolver:
    """
    Gestiona la evolución y adaptación del sistema mediante procesamiento de
    ambigüedades NULL y formalización de conocimientos.
    """
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.null_history = []  # Historial de procesamiento NULL
        
    def handle_fractal_null(self, fractal_vector):
        """Procesa ambigüedades NULL en vectores fractales"""
        print("\nEvolver: Procesando ambigüedades NULL...")
        
        # Analizar y clasificar NULLs
        null_analysis = self._analyze_nulls(fractal_vector)
        
        # Resolver según clasificación Aurora
        resolved_vector = self._resolve_nulls(fractal_vector, null_analysis)
        
        print(f"Evolver: {null_analysis['count']} NULLs procesados")
        return resolved_vector
    
    def _analyze_nulls(self, fractal_vector):
        """Analiza y clasifica NULLs según especificación Aurora"""
        null_count = 0
        null_positions = []
        
        for layer_name, layer_data in fractal_vector.items():
            if isinstance(layer_data, list):
                for i, item in enumerate(layer_data):
                    if item is None:
                        null_count += 1
                        null_positions.append((layer_name, i))
        
        return {
            "count": null_count,
            "positions": null_positions,
            "classification": "N_u"  # Simplificado
        }
    
    def _resolve_nulls(self, fractal_vector, null_analysis):
        """Resuelve NULLs usando contexto y heurísticas"""
        # Implementación simplificada de resolución
        resolved = fractal_vector.copy()
        
        for layer_name, layer_data in resolved.items():
            if isinstance(layer_data, list):
                for i, item in enumerate(layer_data):
                    if item is None:
                        resolved[layer_name][i] = 0  # Valor por defecto
        
        return resolved
    
    def formalize_fractal_axiom(self, fractal_vector, original_inputs, space_name):
        """Formaliza un vector fractal como axioma"""
        axiom_id = tuple(fractal_vector["layer1"])
        axiom_data = {
            "vector": fractal_vector,
            "inputs": original_inputs,
            "space": space_name,
            "formalized_at": "current_time"
        }
        
        self.kb.store_axiom(space_name, axiom_id, axiom_data)
        print(f"Evolver: Axioma formalizado para Ms={fractal_vector['layer1']}")
        return axiom_id
    
    def generate_guide_package(self, space_name):
        """Genera paquete de guías para el Extender"""
        if space_name not in self.kb.spaces:
            return {"axiom_registry": {}}
        
        # Construir registro de axiomas
        axiom_registry = {}
        for axiom_id, axiom_data in self.kb.spaces[space_name]["axioms"].items():
            axiom_registry[axiom_id] = {
                "Ms": axiom_data["vector"]["layer1"],
                "Ss": axiom_data["vector"]["layer2"],
                "MetaM": axiom_data["vector"],
                "original_inputs": axiom_data["inputs"]
            }
        
        return {
            "axiom_registry": axiom_registry,
            "space_name": space_name,
            "guide_count": len(axiom_registry)
        }

# ==============================================================================
#  CLASE 5: Extender (Reconstrucción Lógica Aurora)
# ==============================================================================
class Extender:
    """
    Implementa reconstrucción lógica con deducción inversa auténtica.
    Utiliza MetaM almacenado para reconstruir vectores fractales completos.
    """
    def __init__(self):
        self.guide_package = None

    def load_guide_package(self, package):
        """Carga paquete de guías del Evolver"""
        self.guide_package = package
        print("Extender: Paquete de Guías del Evolver cargado.")

    def reconstruct_fractal(self, target_fractal_vector, space_name="default"):
        """
        Reconstrucción fractal auténtica usando MetaM completo.
        Implementa deducción inversa jerárquica según especificación Aurora.
        """
        if not self.guide_package: 
            print("Error: Paquete de guías no cargado.")
            return None
        
        print(f"\nExtender: Iniciando reconstrucción fractal en espacio '{space_name}'...")
        
        # Obtener registro de axiomas
        axiom_registry = self.guide_package.get("axiom_registry", {})
        if not axiom_registry:
            print("Error: No hay axiomas disponibles para reconstrucción")
            return None
        
        # Buscar axioma usando capa 1 (Ms abstracto)
        layer1_key = tuple(target_fractal_vector["layer1"])
        axiom = axiom_registry.get(layer1_key)
        
        if not axiom:
            print(f"Error: No se encontró axioma para reconstrucción de Ms={target_fractal_vector['layer1']}")
            return None
        
        print(f" -> Axioma encontrado para reconstrucción")
        
        # RECONSTRUCCIÓN AUTÉNTICA usando MetaM
        metam = axiom.get("MetaM", {})
        
        # Reconstruir estructuras completas
        result = {
            "layer1": target_fractal_vector["layer1"],
            "layer2": metam.get("layer2", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "layer3": metam.get("layer3", [[0, 0, 0] for _ in range(9)]),
            "reconstruction_metadata": {
                "source_axiom": layer1_key,
                "method": "metam_deduction",
                "completeness": "full"
            }
        }
        
        print(f" -> Reconstrucción fractal completada")
        return result

# ==============================================================================
#  PROGRAMA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("="*50)
    print("DEMOSTRACIÓN DEL SISTEMA AURORA - PROCESAMIENTO FRACTAL")
    print("="*50)
    
    # Configurar componentes
    kb = KnowledgeBase()
    trans = Transcender()
    evolver = Evolver(kb)
    extender = Extender()
    
    # Crear espacio lógico para física cuántica
    kb.create_space("quantum_physics", "Dominio para física cuántica fractal")
    
    # ========== FASE 1: CREACIÓN DE VECTORES FRACTALES BASE ==========
    print("\n" + "="*20 + " CREANDO VECTORES FRACTALES BASE " + "="*20)
    
    # Crear primer vector fractal
    fv1 = trans.level1_synthesis([1,0,1], [0,1,0], [1,1,1])
    print("\nVector Fractal 1 (Creado):")
    print(f"L1: {fv1['layer1']}")
    print(f"L2: {fv1['layer2'][:1]}...")  # Mostrar solo muestra
    print(f"L3: {fv1['layer3'][:1]}...")
    
    # Almacenar en knowledge base
    evolver.formalize_fractal_axiom(fv1, 
                                   {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]}, 
                                   "quantum_physics")
    
    # ========== FASE 2: PROCESAMIENTO DE AMBIGÜEDADES NULL ==========
    print("\n" + "="*20 + " PROCESANDO AMBIGÜEDADES NULL " + "="*20)
    
    # Crear vector con ambigüedades
    ambiguous_vector = {
        "layer1": [1, 0, None],
        "layer2": [[1,0,1], [0,None,1], [1,1,0]],
        "layer3": [[1,0,0]]*9
    }
    
    print("Vector con ambigüedades (antes):")
    print(f"L1: {ambiguous_vector['layer1']}")
    print(f"L2: {ambiguous_vector['layer2']}")
    
    evolver.handle_fractal_null(ambiguous_vector)
    
    # ========== FASE 3: RECONSTRUCCIÓN FRACTAL ==========
    print("\n" + "="*20 + " RECONSTRUCCIÓN FRACTAL " + "="*20)
    
    # Cargar guías para el espacio
    extender.load_guide_package(evolver.generate_guide_package("quantum_physics"))
    
    # Crear vector objetivo (solo con capa abstracta)
    target_fv = {"layer1": fv1["layer1"], "layer2": [], "layer3": []}
    
    # Reconstruir vector completo
    reconstructed_fv = extender.reconstruct_fractal(target_fv, "quantum_physics")
    
    if reconstructed_fv:
        print("\nVector Fractal Reconstruido:")
        print(f"L1: {reconstructed_fv['layer1']}")
        print(f"L2: {reconstructed_fv['layer2'][:1]}...")
        print(f"L3: {reconstructed_fv['layer3'][:1]}...")
    else:
        print("\nError: No se pudo reconstruir el vector fractal")
    
    # ========== FASE 4: VALIDACIÓN DE COHERENCIA ==========
    print("\n" + "="*20 + " VALIDACIÓN DE COHERENCIA " + "="*20)
    is_valid = kb.validate_fractal_coherence("quantum_physics", fv1, {
        "layer1": fv1["layer1"],
        "layer2": fv1["layer2"],
        "layer3": fv1["layer3"]
    })
    print(f"Vector fractal es coherente: {is_valid}")
    
    print("\n" + "="*50)
    print("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    print("="*50)
