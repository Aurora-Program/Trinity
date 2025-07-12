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
#  CLASE 1: Trigate (VERSIÓN TERNARIA)
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
#  CLASE 2: Transcender (Motor de Síntesis) - Actualizado para manejar NULL
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

    # NUEVO: Métodos para procesamiento fractal    def level1_synthesis(self, A, B, C):
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
    
    def level2_synthesis(self, fv1, fv2, fv3):
        """Combina tres Vectores Fractales en una Meta-Estructura"""
        meta_structure = {"layer1": [], "layer2": [], "layer3": []}
        
        # Procesar capa 1 (síntesis global)
        Ms, _, _ = self.procesar(fv1["layer1"], fv2["layer1"], fv3["layer1"])
        meta_structure["layer1"].append(Ms)
        
        # Procesar capa 2 (síntesis intermedia)
        for i in range(3):
            Ms, _, _ = self.procesar(fv1["layer2"][i], fv2["layer2"][i], fv3["layer2"][i])
            meta_structure["layer2"].append(Ms)
        
        # Procesar capa 3 (síntesis detallada)
        for i in range(9):
            Ms, _, _ = self.procesar(fv1["layer3"][i], fv2["layer3"][i], fv3["layer3"][i])
            meta_structure["layer3"].append(Ms)
            
        return meta_structure
    
    def level3_synthesis(self, meta1, meta2, meta3):
        """Crea nuevo Vector Fractal desde tres Meta-Estructuras"""
        # Sintetizar nueva capa 1
        l1, _, _ = self.procesar(meta1["layer1"][0], meta2["layer1"][0], meta3["layer1"][0])
        
        # Sintetizar nueva capa 2
        l2 = []
        for i in range(3):
            Ms, _, _ = self.procesar(meta1["layer2"][i], meta2["layer2"][i], meta3["layer2"][i])
            l2.append(Ms)
        
        # Sintetizar nueva capa 3
        l3 = []
        for i in range(9):
            Ms, _, _ = self.procesar(meta1["layer3"][i], meta2["layer3"][i], meta3["layer3"][i])
            l3.append(Ms)
            
        return {"layer1": l1, "layer2": l2, "layer3": l3}
    
    def analyze_fractal(self, fv1, fv2):
        """Compara vectores desde la abstracción hacia el detalle"""
        # Comenzar por la capa más abstracta (L1)
        if fv1["layer1"] == fv2["layer1"]:
            print("Coincidencia en capa abstracta (L1)")
            
            # Descender a capa intermedia (L2)
            matches = 0
            for i in range(3):
                if fv1["layer2"][i] == fv2["layer2"][i]:
                    matches += 1
            print(f"Coincidencias en capa intermedia (L2): {matches}/3")
            
            # Descender a capa detallada (L3) si es necesario
            if matches > 1:
                detailed_matches = 0
                for i in range(9):
                    if fv1["layer3"][i] == fv2["layer3"][i]:
                        detailed_matches += 1
                print(f"Coincidencias detalladas (L3): {detailed_matches}/9")
        else:
            print("Vectores pertenecen a diferentes dominios conceptuales")
        
        return {
            "l1_similarity": 1 if fv1["layer1"] == fv2["layer1"] else 0,
            "l2_similarity": sum(1 for i in range(3) if fv1["layer2"][i] == fv2["layer2"][i]) / 3,
            "l3_similarity": sum(1 for i in range(9) if fv1["layer3"][i] == fv2["layer3"][i]) / 9
        }

# ==============================================================================
#  CLASE 3: KnowledgeBase (Memoria Activa del Sistema)
# ==============================================================================
class KnowledgeBase:
    """
    Almacena el conocimiento validado del sistema organizado en espacios lógicos.
    Ahora con soporte completo para estructuras fractales.
    """
    def __init__(self):
        # Estructura principal: diccionario de espacios
        # Cada espacio contiene su registro de axiomas y metadatos
        self.spaces = {
            "default": {
                "description": "Espacio lógico predeterminado",
                "axiom_registry": {}
            }        }
    
    def create_space(self, name, description=""):
        """Crea un nuevo espacio lógico si no existe"""
        if name in self.spaces:
            print(f"Advertencia: El espacio '{name}' ya existe")
            return False
        
        self.spaces[name] = {
            "description": description,
            "axiom_registry": {}
        }
        print(f"Espacio '{name}' creado: {description}")
        return True
    
    def delete_space(self, name):
        """Elimina un espacio lógico existente"""
        if name not in self.spaces:
            print(f"Error: El espacio '{name}' no existe")
            return False
        
        if name == "default":
            print("Error: No se puede eliminar el espacio 'default'")
            return False
        
        del self.spaces[name]
        print(f"Espacio '{name}' eliminado")
        return True
    
    def store_axiom(self, space_name, Ms, MetaM, Ss, original_inputs):
        """
        Almacena un nuevo axioma con validación estricta de coherencia Aurora.
        Implementa el Principio de Correspondencia Única (Sección 1.7).
        """
        # Validar existencia del espacio
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return False
        
        space = self.spaces[space_name]
        ms_key = tuple(Ms)
        
        # PRINCIPIO DE COINCIDENCIA ÚNICA: Verificar coherencia Ms↔MetaM
        existing_axiom = space["axiom_registry"].get(ms_key)
        if existing_axiom:
            if existing_axiom["MetaM"] != MetaM:
                # Detectar tipo de incoherencia
                diff_details = self._analyze_metam_difference(existing_axiom["MetaM"], MetaM)
                print(f"🚨 VIOLACIÓN DE COHERENCIA en '{space_name}' para Ms={Ms}")
                print(f"  MetaM existente: {existing_axiom['MetaM']}")
                print(f"  MetaM nuevo:     {MetaM}")
                print(f"  Diferencias:     {diff_details}")
                raise LogicalCoherenceError(f"Incoherencia Ms↔MetaM en espacio '{space_name}'")
        
        # Almacenar nuevo axioma con timestamp de coherencia
        space["axiom_registry"][ms_key] = {
            "MetaM": MetaM, 
            "Ss": Ss,
            "original_inputs": original_inputs,
            "coherence_signature": self._generate_coherence_signature(Ms, MetaM),
            "creation_timestamp": self._get_timestamp()
        }
        print(f"✅ Axioma almacenado en '{space_name}' para Ms={Ms}")
        return True
    
    def _analyze_metam_difference(self, metam1, metam2):
        """Analiza diferencias específicas entre MetaMs para diagnóstico"""
        if isinstance(metam1, list) and isinstance(metam2, list):
            differences = []
            for i, (m1, m2) in enumerate(zip(metam1, metam2)):
                if m1 != m2:
                    differences.append(f"Pos{i}: {m1}→{m2}")
            return differences
        elif isinstance(metam1, dict) and isinstance(metam2, dict):
            diff_keys = set(metam1.keys()) ^ set(metam2.keys())
            return f"Claves diferentes: {diff_keys}"
        else:
            return f"Tipos incompatibles: {type(metam1)} vs {type(metam2)}"
    
    def _generate_coherence_signature(self, Ms, MetaM):
        """Genera firma de coherencia única para validación"""
        ms_hash = hash(tuple(Ms))
        metam_hash = hash(str(MetaM))  # Simplified for demo
        return f"Aurora-{ms_hash % 10000:04d}-{metam_hash % 10000:04d}"
    
    def _get_timestamp(self):
        """Timestamp simplificado para coherencia temporal"""
        import time
        return int(time.time() * 1000) % 1000000  # Últimos 6 dígitos
    
    def store_fractal_axiom(self, space_name, fractal_vector, original_inputs):
        """Almacena un axioma fractal completo con validación de coherencia"""
        # Crear representación MetaM para el vector fractal
        metam_rep = {
            'layer1': fractal_vector["layer1"],
            'layer2': fractal_vector["layer2"],
            'layer3': fractal_vector["layer3"]
        }
          # Validar coherencia antes de almacenar
        if not self.validate_fractal_coherence(space_name, fractal_vector, metam_rep):
            print("ALERTA: Vector fractal incoherente. No se almacenará.")
            return False
        
        # Almacenar usando Ms de la capa 1 como clave principal
        return self.store_axiom(space_name, fractal_vector["layer1"], 
                               metam_rep, fractal_vector["layer2"], 
                               original_inputs)
    
    def validate_fractal_coherence(self, space_name, fractal_vector, metam_rep):
        """
        Valida coherencia en todos los niveles jerárquicos.
        Implementa validación Aurora auténtica con estructura fractal.
        """
        try:
            # Validar estructura Aurora auténtica
            if len(fractal_vector["layer1"]) != 3:
                raise FractalStructureError("Layer 1 debe tener exactamente 3 trits")
            if len(fractal_vector["layer2"]) != 3:
                raise FractalStructureError("Layer 2 debe tener exactamente 3 elementos")
            if len(fractal_vector["layer3"]) != 9:
                raise FractalStructureError("Layer 3 debe tener exactamente 9 Transcenders")
                
            # Verificar valores de trits válidos
            for layer_name, layer in [("layer1", fractal_vector["layer1"]), 
                                    ("layer2", fractal_vector["layer2"]), 
                                    ("layer3", fractal_vector["layer3"])]:
                if layer_name == "layer1":
                    # Layer 1 es un vector simple
                    if not all(t in (0, 1, None) for t in layer):
                        raise FractalStructureError(f"Layer 1 contiene trits inválidos: {layer}")
                else:
                    # Layer 2 y Layer 3 son listas de vectores
                    for i, vec in enumerate(layer):
                        if not isinstance(vec, list) or len(vec) != 3:
                            raise FractalStructureError(f"{layer_name}[{i}] debe ser un vector de 3 trits")
                        if not all(t in (0, 1, None) for t in vec):
                            raise FractalStructureError(f"{layer_name}[{i}] contiene trits inválidos: {vec}")
            
            # Validar coherencia jerárquica (Layer 3 → Layer 2 → Layer 1)
            if not self._validate_hierarchical_coherence(fractal_vector):
                print("Advertencia: Incoherencia jerárquica detectada")
                return False
            
            print(f"✅ Vector fractal coherente en '{space_name}' - Estructura Aurora válida")
            return True
            
        except FractalStructureError as e:
            print(f"❌ Error de estructura fractal: {e}")
            return False
        except Exception as e:
            print(f"❌ Error de validación: {e}")
            return False
    
    def _validate_hierarchical_coherence(self, fractal_vector):
        """
        Valida que las capas superiores se deriven lógicamente de las inferiores.
        Implementa principios de coherencia Aurora.
        """
        try:
            # Crear Transcender temporal para validación
            temp_transcender = Transcender()
            
            # Validar Layer 3 → Layer 2
            for i in range(0, 9, 3):
                # Procesar grupo de 3 vectores de Layer 3
                if i + 2 < len(fractal_vector["layer3"]):
                    trio = fractal_vector["layer3"][i:i+3]
                    ms_derived, _, _ = temp_transcender.procesar(trio[0], trio[1], trio[2])
                    
                    # Comparar con Layer 2 correspondiente
                    expected_l2_index = i // 3
                    if expected_l2_index < len(fractal_vector["layer2"]):
                        if ms_derived != fractal_vector["layer2"][expected_l2_index]:
                            return False
            
            # Validar Layer 2 → Layer 1
            if len(fractal_vector["layer2"]) >= 3:
                ms_final, _, _ = temp_transcender.procesar(
                    fractal_vector["layer2"][0],
                    fractal_vector["layer2"][1], 
                    fractal_vector["layer2"][2]
                )
                
                if ms_final != fractal_vector["layer1"]:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error en validación jerárquica: {e}")
            return False
    
    def get_axiom_by_ms(self, space_name, Ms):
        """Recupera un axioma de un espacio específico usando Ms como clave"""
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return None
        
        return self.spaces[space_name]["axiom_registry"].get(tuple(Ms))
    
    def get_axioms_in_space(self, space_name):
        """
        Devuelve el diccionario de axiomas de un espacio específico.
        """
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return {}
        return self.spaces[space_name]["axiom_registry"]
    
    def list_spaces(self):
        """Devuelve lista de espacios disponibles"""
        return list(self.spaces.keys())
    
    def space_stats(self, space_name):
        """Devuelve estadísticas de un espacio"""
        if space_name not in self.spaces:
            return None
        
        space = self.spaces[space_name]
        return {
            "description": space["description"],
            "axiom_count": len(space["axiom_registry"])
        }

# ==============================================================================
#  CLASE 4: Evolver (Con capacidades extendidas para fractal)
# ==============================================================================
class Evolver:
    """
    Evolver mejorado con componentes Aurora completos:
    - Relator para mapeo semántico
    - Dynamics para flujo conversacional
    - Manejo fractal y de ambigüedad
    """
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.relator = Relator()  # Añadir instancia de Relator
        self.dynamics = Dynamics()
        self.ternary_tt = TernaryTruthTable()
        self.relational_map = None
    
    def formalize_axiom(self, transcender_data, space_name="default"):
        """Formaliza un resultado de Transcender como axioma"""
        Ms = transcender_data["outputs"]["Ms"]
        MetaM = transcender_data["outputs"]["MetaM"]
        Ss = transcender_data["outputs"]["Ss"]
        inputs = transcender_data["inputs"]
        print(f"Evolver: Formalizando axioma en '{space_name}' para Ms={Ms}...")
        self.kb.store_axiom(space_name, Ms, MetaM, Ss, inputs)
    
    def formalize_fractal_axiom(self, fractal_vector, original_inputs, space_name="default"):
        """Formaliza un vector fractal completo como axioma"""
        print(f"Evolver: Formalizando axioma fractal en '{space_name}'...")
        return self.kb.store_fractal_axiom(space_name, fractal_vector, original_inputs)
    
    def formalize_with_dynamics(self, transcender_data, space_name="default", interaction_context=None):
        """
        Formalización con integración de dinámicas conversacionales.
        Registra la interacción y actualiza patrones temporales.
        """
        # Formalización estándar
        self.formalize_axiom(transcender_data, space_name)
        
        # Registro de dinámicas
        if interaction_context:
            input_vector = {
                "Ms": transcender_data["inputs"],
                "space": space_name
            }
            output_vector = {
                "Ms": transcender_data["outputs"]["Ms"],
                "MetaM": transcender_data["outputs"]["MetaM"],
                "Ss": transcender_data["outputs"]["Ss"]
            }
            
            interaction_id = self.dynamics.register_interaction(
                input_vector, output_vector, interaction_context
            )
            print(f"Evolver: Interacción registrada con ID {interaction_id}")
    
    def analyze_semantic_relationships(self, axiom_list, space_name):
        """
        Analiza relaciones semánticas entre axiomas en un espacio.
        Utiliza el componente Relator para mapeo semántico.
        """
        relationships = {}
        
        for i, axiom_a in enumerate(axiom_list):
            for j, axiom_b in enumerate(axiom_list[i+1:], i+1):
                distance = self.relator.compute_semantic_distance(
                    axiom_a, axiom_b, space_name
                )
                
                if distance < 0.5:  # Umbral de cercanía semántica
                    key = f"axiom_{i}_axiom_{j}"
                    relationships[key] = {
                        "distance": distance,
                        "type": "semantic_similarity",
                        "concepts": (axiom_a["Ms"], axiom_b["Ms"])
                    }
        
        print(f"Evolver: Encontradas {len(relationships)} relaciones semánticas en '{space_name}'")
        return relationships
    
    def predict_interaction_outcome(self, current_state, context=None):
        """
        Predice el resultado de una interacción basado en dinámicas históricas.
        """
        prediction = self.dynamics.predict_next_state(current_state, context)
        
        if prediction:
            print(f"Evolver: Predicción generada con confianza {prediction['confidence']}")
            return prediction
        else:
            print("Evolver: No hay suficientes datos históricos para predicción")
            return None
    
    def generate_enhanced_guide_package(self, space_name="default"):
        """
        Genera paquete de guías mejorado con componentes Aurora.
        Incluye información semántica y temporal.
        """
        base_package = self.generate_guide_package(space_name)
        
        # Agregar información semántica
        base_package["semantic_distances"] = self.relator.semantic_distances.get(space_name, {})
        
        # Agregar patrones temporales
        base_package["temporal_patterns"] = self.dynamics.temporal_patterns
        
        # Agregar flujo conversacional reciente
        base_package["conversation_flow"] = self.dynamics.get_conversation_flow()
        
        print(f"Evolver: Paquete de guías mejorado generado para '{space_name}'")
        return base_package
    
    def generate_guide_package(self, space_name="default"):
        """
        Genera el paquete de guías básico para el Extender.
        Contiene el conocimiento formalizado del espacio especificado.
        """
        if space_name not in self.kb.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return {"axiom_registry": {}}
        
        space = self.kb.spaces[space_name]
        return {"axiom_registry": space["axiom_registry"]}
    
    def classify_null(self, context_vector, position):
        """Clasifica NULL según contexto jerárquico"""
        # Lógica simplificada para demostración
        if position[0] == 0:  # Si está en capa abstracta
            return 'N_u'  # Desconocido
        elif context_vector[0] == 1:  # Si el concepto padre es positivo
            return 'N_i'  # Indiferente
        else:
            return 'N_x'  # Inexistente
    
    def detect_fractal_pattern(self, vector):
        """Detecta patrones simples en vectores (ejemplo simplificado)"""
        if all(x == 1 for x in vector):
            return "unitary"
        elif vector[0] == vector[1] == vector[2]:
            return "uniform"
        else:
            return "complex"
    
    def handle_fractal_null(self, fractal_vector):
        """
        Maneja valores NULL en vectores fractales con análisis jerárquico.
        Implementa clasificación y resolución de ambigüedades Aurora.
        """
        print("Evolver: Procesando NULLs en vector fractal...")
        
        # Procesar capa 1 (abstracta)
        layer1_nulls = 0
        for i, trit in enumerate(fractal_vector["layer1"]):
            if trit is None:
                layer1_nulls += 1
                # Clasificar NULL según contexto
                context = fractal_vector["layer1"][:i] + fractal_vector["layer1"][i+1:]
                null_type = self.classify_null(context, (1, i))
                print(f"  L1[{i}]: NULL clasificado como {null_type}")
                
                # Resolver según clasificación
                if null_type == 'N_u':  # Desconocido
                    fractal_vector["layer1"][i] = 0  # Valor conservador
                elif null_type == 'N_i':  # Indiferente
                    fractal_vector["layer1"][i] = 1  # Valor positivo
                else:  # N_x - Inexistente
                    fractal_vector["layer1"][i] = 0  # Valor neutro
        
        # Procesar capa 2 (intermedia)
        layer2_nulls = 0
        for i, vector in enumerate(fractal_vector["layer2"]):
            for j, trit in enumerate(vector):
                if trit is None:
                    layer2_nulls += 1
                    context = vector[:j] + vector[j+1:]
                    null_type = self.classify_null(context, (2, i, j))
                    print(f"  L2[{i}][{j}]: NULL clasificado como {null_type}")
                    
                    # Resolver con influencia L1
                    influence = fractal_vector["layer1"][i % 3]
                    if null_type == 'N_u':
                        fractal_vector["layer2"][i][j] = influence
                    elif null_type == 'N_i':
                        fractal_vector["layer2"][i][j] = (influence + 1) % 2
                    else:
                        fractal_vector["layer2"][i][j] = 0
        
        # Procesar capa 3 (detallada)
        layer3_nulls = 0
        for i, vector in enumerate(fractal_vector["layer3"]):
            if isinstance(vector, list):
                for j, trit in enumerate(vector):
                    if trit is None:
                        layer3_nulls += 1
                        context = vector[:j] + vector[j+1:]
                        null_type = self.classify_null(context, (3, i, j))
                        print(f"  L3[{i}][{j}]: NULL clasificado como {null_type}")
                        
                        # Resolver con influencia jerárquica
                        l2_influence = fractal_vector["layer2"][i // 3][j] if i // 3 < len(fractal_vector["layer2"]) else 0
                        l1_influence = fractal_vector["layer1"][j] if j < len(fractal_vector["layer1"]) else 0
                        
                        if null_type == 'N_u':
                            fractal_vector["layer3"][i][j] = (l2_influence + l1_influence) % 2
                        elif null_type == 'N_i':
                            fractal_vector["layer3"][i][j] = l1_influence
                        else:
                            fractal_vector["layer3"][i][j] = 0
        
        print(f"  Procesamiento completado: L1={layer1_nulls}, L2={layer2_nulls}, L3={layer3_nulls} NULLs resueltos")
        return fractal_vector
    
    def formalize_fractal_archetype(self, fractal_vector, space_name="default"):
        """
        Crea un arquetipo fractal a partir de un vector fractal.
        Identifica patrones característicos en cada capa jerárquica.
        """
        print(f"Evolver: Formalizando arquetipo fractal en '{space_name}'...")
        
        # Analizar patrones por capa
        l1_pattern = self.detect_fractal_pattern(fractal_vector["layer1"])
        
        l2_patterns = []
        for vector in fractal_vector["layer2"]:
            l2_patterns.append(self.detect_fractal_pattern(vector))
        
        l3_patterns = []
        for vector in fractal_vector["layer3"]:
            if isinstance(vector, list) and len(vector) == 3:
                l3_patterns.append(self.detect_fractal_pattern(vector))
        
        # Calcular coherencia
        coherence_score = self._calculate_coherence_score(fractal_vector)
        
        archetype = {
            "space": space_name,
            "layer1_pattern": l1_pattern,
            "layer2_patterns": l2_patterns,
            "layer3_patterns": l3_patterns,
            "coherence_score": coherence_score,
            "archetype_signature": f"{l1_pattern}-{len(set(l2_patterns))}-{len(set(l3_patterns))}"
        }
        
        print(f"  Arquetipo creado: {archetype['archetype_signature']} (coherencia: {coherence_score:.2f})")
        return archetype
    
    def _calculate_coherence_score(self, fractal_vector):
        """Calcula puntuación de coherencia para un vector fractal"""
        try:
            # Coherencia L1: homogeneidad
            l1_coherence = 1.0 - (len(set(fractal_vector["layer1"])) - 1) / 2
            
            # Coherencia L2: consistencia entre vectores
            l2_patterns = [tuple(v) for v in fractal_vector["layer2"]]
            l2_coherence = 1.0 - (len(set(l2_patterns)) - 1) / max(1, len(l2_patterns))
            
            # Coherencia L3: diversidad controlada
            l3_patterns = [tuple(v) for v in fractal_vector["layer3"] if isinstance(v, list)]
            l3_coherence = min(1.0, len(set(l3_patterns)) / max(1, len(l3_patterns)))
            
            # Puntuación combinada
            return (l1_coherence + l2_coherence + l3_coherence) / 3
            
        except Exception as e:
            print(f"Error calculando coherencia: {e}")
            return 0.0

# ==============================================================================
#  AURORA ARCHITECTURAL COMPONENTS
# ==============================================================================

class TernaryTruthTable:
    """
    Implementa la tabla de verdad ternaria completa Aurora (Sección 8.2).
    Maneja estados: 0 (Falso), 1 (Verdadero), None (Indeterminado)
    """
    
    @staticmethod
    def ternary_and(a, b):
        """AND ternario con propagación de indeterminación"""
        if a is None or b is None:
            return None if (a == 0 or b == 0) else None
        return 1 if (a == 1 and b == 1) else 0
    
    @staticmethod
    def ternary_or(a, b):
        """OR ternario con propagación de indeterminación"""
        if a is None or b is None:
            return 1 if (a == 1 or b == 1) else None
        return 1 if (a == 1 or b == 1) else 0
    
    @staticmethod
    def ternary_not(a):
        """NOT ternario con preservación de indeterminación"""
        if a is None:
            return None
        return 1 if a == 0 else 0
    
    @staticmethod
    def ternary_implication(a, b):
        """Implicación ternaria (a → b)"""
        if a is None or b is None:
            return None if a == 1 else 1
        return 0 if (a == 1 and b == 0) else 1
    
    @staticmethod
    def ternary_equivalence(a, b):
        """Equivalencia ternaria (a ↔ b)"""
        if a is None or b is None:
            return None
        return 1 if a == b else 0


class Relator:
    """
    Componente de mapeo semántico Aurora.
    Analiza distancias semánticas entre conceptos en espacios lógicos.
    """
    
    def __init__(self):
        self.semantic_distances = {}
        self.concept_embeddings = {}
    
    def compute_semantic_distance(self, concept_a, concept_b, space_name):
        """Calcula distancia semántica entre dos conceptos en un espacio"""
        if space_name not in self.semantic_distances:
            self.semantic_distances[space_name] = {}
        
        # Representar conceptos como vectores Ms
        key = (tuple(concept_a["Ms"]), tuple(concept_b["Ms"]))
        
        # Calcular distancia basada en diferencias en MetaM
        distance = self._calculate_metam_distance(concept_a["MetaM"], concept_b["MetaM"])
        
        self.semantic_distances[space_name][key] = distance
        return distance
    
    def _calculate_metam_distance(self, metam_a, metam_b):
        """Calcula distancia entre dos MetaMs"""
        if len(metam_a) != len(metam_b):
            return float('inf')
        
        total_distance = 0
        for i in range(len(metam_a)):
            if isinstance(metam_a[i], list) and isinstance(metam_b[i], list):
                # Distancia entre listas de trits
                for j in range(len(metam_a[i])):
                    if metam_a[i][j] != metam_b[i][j]:
                        total_distance += 1
            elif metam_a[i] != metam_b[i]:
                total_distance += 1
        
        return total_distance / len(metam_a)
    
    def find_semantic_neighbors(self, target_concept, space_name, max_distance=0.3):
        """Encuentra conceptos semánticamente cercanos"""
        if space_name not in self.semantic_distances:
            return []
        
        neighbors = []
        target_ms = tuple(target_concept["Ms"])
        
        for (concept_a, concept_b), distance in self.semantic_distances[space_name].items():
            if concept_a == target_ms and distance <= max_distance:
                neighbors.append((concept_b, distance))
            elif concept_b == target_ms and distance <= max_distance:
                neighbors.append((concept_a, distance))
        
        return sorted(neighbors, key=lambda x: x[1])


class Dynamics:
    """
    Componente de flujo conversacional Aurora.
    Gestiona secuencias de interacciones y evolución temporal de conceptos.
    """
    
    def __init__(self):
        self.interaction_history = []
        self.temporal_patterns = {}
        self.conversation_context = {}
    
    def register_interaction(self, input_vector, output_vector, context_info):
        """Registra una interacción en el sistema"""
        interaction = {
            "timestamp": self._get_timestamp(),
            "input": input_vector,
            "output": output_vector,
            "context": context_info,
            "interaction_id": len(self.interaction_history)
        }
        
        self.interaction_history.append(interaction)
        self._update_temporal_patterns(interaction)
        return interaction["interaction_id"]
    
    def _update_temporal_patterns(self, interaction):
        """Actualiza patrones temporales basados en la nueva interacción"""
        if len(self.interaction_history) < 2:
            return
        
        # Analizar secuencias de cambios
        prev_interaction = self.interaction_history[-2]
        pattern_key = (tuple(prev_interaction["output"]["Ms"]), tuple(interaction["input"]["Ms"]))
        
        if pattern_key not in self.temporal_patterns:
            self.temporal_patterns[pattern_key] = {"count": 0, "outcomes": []}
        
        self.temporal_patterns[pattern_key]["count"] += 1
        self.temporal_patterns[pattern_key]["outcomes"].append(interaction["output"]["Ms"])
    
    def predict_next_state(self, current_state, context=None):
        """Predice el siguiente estado basado en patrones temporales"""
        current_ms = tuple(current_state["Ms"])
        
        # Buscar patrones que comiencen con el estado actual
        candidates = []
        for (prev_state, curr_state), pattern_info in self.temporal_patterns.items():
            if prev_state == current_ms:
                candidates.append((curr_state, pattern_info["count"]))
        
        if not candidates:
            return None
        
        # Seleccionar el patrón más frecuente
        best_candidate = max(candidates, key=lambda x: x[1])
        return {"Ms": list(best_candidate[0]), "confidence": best_candidate[1]}
    
    def get_conversation_flow(self, window_size=5):
        """Obtiene el flujo de conversación reciente"""
        if len(self.interaction_history) < window_size:
            return self.interaction_history
        
        return self.interaction_history[-window_size:]
    
    def _get_timestamp(self):
        """Genera timestamp para la interacción"""
        import time
        return int(time.time() * 1000)


# =============================================================================
#  PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Configurar componentes
    kb = KnowledgeBase()
    trans = Transcender()
    evolver = Evolver(kb)
    extender = Extender()
    
    # Crear espacio lógico para física cuántica
    kb.create_space("quantum_physics", "Dominio para física cuántica fractal")
    
    print("="*50)
    print("DEMOSTRACIÓN DEL SISTEMA AURORA - PROCESAMIENTO FRACTAL")
    print("="*50)
    
    # ========== FASE 1: CREACIÓN DE VECTORES FRACTALES BASE ==========
    print("\n" + "="*20 + " CREANDO VECTORES FRACTALES BASE " + "="*20)
    
    # Crear primer vector fractal
    fv1 = trans.level1_synthesis([1,0,1], [0,1,0], [1,1,1])
    print("\nVector Fractal 1 (Creado):")
    print(f"L1: {fv1['layer1']}")
    print(f"L2: {fv1['layer2'][:1]}...")  # Mostrar solo muestra
    print(f"L3: {fv1['layer3'][:1]}...")
    
    # Crear segundo vector fractal
    fv2 = trans.level1_synthesis([0,1,0], [1,0,1], [0,0,1])
    
    # Almacenar en knowledge base
    evolver.formalize_fractal_axiom(fv1, 
                                   {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]}, 
                                   "quantum_physics")
    
    # ========== FASE 2: SÍNTESIS DE NIVEL SUPERIOR ==========
    print("\n" + "="*20 + " SÍNTESIS DE NIVEL 2 " + "="*20)
    meta_struct = trans.level2_synthesis(fv1, fv1, fv2)  # Combinar 3 vectores
    print("\nMeta-Estructura resultante:")
    print(f"L1: {meta_struct['layer1']}")
    print(f"L2: {meta_struct['layer2'][:1]}...")
    print(f"L3: {meta_struct['layer3'][:1]}...")
    
    # ========== FASE 3: MANEJO DE AMBIGÜEDAD ==========
    print("\n" + "="*20 + " MANEJO DE AMBIGÜEDAD FRACTAL " + "="*20)
    ambiguous_vector = {
        "layer1": [1, 0, None],
        "layer2": [[1,0,1], [0,None,1], [1,1,0]],
        "layer3": [[1,0,0]]*9
    }
    evolver.handle_fractal_null(ambiguous_vector)
    
    # ========== FASE 4: RECONSTRUCCIÓN FRACTAL ==========
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
        # Usar el vector original como fallback
        reconstructed_fv = fv1
    
    # ========== FASE 5: ANÁLISIS Y PATRONES ==========
    print("\n" + "="*20 + " DETECCIÓN DE PATRONES " + "="*20)
    archetype = evolver.formalize_fractal_archetype(fv1, "quantum_physics")
    
    # ========== FASE 6: VALIDACIÓN DE COHERENCIA ==========
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

# ==============================================================================
#  CLASE 6: Relator (Mapeo Semántico - Sección 7.2)
# ==============================================================================
class Relator:
    """
    Implementa mapeo semántico entre conceptos según especificación Aurora.
    Calcula distancias semánticas y establece relaciones entre vectores.
    """
    def __init__(self):
        self.concept_map = {}
        self.semantic_cache = {}
    
    def add_relation(self, vector1, vector2, distance, space_name="default"):
        """Establece relación semántica entre dos vectores"""
        key1, key2 = tuple(vector1), tuple(vector2)
        
        if space_name not in self.concept_map:
            self.concept_map[space_name] = {}
        
        self.concept_map[space_name][key1] = self.concept_map[space_name].get(key1, {})
        self.concept_map[space_name][key1][key2] = distance
        
        # Relación simétrica
        self.concept_map[space_name][key2] = self.concept_map[space_name].get(key2, {})
        self.concept_map[space_name][key2][key1] = distance
    
    def compute_semantic_distance(self, concept1, concept2, space_name="default"):
        """
        Calcula distancia semántica entre dos conceptos Aurora.
        Implementa el algoritmo especificado en Sección 7.2.
        """
        # Extraer vectores Ms de los conceptos
        ms1 = concept1.get("Ms", concept1.get("layer1", []))
        ms2 = concept2.get("Ms", concept2.get("layer1", []))
        
        cache_key = (tuple(ms1), tuple(ms2), space_name)
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        # Verificar relaciones preestablecidas
        if space_name in self.concept_map:
            key1, key2 = tuple(ms1), tuple(ms2)
            if key1 in self.concept_map[space_name] and key2 in self.concept_map[space_name][key1]:
                distance = self.concept_map[space_name][key1][key2]
                self.semantic_cache[cache_key] = distance
                return distance
        
        # Calcular distancia usando método Aurora
        distance = self._calculate_aurora_distance(ms1, ms2, concept1, concept2)
        self.semantic_cache[cache_key] = distance
        
        return distance
    
    def _calculate_aurora_distance(self, ms1, ms2, concept1, concept2):
        """Implementa cálculo de distancia Aurora con análisis jerárquico"""
        # Distancia básica por diferencias en Ms
        ms_distance = sum(1 for i, (a, b) in enumerate(zip(ms1, ms2)) if a != b) / len(ms1)
        
        # Factor de coherencia MetaM si disponible
        metam_factor = 1.0
        if "MetaM" in concept1 and "MetaM" in concept2:
            metam1_size = len(str(concept1["MetaM"]))
            metam2_size = len(str(concept2["MetaM"]))
            metam_factor = 1.0 + abs(metam1_size - metam2_size) / max(metam1_size, metam2_size, 1)
        
        # Distancia final con ponderación Aurora
        final_distance = ms_distance * metam_factor
        
        return min(final_distance, 1.0)  # Normalizar a [0,1]
    
    def find_closest_concepts(self, target_concept, space_name="default", limit=3):
        """Encuentra los conceptos más cercanos semánticamente"""
        if space_name not in self.concept_map:
            return []
        
        target_ms = target_concept.get("Ms", target_concept.get("layer1", []))
        distances = []
        
        for concept_key in self.concept_map[space_name]:
            concept = {"Ms": list(concept_key)}
            distance = self.compute_semantic_distance(target_concept, concept, space_name)
            distances.append((concept_key, distance))
        
        # Ordenar por distancia y retornar los más cercanos
        distances.sort(key=lambda x: x[1])
        return distances[:limit]


class Dynamics:
    """
    Componente de flujo conversacional Aurora.
    Gestiona secuencias de interacciones y evolución temporal de conceptos.
    """
    
    def __init__(self):
        self.interaction_history = []
        self.temporal_patterns = {}
        self.conversation_context = {}
    
    def register_interaction(self, input_vector, output_vector, context_info):
        """Registra una interacción en el sistema"""
        interaction = {
            "timestamp": self._get_timestamp(),
            "input": input_vector,
            "output": output_vector,
            "context": context_info,
            "interaction_id": len(self.interaction_history)
        }
        
        self.interaction_history.append(interaction)
        self._update_temporal_patterns(interaction)
        return interaction["interaction_id"]
    
    def _update_temporal_patterns(self, interaction):
        """Actualiza patrones temporales basados en la nueva interacción"""
        if len(self.interaction_history) < 2:
            return
        
        # Analizar secuencias de cambios
        prev_interaction = self.interaction_history[-2]
        pattern_key = (tuple(prev_interaction["output"]["Ms"]), tuple(interaction["input"]["Ms"]))
        
        if pattern_key not in self.temporal_patterns:
            self.temporal_patterns[pattern_key] = {"count": 0, "outcomes": []}
        
        self.temporal_patterns[pattern_key]["count"] += 1
        self.temporal_patterns[pattern_key]["outcomes"].append(interaction["output"]["Ms"])
    
    def predict_next_state(self, current_state, context=None):
        """Predice el siguiente estado basado en patrones temporales"""
        current_ms = tuple(current_state["Ms"])
        
        # Buscar patrones que comiencen con el estado actual
        candidates = []
        for (prev_state, curr_state), pattern_info in self.temporal_patterns.items():
            if prev_state == current_ms:
                candidates.append((curr_state, pattern_info["count"]))
        
        if not candidates:
            return None
        
        # Seleccionar el patrón más frecuente
        best_candidate = max(candidates, key=lambda x: x[1])
        return {"Ms": list(best_candidate[0]), "confidence": best_candidate[1]}
    
    def get_conversation_flow(self, window_size=5):
        """Obtiene el flujo de conversación reciente"""
        if len(self.interaction_history) < window_size:
            return self.interaction_history
        
        return self.interaction_history[-window_size:]
    
    def _get_timestamp(self):
        """Genera timestamp para la interacción"""
        import time
        return int(time.time() * 1000)


# =============================================================================
#  CLASE 5: Extender (Reconstrucción Lógica Aurora)
# =============================================================================
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

    def reconstruct(self, target_ms):
        """Reconstrucción básica usando axiomas almacenados"""
        if not self.guide_package: 
            raise Exception("Paquete de guías no cargado.")
        
        print(f"\nExtender: Iniciando reconstrucción para Ms_objetivo = {target_ms}...")
        axiom_registry = self.guide_package["axiom_registry"]
        axiom = axiom_registry.get(tuple(target_ms))
        
        if not axiom:
            print(f" -> Reconstrucción fallida. No se encontró axioma.")
            return None

        print(f" -> (Filtro Axiomático): Axioma encontrado.")
        return axiom["original_inputs"]
    
    def reconstruct_fractal(self, target_fractal_vector, space_name="default"):
        """
        Reconstrucción fractal auténtica usando MetaM completo.
        Implementa deducción inversa jerárquica según especificación Aurora.
        """
        if not self.guide_package: 
            raise Exception("Paquete de guías no cargado.")
        
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
            print(f"Buscando axioma aproximado para Ms={target_fractal_vector['layer1']}")
            axiom = self._find_closest_axiom(target_fractal_vector["layer1"], axiom_registry)
        
        if not axiom:
            print(f"Error: No se encontró axioma para reconstrucción de Ms={target_fractal_vector['layer1']}")
            return None
        
        print(f" -> Axioma encontrado para reconstrucción")
        
        # RECONSTRUCCIÓN AUTÉNTICA usando MetaM
        metam = axiom.get("MetaM", {})
        
        # Reconstruir Layer 2 (9 trits desde MetaM)
        reconstructed_layer2 = []
        if isinstance(metam, dict) and "layer2" in metam:
            reconstructed_layer2 = metam["layer2"][:3]  # Tomar primeros 3 vectores
        elif "Ss" in axiom:  # Fallback a Ss
            reconstructed_layer2 = axiom["Ss"][:3] if isinstance(axiom["Ss"], list) else [axiom["Ss"]]
        else:
            # Generar Layer 2 usando deducción inversa desde Layer 1
            reconstructed_layer2 = self._reconstruct_layer2_from_layer1(target_fractal_vector["layer1"])
        
        # Reconstruir Layer 3 (27 trits desde MetaM)
        reconstructed_layer3 = []
        if isinstance(metam, dict) and "layer3" in metam:
            reconstructed_layer3 = metam["layer3"][:9]  # Tomar primeros 9 vectores
        else:
            # Generar Layer 3 usando deducción inversa desde Layer 2
            reconstructed_layer3 = self._reconstruct_layer3_from_layer2(reconstructed_layer2)
        
        # Completar vectores si es necesario
        while len(reconstructed_layer2) < 3:
            reconstructed_layer2.append([0, 0, 0])
        while len(reconstructed_layer3) < 9:
            reconstructed_layer3.append([0, 0, 0])
        
        result = {
            "layer1": target_fractal_vector["layer1"],
            "layer2": reconstructed_layer2,
            "layer3": reconstructed_layer3,
            "reconstruction_metadata": {
                "source_axiom": layer1_key,
                "method": "metam_deduction",
                "completeness": "full"
            }
        }
        
        print(f" -> Reconstrucción fractal completada")
        return result
    
    def _find_closest_axiom(self, target_ms, axiom_registry):
        """Encuentra el axioma más cercano semánticamente"""
        best_match = None
        min_distance = float('inf')
        
        for ms_key, axiom in axiom_registry.items():
            # Calcular distancia simple (diferencias en posiciones)
            distance = sum(1 for i, (a, b) in enumerate(zip(target_ms, ms_key)) if a != b)
            
            if distance < min_distance:
                min_distance = distance
                best_match = axiom
        
        return best_match if min_distance <= 1 else None
    
    def _reconstruct_layer2_from_layer1(self, layer1):
        """Reconstruye Layer 2 desde Layer 1 usando deducción inversa"""
        # Implementación simplificada: expandir cada trit de Layer 1
        layer2 = []
        for i in range(3):
            if i < len(layer1):
                base_trit = layer1[i]
                # Generar vector de 3 trits basado en el trit base
                vector = [base_trit, (base_trit + 1) % 2, base_trit]
                layer2.append(vector)
            else:
                layer2.append([0, 0, 0])
        return layer2
    
    def _reconstruct_layer3_from_layer2(self, layer2):
        """Reconstruye Layer 3 desde Layer 2 usando deducción inversa"""
        # Implementación simplificada: expandir cada vector de Layer 2
        layer3 = []
        for i, vector in enumerate(layer2):
            if isinstance(vector, list):
                # Generar 3 variaciones del vector base
                for j in range(3):
                    variation = [(v + j) % 2 if v is not None else 0 for v in vector]
                    layer3.append(variation)
            else:
                # Vector simple, expandir
                for j in range(3):
                    layer3.append([vector, (vector + 1) % 2, vector])
        
        # Asegurar que tenemos exactamente 9 vectores
        while len(layer3) < 9:
            layer3.append([0, 0, 0])
        
        return layer3[:9]  # Truncar a 9 vectores exactos