import random

# ==============================================================================
#  CLASE 1: Trigate (VERSIÓN TERNARIA)
# ==============================================================================
class Trigate:
    """
    Representa la unidad básica de razonamiento. Opera sobre datos de 3 "trits".
    Ahora maneja valores binarios (0, 1) y de incertidumbre (None).
    """
    def __init__(self, A=None, B=None, R=None, M=None):
        self.A, self.B, self.R, self.M = A, B, R, M

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
        self._validate(self.A, "A"); self._validate(self.B, "B"); self._validate(self.M, "M")
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

    # NUEVO: Métodos para procesamiento fractal
    def level1_synthesis(self, A, B, C):
        """Crea un Vector Fractal a partir de tres vectores básicos"""
        # Capa 3 (27 dimensiones): 9 vectores de 3 trits, repetidos 3 veces cada uno
        base_vectors = [A, B, C, A, B, C, A, B, C]  # 9 vectores de 3 trits
        layer3 = base_vectors * 3  # 27 vectores de 3 trits

        # Capa 2 (9 dimensiones): síntesis intermedia
        layer2 = []
        for i in range(0, 27, 3):  # Procesar en grupos de 3 vectores
            trio = layer3[i:i+3]  # trio = [vec1, vec2, vec3], cada uno lista de 3 trits
            Ms, _, _ = self.procesar(trio[0], trio[1], trio[2])
            layer2.append(Ms)

        # Capa 1 (3 dimensiones): síntesis global
        Ms, Ss, MetaM = self.procesar(layer2[0], layer2[1], layer2[2])
        return {"layer1": Ms, "layer2": layer2, "layer3": layer3}
    
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
            }
        }
    
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
        Almacena un nuevo axioma en un espacio lógico específico.
        Verifica coherencia según el principio de correspondencia única.
        """
        # Validar existencia del espacio
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return False
        
        space = self.spaces[space_name]
        ms_key = tuple(Ms)
        
        # Verificar correspondencia única (Ms <-> MetaM)
        existing_axiom = space["axiom_registry"].get(ms_key)
        if existing_axiom and existing_axiom["MetaM"] != MetaM:
            print(f"ALERTA: Incoherencia en '{space_name}' para Ms={Ms}")
            print(f"  MetaM existente: {existing_axiom['MetaM']}")
            print(f"  MetaM nuevo:     {MetaM}")
            return False
        
        # Almacenar nuevo axioma
        space["axiom_registry"][ms_key] = {
            "MetaM": MetaM, 
            "Ss": Ss,
            "original_inputs": original_inputs
        }
        print(f"Axioma almacenado en '{space_name}' para Ms={Ms}")
        return True
    
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
        """Valida coherencia en todos los niveles jerárquicos"""
        # Validar capa 1
        if not self.get_axiom_by_ms(space_name, fractal_vector["layer1"]) is None:
            if self.get_axiom_by_ms(space_name, fractal_vector["layer1"])["MetaM"] != metam_rep:
                return False
        
        # Validar capa 2
        for i in range(3):
            axiom = self.get_axiom_by_ms(space_name, fractal_vector["layer2"][i])
            if axiom and axiom["MetaM"] != metam_rep["layer2"][i]:
                return False
        
        # Validar capa 3
        for i in range(9):
            axiom = self.get_axiom_by_ms(space_name, fractal_vector["layer3"][i])
            if axiom and axiom["MetaM"] != metam_rep["layer3"][i]:
                return False
        
        return True
    
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
            print(f"Error: Espacio '{name}' no encontrado")
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
    """Ahora incluye capacidades para manejo fractal y de ambigüedad"""
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
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
    
    def classify_null(self, context_vector, position):
        """Clasifica NULL según contexto jerárquico"""
        # Lógica simplificada para demostración
        if position[0] == 0:  # Si está en capa abstracta
            return 'N_u'  # Desconocido
        elif context_vector[0] == 1:  # Si el concepto padre es positivo
            return 'N_i'  # Indiferente
        else:
            return 'N_x'  # Inexistente
    
    def handle_fractal_null(self, fractal_vector):
        """Procesa NULLs en un vector fractal completo"""
        # Capa 1
        for i in range(3):
            if fractal_vector["layer1"][i] is None:
                null_type = self.classify_null([1,1,1], (0, i))
                print(f"NULL en L1[{i}]: {null_type}")
        
        # Capa 2
        for i in range(3):
            for j in range(3):
                if fractal_vector["layer2"][i][j] is None:
                    null_type = self.classify_null(fractal_vector["layer1"], (1, i, j))
                    print(f"NULL en L2[{i}][{j}]: {null_type}")
        
        # Capa 3
        for i in range(9):
            for j in range(3):
                if fractal_vector["layer3"][i][j] is None:
                    null_type = self.classify_null(fractal_vector["layer2"][i//3], (2, i, j))
                    print(f"NULL en L3[{i}][{j}]: {null_type}")
    
    def detect_fractal_pattern(self, vector):
        """Detecta patrones simples en vectores (ejemplo simplificado)"""
        if all(x == 1 for x in vector):
            return "unitary"
        elif vector[0] == vector[1] == vector[2]:
            return "uniform"
        else:
            return "complex"
    
    def formalize_fractal_archetype(self, fractal_vector, space_name):
        """Crea arquetipos desde patrones fractales"""
        # Identificar patrones recurrentes en capas
        layer1_pattern = self.detect_fractal_pattern(fractal_vector["layer1"])
        layer2_patterns = [self.detect_fractal_pattern(vec) for vec in fractal_vector["layer2"]]
        
        print(f"Arquetipo fractal identificado en espacio '{space_name}':")
        print(f"Patrón L1: {layer1_pattern}")
        print(f"Patrones L2: {layer2_patterns}")
        
        return {
            "layer1": layer1_pattern,
            "layer2": layer2_patterns
        }
    
    def generate_guide_package(self, space_name=None):
        """Genera paquete de guías para espacio específico o todos los espacios"""
        if space_name:
            return {
                "space": space_name,
                "axiom_registry": self.kb.get_axioms_in_space(space_name)
            }
        else:
            return {
                "all_spaces": {name: space["axiom_registry"] 
                              for name, space in self.kb.spaces.items()}
            }

# ==============================================================================
#  CLASE 5: Extender (Con capacidades extendidas para fractal)
# ==============================================================================
class Extender:
    """Ahora incluye capacidades para reconstrucción fractal"""
    def __init__(self):
        self.guide_package = None
        self.transcender = Transcender()

    def load_guide_package(self, package):
        self.guide_package = package
        print("Extender: Paquete de Guías del Evolver cargado.")

    def reconstruct(self, target_ms):
        if not self.guide_package: raise Exception("Paquete de guías no cargado.")
        
        print(f"\nExtender: Iniciando reconstrucción para Ms_objetivo = {target_ms}...")
        axiom_registry = self.guide_package["axiom_registry"]
        axiom = axiom_registry.get(tuple(target_ms))
        
        if not axiom:
            print(f" -> Reconstrucción fallida. No se encontró axioma.")
            return None

        print(f" -> (Filtro Axiomático): Axioma encontrado.")
        return axiom["original_inputs"]
    
    def reconstruct_fractal(self, target_fractal_vector, space_name="default"):
        """Reconstruye un vector fractal completo desde representación abstracta"""
        if not self.guide_package: 
            raise Exception("Paquete de guías no cargado.")
        
        # Determinar el registro de axiomas a usar
        if "space" in self.guide_package:  # Paquete de espacio único
            axiom_registry = self.guide_package["axiom_registry"]
        elif "all_spaces" in self.guide_package:  # Paquete multi-espacio
            if space_name not in self.guide_package["all_spaces"]:
                print(f"Error: Espacio '{space_name}' no disponible en paquete")
                return None
            axiom_registry = self.guide_package["all_spaces"][space_name]
        else:
            print("Error: Formato de paquete de guías inválido")
            return None
        
        # Obtener axioma principal usando capa 1
        axiom = axiom_registry.get(tuple(target_fractal_vector["layer1"]))
        if not axiom:
            print(f"Error: No se encontró axioma para Ms={target_fractal_vector['layer1']}")
            return None
        
        # Reconstruir capa 2
        reconstructed_layer2 = []
        for i in range(3):
            # En un caso real, usaríamos deducción inversa con Trigates
            # Aquí simplificamos usando los valores almacenados
            reconstructed_layer2.append(axiom["Ss"][i])
        
        # Reconstruir capa 3
        reconstructed_layer3 = []
        for i in range(9):
            # En un caso real, usaríamos deducción inversa con Trigates
            # Aquí simplificamos usando los valores almacenados
            reconstructed_layer3.append(axiom["MetaM"]["layer3"][i])
        
        return {
            "layer1": target_fractal_vector["layer1"],
            "layer2": reconstructed_layer2,
            "layer3": reconstructed_layer3
        }

# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL: DEMO COMPLETA DEL SISTEMA
# ==============================================================================
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
    print("\nVector Fractal Reconstruido:")
    print(f"L1: {reconstructed_fv['layer1']}")
    print(f"L2: {reconstructed_fv['layer2'][:1]}...")
    print(f"L3: {reconstructed_fv['layer3'][:1]}...")
    
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