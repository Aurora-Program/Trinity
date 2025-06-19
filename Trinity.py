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

# ==============================================================================
#  CLASE 3: KnowledgeBase (Memoria Activa del Sistema) - Sin cambios
# ==============================================================================
class KnowledgeBase:
    """
    Almacena el conocimiento validado del sistema organizado en espacios lógicos.
    Cada espacio representa un dominio de conocimiento independiente (médico, financiero, etc.)
    con sus propias reglas y correspondencias.
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
#  CLASE 4: Evolver (VERSIÓN COMPLETA)
# ==============================================================================
class Evolver:
    """
    Analiza resultados para destilar Arquetipos, Dinámicas y Relaciones.
    """
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        # NUEVO: Almacenes para los modelos de Dinámicas y Relator
        self.dynamic_models = {}
        self.relational_maps = {}

    def formalize_axiom(self, transcender_data, space_name="default"):
        """Formaliza un único resultado de Transcender como un axioma."""
        Ms = transcender_data["outputs"]["Ms"]
        MetaM = transcender_data["outputs"]["MetaM"]
        Ss = transcender_data["outputs"]["Ss"]
        inputs = transcender_data["inputs"]
        print(f"Evolver (Archetype): Formalizando axioma en '{space_name}' para Ms={Ms}...")
        self.kb.store_axiom(space_name, Ms, MetaM, Ss, inputs)

    # NUEVO: Método para formalizar Dinámicas
    def formalize_dynamics(self, interaction_sequence, space_name="default"):
        """
        Analiza una secuencia de interacciones (representadas por sus Ss)
        y la almacena como una "coreografía" exitosa.
        """
        print(f"Evolver (Dynamics): Formalizando secuencia de interacción en '{space_name}'...")
        if space_name not in self.dynamic_models:
            self.dynamic_models[space_name] = []
        self.dynamic_models[space_name].append(interaction_sequence)

    # NUEVO: Método para construir el mapa del Relator
    def build_relational_map(self, space_name="default"):
        """
        Analiza todos los axiomas en un espacio para mapear las distancias
        conceptuales (distancia de Hamming) entre ellos.
        """
        print(f"Evolver (Relator): Construyendo mapa relacional para el espacio '{space_name}'...")
        axioms = self.kb.get_axioms_in_space(space_name)
        if len(axioms) < 2:
            print(" -> No hay suficientes axiomas para construir un mapa.")
            return

        # Extraer todos los vectores Ms y Ss del espacio
        concepts = {ms: data["Ss"] for ms, data in axioms.items()}
        map_matrix = {}

        for ms1, ss1 in concepts.items():
            distances = {}
            for ms2, ss2 in concepts.items():
                if ms1 == ms2: continue
                # Calcular distancia de Hamming
                dist = self._hamming_distance(ss1, ss2)
                distances[ms2] = dist
            map_matrix[ms1] = distances
        
        self.relational_maps[space_name] = map_matrix

    def _hamming_distance(self, v1, v2):
        """Calcula la distancia de Hamming. Un `None` se trata como una diferencia."""
        distance = 0
        for i in range(len(v1)):
            if v1[i] is None or v2[i] is None or v1[i] != v2[i]:
                if v1[i] == v2[i]: # ambos son None
                    continue
                distance += 1
        return distance
        
    # MODIFICADO: Ahora genera un paquete de guías mucho más rico
    def generate_guide_package(self, space_name):
        """Genera un paquete de guías completo para un espacio."""
        if space_name not in self.kb.spaces: return None
        return {
            "space": space_name,
            "axiom_registry": self.kb.get_axioms_in_space(space_name),
            "dynamic_model": self.dynamic_models.get(space_name, []),
            "relational_map": self.relational_maps.get(space_name, {})
        }

# ==============================================================================
#  CLASE 5: Extender (VERSIÓN MEJORADA)
# ==============================================================================
class Extender:
    def __init__(self):
        self.guide_package = None

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

    # NUEVO: Método para usar el mapa del Relator
    def suggest_alternatives(self, target_ms, max_suggestions=2):
        """
        Usa el mapa relacional para sugerir conceptos 'cercanos' al objetivo.
        """
        if not self.guide_package or not self.guide_package["relational_map"]:
            print("Extender (Relator): No hay mapa relacional cargado para sugerir alternativas.")
            return []
        
        print(f"Extender (Relator): Buscando alternativas para Ms = {target_ms}...")
        relational_map = self.guide_package["relational_map"]
        
        # Obtener las distancias para nuestro Ms objetivo
        distances = relational_map.get(tuple(target_ms))
        if not distances:
            print(" -> No se encontraron distancias para este concepto.")
            return []

        # Ordenar por distancia (más cercano primero)
        sorted_alternatives = sorted(distances.items(), key=lambda item: item[1])
        
        print(f" -> Conceptos cercanos encontrados: {sorted_alternatives}")
        return sorted_alternatives[:max_suggestions]
# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL: DEMO DE EVOLVER COMPLETO
# ==============================================================================
if __name__ == "__main__":
    kb = KnowledgeBase()
    evolver = Evolver(kb)
    trans = Transcender()
    
    # 1. Crear un espacio y poblarlo con varios conceptos (axiomas)
    print("="*20 + " FASE 1: POBLANDO EL ESPACIO 'creative_writing' " + "="*20)
    kb.create_space("creative_writing", "Conceptos para escritura de ficción")
    
    concepts = {
        "protagonista_heroico": {"InA": [1,1,1], "InB": [0,0,0], "InC": [1,0,1]},
        "protagonista_atormentado": {"InA": [1,1,0], "InB": [0,0,1], "InC": [1,0,0]},
        "villano_carismatico": {"InA": [0,0,0], "InB": [1,1,1], "InC": [0,1,0]},
        "villano_brutal": {"InA": [0,0,1], "InB": [1,1,0], "InC": [0,1,1]},
        "aliado_leal": {"InA": [1,0,1], "InB": [0,1,0], "InC": [1,1,1]}
    }

    # Diccionario para mapear Ms a nombres de conceptos para el reporte final
    ms_to_name_map = {}

    for name, inputs in concepts.items():
        print(f"\nProcesando concepto: '{name}'")
        ms, ss, metam = trans.procesar(**inputs)
        evolver.formalize_axiom(trans.last_run_data, "creative_writing")
        ms_to_name_map[tuple(ms)] = name

    # 2. Formalizar una secuencia de interacción (Dinámicas)
    print("\n" + "="*20 + " FASE 2: APRENDIENDO DINÁMICAS " + "="*20)
    interaction = [[1,0,0], [0,1,0], [1,1,1]]
    evolver.formalize_dynamics(interaction, "creative_writing")

    # 3. Construir el mapa conceptual (Relator)
    print("\n" + "="*20 + " FASE 3: CONSTRUYENDO MAPA RELACIONAL " + "="*20)
    evolver.build_relational_map("creative_writing")

    # 4. Usar el Extender con el conocimiento completo
    print("\n" + "="*20 + " FASE 4: USANDO EL EXTENDER MEJORADO " + "="*20)
    
    guide_pkg = evolver.generate_guide_package("creative_writing")
    extender = Extender()
    extender.load_guide_package(guide_pkg)
    
    print("\n--- Conocimiento en Paquete de Guías ---")
    print(f"Modelo de Dinámicas aprendido: {guide_pkg['dynamic_model']}")
    print(f"Mapa Relacional (distancias desde cada Ms):")
    for ms_tuple, distances in guide_pkg['relational_map'].items():
        print(f"  Desde '{ms_to_name_map.get(ms_tuple, 'desconocido')}' {ms_tuple}:")
        for alt_ms_tuple, dist in distances.items():
            alt_name = ms_to_name_map.get(alt_ms_tuple, 'desconocido')
            print(f"    -> a '{alt_name}' {alt_ms_tuple}: {dist}")
    print("----------------------------------------")
    
    # Pedir al Extender que sugiera alternativas a "protagonista_heroico"
    ms_heroico_tuple = next(ms for ms, name in ms_to_name_map.items() if name == "protagonista_heroico")
    
    suggestions = extender.suggest_alternatives(list(ms_heroico_tuple))
    
    print(f"\nSugerencias de alternativas para 'protagonista_heroico' (Ms={list(ms_heroico_tuple)}):")
    for alt_ms, dist in suggestions:
        alt_name = ms_to_name_map.get(alt_ms, 'desconocido')
        print(f" -> Concepto cercano: '{alt_name}' (Ms={list(alt_ms)}) con una distancia de {dist}")