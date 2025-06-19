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
class FractalVector:
    """Representa un vector de conocimiento fractal con estructura jerárquica"""
    def __init__(self, layer1, layer2, layer3):
        """
        Inicializa el vector fractal con sus tres capas:
        - layer1: 3 dimensiones (síntesis global)
        - layer2: 9 dimensiones (3 vectores de 3)
        - layer3: 27 dimensiones (9 vectores de 3)
        """
        self.layer1 = layer1  # Vector de 3 dimensiones
        self.layer2 = layer2  # Lista de 3 vectores (cada uno de 3 dimensiones)
        self.layer3 = layer3  # Lista de 9 vectores (cada uno de 3 dimensiones)
    
    def __str__(self):
        """Representación visual de la estructura fractal"""
        l1_str = f"L1: {self.layer1}"
        l2_str = f"L2: {[vec for vec in self.layer2]}"
        l3_str = f"L3: {[vec for vec in self.layer3]}"
        return f"FractalVector(\n  {l1_str}\n  {l2_str}\n  {l3_str}\n)"


class FractalProcessor:
    """Gestiona el procesamiento fractal a través de los tres niveles de síntesis"""
    def __init__(self):
        self.transcender = Transcender()
    
    def level1_synthesis(self, A, B, C):
        """
        Síntesis Nivel 1: Crea un Vector Fractal a partir de tres vectores básicos
        Documentación: 4.2. Level 1 Synthesis: Creating a Fractal Vector
        """
        # Capa 3 (27 dimensiones): 9 vectores de 3 trits, repetidos 3 veces cada uno
        base_vectors = [A, B, C, A, B, C, A, B, C]  # 9 vectores de 3 trits
        layer3 = base_vectors * 3  # 27 vectores de 3 trits

        # Capa 2 (9 dimensiones): síntesis intermedia
        layer2 = []
        for i in range(0, 27, 3):  # Procesar en grupos de 3 vectores
            trio = layer3[i:i+3]  # trio = [vec1, vec2, vec3], cada uno lista de 3 trits
            Ms, _, _ = self.transcender.procesar(trio[0], trio[1], trio[2])
            layer2.append(Ms)

        # Capa 1 (3 dimensiones): síntesis global
        Ms, Ss, MetaM = self.transcender.procesar(
            layer2[0], layer2[1], layer2[2]
        )
        return FractalVector(Ms, layer2, layer3)
    
    def level2_synthesis(self, fv1, fv2, fv3):
        """
        Síntesis Nivel 2: Combina tres Vectores Fractales en una Meta-Estructura
        Documentación: 4.3. Level 2 Synthesis: The Interaction of Fractal Vectors
        """
        meta_structure = {"layer1": [], "layer2": [], "layer3": []}
        
        # Procesar capa 1 (síntesis global)
        Ms, _, _ = self.transcender.procesar(
            fv1.layer1, fv2.layer1, fv3.layer1
        )
        meta_structure["layer1"].append(Ms)
        
        # Procesar capa 2 (síntesis intermedia)
        for i in range(3):
            Ms, _, _ = self.transcender.procesar(
                fv1.layer2[i], fv2.layer2[i], fv3.layer2[i]
            )
            meta_structure["layer2"].append(Ms)
        
        # Procesar capa 3 (síntesis detallada)
        for i in range(9):
            Ms, _, _ = self.transcender.procesar(
                fv1.layer3[i], fv2.layer3[i], fv3.layer3[i]
            )
            meta_structure["layer3"].append(Ms)
            
        return meta_structure
    
    def level3_synthesis(self, meta1, meta2, meta3):
        """
        Síntesis Nivel 3: Crea nuevo Vector Fractal desde tres Meta-Estructuras
        Documentación: 4.4. Level 3 Synthesis: The Recursive Leap to Higher Abstraction
        """
        # Sintetizar nueva capa 1
        l1 = self.transcender.procesar(
            meta1["layer1"][0], meta2["layer1"][0], meta3["layer1"][0]
        )[0]
        
        # Sintetizar nueva capa 2
        l2 = []
        for i in range(3):
            Ms, _, _ = self.transcender.procesar(
                meta1["layer2"][i], meta2["layer2"][i], meta3["layer2"][i]
            )
            l2.append(Ms)
        
        # Sintetizar nueva capa 3
        l3 = []
        for i in range(9):
            Ms, _, _ = self.transcender.procesar(
                meta1["layer3"][i], meta2["layer3"][i], meta3["layer3"][i]
            )
            l3.append(Ms)
            
        return FractalVector(l1, l2, l3)
    
    def analyze_fractal(self, fv1, fv2):
        """
        Análisis Fractal: Compara vectores desde la abstracción hacia el detalle
        Documentación: 4.5. Analysis and Extension
        """
        # Comenzar por la capa más abstracta (L1)
        if fv1.layer1 == fv2.layer1:
            print("Coincidencia en capa abstracta (L1)")
            
            # Descender a capa intermedia (L2)
            matches = 0
            for i in range(3):
                if fv1.layer2[i] == fv2.layer2[i]:
                    matches += 1
            print(f"Coincidencias en capa intermedia (L2): {matches}/3")
            
            # Descender a capa detallada (L3) si es necesario
            if matches > 1:
                detailed_matches = 0
                for i in range(9):
                    if fv1.layer3[i] == fv2.layer3[i]:
                        detailed_matches += 1
                print(f"Coincidencias detalladas (L3): {detailed_matches}/9")
        else:
            print("Vectores pertenecen a diferentes dominios conceptuales")
        
        return {
            "l1_similarity": 1 if fv1.layer1 == fv2.layer1 else 0,
            "l2_similarity": sum(1 for i in range(3) if fv1.layer2[i] == fv2.layer2[i]) / 3,
            "l3_similarity": sum(1 for i in range(9) if fv1.layer3[i] == fv2.layer3[i]) / 9
        }
# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL: DEMO DEL PROCESO FRACTAL
# ==============================================================================

if __name__ == "__main__":
    # Configurar componentes
    fp = FractalProcessor()
    kb = KnowledgeBase()
    
    # Crear espacio lógico para conceptos fractales
    kb.create_space("fractal_concepts", "Dominio para conocimiento fractal")
    
    print("="*20 + " SÍNTESIS NIVEL 1: CREANDO VECTORES BASE " + "="*20)
    # Crear vectores base (documentación: 4.2)
    fv_base1 = fp.level1_synthesis([1,0,1], [0,1,0], [1,1,1])
    fv_base2 = fp.level1_synthesis([0,1,0], [1,0,1], [0,0,1])
    fv_base3 = fp.level1_synthesis([1,1,0], [0,0,1], [1,0,0])
    
    print("\nVector Fractal 1:")
    print(fv_base1)
    
    # Almacenar en base de conocimiento
    # Guardar Ms, Ss y MetaM reales del vector fractal base
    Ms = fv_base1.layer1
    Ss = fv_base1.layer2  # O puedes definir una capa específica como forma
    MetaM = fv_base1.layer3  # O puedes definir una estructura más abstracta

    # Formalizar el axioma con los datos correctos
    # Si prefieres usar solo la capa 1 como Ms y las otras como Ss y MetaM, ajusta aquí
    # Aquí se usa layer1 como Ms, layer2 como Ss, layer3 como MetaM

    evolver = Evolver(kb)
    evolver.formalize_axiom({
        "inputs": {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]},
        "outputs": {
            "Ms": Ms,
            "Ss": Ss,
            "MetaM": MetaM
        }
    }, "fractal_concepts")
    
    print("\n" + "="*20 + " SÍNTESIS NIVEL 2: COMBINANDO VECTORES " + "="*20)
    # Crear meta-estructura (documentación: 4.3)
    meta_struct = fp.level2_synthesis(fv_base1, fv_base2, fv_base3)
    print("\nMeta-Estructura resultante:")
    print(f"L1: {meta_struct['layer1']}")
    print(f"L2: {meta_struct['layer2']}")
    print(f"L3: {meta_struct['layer3'][:2]}...")  # Mostrar solo muestra por brevedad
    
    print("\n" + "="*20 + " SÍNTESIS NIVEL 3: SALTO RECURSIVO " + "="*20)
    # Crear meta-estructuras adicionales
    meta_struct2 = fp.level2_synthesis(fv_base2, fv_base3, fv_base1)
    meta_struct3 = fp.level2_synthesis(fv_base3, fv_base1, fv_base2)
    
    # Crear nuevo vector fractal de alto nivel (documentación: 4.4)
    fv_high_level = fp.level3_synthesis(meta_struct, meta_struct2, meta_struct3)
    print("\nVector Fractal de Alto Nivel:")
    print(fv_high_level)
    
    print("\n" + "="*20 + " ANÁLISIS FRACTAL " + "="*20)
    # Comparar vectores (documentación: 4.5)
    print("\nComparando vectores base:")
    similarity = fp.analyze_fractal(fv_base1, fv_base2)
    
    print("\nComparando vector base con vector de alto nivel:")
    fp.analyze_fractal(fv_base1, fv_high_level)
    
    print("\n" + "="*20 + " RECONSTRUCCIÓN DESDE MEMORIA " + "="*20)
    # Reconstrucción usando el Extender
    extender = Extender()
    extender.load_guide_package(evolver.generate_guide_package("fractal_concepts"))
    
    target_ms = fv_base1.layer1
    reconstructed = extender.reconstruct(target_ms)
    print(f"\nReconstrucción para Ms={target_ms}:")
    print(f"Entradas originales: {reconstructed}")