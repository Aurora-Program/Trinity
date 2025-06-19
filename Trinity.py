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
    Almacena el conocimiento validado del sistema. 
    La estructura central es un registro de correspondencias unívocas entre
    la Estructura (Ms) y su Función (MetaM) y Forma (Ss). 
    """
    def __init__(self):
        self.axiom_registry = {}

    def store_axiom(self, Ms, MetaM, Ss, original_inputs):
        """Almacena una nueva regla coherente en la base de conocimiento."""
        ms_key = tuple(Ms)
        if ms_key in self.axiom_registry and self.axiom_registry[ms_key]["MetaM"] != MetaM:
            print(f"ALERTA: Incoherencia Lógica Detectada para Ms={Ms}. Se rechaza el nuevo patrón.")
            return False
        
        self.axiom_registry[ms_key] = {
            "MetaM": MetaM, 
            "Ss": Ss,
            "original_inputs": original_inputs
        }
        return True

    def get_axiom_by_ms(self, Ms):
        """Recupera un axioma usando la Estructura (Ms) como clave."""
        return self.axiom_registry.get(tuple(Ms))

# ==============================================================================
#  CLASE 4: Evolver (Motor de Formalización del Conocimiento) - Sin cambios
# ==============================================================================
class Evolver:
    """
    Analiza los resultados de múltiples Transcenders para destilar principios
    universales (Arquetipos) y formalizar el conocimiento. 
    """
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def formalize(self, transcender_run_data):
        """
        Proceso principal del Evolver. Toma la salida de un Transcender y la formaliza.
        """
        self._formalize_archetype(transcender_run_data)
        self._formalize_dynamics(transcender_run_data)
        self._formalize_relator(transcender_run_data)

    def _formalize_archetype(self, data):
        """Establece la relación Ms -> (MetaM, Ss) como un axioma en la KB. """
        Ms = data["outputs"]["Ms"]
        MetaM = data["outputs"]["MetaM"]
        Ss = data["outputs"]["Ss"]
        inputs = data["inputs"]
        print(f"Evolver (Archetype): Formalizando axioma para Ms={Ms}...")
        self.kb.store_axiom(Ms, MetaM, Ss, inputs)

    def _formalize_dynamics(self, data):
        """Placeholder: Analizaría secuencias de interacciones (Ss) a lo largo del tiempo. """
        pass

    def _formalize_relator(self, data):
        """Placeholder: Analizaría las distancias semánticas entre conceptos en un mismo espacio. """
        pass
        
    def generate_guide_package(self):
        """
        Genera el paquete de guías para el Extender, que contiene el conocimiento formalizado. 
        """
        return {"axiom_registry": self.kb.axiom_registry}

# ==============================================================================
#  CLASE 5: Extender (Motor de Reconstrucción Guiada) - Sin cambios
# ==============================================================================
class Extender:
    """
    Utiliza las leyes formalizadas por el Evolver para construir resultados
    tangibles a partir de conocimiento abstracto. 
    """
    def __init__(self):
        self.guide_package = None

    def load_guide_package(self, package):
        """Carga el paquete de conocimiento generado por el Evolver. """
        self.guide_package = package
        print("Extender: Paquete de Guías del Evolver cargado.")

    def reconstruct(self, target_ms):
        """
        Reconstruye la información detallada a partir de una Estructura (Ms) objetivo.
        """
        if not self.guide_package:
            raise Exception("El Extender no puede operar sin un Paquete de Guías.")
            
        print(f"\nExtender: Iniciando reconstrucción para Ms_objetivo = {target_ms}...")
        axiom = self.guide_package["axiom_registry"].get(tuple(target_ms))
        
        if not axiom:
            print(f"Extender: No se encontró ningún axioma para Ms={target_ms}. Reconstrucción fallida.")
            return None

        print(f"Extender (Filtro Axiomático): Axioma encontrado. MetaM={axiom['MetaM']}.")
        reconstructed_data = axiom["original_inputs"]
        print(f"Extender: Reconstrucción completada. Datos originales recuperados: {reconstructed_data}")
        return reconstructed_data

# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Prueba del sistema binario original ---
    print("="*20 + " PRUEBA DEL SISTEMA BINARIO " + "="*20)
    entradas_A = [1, 0, 1]
    entradas_B = [1, 1, 0]
    entradas_C = [0, 0, 1]

    # Fases de síntesis, formalización y aplicación
    trans = Transcender()
    Ms_calculado, Ss_calculado, MetaM_calculado = trans.procesar(entradas_A, entradas_B, entradas_C)
    print(f"Transcender ha procesado las entradas.")
    print(f"  > Ms (Estructura) generado: {Ms_calculado}")
    print(f"  > Ss (Forma) generado:      {Ss_calculado}")
    print(f"  > MetaM (Función) generado: {MetaM_calculado}")

    kb = KnowledgeBase()
    evolver = Evolver(kb)
    evolver.formalize(trans.last_run_data)
    guide_pkg = evolver.generate_guide_package()

    extender = Extender()
    extender.load_guide_package(guide_pkg)
    reconstruccion = extender.reconstruct(Ms_calculado)

    # Verificación
    if reconstruccion:
        print("\nVerificación binaria exitosa:", reconstruccion == {"InA": entradas_A, "InB": entradas_B, "InC": entradas_C})
    
    # --- 2. Prueba de lógica ternaria ---
    print("\n" + "="*20 + " PRUEBA DE LÓGICA TERNARIA " + "="*20)
    
    # Caso 1: Inferencia con NULL en la entrada
    tg_ternario_1 = Trigate(A=[1, 0, None], B=[1, 1, 0], M=[0, 1, 1])
    R_inferido = tg_ternario_1.inferir()
    print(f"Caso 1 (Inferencia): A=[1,0,None], B=[1,1,0], M=[0,1,1] => R = {R_inferido}")
    
    # Caso 2: Aprendizaje con NULL en el resultado
    tg_ternario_2 = Trigate(A=[1, 0, 1], B=[1, 1, 0], R=[0, 1, None])
    M_aprendido = tg_ternario_2.aprender()
    print(f"Caso 2 (Aprendizaje): A=[1,0,1], B=[1,1,0], R=[0,1,None] => M = {M_aprendido}")
    
    # Caso 3: Síntesis de S con NULL
    tg_ternario_3 = Trigate(A=[1, 0, 1], B=[0, 1, 0], R=[1, None, 0])
    S_sintetizado = tg_ternario_3.sintesis_S()
    print(f"Caso 3 (Síntesis S): A=[1,0,1], B=[0,1,0], R=[1,None,0] => S = {S_sintetizado}")
    
    # Caso 4: Sistema completo con incertidumbre
    print("\n" + "="*20 + " SISTEMA COMPLETO CON INCERTIDUMBRE " + "="*20)
    entradas_ternarias = {
        "InA": [1, 0, None], 
        "InB": [1, 1, 0], 
        "InC": [0, None, 1]
    }
    
    trans_ternario = Transcender()
    Ms_ternario, Ss_ternario, MetaM_ternario = trans_ternario.procesar(**entradas_ternarias)
    print(f"\nTranscender con entradas ternarias:")
    print(f"  > Ms (Estructura) generado: {Ms_ternario}")
    print(f"  > Ss (Forma) generado:      {Ss_ternario}")
    print(f"  > MetaM (Función) generado: {MetaM_ternario}")
    
    kb_ternario = KnowledgeBase()
    evolver_ternario = Evolver(kb_ternario)
    evolver_ternario.formalize(trans_ternario.last_run_data)
    
    extender_ternario = Extender()
    extender_ternario.load_guide_package(evolver_ternario.generate_guide_package())
    reconstruccion_ternaria = extender_ternario.reconstruct(Ms_ternario)
    
    print("\nVerificación ternaria:", reconstruccion_ternaria == entradas_ternarias)