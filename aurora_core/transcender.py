from Trinity import Trigate


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
