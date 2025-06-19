from transcender import Transcender
from fractalVector import FractalVector





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