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