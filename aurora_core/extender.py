
from transcender import Transcender 

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
