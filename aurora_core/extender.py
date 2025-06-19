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