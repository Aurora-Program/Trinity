class Evolver:
    """El Evolver ahora incluye la lógica del Relator para clustering."""
    def __init__(self):
        self.relational_map = None

    def _hamming_distance(self, v1, v2):
        # Compara las capas más abstractas (L1) para una similitud fundamental
        distance = 0
        for i in range(len(v1.layer1)):
            if v1.layer1[i] != v2.layer1[i]:
                distance += 1
        return distance

    def build_relational_map(self, concepts_dict):
        """Construye un mapa de distancias entre todos los conceptos."""
        self.relational_map = {}
        letters = list(concepts_dict.keys())
        for i in range(len(letters)):
            for j in range(i, len(letters)):
                letter1, letter2 = letters[i], letters[j]
                fv1, fv2 = concepts_dict[letter1], concepts_dict[letter2]
                dist = self._hamming_distance(fv1, fv2)
                
                # Almacenar distancia simétricamente
                self.relational_map.setdefault(letter1, {})[letter2] = dist
                self.relational_map.setdefault(letter2, {})[letter1] = dist

    def discover_clusters(self, threshold=1):
        """Descubre clústeres de conceptos basados en un umbral de distancia."""
        if not self.relational_map:
            print("Error: El mapa relacional debe ser construido primero.")
            return []
        
        clusters = []
        unclustered = set(self.relational_map.keys())

        while unclustered:
            # Iniciar un nuevo clúster
            seed = unclustered.pop()
            new_cluster = {seed}
            
            # Buscar miembros para el clúster
            queue = [seed]
            while queue:
                current_letter = queue.pop(0)
                # Encontrar vecinos cercanos
                for neighbor, distance in self.relational_map[current_letter].items():
                    if neighbor in unclustered and distance <= threshold:
                        new_cluster.add(neighbor)
                        unclustered.remove(neighbor)
                        queue.append(neighbor)
            clusters.append(sorted(list(new_cluster)))
        return clusters
