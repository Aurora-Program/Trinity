

class Evolver:
    def analyze_dynamics(self, sequence, metric='hamming'):
        """
        Dynamics: analiza la evolución temporal de una secuencia de vectores (MetaMs, Ms, etc).
        Devuelve diferencias sucesivas y métricas de cambio.
        """
        if not sequence or len(sequence) < 2:
            return {'error': 'Se requieren al menos dos elementos para dinámica temporal.'}
        diffs = []
        for i in range(1, len(sequence)):
            a, b = sequence[i-1], sequence[i]
            if metric == 'hamming':
                diff = sum((x != y) if (x is not None and y is not None) else 0 for x, y in zip(a, b))
            elif metric == 'euclidean':
                diff = sum(((x - y) ** 2) if (x is not None and y is not None) else 0 for x, y in zip(a, b)) ** 0.5
            else:
                diff = None
            diffs.append(diff)
        return {
            'metric': metric,
            'diffs': diffs,
            'mean_change': sum(diffs)/len(diffs) if diffs else 0,
            'max_change': max(diffs) if diffs else 0,
            'min_change': min(diffs) if diffs else 0
        }
    """
    El Evolver analiza los resultados de los Transcender (y sus MetaM) para encontrar patrones, reglas y axiomas entre espacios.
    Puede operar sobre MetaMs, KnowledgeBase, o cualquier estructura jerárquica de resultados.
    Incluye lógica de Relator para analizar relaciones internas en un mismo espacio.
    """
    def __init__(self):
        self.axioms = []  # Aquí se guardarán reglas o patrones detectados

    def relate_vectors(self, vectors, context_id=None, distance_metric='hamming'):
        """
        El Relator: analiza relaciones entre vectores en el mismo espacio/contexto.
        Calcula matriz de distancias y detecta agrupamientos, opuestos y afinidades.
        Opcionalmente, context_id puede usarse para agrupar por axioma/contexto.
        distance_metric: 'hamming' (por defecto) o 'euclidean'.
        Devuelve un dict con la matriz de distancias y relaciones clave.
        """
        if not vectors or len(vectors) < 2:
            return {'error': 'Se requieren al menos dos vectores.'}
        n = len(vectors)
        dist_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if distance_metric == 'hamming':
                    dist = sum(a != b for a, b in zip(vectors[i], vectors[j]))
                elif distance_metric == 'euclidean':
                    dist = sum((a-b)**2 for a, b in zip(vectors[i], vectors[j])) ** 0.5
                else:
                    raise ValueError('Métrica no soportada')
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        # Buscar pares más cercanos y más lejanos
        min_dist = float('inf')
        max_dist = float('-inf')
        min_pairs = []
        max_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                d = dist_matrix[i][j]
                if d < min_dist:
                    min_dist = d
                    min_pairs = [(i, j)]
                elif d == min_dist:
                    min_pairs.append((i, j))
                if d > max_dist:
                    max_dist = d
                    max_pairs = [(i, j)]
                elif d == max_dist:
                    max_pairs.append((i, j))
        return {
            'context_id': context_id,
            'distance_metric': distance_metric,
            'dist_matrix': dist_matrix,
            'min_dist': min_dist,
            'min_pairs': min_pairs,
            'max_dist': max_dist,
            'max_pairs': max_pairs,
            'n_vectors': n
        }

    def resolve_ambiguity(self, metaMs_list, pattern_result, metaM_target=None):
        """
        Dada una lista de MetaMs y el resultado del análisis de patrones,
        intenta resolver ambigüedades o descartar axiomas inválidos.
        Si se provee metaM_target (esperado), verifica si algún patrón lo predice.
        Devuelve dict con decisión y justificación.
        """
        if not metaMs_list or any(m is None for m in metaMs_list):
            return {'error': 'Faltan MetaMs para analizar.'}
        justificacion = []
        # Si todos son iguales, no hay ambigüedad
        if pattern_result.get('iguales'):
            if metaM_target is not None and metaM_target != metaMs_list[0]:
                justificacion.append('Axioma de igualdad descartado: no predice el target.')
                return {'resuelto': False, 'descartado': True, 'razon': justificacion}
            justificacion.append('Axioma de igualdad aceptado.')
            return {'resuelto': True, 'axioma': f"MetaM = {metaMs_list[0]}", 'razon': justificacion}
        # Probar si alguna operación predice el target
        if metaM_target is not None:
            if pattern_result['xor'] == list(metaM_target):
                justificacion.append('Axioma XOR predice el target.')
                return {'resuelto': True, 'axioma': f"MetaM = XOR({metaMs_list})", 'razon': justificacion}
            if pattern_result['and'] == list(metaM_target):
                justificacion.append('Axioma AND predice el target.')
                return {'resuelto': True, 'axioma': f"MetaM = AND({metaMs_list})", 'razon': justificacion}
            if pattern_result['or'] == list(metaM_target):
                justificacion.append('Axioma OR predice el target.')
                return {'resuelto': True, 'axioma': f"MetaM = OR({metaMs_list})", 'razon': justificacion}
            justificacion.append('Ningún axioma simple predice el target. Ambigüedad no resuelta.')
            return {'resuelto': False, 'descartado': True, 'razon': justificacion}
        # Si no hay target, solo reportar ambigüedad
        justificacion.append('Ambigüedad detectada: no hay patrón dominante.')
        return {'resuelto': False, 'descartado': False, 'razon': justificacion}
    """
    El Evolver analiza los resultados de los Transcender (y sus MetaM) para encontrar patrones, reglas y axiomas entre espacios.
    Puede operar sobre MetaMs, KnowledgeBase, o cualquier estructura jerárquica de resultados.
    """
    def __init__(self):
        self.axioms = []  # Aquí se guardarán reglas o patrones detectados

    def analyze_metaMs(self, metaMs_list):
        """
        Analiza una lista de MetaMs (cada uno vector de 3 bits) y busca patrones simples: igualdad, XOR, AND, OR, etc.
        Devuelve un dict con los resultados del análisis.
        """
        if not metaMs_list or any(m is None for m in metaMs_list):
            return {'error': 'Faltan MetaMs para analizar.'}
        iguales = all(m == metaMs_list[0] for m in metaMs_list)
        xor = [
            None if (metaMs_list[0][i] is None or metaMs_list[1][i] is None or metaMs_list[2][i] is None)
            else (metaMs_list[0][i] ^ metaMs_list[1][i] ^ metaMs_list[2][i])
            for i in range(3)
        ]
        anded = [
            None if (metaMs_list[0][i] is None or metaMs_list[1][i] is None or metaMs_list[2][i] is None)
            else (metaMs_list[0][i] & metaMs_list[1][i] & metaMs_list[2][i])
            for i in range(3)
        ]
        ored = [
            None if (metaMs_list[0][i] is None or metaMs_list[1][i] is None or metaMs_list[2][i] is None)
            else (metaMs_list[0][i] | metaMs_list[1][i] | metaMs_list[2][i])
            for i in range(3)
        ]
        return {
            'iguales': iguales,
            'xor': xor,
            'and': anded,
            'or': ored
        }

    def formalize_axiom(self, pattern_result, metaMs_list):
        """
        Formaliza un axioma a partir de un patrón detectado entre MetaMs.
        """
        if pattern_result.get('iguales'):
            axiom = f"Todos los MetaM globales son iguales: {metaMs_list[0]}"
        else:
            axiom = f"Patrón detectado - XOR: {pattern_result['xor']}, AND: {pattern_result['and']}, OR: {pattern_result['or']}"
        self.axioms.append(axiom)
        return axiom

    def get_axioms(self):
        """Devuelve la lista de axiomas formalizados."""
        return self.axioms
