
class Transcender:
    def __init__(self):
        self.trigate_AB = Trigate()
        self.trigate_BC = Trigate()
        self.trigate_CA = Trigate()
        self.trigate_superior = Trigate()
        self.trigate_Ms = Trigate()

    def deep_learning(self, A, B, C, M_emergent):
        self.A = list(A)
        self.B = list(B)
        self.C = list(C)
        self.M_emergent = list(M_emergent)

        R_hipotesis = []
        all_none = True
        for i in range(3):
            if self.A[i] is None or self.B[i] is None or self.M_emergent[i] is None:
                R_hipotesis.append(None)
                continue
            found = False
            for r in [0, 1]:
                A_vec = [None, None, None]
                B_vec = [None, None, None]
                R_vec = [None, None, None]
                A_vec[i] = self.A[i]
                B_vec[i] = self.B[i]
                R_vec[i] = r
                result = self.trigate_superior.synthesize(A_vec, B_vec, R_vec)
                if isinstance(result, tuple) and len(result) == 2:
                    m, _ = result
                else:
                    m = result
                if (isinstance(m, list) and m[i] == self.M_emergent[i]) or (not isinstance(m, list) and m == self.M_emergent[i]):
                    R_hipotesis.append(r)
                    found = True
                    all_none = False
                    break
            if not found:
                R_hipotesis.append(None)

        if all_none:
            return None

        if any(x is None for x in self.M_emergent) or all(x is None for x in R_hipotesis):
            return None

        self.S_emergent = []
        for i in range(3):
            if self.A[i] is None or self.B[i] is None or R_hipotesis[i] is None:
                self.S_emergent.append(None)
                continue
            A_vec = [None, None, None]
            B_vec = [None, None, None]
            R_vec = [None, None, None]
            A_vec[i] = self.A[i]
            B_vec[i] = self.B[i]
            R_vec[i] = R_hipotesis[i]
            result = self.trigate_superior.synthesize(A_vec, B_vec, R_vec)
            if isinstance(result, tuple) and len(result) == 2:
                _, s_em = result
            else:
                s_em = result
            if isinstance(s_em, list):
                self.S_emergent.append(s_em[i])
            else:
                self.S_emergent.append(s_em)

        self.M_AB, self.S_AB = [], []
        self.M_BC, self.S_BC = [], []
        self.M_CA, self.S_CA = [], []
        for i in range(3):
            if self.A[i] is None or self.B[i] is None or self.C[i] is None or R_hipotesis[i] is None:
                self.M_AB.append(None)
                self.S_AB.append(None)
                self.M_BC.append(None)
                self.S_BC.append(None)
                self.M_CA.append(None)
                self.S_CA.append(None)
                continue
            A_vec = [None, None, None]
            B_vec = [None, None, None]
            C_vec = [None, None, None]
            R_vec = [None, None, None]
            A_vec[i] = self.A[i]
            B_vec[i] = self.B[i]
            C_vec[i] = self.C[i]
            R_vec[i] = R_hipotesis[i]
            result_ab = self.trigate_AB.synthesize(A_vec, B_vec, R_vec)
            result_bc = self.trigate_BC.synthesize(B_vec, C_vec, R_vec)
            result_ca = self.trigate_CA.synthesize(C_vec, A_vec, R_vec)
            if isinstance(result_ab, tuple) and len(result_ab) == 2:
                m_ab, s_ab = result_ab
            else:
                m_ab, s_ab = result_ab, None
            if isinstance(result_bc, tuple) and len(result_bc) == 2:
                m_bc, s_bc = result_bc
            else:
                m_bc, s_bc = result_bc, None
            if isinstance(result_ca, tuple) and len(result_ca) == 2:
                m_ca, s_ca = result_ca
            else:
                m_ca, s_ca = result_ca, None
            self.M_AB.append(m_ab[i] if isinstance(m_ab, list) else m_ab)
            self.S_AB.append(s_ab[i] if isinstance(s_ab, list) else s_ab)
            self.M_BC.append(m_bc[i] if isinstance(m_bc, list) else m_bc)
            self.S_BC.append(s_bc[i] if isinstance(s_bc, list) else s_bc)
            self.M_CA.append(m_ca[i] if isinstance(m_ca, list) else m_ca)
            self.S_CA.append(s_ca[i] if isinstance(s_ca, list) else s_ca)

        self.M_intermediate = []
        for i in range(3):
            if self.M_AB[i] is None or self.M_BC[i] is None or R_hipotesis[i] is None:
                self.M_intermediate.append(None)
                continue
            MAB_vec = [None, None, None]
            MBC_vec = [None, None, None]
            R_vec = [None, None, None]
            MAB_vec[i] = self.M_AB[i]
            MBC_vec[i] = self.M_BC[i]
            R_vec[i] = R_hipotesis[i]
            result_int = self.trigate_Ms.synthesize(MAB_vec, MBC_vec, R_vec)
            if isinstance(result_int, tuple) and len(result_int) == 2:
                m_int, _ = result_int
            else:
                m_int = result_int
            self.M_intermediate.append(m_int[i] if isinstance(m_int, list) else m_int)

        self.MetaM = [((int(a) ^ int(b)) if a is not None and b is not None else None)
                      for a, b in zip(self.M_intermediate, self.M_emergent)]
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'M_AB': self.M_AB,
            'S_AB': self.S_AB,
            'M_BC': self.M_BC,
            'S_BC': self.S_BC,
            'M_CA': self.M_CA,
            'S_CA': self.S_CA,
            'M_emergent': self.M_emergent,
            'S_emergent': self.S_emergent,
            'M_intermediate': self.M_intermediate,
            'MetaM': self.MetaM
        }

    def compute(self, A, B, C):
        self.A = list(A)
        self.B = list(B)
        self.C = list(C)

        self.M_AB, self.S_AB = self.trigate_AB.synthesize(self.A, self.B)
        self.M_BC, self.S_BC = self.trigate_BC.synthesize(self.B, self.C)
        self.M_CA, self.S_CA = self.trigate_CA.synthesize(self.C, self.A)

        self.M_emergent, self.S_emergent = self.trigate_superior.synthesize(self.M_AB, self.M_BC)

        interm1, _ = self.trigate_Ms.synthesize(self.M_AB, self.M_BC)
        self.M_intermediate, _ = self.trigate_Ms.synthesize(interm1, self.M_CA)

        self.MetaM = [((int(a) ^ int(b)) if a is not None and b is not None else None)
                      for a, b in zip(self.M_intermediate, self.M_emergent)]

        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'M_AB': self.M_AB,
            'S_AB': self.S_AB,
            'M_BC': self.M_BC,
            'S_BC': self.S_BC,
            'M_CA': self.M_CA,
            'S_CA': self.S_CA,
            'M_emergent': self.M_emergent,
            'S_emergent': self.S_emergent,
            'M_intermediate': self.M_intermediate,
            'MetaM': self.MetaM
        }

    def compute_fractal(self, fractal_tensor):
        resultados = {}
        for nivel, vectores in [
            ('nivel_3', fractal_tensor.nivel_3),
            ('nivel_9', fractal_tensor.nivel_9),
            ('nivel_27', fractal_tensor.nivel_27)
        ]:
            resultados[nivel] = []
            for v in vectores:
                res = self.compute(v, v, v)
                resultados[nivel].append(res)
        return resultados


class FractalTensor:
    """
    Representa un tensor fractal con 3 niveles:
    - nivel_3: lista de 3 vectores ternarios (cada uno de 3 elementos)
    - nivel_9: lista de 9 vectores ternarios (cada uno de 3 elementos)
    - nivel_27: lista de 27 vectores ternarios (cada uno de 3 elementos)
    """
    def __init__(self, nivel_3=None, nivel_9=None, nivel_27=None):
        self.nivel_3 = nivel_3 if nivel_3 is not None else [[None, None, None] for _ in range(3)]
        self.nivel_9 = nivel_9 if nivel_9 is not None else [[None, None, None] for _ in range(9)]
        self.nivel_27 = nivel_27 if nivel_27 is not None else [[None, None, None] for _ in range(27)]

    @staticmethod
    def random():
        import random
        def rand_vec():
            return [random.choice([0, 1, None]) for _ in range(3)]
        return FractalTensor(
            nivel_3=[rand_vec() for _ in range(3)],
            nivel_9=[rand_vec() for _ in range(9)],
            nivel_27=[rand_vec() for _ in range(27)]
        )

    def as_dict(self):
        return {'nivel_3': self.nivel_3, 'nivel_9': self.nivel_9, 'nivel_27': self.nivel_27}

    def __repr__(self):
        return f"FractalTensor(3={self.nivel_3}, 9={self.nivel_9}, 27={self.nivel_27})"


class Evolver:
    def __init__(self):
        self.axioms = []

    def _validate_vector_list(self, vectors, min_len, name):
        """
        Valida que 'vectors' sea una lista de al menos min_len vectores ternarios de longitud 3.
        Cada valor debe ser 0, 1 o None.
        Devuelve (True, None) si es válido, (False, mensaje) si no.
        """
        if not isinstance(vectors, (list, tuple)):
            return False, f"{name} debe ser una lista o tupla."
        if len(vectors) < min_len:
            return False, f"{name} debe tener al menos {min_len} elementos."
        for idx, v in enumerate(vectors):
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                return False, f"Elemento {idx} de {name} no es un vector de longitud 3."
            for j, val in enumerate(v):
                if val not in (0, 1, None):
                    return False, f"Valor inválido en {name}[{idx}][{j}]: {val}. Solo se permiten 0, 1 o None."
        return True, None

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

    def relate_vectors(self, vectors, context_id=None, distance_metric='hamming'):
        """
        El Relator: analiza relaciones entre vectores en el mismo espacio/contexto.
        Calcula matriz de distancias y detecta agrupamientos, opuestos y afinidades.
        Opcionalmente, context_id puede usarse para agrupar por axioma/contexto.
        distance_metric: 'hamming' (por defecto) o 'euclidean'.
        Devuelve un dict con la matriz de distancias y relaciones clave.
        """
        ok, err = self._validate_vector_list(vectors, 2, "vectors")
        if not ok:
            return {'error': err}
        n = len(vectors)
        dist_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                try:
                    if distance_metric == 'hamming':
                        dist = sum((a != b) if (a is not None and b is not None) else 0 for a, b in zip(vectors[i], vectors[j]))
                    elif distance_metric == 'euclidean':
                        dist = sum(((a-b)**2) if (a is not None and b is not None) else 0 for a, b in zip(vectors[i], vectors[j])) ** 0.5
                    else:
                        raise ValueError('Métrica no soportada')
                except Exception:
                    dist = None
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
        ok, err = self._validate_vector_list(metaMs_list, 3, "metaMs_list")
        if not ok:
            return {'error': err}
        iguales = all(m == metaMs_list[0] for m in metaMs_list)
        xor = []
        anded = []
        ored = []
        for i in range(3):
            bits = [m[i] for m in metaMs_list]
            if any(b is None for b in bits):
                xor.append(None)
                anded.append(None)
                ored.append(None)
            else:
                try:
                    x = bits[0]
                    for b in bits[1:]:
                        x ^= b
                    xor.append(x)
                    a = bits[0]
                    for b in bits[1:]:
                        a &= b
                    anded.append(a)
                    o = bits[0]
                    for b in bits[1:]:
                        o |= b
                    ored.append(o)
                except Exception:
                    xor.append(None)
                    anded.append(None)
                    ored.append(None)
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

    def analyze_dynamics_adaptive(self, sequence, metric='auto', window=3):
        """
        Dinámica adaptativa: detecta cambios de régimen, ciclos, rupturas y ajusta la métrica según la variabilidad.
        Si metric='auto', selecciona la mejor métrica según la dispersión local.
        window: tamaño de ventana para análisis local (por defecto 3).
        Devuelve dict con métricas, rupturas, ciclos y sugerencias de arquetipo.
        """
        if not sequence or len(sequence) < 2:
            return {'error': 'Se requieren al menos dos elementos para dinámica temporal.'}
        # Selección adaptativa de métrica
        if metric == 'auto':
            # Prueba ambas y elige la de mayor varianza
            hamming_diffs = [sum((x != y) if (x is not None and y is not None) else 0 for x, y in zip(sequence[i-1], sequence[i])) for i in range(1, len(sequence))]
            euclidean_diffs = [sum(((x - y) ** 2) if (x is not None and y is not None) else 0 for x, y in zip(sequence[i-1], sequence[i])) ** 0.5 for i in range(1, len(sequence))]
            var_h = (max(hamming_diffs) - min(hamming_diffs)) if hamming_diffs else 0
            var_e = (max(euclidean_diffs) - min(euclidean_diffs)) if euclidean_diffs else 0
            metric = 'hamming' if var_h >= var_e else 'euclidean'
        # Calcula diferencias
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
        # Detecta rupturas y ciclos
        rupturas = [i for i, d in enumerate(diffs) if d > (sum(diffs)/len(diffs) + 2 * (max(diffs)-min(diffs))/max(1,len(diffs)))]
        ciclos = []
        if window > 1 and len(sequence) >= window:
            for i in range(len(sequence)-window):
                if sequence[i] == sequence[i+window]:
                    ciclos.append((i, i+window))
        # Sugerencia de arquetipo: si hay estabilidad, sugerir "estático"; si hay rupturas, sugerir "dinámico".
        if len(rupturas) > 0:
            arquetipo = 'dinámico'
        elif len(ciclos) > 0:
            arquetipo = 'cíclico'
        else:
            arquetipo = 'estático'
        return {
            'metric': metric,
            'diffs': diffs,
            'rupturas': rupturas,
            'ciclos': ciclos,
            'arquetipo_sugerido': arquetipo,
            'mean_change': sum(diffs)/len(diffs) if diffs else 0,
            'max_change': max(diffs) if diffs else 0,
            'min_change': min(diffs) if diffs else 0
        }

    def relate_vectors_adaptive(self, vectors, context_id=None, distance_metric='auto', affinity_threshold=None):
        """
        Relator adaptativo: detecta clústeres, outliers y ajusta el umbral de afinidad según la dispersión.
        Si distance_metric='auto', selecciona la mejor métrica según la dispersión local.
        affinity_threshold: si None, se calcula automáticamente.
        Devuelve dict con clústeres, outliers y matriz de distancias.
        """
        if not vectors or len(vectors) < 2:
            return {'error': 'Se requieren al menos dos vectores.'}
        # Selección adaptativa de métrica
        if distance_metric == 'auto':
            hamming_diffs = []
            euclidean_diffs = []
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    h = sum(a != b for a, b in zip(vectors[i], vectors[j]))
                    e = sum(((a - b) ** 2) if (a is not None and b is not None) else 0 for a, b in zip(vectors[i], vectors[j])) ** 0.5
                    hamming_diffs.append(h)
                    euclidean_diffs.append(e)
            var_h = (max(hamming_diffs) - min(hamming_diffs)) if hamming_diffs else 0
            var_e = (max(euclidean_diffs) - min(euclidean_diffs)) if euclidean_diffs else 0
            distance_metric = 'hamming' if var_h >= var_e else 'euclidean'
        # Calcula matriz de distancias
        n = len(vectors)
        dist_matrix = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if distance_metric == 'hamming':
                    dist = sum(a != b for a, b in zip(vectors[i], vectors[j]))
                elif distance_metric == 'euclidean':
                    dist = sum(((a - b) ** 2) if (a is not None and b is not None) else 0 for a, b in zip(vectors[i], vectors[j])) ** 0.5
                else:
                    raise ValueError('Métrica no soportada')
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        # Umbral adaptativo
        all_dists = [dist_matrix[i][j] for i in range(n) for j in range(i+1, n)]
        if affinity_threshold is None:
            if all_dists:
                affinity_threshold = sum(all_dists)/len(all_dists)
            else:
                affinity_threshold = 1
        # Clustering simple: afinidad por debajo del umbral
        clusters = []
        assigned = set()
        for i in range(n):
            if i in assigned:
                continue
            cluster = [i]
            for j in range(n):
                if i != j and dist_matrix[i][j] <= affinity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            assigned.add(i)
            clusters.append(cluster)
        # Outliers: clusters de tamaño 1
        outliers = [c[0] for c in clusters if len(c) == 1]
        return {
            'context_id': context_id,
            'distance_metric': distance_metric,
            'dist_matrix': dist_matrix,
            'clusters': clusters,
            'outliers': outliers,
            'affinity_threshold': affinity_threshold,
            'n_vectors': n
        }

    def promote_archetypes(self, archetype_scores, min_score=0.7):
        """
        Promueve arquetipos (axiomas) con score alto y degrada los que fallan.
        archetype_scores: dict {arquetipo: [aciertos, total]}
        Devuelve lista de arquetipos promovidos y degradados.
        """
        promoted = []
        degraded = []
        for arch, (hits, total) in archetype_scores.items():
            score = hits/total if total > 0 else 0
            if score >= min_score:
                promoted.append((arch, score))
            elif total > 0:
                degraded.append((arch, score))
        return {'promoted': promoted, 'degraded': degraded}
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
        ok, err = self._validate_vector_list(metaMs_list, 3, "metaMs_list")
        if not ok:
            return {'error': err}
        iguales = all(m == metaMs_list[0] for m in metaMs_list)
        xor = []
        anded = []
        ored = []
        for i in range(3):
            bits = [m[i] for m in metaMs_list]
            if any(b is None for b in bits):
                xor.append(None)
                anded.append(None)
                ored.append(None)
            else:
                try:
                    x = bits[0]
                    for b in bits[1:]:
                        x ^= b
                    xor.append(x)
                    a = bits[0]
                    for b in bits[1:]:
                        a &= b
                    anded.append(a)
                    o = bits[0]
                    for b in bits[1:]:
                        o |= b
                    ored.append(o)
                except Exception:
                    xor.append(None)
                    anded.append(None)
                    ored.append(None)
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




class Extender:
    """
    El Extender reconstruye información concreta a partir de conocimiento abstracto (Ss, MetaM, axiomas, dinámicas, relaciones).
    Utiliza la dinámica para identificar el arquetipo adecuado y luego extiende (reconstruye) los tensores originales usando la información de los Transcender, pero en sentido inverso.
    """
    def __init__(self, knowledge_base=None, evolver=None):
        self.knowledge_base = knowledge_base  # Puede ser None o una instancia de KnowledgeBase
        self.evolver = evolver or Evolver()

    def extend(self, Ss, contexto=None):
        """
        Reconstruye la información detallada a partir de un input abstracto Ss.
        1. Usa la dinámica para identificar el arquetipo (axioma) relevante.
        2. Recupera el MetaM y otros datos del Evolver/KnowledgeBase.
        3. Realiza la extensión inversa: reconstruye los tensores originales (A, B, C, ...).
        4. Devuelve la reconstrucción y justificación.
        """
        # Validar que Ss tenga al menos 3 elementos y que cada uno sea vector de 3
        if not isinstance(Ss, list) or len(Ss) < 3:
            return {'error': 'Se requieren al menos 3 elementos en Ss para extensión.'}
        for idx, v in enumerate(Ss[:3]):
            if not isinstance(v, list) or len(v) != 3:
                return {'error': f'Elemento Ss[{idx}] debe ser vector de 3 elementos.'}
        # 1. Identificar arquetipo usando dinámica
        if self.evolver and hasattr(self.evolver, 'analyze_metaMs'):
            # Suponemos que Ss puede ser una lista de MetaMs o similar
            dinamica = self.evolver.analyze_metaMs(Ss)
            arquetipo = self.evolver.formalize_axiom(dinamica, Ss)
        else:
            arquetipo = None

        # 2. Recuperar MetaM y detalles de la KnowledgeBase (si existe)
        detalles = None
        if self.knowledge_base:
            # Buscar entradas que coincidan con Ss o arquetipo
            # find_by_inputs requiere 4 argumentos: A, B, C, M_emergent
            if hasattr(self.knowledge_base, 'find_by_inputs'):
                A, B, C = Ss[0], Ss[1], Ss[2]
                # Intentar obtener M_emergent del contexto, si no existe usar None
                M_emergent = None
                if contexto and isinstance(contexto, dict):
                    M_emergent = contexto.get('M_emergent', None)
                # Solo llamar si M_emergent no es None, para evitar errores de tipo
                if M_emergent is not None:
                    detalles = self.knowledge_base.find_by_inputs(A, B, C, M_emergent)
                else:
                    detalles = []

        # 3. Extensión inversa: reconstruir tensores originales
        tensores_candidatos = None
        relator_result = None
        if detalles and isinstance(detalles, list) and len(detalles) > 0:
            # Extraer posibles salidas (C) de los detalles encontrados
            tensores_candidatos = [d.get('C') for d in detalles if 'C' in d]
            objetivo = contexto.get('objetivo') if contexto and isinstance(contexto, dict) and 'objetivo' in contexto else None
            # Si hay más de un candidato y un objetivo, aplicar rotación áurea
            if tensores_candidatos and objetivo:
                phi = (1 + 5 ** 0.5) / 2  # Proporción áurea
                mejor_dist = float('inf')
                mejor_rot = None
                mejor_candidato = None
                for idx, cand in enumerate(tensores_candidatos):
                    for rot in range(len(cand)):
                        rotado = cand[rot:] + cand[:rot]
                        dist = sum(a != b for a, b in zip(rotado, objetivo))
                        # Penalización por alejarse de la rotación áurea
                        golden_rot = int(round(len(cand) / phi)) % len(cand)
                        penal = abs(rot - golden_rot)
                        score = dist + penal * 0.1  # Peso bajo a la penalización
                        if score < mejor_dist:
                            mejor_dist = score
                            mejor_rot = rot
                            mejor_candidato = rotado
                tensores_reconstruidos = mejor_candidato if mejor_candidato is not None else tensores_candidatos[0]
            else:
                tensores_reconstruidos = tensores_candidatos[0] if tensores_candidatos else None
            # Si no hay candidatos válidos, propagar None
            if tensores_reconstruidos is None or all(x is None for x in tensores_reconstruidos):
                tensores_reconstruidos = None
        else:
            tensores_reconstruidos = None
        reconstruccion = {
            'arquetipo_utilizado': arquetipo,
            'detalles_encontrados': detalles,
            'tensores_reconstruidos': tensores_reconstruidos,
            'relator_result': relator_result
        }

        # 4. Justificación y salida
        return {
            'input_Ss': Ss,
            'contexto': contexto,
            'reconstruccion': reconstruccion
        }

    def extend_fractal(self, fractal_tensor, contexto=None):
        """
        Reconstruye/infiere un FractalTensor a partir de abstracciones, operando en paralelo por nivel.
        Devuelve un nuevo FractalTensor con los resultados reconstruidos.
        """
        # Aquí se asume que la extensión inversa es aplicar extend() a cada subvector
        nivel_3 = []
        nivel_9 = []
        nivel_27 = []
        for v in fractal_tensor.nivel_3:
            res = self.extend(v, contexto)
            nivel_3.append(res)
        for v in fractal_tensor.nivel_9:
            res = self.extend(v, contexto)
            nivel_9.append(res)
        for v in fractal_tensor.nivel_27:
            res = self.extend(v, contexto)
            nivel_27.append(res)
        return FractalTensor(nivel_3=nivel_3, nivel_9=nivel_9, nivel_27=nivel_27)

    def extend(self, Ss, contexto=None):
        """
        Reconstruye la información detallada a partir de un input abstracto Ss.
        1. Usa la dinámica para identificar el arquetipo (axioma) relevante.
        2. Recupera el MetaM y otros datos del Evolver/KnowledgeBase.
        3. Realiza la extensión inversa: reconstruye los tensores originales (A, B, C, ...).
        4. Devuelve la reconstrucción y justificación.
        """
        # Validar que Ss tenga al menos 3 elementos y que cada uno sea vector de 3
        if not isinstance(Ss, list) or len(Ss) < 3:
            return {'error': 'Se requieren al menos 3 elementos en Ss para extensión.'}
        for idx, v in enumerate(Ss[:3]):
            if not isinstance(v, list) or len(v) != 3:
                return {'error': f'Elemento Ss[{idx}] debe ser vector de 3 elementos.'}
        # 1. Identificar arquetipo usando dinámica
        if self.evolver and hasattr(self.evolver, 'analyze_metaMs'):
            # Suponemos que Ss puede ser una lista de MetaMs o similar
            dinamica = self.evolver.analyze_metaMs(Ss)
            arquetipo = self.evolver.formalize_axiom(dinamica, Ss)
        else:
            arquetipo = None

        # 2. Recuperar MetaM y detalles de la KnowledgeBase (si existe)
        detalles = None
        if self.knowledge_base:
            # Buscar entradas que coincidan con Ss o arquetipo
            # find_by_inputs requiere 4 argumentos: A, B, C, M_emergent
            if hasattr(self.knowledge_base, 'find_by_inputs'):
                A, B, C = Ss[0], Ss[1], Ss[2]
                # Intentar obtener M_emergent del contexto, si no existe usar None
                M_emergent = None
                if contexto and isinstance(contexto, dict):
                    M_emergent = contexto.get('M_emergent', None)
                # Solo llamar si M_emergent no es None, para evitar errores de tipo
                if M_emergent is not None:
                    detalles = self.knowledge_base.find_by_inputs(A, B, C, M_emergent)
                else:
                    detalles = []

        # 3. Extensión inversa: reconstruir tensores originales
        tensores_candidatos = None
        relator_result = None
        if detalles and isinstance(detalles, list) and len(detalles) > 0:
            # Extraer posibles salidas (C) de los detalles encontrados
            tensores_candidatos = [d.get('C') for d in detalles if 'C' in d]
            objetivo = contexto.get('objetivo') if contexto and isinstance(contexto, dict) and 'objetivo' in contexto else None
            # Si hay más de un candidato y un objetivo, aplicar rotación áurea
            if tensores_candidatos and objetivo:
                phi = (1 + 5 ** 0.5) / 2  # Proporción áurea
                mejor_dist = float('inf')
                mejor_rot = None
                mejor_candidato = None
                for idx, cand in enumerate(tensores_candidatos):
                    for rot in range(len(cand)):
                        rotado = cand[rot:] + cand[:rot]
                        dist = sum(a != b for a, b in zip(rotado, objetivo))
                        # Penalización por alejarse de la rotación áurea
                        golden_rot = int(round(len(cand) / phi)) % len(cand)
                        penal = abs(rot - golden_rot)
                        score = dist + penal * 0.1  # Peso bajo a la penalización
                        if score < mejor_dist:
                            mejor_dist = score
                            mejor_rot = rot
                            mejor_candidato = rotado
                tensores_reconstruidos = mejor_candidato if mejor_candidato is not None else tensores_candidatos[0]
            else:
                tensores_reconstruidos = tensores_candidatos[0] if tensores_candidatos else None
            # Si no hay candidatos válidos, propagar None
            if tensores_reconstruidos is None or all(x is None for x in tensores_reconstruidos):
                tensores_reconstruidos = None
        else:
            tensores_reconstruidos = None
        reconstruccion = {
            'arquetipo_utilizado': arquetipo,
            'detalles_encontrados': detalles,
            'tensores_reconstruidos': tensores_reconstruidos,
            'relator_result': relator_result
        }

        # 4. Justificación y salida
        return {
            'input_Ss': Ss,
            'contexto': contexto,
            'reconstruccion': reconstruccion
        }

    def extend_fractal(self, fractal_tensor, contexto=None):
        """
        Reconstruye/infiere un FractalTensor a partir de abstracciones, operando en paralelo por nivel.
        Devuelve un nuevo FractalTensor con los resultados reconstruidos.
        """
        # Aquí se asume que la extensión inversa es aplicar extend() a cada subvector
        nivel_3 = []
        nivel_9 = []
        nivel_27 = []
        for v in fractal_tensor.nivel_3:
            res = self.extend(v, contexto)
            nivel_3.append(res)
        for v in fractal_tensor.nivel_9:
            res = self.extend(v, contexto)
            nivel_9.append(res)
        for v in fractal_tensor.nivel_27:
            res = self.extend(v, contexto)
            nivel_27.append(res)
        return FractalTensor(nivel_3=nivel_3, nivel_9=nivel_9, nivel_27=nivel_27)

class KnowledgeBase:
    def check_coherence(self, Ms, MetaM):
        """
        Valida la coherencia entre Ms (lista de 3 vectores de 3 bits) y MetaM (vector de 3 bits).
        Regla simple: para cada bit, si todos los Ms son iguales y no None, MetaM debe ser 0; si son distintos, MetaM debe ser 1; si hay None, coherencia es None.
        Devuelve lista de bool/None por bit.
        """
        coherence = []
        for i in range(3):
            ms_bits = [m[i] if m is not None else None for m in Ms]
            if any(b is None for b in ms_bits) or MetaM[i] is None:
                coherence.append(None)
            elif ms_bits[0] == ms_bits[1] == ms_bits[2]:
                coherence.append(MetaM[i] == 0)
            else:
                coherence.append(MetaM[i] == 1)
        return coherence
    def __init__(self):
        # Cada entrada es un dict con las claves:
        # 'A', 'B', 'C', 'M_emergent', 'MetaM', 'R_validos', 'transcender_id'
        self.knowledge = []

    def add_entry(self, A, B, C, M_emergent, MetaM, R_validos, transcender_id=None, Ms=None):
        """
        Guarda una nueva entrada en la base de conocimiento.
        Valida coherencia: si ya existe una entrada con los mismos Ms (M_emergent), su MetaM debe coincidir.
        Si no, lanza una excepción.
        Args:
            A, B, C: vectores de entrada (listas de 3 bits)
            M_emergent: vector de 3 bits
            MetaM: vector de 3 bits
            R_validos: lista de R válidos (puede ser lista de listas o lista de ints)
            transcender_id: identificador o referencia al transcender (opcional)
            Ms: lista de 3 vectores de 3 bits (opcional, para validación de coherencia)
        """
        # Validar longitud de vectores
        for name, v in zip(['A','B','C','M_emergent','MetaM'], [A,B,C,M_emergent,MetaM]):
            if not isinstance(v, list) or len(v) != 3:
                raise Exception(f"{name} debe ser un vector de 3 elementos: {v}")
        # Validar que no haya None en M_emergent ni MetaM
        if any(x is None for x in M_emergent):
            raise Exception(f"No se puede almacenar entrada: M_emergent contiene None: {M_emergent}")
        if any(x is None for x in MetaM):
            raise Exception(f"No se puede almacenar entrada: MetaM contiene None: {MetaM}")
        # Validación de coherencia: si ya existe una entrada con el mismo M_emergent, su MetaM debe coincidir
        for entry in self.knowledge:
            if entry['M_emergent'] == list(M_emergent):
                if entry['MetaM'] != list(MetaM):
                    raise Exception(f"Coherencia violada: Ms existente con MetaM diferente. Ms={M_emergent}, MetaM nuevo={MetaM}, MetaM previo={entry['MetaM']}")
        entry = {
            'A': list(A),
            'B': list(B),
            'C': list(C),
            'M_emergent': list(M_emergent),
            'MetaM': list(MetaM),
            'R_validos': R_validos,
            'transcender_id': transcender_id
        }
        # Validar coherencia si se proveen Ms
        if Ms is not None:
            entry['coherence_Ms_MetaM'] = self.check_coherence(Ms, MetaM)
        self.knowledge.append(entry)

    def find_by_inputs(self, A, B, C, M_emergent):
        """
        Busca entradas que coincidan exactamente con los vectores dados.
        """
        results = []
        for entry in self.knowledge:
            if (entry['A'] == list(A) and entry['B'] == list(B) and entry['C'] == list(C)
                and entry['M_emergent'] == list(M_emergent)):
                results.append(entry)
        return results

    def all_entries(self):
        """Devuelve todas las entradas almacenadas."""
        return self.knowledge




        # 2. Calcular S_emergent usando la hipótesis R
        self.S_emergent = []
        for i in range(3):
            if self.A[i] is None or self.B[i] is None or R_hipotesis[i] is None:
                self.S_emergent.append(None)
                continue
            A_vec = [None, None, None]
            B_vec = [None, None, None]
            R_vec = [None, None, None]
            A_vec[i] = self.A[i]
            B_vec[i] = self.B[i]
            R_vec[i] = R_hipotesis[i]
            result = self.trigate_superior.synthesize(A_vec, B_vec, R_vec)
            if isinstance(result, tuple) and len(result) == 2:
                _, s_em = result
            else:
                s_em = result
            if isinstance(s_em, list):
                self.S_emergent.append(s_em[i])
            else:
                self.S_emergent.append(s_em)

        # 3. Calcular M_AB, M_BC, M_CA y S_AB, S_BC, S_CA usando R_hipotesis
        self.M_AB, self.S_AB = [], []
        self.M_BC, self.S_BC = [], []
        self.M_CA, self.S_CA = [], []
        for i in range(3):
            if self.A[i] is None or self.B[i] is None or self.C[i] is None or R_hipotesis[i] is None:
                self.M_AB.append(None)
                self.S_AB.append(None)
                self.M_BC.append(None)
                self.S_BC.append(None)
                self.M_CA.append(None)
                self.S_CA.append(None)
                continue
            A_vec = [None, None, None]
            B_vec = [None, None, None]
            C_vec = [None, None, None]
            R_vec = [None, None, None]
            A_vec[i] = self.A[i]
            B_vec[i] = self.B[i]
            C_vec[i] = self.C[i]
            R_vec[i] = R_hipotesis[i]
            result_ab = self.trigate_AB.synthesize(A_vec, B_vec, R_vec)
            result_bc = self.trigate_BC.synthesize(B_vec, C_vec, R_vec)
            result_ca = self.trigate_CA.synthesize(C_vec, A_vec, R_vec)
            if isinstance(result_ab, tuple) and len(result_ab) == 2:
                m_ab, s_ab = result_ab
            else:
                m_ab, s_ab = result_ab, None
            if isinstance(result_bc, tuple) and len(result_bc) == 2:
                m_bc, s_bc = result_bc
            else:
                m_bc, s_bc = result_bc, None
            if isinstance(result_ca, tuple) and len(result_ca) == 2:
                m_ca, s_ca = result_ca
            else:
                m_ca, s_ca = result_ca, None
            self.M_AB.append(m_ab[i] if isinstance(m_ab, list) else m_ab)
            self.S_AB.append(s_ab[i] if isinstance(s_ab, list) else s_ab)
            self.M_BC.append(m_bc[i] if isinstance(m_bc, list) else m_bc)
            self.S_BC.append(s_bc[i] if isinstance(s_bc, list) else s_bc)
            self.M_CA.append(m_ca[i] if isinstance(m_ca, list) else m_ca)
            self.S_CA.append(s_ca[i] if isinstance(s_ca, list) else s_ca)

        # 4. Calcular M_intermediate usando TrigateMs
        self.M_intermediate = []
        for i in range(3):
            if self.M_AB[i] is None or self.M_BC[i] is None or R_hipotesis[i] is None:
                self.M_intermediate.append(None)
                continue
            MAB_vec = [None, None, None]
            MBC_vec = [None, None, None]
            R_vec = [None, None, None]
            MAB_vec[i] = self.M_AB[i]
            MBC_vec[i] = self.M_BC[i]
            R_vec[i] = R_hipotesis[i]
            result_int = self.trigate_Ms.synthesize(MAB_vec, MBC_vec, R_vec)
            if isinstance(result_int, tuple) and len(result_int) == 2:
                m_int, _ = result_int
            else:
                m_int = result_int
            self.M_intermediate.append(m_int[i] if isinstance(m_int, list) else m_int)

        # 5. Calcular MetaM como XOR entre M_intermediate y M_emergent (ternario)
        self.MetaM = [((int(a) ^ int(b)) if a is not None and b is not None else None)
                      for a, b in zip(self.M_intermediate, self.M_emergent)]
        # Retornar todos los resultados relevantes
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'M_AB': self.M_AB,
            'S_AB': self.S_AB,
            'M_BC': self.M_BC,
            'S_BC': self.S_BC,
            'M_CA': self.M_CA,
            'S_CA': self.S_CA,
            'M_emergent': self.M_emergent,
            'S_emergent': self.S_emergent,
            'M_intermediate': self.M_intermediate,
            'MetaM': self.MetaM
        }

    def compute(self, A, B, C):
        """
        Main computation method for Transcender.
        Args:
            A, B, C: Three 3-bit vectors (lists or tuples of 3 ints: 0 or 1)
        Sets all properties and computes the full hierarchical logic flow.
        """
        self.A = list(A)
        self.B = list(B)
        self.C = list(C)

        # 1. Compute Trigate outputs for (A,B), (B,C), (C,A)
        self.M_AB, self.S_AB = self.trigate_AB.synthesize(self.A, self.B)
        self.M_BC, self.S_BC = self.trigate_BC.synthesize(self.B, self.C)
        self.M_CA, self.S_CA = self.trigate_CA.synthesize(self.C, self.A)

        # 2. Compute M_emergent and S_emergent using the superior Trigate
        self.M_emergent, self.S_emergent = self.trigate_superior.synthesize(self.M_AB, self.M_BC)

        # 3. Compute M_intermediate using the three M vectors
        interm1, _ = self.trigate_Ms.synthesize(self.M_AB, self.M_BC)
        self.M_intermediate, _ = self.trigate_Ms.synthesize(interm1, self.M_CA)

        # 4. Compute MetaM as bitwise XOR between M_intermediate and M_emergent
        self.MetaM = [((int(a) ^ int(b)) if a is not None and b is not None else None)
                      for a, b in zip(self.M_intermediate, self.M_emergent)]

        # Return all relevant results
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'M_AB': self.M_AB,
            'S_AB': self.S_AB,
            'M_BC': self.M_BC,
            'S_BC': self.S_BC,
            'M_CA': self.M_CA,
            'S_CA': self.S_CA,
            'M_emergent': self.M_emergent,
            'S_emergent': self.S_emergent,
            'M_intermediate': self.M_intermediate,
            'MetaM': self.MetaM
        }
    def compute_fractal(self, fractal_tensor):
        """
        Procesa un FractalTensor y retorna los resultados de compute para cada nivel.
        Nunca retorna None; si hay error, retorna un diccionario con mensaje de error.
        """
        resultados = {}
        for nivel, vectores in [
            ('nivel_3', getattr(fractal_tensor, 'nivel_3', [])),
            ('nivel_9', getattr(fractal_tensor, 'nivel_9', [])),
            ('nivel_27', getattr(fractal_tensor, 'nivel_27', []))
        ]:
            resultados[nivel] = []
            for v in vectores:
                try:
                    res = self.compute(v, v, v)
                except Exception as e:
                    res = {'error': str(e), 'input': v}
                if res is None:
                    res = {'error': 'compute devolvió None', 'input': v}
                resultados[nivel].append(res)
        return resultados


class Trigate:
    """
    Trigate: The fundamental logical module of the Aurora system.
    
    Based on the geometric principle of a triangle where given two angles,
    the third can be deduced. In Aurora's Boolean logic:
    - A and B are the inputs
    - M is the logical function (control vector)
    - R is the result
    
    The Trigate can operate in three modes:
    1. Inference: Calculate R from A, B, and M
    2. Learning: Discover M from A, B, and R
    3. Inverse Deduction: Find missing input
    """
    
    def __init__(self):
        """Initialize a new Trigate instance."""
        # Properties to store the last operation
        self.last_operation = None
        self.last_inputs = None
        self.last_result = None
        
        # Properties for internal state
        self.A = None  # First input (3 bits)
        self.B = None  # Second input (3 bits)
        self.M = None  # Control vector (3 bits)
        self.R = None  # Result (3 bits)
        self.S = None  # Synthesis value (3 bits) - combines inputs and result
    
    def infer(self, A, B, M):
        """
        Mode 1: Inference - Calculate R from A, B, and M
        
        Args:
            A: First input (3-bit list like [0,1,1])
            B: Second input (3-bit list like [1,0,1]) 
            M: Control vector (3-bit list like [1,1,1])
            
        Returns:
            R: Result (3-bit list)
        """
        # Store inputs in instance properties
        self.A = A
        self.B = B
        self.M = M
        
        # Calculate R bit by bit (ternary logic)
        R = []
        for i in range(3):
            if A[i] is None or B[i] is None or M[i] is None:
                r_bit = None
            elif M[i] == 1:
                r_bit = A[i] ^ B[i]
            elif M[i] == 0:
                r_bit = 1 - (A[i] ^ B[i])
            else:
                r_bit = None
            R.append(r_bit)
        
        # Store result
        self.R = R
        self.last_operation = "infer"
        self.last_inputs = {"A": A, "B": B, "M": M}
        self.last_result = R
        
        return R
    
    def learn(self, A, B, R):
        """
        Mode 2: Learning - Discover M from A, B, and R
        
        Args:
            A: First input (3-bit list)
            B: Second input (3-bit list)
            R: Expected result (3-bit list)
            
        Returns:
            M: Learned control vector (3-bit list)
        """
        # Store inputs
        self.A = A
        self.B = B
        self.R = R
        
        # Learn M bit by bit (ternary logic)
        M = []
        for i in range(3):
            if A[i] is None or B[i] is None or R[i] is None:
                M.append(None)
            elif (A[i] ^ B[i]) == R[i]:
                M.append(1)  # XOR was used
            elif (1 - (A[i] ^ B[i])) == R[i]:
                M.append(0)  # XNOR was used
            else:
                M.append(None)
        
        # Store result
        self.M = M
        self.last_operation = "learn"
        self.last_inputs = {"A": A, "B": B, "R": R}
        self.last_result = M
        
        return M
    
    def deduce_B(self, A, M, R):
        """
        Mode 3: Inverse Deduction - Find B from A, M, and R
        
        Args:
            A: Known input (3-bit list)
            M: Control vector (3-bit list)
            R: Expected result (3-bit list)
            
        Returns:
            B: Deduced input (3-bit list)
        """
        # Store known values
        self.A = A
        self.M = M
        self.R = R
        
        # Deduce B bit by bit (ternary logic)
        B = []
        for i in range(3):
            if A[i] is None or M[i] is None or R[i] is None:
                b_bit = None
            elif M[i] == 1:
                b_bit = A[i] ^ R[i]
            elif M[i] == 0:
                b_bit = 1 - (A[i] ^ R[i])
            else:
                b_bit = None
            B.append(b_bit)
        
        # Store result
        self.B = B
        self.last_operation = "deduce_B"
        self.last_inputs = {"A": A, "M": M, "R": R}
        self.last_result = B
        
        return B
    
    def deduce_A(self, B, M, R):
        """
        Mode 3: Inverse Deduction - Find A from B, M, and R
        
        Args:
            B: Known input (3-bit list)
            M: Control vector (3-bit list)
            R: Expected result (3-bit list)
            
        Returns:
            A: Deduced input (3-bit list)
        """
        # Since XOR and XNOR are symmetric, A and B are interchangeable
        return self.deduce_B(B, M, R)
    
    def solve(self, A=None, B=None, M=None, R=None):
        """
        Smart resolver - Automatically determines which operation to perform
        based on which values are provided and which are missing.
        
        Args:
            A: First input (3-bit list or None)
            B: Second input (3-bit list or None)
            M: Control vector (3-bit list or None)
            R: Result (3-bit list or None)
            
        Returns:
            dict: Contains the missing value and operation performed
        """
        # Count how many values are provided
        provided = []
        missing = []
        
        if A is not None:
            provided.append('A')
        else:
            missing.append('A')
            
        if B is not None:
            provided.append('B')
        else:
            missing.append('B')
            
        if M is not None:
            provided.append('M')
        else:
            missing.append('M')
            
        if R is not None:
            provided.append('R')
        else:
            missing.append('R')
        
        # Check if we have exactly 3 values (need to find 1)
        if len(provided) != 3:
            return {
                "error": f"Need exactly 3 values to solve, got {len(provided)}",
                "provided": provided,
                "missing": missing
            }
        
        # Determine which operation to perform based on what's missing
        if 'R' in missing:
            # Missing R: Use inference (A, B, M -> R)
            result_value = self.infer(A, B, M)
            return {
                "operation": "inference",
                "missing_value": "R",
                "result": result_value,
                "inputs_used": {"A": A, "B": B, "M": M}
            }
            
        elif 'M' in missing:
            # Missing M: Use learning (A, B, R -> M)
            result_value = self.learn(A, B, R)
            return {
                "operation": "learning",
                "missing_value": "M",
                "result": result_value,
                "inputs_used": {"A": A, "B": B, "R": R}
            }
            
        elif 'B' in missing:
            # Missing B: Use inverse deduction (A, M, R -> B)
            result_value = self.deduce_B(A, M, R)
            return {
                "operation": "deduce_B",
                "missing_value": "B",
                "result": result_value,
                "inputs_used": {"A": A, "M": M, "R": R}
            }
            
        elif 'A' in missing:
            # Missing A: Use inverse deduction (B, M, R -> A)
            result_value = self.deduce_A(B, M, R)
            return {
                "operation": "deduce_A",
                "missing_value": "A",
                "result": result_value,
                "inputs_used": {"B": B, "M": M, "R": R}
            }
        
        # This shouldn't happen if our logic is correct
        return {
            "error": "Unexpected state in solve function",
            "provided": provided,
            "missing": missing
        }
    
    def synthesize(self, A, B, R=None):
        """
        Synthesize logic for Trigate.
        - Si se llama con A, B: retorna (M, S) donde M = A XOR B, S = A XNOR B
        - Si se llama con A, B, R: retorna S, donde S = f(A, B, R) según la documentación
        Args:
            A: Primer input (lista de 3 bits)
            B: Segundo input (lista de 3 bits)
            R: (opcional) Resultado (lista de 3 bits)
        Returns:
            (M, S) si R es None, si no retorna S
        """
        if R is None:
            M = []
            S = []
            for i in range(3):
                if A[i] is None or B[i] is None:
                    m_bit = None
                    s_bit = None
                else:
                    m_bit = A[i] ^ B[i]
                    s_bit = 1 - (A[i] ^ B[i])  # XNOR
                M.append(m_bit)
                S.append(s_bit)
            self.last_operation = "synthesize_AB"
            self.last_inputs = {"A": A, "B": B}
            self.last_result = (M, S)
            return M, S
        else:
            S = []
            for i in range(3):
                if A[i] is None or B[i] is None or R[i] is None:
                    s_bit = None
                elif R[i] == 1:
                    temp1 = R[i] ^ A[i]
                    temp2 = R[i] ^ B[i]
                    s_bit = temp1 ^ temp2
                elif R[i] == 0:
                    temp1 = 1 - (R[i] ^ A[i])  # XNOR
                    temp2 = 1 - (R[i] ^ B[i])  # XNOR
                    s_bit = temp1 ^ temp2
                else:
                    s_bit = None
                S.append(s_bit)
            self.last_operation = "synthesize_ABR"
            self.last_inputs = {"A": A, "B": B, "R": R}
            self.last_result = S
            return S



class TrigateSinLut:
    # Tablas LUT precalculadas para todas las operaciones
    _LUT_INFER = {}      # (A, B, M) -> R
    _LUT_LEARN = {}      # (A, B, R) -> M
    _LUT_SYNTH = {}      # (A, B, R) -> S
    _LUT_DEDUCE_B = {}   # (A, M, R) -> B
    _LUT_DEDUCE_A = {}   # (B, M, R) -> A
    
    def __init__(self):
        if not Trigate._LUT_INFER:
            self._build_luts()
    
    def _build_luts(self):
        # Precalcular todas las combinaciones de 1 bit (8 valores posibles)
        for a in [0, 1]:
            for b in [0, 1]:
                for r in [0, 1]:
                    for m in [0, 1]:
                        # LUT para infer(): (A, B, M) -> R
                        Trigate._LUT_INFER[(a, b, m)] = a ^ b if m == 1 else 1 - (a ^ b)
                        
                        # LUT para learn(): (A, B, R) -> M
                        if (a ^ b) == r:
                            Trigate._LUT_LEARN[(a, b, r)] = 1
                        else:
                            Trigate._LUT_LEARN[(a, b, r)] = 0
                        
                        # LUT para synthesize(): (A, B, R) -> S
                        if r == 1:
                            Trigate._LUT_SYNTH[(a, b, r)] = (1 ^ a) ^ (1 ^ b)
                        else:
                            Trigate._LUT_SYNTH[(a, b, r)] = (1 - (0 ^ a)) ^ (1 - (0 ^ b))
                        
                        # LUT para deduce_B(): (A, M, R) -> B

                        if m == 1:
                            Trigate._LUT_DEDUCE_B[(a, m, r)] = a ^ r
                        else:
                            Trigate._LUT_DEDUCE_B[(a, m, r)] = 1 - (a ^ r)
                        
                        # LUT para deduce_A(): (B, M, R) -> A
                        if m == 1:
                            Trigate._LUT_DEDUCE_A[(b, m, r)] = b ^ r
                        else:
                            Trigate._LUT_DEDUCE_A[(b, m, r)] = 1 - (b ^ r)

    # Métodos optimizados con LUTs
    def infer(self, A, B, M):
        return [self._get_lut_value(Trigate._LUT_INFER, (a, b, m)) 
                for a, b, m in zip(A, B, M)]
    
    def learn(self, A, B, R):
        return [self._get_lut_value(Trigate._LUT_LEARN, (a, b, r)) 
                for a, b, r in zip(A, B, R)]
    
    def synthesize(self, A, B, R):
        return [self._get_lut_value(Trigate._LUT_SYNTH, (a, b, r)) 
                for a, b, r in zip(A, B, R)]
    
    def deduce_B(self, A, M, R):
        return [self._get_lut_value(Trigate._LUT_DEDUCE_B, (a, m, r)) 
                for a, m, r in zip(A, M, R)]
    
    def deduce_A(self, B, M, R):
        return [self._get_lut_value(Trigate._LUT_DEDUCE_A, (b, m, r)) 
                for b, m, r in zip(B, M, R)]
    
    def _get_lut_value(self, lut, key):
        """
        Obtiene el valor de la tabla LUT para una clave dada.
        Si la clave no existe, lanza una excepción.
        """
        if key in lut:
            return lut[key]
        else:
            raise ValueError(f"Entrada no válida para la LUT: {key}")

    def _validate_vector_list(self, vectors, min_len, name):
        """
        Valida que 'vectors' sea una lista de al menos min_len vectores ternarios de longitud 3.
        Cada valor debe ser 0, 1 o None.
        Devuelve (True, None) si es válido, (False, mensaje) si no.
        """
        if not isinstance(vectors, (list, tuple)):
            return False, f"{name} debe ser una lista o tupla."
        if len(vectors) < min_len:
            return False, f"{name} debe tener al menos {min_len} elementos."
        for idx, v in enumerate(vectors):
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                return False, f"Elemento {idx} de {name} no es un vector de longitud 3."
            for j, val in enumerate(v):
                if val not in (0, 1, None):
                    return False, f"Valor inválido en {name}[{idx}][{j}]: {val}. Solo se permiten 0, 1 o None."
        return True, None

    # Eliminada definición residual/incompleta de Transcender
        pass