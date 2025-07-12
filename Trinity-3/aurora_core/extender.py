from aurora_core.evolver import Evolver

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
