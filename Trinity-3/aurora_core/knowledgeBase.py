
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
