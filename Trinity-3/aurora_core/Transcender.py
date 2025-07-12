from .Trigate import Trigate
from .FractalVector import FractalVector

class Transcender:
    def __init__(self, fractal_vector):
        self.fractal_vector = fractal_vector
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

    def sintetizar_triada(self, A, B, C):
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
                res = self.sintetizar_triada(v, v, v)
                resultados[nivel].append(res)
        return resultados

    def extender_fractal(self, dynamic_vector=None):
        nivel_3 = self.fractal_vector.nivel_3
        nivel_9 = self.fractal_vector.nivel_9

        if not nivel_3 or not nivel_9:
            return self.fractal_vector, {}

        A, B, C = nivel_9[0], nivel_9[1], nivel_9[2]
        # Si se proporciona dinámica, aplicar como sesgo XOR a los vectores de nivel 9
        if dynamic_vector and isinstance(dynamic_vector, list) and len(dynamic_vector) == len(A):
            A = [a ^ d for a, d in zip(A, dynamic_vector)]
            B = [b ^ d for b, d in zip(B, dynamic_vector)]
            C = [c ^ d for c, d in zip(C, dynamic_vector)]

        M_AB, S_AB = self.trigate_AB.synthesize(A, B)
        M_BC, S_BC = self.trigate_BC.synthesize(B, C)
        M_CA, S_CA = self.trigate_CA.synthesize(C, A)

        # Usar el vector semilla completo nivel_3 como 'semilla' para Nivel 27
        semilla = nivel_3 if isinstance(nivel_3, list) else [None, None, None]

        nivel_27 = [
            [M_AB, S_AB, semilla],
            [M_BC, S_BC, semilla],
            [M_CA, S_CA, semilla]
        ]
        
        componentes_extension = {
            "M_AB": M_AB, "S_AB": S_AB,
            "M_BC": M_BC, "S_BC": S_BC,
            "M_CA": M_CA, "S_CA": S_CA
        }

        return FractalVector(nivel_3, nivel_9, nivel_27), componentes_extension
