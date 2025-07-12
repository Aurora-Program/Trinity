from .Trigate import Trigate

class FractalVector:
    def __init__(self, nivel_3=None, nivel_9=None, nivel_27=None):
        self.nivel_3 = nivel_3 if nivel_3 is not None else []
        self.nivel_9 = nivel_9 if nivel_9 is not None else []
        self.nivel_27 = nivel_27 if nivel_27 is not None else []

    def __repr__(self):
        return f"FractalVector(nivel_3={self.nivel_3}, nivel_9={self.nivel_9}, nivel_27={self.nivel_27})"

    def to_dict(self):
        return {
            "nivel_3": self.nivel_3,
            "nivel_9": self.nivel_9,
            "nivel_27": self.nivel_27
        }

    def to_list(self):
        return [self.nivel_3, self.nivel_9, self.nivel_27]

    @classmethod
    def from_dict(cls, data):
        return cls(
            nivel_3=data.get('nivel_3'),
            nivel_9=data.get('nivel_9'),
            nivel_27=data.get('nivel_27')
        )

    @classmethod
    def from_list(cls, data):
        # Ensure each dimension value is within 0–7 (3-bit range)
        def clamp3(x):
            try:
                xi = int(x)
            except:
                return 0
            return max(0, min(7, xi))

        # Nivel 3: single list of ints
        raw_n3 = data[0] if isinstance(data[0], list) else []
        nivel_3 = [clamp3(v) for v in raw_n3]
        # Nivel 9: list of lists
        raw_n9 = data[1] if isinstance(data[1], list) else []
        nivel_9 = []
        for row in raw_n9:
            if isinstance(row, list):
                nivel_9.append([clamp3(v) for v in row])
        # Nivel 27: nested lists
        raw_n27 = data[2] if isinstance(data[2], list) else []
        nivel_27 = []
        for block in raw_n27:
            if isinstance(block, list):
                nivel_27.append([[clamp3(v) for v in triple] for triple in block if isinstance(triple, list)])
        return cls(nivel_3=nivel_3, nivel_9=nivel_9, nivel_27=nivel_27)
