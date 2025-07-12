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

    @classmethod
    def from_dict(cls, data):
        return cls(
            nivel_3=data.get('nivel_3'),
            nivel_9=data.get('nivel_9'),
            nivel_27=data.get('nivel_27')
        )
