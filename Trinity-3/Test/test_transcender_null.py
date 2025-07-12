import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.Transcender import Transcender

def print_result(label, result):
    print(f"{label}: {result}")

if __name__ == "__main__":
    transcender = Transcender()
    # Caso 1: A, B, C con None en distintas posiciones
    A = [1, None, 0]
    B = [0, 1, None]
    C = [None, 0, 1]
    M_emergent = [1, 0, None]
    result = transcender.deep_learning(A, B, C, M_emergent)
    print_result("Caso ternario con None", result)

    # Caso 2: Todos None
    A = [None, None, None]
    B = [None, None, None]
    C = [None, None, None]
    M_emergent = [None, None, None]
    result = transcender.deep_learning(A, B, C, M_emergent)
    print_result("Todos None", result)

    # Caso 3: Solo un None en M_emergent
    A = [1, 0, 1]
    B = [0, 1, 0]
    C = [1, 1, 0]
    M_emergent = [1, None, 0]
    result = transcender.deep_learning(A, B, C, M_emergent)
    print_result("Solo un None en M_emergent", result)

    # Caso 4: Sin None (control)
    A = [1, 0, 1]
    B = [0, 1, 0]
    C = [1, 1, 0]
    M_emergent = [1, 0, 0]
    result = transcender.deep_learning(A, B, C, M_emergent)
    print_result("Sin None (control)", result)
