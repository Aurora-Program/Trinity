import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.Transcender import Transcender

def test_deep_learning_no_solution():
    transcender = Transcender()
    # Caso: No hay ninguna combinación posible que produzca el M_emergent deseado
    A = [1, 1, 1]
    B = [0, 0, 0]
    C = [1, 1, 1]
    # M_emergent imposible para estos valores
    M_emergent = [0, 0, 0]  # Forzamos un caso sin solución
    result = transcender.deep_learning(A, B, C, M_emergent)
    print("Caso sin solución posible:", result)
    assert result is None, "Debe devolver None cuando no hay solución posible"
    print("Test de propagación de ambigüedad (sin solución) completado.")

if __name__ == "__main__":
    test_deep_learning_no_solution()
