import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aurora_core.Trigate import Trigate
import random

def test_recursive_synthesis_deep():
    print("\n=== Test de síntesis recursiva profunda (nivel 3+) ===")
    trigate = Trigate()
    # Generar una secuencia de 5 vectores ternarios (0, 1, None)
    def random_ternary():
        return [random.choice([0, 1, None]) for _ in range(3)]
    inputs = [random_ternary() for _ in range(5)]
    print(f"Entradas: {inputs}")
    # Usar la versión compatible de recursive_synthesis (levels)
    final, history = trigate.recursive_synthesis(inputs)
    for i, step in enumerate(history):
        print(f"Nivel {i+1}: S={step}")
    print(f"Resultado final de síntesis recursiva: {final}")
    # Validación de fractalidad: el resultado debe ser coherente con la lógica ternaria
    assert len(final) == 3, "El resultado final debe ser un vector de 3 elementos"
    assert all(x in [0, 1, None] for x in final), "Todos los valores deben ser 0, 1 o None"
    print("Test de síntesis recursiva profunda completado y validado.")

if __name__ == "__main__":
    test_recursive_synthesis_deep()
