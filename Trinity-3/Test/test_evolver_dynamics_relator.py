import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.evolver import Evolver

def print_dict(title, d):
    print(f"\n{title}")
    for k, v in d.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    evolver = Evolver()
    # Test Dynamics: Secuencia temporal de MetaMs
    sequence = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [None, 1, 1],
        [1, None, 1],
    ]
    dyn = evolver.analyze_dynamics(sequence)
    print_dict("Dynamics (cambios temporales)", dyn)

    # Test Relator: Relaciones internas entre vectores
    vectors = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [1, 0, 1],
        [None, 1, 0],
    ]
    rel = evolver.relate_vectors(vectors)
    print_dict("Relator (relaciones internas)", rel)
