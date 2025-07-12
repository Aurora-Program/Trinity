import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.Trigate import Trigate

def int_to_bits(n):
    return [(n >> 2) & 1, (n >> 1) & 1, n & 1]

def bits_to_str(bits):
    return ''.join(str(b) for b in bits)

if __name__ == "__main__":
    # Example: A=1, B=1, S=2
    A = int_to_bits(1)  # [0, 0, 1]
    B = int_to_bits(1)  # [0, 1, 0]
    S = int_to_bits(0)  # [1,1 , 1]

    trigate = Trigate()
    results = trigate.deep_learning(A, B, S)
    print(f"A: {A} ({bits_to_str(A)})  B: {B} ({bits_to_str(B)})  S: {S} ({bits_to_str(S)})\n")
    if results:
        for idx, sol in enumerate(results):
            print(f"Solution {idx+1}:")
            print(f"  R: {sol['R']} ({bits_to_str(sol['R'])})")
            print(f"  M: {sol['M']} ({bits_to_str(sol['M'])})\n")
    else:
        print("No valid (R, M) found for these inputs.")


