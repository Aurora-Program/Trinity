import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.Trigate import Trigate

def int_to_bits(n):
    return [(n >> 2) & 1, (n >> 1) & 1, n & 1]

def bits_to_str(bits):
    return ''.join(str(b) for b in bits)

if __name__ == "__main__":
    trigate = Trigate()
    print("A   B   R   M   S")
    print("====================")
    for a in range(8):
        for b in range(8):
            for r in range(8):
                A = int_to_bits(a)
                B = int_to_bits(b)
                R = int_to_bits(r)
                M = trigate.learn(A, B, R)
                S = trigate.synthesize(A, B, R)
                print(f"{bits_to_str(A)} {bits_to_str(B)} {bits_to_str(R)} {bits_to_str(M)} {bits_to_str(S)}")
