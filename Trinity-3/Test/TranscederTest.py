from aurora_core.Transcender import Transcender
from itertools import product

def binvec(n):
    return [int(x) for x in f'{n:03b}']

def test_deep_learning():
    # Puedes cambiar estos valores para probar otros casos
    A = [1, 0, 1]
    B = [0, 1, 1]
    C = [1, 1, 0]
    M_emergent = [1, 0, 0]

    transcender = Transcender()
    result = transcender.deep_learning(A, B, C, M_emergent)

    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    print(f"M_emergent: {M_emergent}")
    print(f"MetaM: {result['MetaM']}")
    print(f"S_emergent: {result['S_emergent']}")
    print(f"R_hipotesis: {result['R_hipotesis']}")

def test_all_combinations():
    transcender = Transcender()
    total = 0
    found = 0
    print("A     B     C     M_emergent   R_hipotesis   MetaM   S_emergent")
    for a, b, c, m in product(range(8), repeat=4):
        A = binvec(a)
        B = binvec(b)
        C = binvec(c)
        M_emergent = binvec(m)
        result = transcender.deep_learning(A, B, C, M_emergent)
        total += 1
        if None not in result['R_hipotesis']:
            found += 1
            print(f"{A} {B} {C} {M_emergent}   {result['R_hipotesis']}   {result['MetaM']}   {result['S_emergent']}")
    print(f"\nCasos con R válidos: {found} de {total}")

if __name__ == "__main__":
    test_all_combinations()
