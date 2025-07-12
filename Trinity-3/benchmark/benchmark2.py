# Aurora minimal proof‑of‑concept: Trigate + Transcender
#
# A Trigate is a ternary logic gate that works on 3‑bit integers.
# For each bit position i:
#   – if the control bit M_i == 1  → output = A_i XOR B_i
#   – if the control bit M_i == 0  → output = XNOR(A_i, B_i)
#
# The gate can run in three modes:
#   • Inference     (A, B, M) -> R
#   • Learning      (A, B, R) -> M
#   • Inv. deduct   (M, R, A) -> B  (or vice‑versa)
#
# A Transcender combines three Trigates:
#     T1(A,B) , T2(B,C) , T3(C,A)
# and produces a higher‑level control Ms, a factual snapshot Ss,
# and a full logical map MetaM = [M1, M2, M3, Ms].
#
# This is **only** a *toy* implementation to illustrate the idea – not
# the full Aurora math.

import random

# ---------- helpers ----------
def int3(x: int) -> str:
    """3‑bit binary string."""
    return format(x & 0b111, "03b")


def bit_op(a_bit: int, b_bit: int, m_bit: int) -> int:
    """Core per‑bit operation."""
    xor = a_bit ^ b_bit
    return xor if m_bit else (1 - xor)       # XOR vs XNOR


def trigate_apply(A: int, B: int, M: int) -> int:
    """(A,B,M) ➜ R   (inference mode)"""
    r = 0
    for i in range(3):
        res_bit = bit_op((A >> i) & 1, (B >> i) & 1, (M >> i) & 1)
        r |= (res_bit << i)
    return r


def trigate_learn_M(A: int, B: int, R: int) -> int:
    """(A,B,R) ➜ inferred M   (learning mode)"""
    m = 0
    for i in range(3):
        a, b, r = (A >> i) & 1, (B >> i) & 1, (R >> i) & 1
        m_bit = 1 if (a ^ b) == r else 0
        m |= (m_bit << i)
    return m


def trigate_inverse(M: int, known: int, R: int, solve_for: str = "B") -> int | None:
    """Recover missing input (inverse deduction)."""
    for candidate in range(8):
        if solve_for == "B":
            ok = trigate_apply(known, candidate, M) == R
        else:  # solve for A
            ok = trigate_apply(candidate, known, M) == R
        if ok:
            return candidate
    return None


# ---------- core classes ----------
class Trigate:
    """A single Aurora Trigate."""

    def __init__(self, *, A: int | None, B: int | None, M: int | None, R: int | None):
        # copy args
        self.A, self.B, self.M, self.R = A, B, M, R

        # --- resolve missing element ---------------------------------------
        if None not in (self.A, self.B, self.M) and self.R is None:
            # inference
            self.R = trigate_apply(self.A, self.B, self.M)
        elif None not in (self.A, self.B, self.R) and self.M is None:
            # learning
            self.M = trigate_learn_M(self.A, self.B, self.R)
        elif None not in (self.M, self.R) and (
            (self.A is None) ^ (self.B is None)
        ):
            # inverse deduction
            if self.A is None:
                self.A = trigate_inverse(self.M, self.B, self.R, solve_for="A")
            else:
                self.B = trigate_inverse(self.M, self.A, self.R, solve_for="B")

        # quick validation
        assert None not in (
            self.A,
            self.B,
            self.M,
            self.R,
        ), "Trigate needs 3 of 4 inputs to solve the 4th."

    # pretty‑print
    def __repr__(self):
        return (
            f"Trigate(A={int3(self.A)}, "
            f"B={int3(self.B)}, M={int3(self.M)}, R={int3(self.R)})"
        )


class Transcender:
    """Minimal proof‑of‑concept Transcender."""

    def __init__(self, A: int, B: int, C: int):
        # choose random lower‑level control vectors
        m1, m2, m3 = (random.randint(0, 7) for _ in range(3))

        # three lower Trigates
        self.T1 = Trigate(A=A, B=B, M=m1, R=None)
        self.T2 = Trigate(A=B, B=C, M=m2, R=None)
        self.T3 = Trigate(A=C, B=A, M=m3, R=None)

        # --- higher‑level synthesis (toy version) --------------------------
        # Structure Ms = XOR of the three controls (placeholder rule)
        self.Ms = self.T1.M ^ self.T2.M ^ self.T3.M

        # Form Ss = tuple of the factual outcomes
        self.Ss = (self.T1.R, self.T2.R, self.T3.R)

        # Function MetaM = full logical map
        self.MetaM = (self.T1.M, self.T2.M, self.T3.M, self.Ms)

    # quick dump
    def dump(self):
        print(" ↧ Lower trigates")
        print("   ", self.T1)
        print("   ", self.T2)
        print("   ", self.T3)
        print(f"\n ↧    Ms (Structure): {int3(self.Ms)}")
        print(
            f" ↧    Ss (Form)     : {[int3(s) for s in self.Ss]}",
        )
        print(
            f" ↧    MetaM (Function): {[int3(m) for m in self.MetaM]}",
        )


# ---------- demo / quick test ----------
if __name__ == "__main__":
    random.seed(42)  # reproducible

    # three random 3‑bit inputs
    A, B, C = (random.randint(0, 7) for _ in range(3))
    print(f"\n🟢  Inputs  A={int3(A)}, B={int3(B)}, C={int3(C)}")

    # build a Transcender
    T = Transcender(A, B, C)
    T.dump()

    # ---- Inference demo ---------------------------------------------------
    print("\n🟡  Inference on T1  (given A,B,M → R)")
    inf_R = trigate_apply(T.T1.A, T.T1.B, T.T1.M)
    print(
        f"       computed R = {int3(inf_R)}   "
        f"(stored R = {int3(T.T1.R)})"
    )

    # ---- Inverse deduction demo ------------------------------------------
    print("\n🔵  Inverse deduction on T1  (given M,R,A → B)")
    recovered_B = trigate_inverse(T.T1.M, T.T1.A, T.T1.R, solve_for="B")
    print(
        f"       recovered B = {int3(recovered_B)}   "
        f"(actual B = {int3(T.T1.B)})"
    )


