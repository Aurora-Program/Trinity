class Trigate:
    """
    Trigate: The fundamental logical module of the Aurora system.
    
    Based on the geometric principle of a triangle where given two angles,
    the third can be deduced. In Aurora's Boolean logic:
    - A and B are the inputs
    - M is the logical function (control vector)
    - R is the result
    
    The Trigate can operate in three modes:
    1. Inference: Calculate R from A, B, and M
    2. Learning: Discover M from A, B, and R
    3. Inverse Deduction: Find missing input
    """
    
    def __init__(self):
        """Initialize a new Trigate instance."""
        # Properties to store the last operation
        self.last_operation = None
        self.last_inputs = None
        self.last_result = None
        
        # Properties for internal state
        self.A = None  # First input (3 bits)
        self.B = None  # Second input (3 bits)
        self.M = None  # Control vector (3 bits)
        self.R = None  # Result (3 bits)
        self.S = None  # Synthesis value (3 bits) - combines inputs and result
    
    def infer(self, A, B, M):
        """
        Mode 1: Inference - Calculate R from A, B, and M
        
        Args:
            A: First input (3-bit list like [0,1,1])
            B: Second input (3-bit list like [1,0,1]) 
            M: Control vector (3-bit list like [1,1,1])
            
        Returns:
            R: Result (3-bit list)
        """
        # Store inputs in instance properties
        self.A = A
        self.B = B
        self.M = M
        
        # Calculate R bit by bit (ternary logic)
        R = []
        for i in range(3):
            if A[i] is None or B[i] is None or M[i] is None:
                r_bit = None
            elif M[i] == 1:
                r_bit = A[i] ^ B[i]
            elif M[i] == 0:
                r_bit = 1 - (A[i] ^ B[i])
            else:
                r_bit = None
            R.append(r_bit)
        
        # Store result
        self.R = R
        self.last_operation = "infer"
        self.last_inputs = {"A": A, "B": B, "M": M}
        self.last_result = R
        
        return R
    
    def learn(self, A, B, R):
        """
        Mode 2: Learning - Discover M from A, B, and R
        
        Args:
            A: First input (3-bit list)
            B: Second input (3-bit list)
            R: Expected result (3-bit list)
            
        Returns:
            M: Learned control vector (3-bit list)
        """
        # Store inputs
        self.A = A
        self.B = B
        self.R = R
        
        # Learn M bit by bit (ternary logic)
        M = []
        for i in range(3):
            if A[i] is None or B[i] is None or R[i] is None:
                M.append(None)
            elif (A[i] ^ B[i]) == R[i]:
                M.append(1)  # XOR was used
            elif (1 - (A[i] ^ B[i])) == R[i]:
                M.append(0)  # XNOR was used
            else:
                M.append(None)
        
        # Store result
        self.M = M
        self.last_operation = "learn"
        self.last_inputs = {"A": A, "B": B, "R": R}
        self.last_result = M
        
        return M
    
    def deduce_B(self, A, M, R):
        """
        Mode 3: Inverse Deduction - Find B from A, M, and R
        
        Args:
            A: Known input (3-bit list)
            M: Control vector (3-bit list)
            R: Expected result (3-bit list)
            
        Returns:
            B: Deduced input (3-bit list)
        """
        # Store known values
        self.A = A
        self.M = M
        self.R = R
        
        # Deduce B bit by bit (ternary logic)
        B = []
        for i in range(3):
            if A[i] is None or M[i] is None or R[i] is None:
                b_bit = None
            elif M[i] == 1:
                b_bit = A[i] ^ R[i]
            elif M[i] == 0:
                b_bit = 1 - (A[i] ^ R[i])
            else:
                b_bit = None
            B.append(b_bit)
        
        # Store result
        self.B = B
        self.last_operation = "deduce_B"
        self.last_inputs = {"A": A, "M": M, "R": R}
        self.last_result = B
        
        return B
    
    def deduce_A(self, B, M, R):
        """
        Mode 3: Inverse Deduction - Find A from B, M, and R
        
        Args:
            B: Known input (3-bit list)
            M: Control vector (3-bit list)
            R: Expected result (3-bit list)
            
        Returns:
            A: Deduced input (3-bit list)
        """
        # Since XOR and XNOR are symmetric, A and B are interchangeable
        return self.deduce_B(B, M, R)
    
    def solve(self, A=None, B=None, M=None, R=None):
        """
        Smart resolver - Automatically determines which operation to perform
        based on which values are provided and which are missing.
        
        Args:
            A: First input (3-bit list or None)
            B: Second input (3-bit list or None)
            M: Control vector (3-bit list or None)
            R: Result (3-bit list or None)
            
        Returns:
            dict: Contains the missing value and operation performed
        """
        # Count how many values are provided
        provided = []
        missing = []
        
        if A is not None:
            provided.append('A')
        else:
            missing.append('A')
            
        if B is not None:
            provided.append('B')
        else:
            missing.append('B')
            
        if M is not None:
            provided.append('M')
        else:
            missing.append('M')
            
        if R is not None:
            provided.append('R')
        else:
            missing.append('R')
        
        # Check if we have exactly 3 values (need to find 1)
        if len(provided) != 3:
            return {
                "error": f"Need exactly 3 values to solve, got {len(provided)}",
                "provided": provided,
                "missing": missing
            }
        
        # Determine which operation to perform based on what's missing
        if 'R' in missing:
            # Missing R: Use inference (A, B, M -> R)
            result_value = self.infer(A, B, M)
            return {
                "operation": "inference",
                "missing_value": "R",
                "result": result_value,
                "inputs_used": {"A": A, "B": B, "M": M}
            }
            
        elif 'M' in missing:
            # Missing M: Use learning (A, B, R -> M)
            result_value = self.learn(A, B, R)
            return {
                "operation": "learning",
                "missing_value": "M",
                "result": result_value,
                "inputs_used": {"A": A, "B": B, "R": R}
            }
            
        elif 'B' in missing:
            # Missing B: Use inverse deduction (A, M, R -> B)
            result_value = self.deduce_B(A, M, R)
            return {
                "operation": "deduce_B",
                "missing_value": "B",
                "result": result_value,
                "inputs_used": {"A": A, "M": M, "R": R}
            }
            
        elif 'A' in missing:
            # Missing A: Use inverse deduction (B, M, R -> A)
            result_value = self.deduce_A(B, M, R)
            return {
                "operation": "deduce_A",
                "missing_value": "A",
                "result": result_value,
                "inputs_used": {"B": B, "M": M, "R": R}
            }
        
        # This shouldn't happen if our logic is correct
        return {
            "error": "Unexpected state in solve function",
            "provided": provided,
            "missing": missing
        }
    
    def synthesize(self, A, B, R=None):
        """
        Synthesize logic for Trigate.
        - Si se llama con A, B: retorna (M, S) donde M = A XOR B, S = A XNOR B
        - Si se llama con A, B, R: retorna S, donde S = f(A, B, R) según la documentación
        Args:
            A: Primer input (lista de 3 bits)
            B: Segundo input (lista de 3 bits)
            R: (opcional) Resultado (lista de 3 bits)
        Returns:
            (M, S) si R es None, si no retorna S
        """
        if R is None:
            M = []
            S = []
            for i in range(3):
                if A[i] is None or B[i] is None:
                    m_bit = None
                    s_bit = None
                else:
                    m_bit = A[i] ^ B[i]
                    s_bit = 1 - (A[i] ^ B[i])  # XNOR
                M.append(m_bit)
                S.append(s_bit)
            self.last_operation = "synthesize_AB"
            self.last_inputs = {"A": A, "B": B}
            self.last_result = (M, S)
            return M, S
        else:
            S = []
            for i in range(3):
                if A[i] is None or B[i] is None or R[i] is None:
                    s_bit = None
                elif R[i] == 1:
                    temp1 = R[i] ^ A[i]
                    temp2 = R[i] ^ B[i]
                    s_bit = temp1 ^ temp2
                elif R[i] == 0:
                    temp1 = 1 - (R[i] ^ A[i])  # XNOR
                    temp2 = 1 - (R[i] ^ B[i])  # XNOR
                    s_bit = temp1 ^ temp2
                else:
                    s_bit = None
                S.append(s_bit)
            self.last_operation = "synthesize_ABR"
            self.last_inputs = {"A": A, "B": B, "R": R}
            self.last_result = S
            return S

