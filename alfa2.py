# ==============================================================================
#  CLASE 1: Trigate (Unidad Básica de Razonamiento)
# ==============================================================================
class Trigate:
    """
    Representa la unidad básica de razonamiento. Opera sobre datos de 3 bits.
    A y B son entradas. M es el modo de operación (0=XNOR, 1=XOR). R es el resultado.
    """
    def __init__(self, A=None, B=None, R=None, M=None):
        self.A = A
        self.B = B
        self.R = R
        self.M = M

    def _xor(self, bit1, bit2):
        """Realiza la operación XOR sobre dos bits."""
        return 1 if bit1 != bit2 else 0

    def _xnor(self, bit1, bit2):
        """Realiza la operación XNOR sobre dos bits."""
        return 1 if bit1 == bit2 else 0

    def _validate_3bit_list(self, val, name):
        """Validador para listas de 3 bits (0 o 1)."""
        if not isinstance(val, list) or len(val) != 3:
            raise ValueError(f"{name} debe ser una lista de 3 bits. Se recibió: {val}")
        for bit in val:
            if bit not in (0, 1):
                raise ValueError(f"Los bits en {name} deben ser 0 o 1. Encontrado: {bit}")

    def inferir(self):
        """Calcula R basado en A, B y M."""
        self._validate_3bit_list(self.A, "A"); self._validate_3bit_list(self.B, "B"); self._validate_3bit_list(self.M, "M")
        self.R = []
        for i in range(3):
            if self.M[i] == 0: self.R.append(self._xnor(self.A[i], self.B[i]))
            else: self.R.append(self._xor(self.A[i], self.B[i]))
        return self.R

    def aprender(self):
        """Aprende M basado en A, B y R."""
        self._validate_3bit_list(self.A, "A"); self._validate_3bit_list(self.B, "B"); self._validate_3bit_list(self.R, "R")
        self.M = []
        for i in range(3):
            if self.R[i] == self._xor(self.A[i], self.B[i]): self.M.append(1)
            elif self.R[i] == self._xnor(self.A[i], self.B[i]): self.M.append(0)
            else: raise ValueError(f"Valor R[{i}]={self.R[i]} inconsistente con A[{i}]={self.A[i]} y B[{i}]={self.B[i]}")
        return self.M

    def sintesis(self):
        """Calcula el valor de síntesis S de 3 bits."""
        self._validate_3bit_list(self.A, "A"); self._validate_3bit_list(self.B, "B"); self._validate_3bit_list(self.R, "R")
        S_calculado = []
        for i in range(3):
            if self.R[i] == 0: S_calculado.append(self.A[i])
            else: S_calculado.append(self.B[i])
        return S_calculado

# ==============================================================================
#  CLASE 2: Trancender (Estructura Jerárquica)
# ==============================================================================
class Trancender:
    def __init__(self):
        """
        Inicializa un Trancender. Su configuración (MetaM) se determinará
        proporcionando un conjunto completo de datos iniciales a través del 
        método 'aprender_y_calcular'.
        """
        # --- Configuración interna (será aprendida/calculada) ---
        self.M1, self.M2, self.M3, self.MS = None, None, None, None
        self.MetaM = None

        # --- Estado interno del último cálculo ---
        self.InA, self.InB, self.InC = None, None, None
        self.R1_dado, self.R2_dado, self.R3_dado = None, None, None
        self.S1, self.S2, self.S3 = None, None, None
        self.RS_superior = None
        self.SintesisSuperior = None

        # Instancias de Trigate para uso interno
        self._TG1, self._TG2, self._TG3, self._TG_S = Trigate(), Trigate(), Trigate(), Trigate()

    def _validate_3bit_list(self, val, name_val):
        """Validador interno para listas de 3 bits (0 o 1)."""
        if not isinstance(val, list) or len(val) != 3:
            raise ValueError(f"{name_val} debe ser una lista de 3 bits.")
        for bit in val:
            if bit not in (0, 1):
                raise ValueError(f"Los bits en {name_val} deben ser 0 o 1. Encontrado: {bit}")

    def aprender_y_calcular(self, InA, InB, InC, R1, R2, R3):
        """
        Aprende/calcula la configuración completa del Trancender (MetaM) y su resultado
        (SintesisSuperior) a partir de las 3 entradas y los 3 resultados de la capa inferior.

        Args:
            InA, InB, InC (list): Las tres entradas principales de 3 bits.
            R1, R2, R3 (list): Los tres resultados de 3 bits de la capa inferior.

        Returns:
            tuple: Una tupla conteniendo (SintesisSuperior, MetaM).
        """
        # 1. Validar todas las entradas
        self._validate_3bit_list(InA, "InA"); self._validate_3bit_list(InB, "InB"); self._validate_3bit_list(InC, "InC")
        self._validate_3bit_list(R1, "R1"); self._validate_3bit_list(R2, "R2"); self._validate_3bit_list(R3, "R3")

        # Guardar datos iniciales
        self.InA, self.InB, self.InC = InA, InB, InC
        self.R1_dado, self.R2_dado, self.R3_dado = R1, R2, R3

        # 2. Aprender M y calcular S para la primera capa
        # TG1 (A,B) -> R1 => se obtienen M1 y S1
        self._TG1.A, self._TG1.B, self._TG1.R = self.InA, self.InB, self.R1_dado
        self.M1 = self._TG1.aprender()
        self.S1 = self._TG1.sintesis()

        # TG2 (B,C) -> R2 => se obtienen M2 y S2
        self._TG2.A, self._TG2.B, self._TG2.R = self.InB, self.InC, self.R2_dado
        self.M2 = self._TG2.aprender()
        self.S2 = self._TG2.sintesis()

        # TG3 (C,A) -> R3 => se obtienen M3 y S3
        self._TG3.A, self._TG3.B, self._TG3.R = self.InC, self.InA, self.R3_dado
        self.M3 = self._TG3.aprender()
        self.S3 = self._TG3.sintesis()

        # 3. Determinar MS (Modo del Trigate Superior)
        # HIPÓTESIS CLAVE: El modo del Trigate superior (MS) se determina a partir de la
        # síntesis del Trigate que cierra el ciclo (S3).
        self.MS = self.S3

        # 4. Ensamblar MetaM
        self.MetaM = [self.M1, self.M2, self.M3, self.MS]

        # 5. Calcular los resultados de la capa superior
        # Asumimos que A_superior = S1 y B_superior = S2
        self._TG_S.A, self._TG_S.B, self._TG_S.M = self.S1, self.S2, self.MS
        self.RS_superior = self._TG_S.inferir()
        self.SintesisSuperior = self._TG_S.sintesis()

        return self.SintesisSuperior, self.MetaM

    def __str__(self):
        """Representación en cadena de texto del estado del Trancender."""
        if self.MetaM is None:
            return "Trancender (no calculado/aprendido)."
        
        return (
            f"================ Trancender State ================\n"
            f"  Inputs: \n"
            f"    InA: {self.InA}   InB: {self.InB}   InC: {self.InC}\n"
            f"    R1_dado: {self.R1_dado}   R2_dado: {self.R2_dado}   R3_dado: {self.R3_dado}\n"
            f"--------------------------------------------------\n"
            f"  Layer 1 Results:\n"
            f"    TG1(A,B): M1 = {self.M1}   =>   S1 = {self.S1}\n"
            f"    TG2(B,C): M2 = {self.M2}   =>   S2 = {self.S2}\n"
            f"    TG3(C,A): M3 = {self.M3}   =>   S3 = {self.S3}\n"
            f"--------------------------------------------------\n"
            f"  Superior Layer Results:\n"
            f"    MS (derived from S3): {self.MS}\n"
            f"    Inputs to TG_S: A={self.S1}, B={self.S2}\n"
            f"    RS_superior: {self.RS_superior}\n"
            f"--------------------------------------------------\n"
            f"  Final Outputs:\n"
            f"    >> SintesisSuperior: {self.SintesisSuperior}\n"
            f"    >> MetaM: {self.MetaM}\n"
            f"=================================================="
        )

# ==============================================================================
#  BLOQUE DE EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # 1. Crear una instancia de Trancender
    trans = Trancender()

    # 2. Definir un conjunto completo de datos de entrada/entrenamiento
    #    Estos valores son un ejemplo para demostrar el funcionamiento.
    entradas_A = [1, 0, 1]
    entradas_B = [1, 1, 0]
    entradas_C = [0, 0, 1]

    # Los resultados 'R' se asumen como dados para este escenario de aprendizaje
    resultados_R1 = [1, 1, 0] # R para (A,B) -> M1=[0,1,1]
    resultados_R2 = [1, 1, 1] # R para (B,C) -> M2=[0,1,1]
    resultados_R3 = [0, 1, 0] # R para (C,A) -> M3=[0,0,0]

    print("--- Calculando la configuración del Trancender ---")
    
    # 3. Ejecutar el proceso de cálculo y aprendizaje
    try:
        s_sup_final, meta_m_final = trans.aprender_y_calcular(
            entradas_A, entradas_B, entradas_C,
            resultados_R1, resultados_R2, resultados_R3
        )

        # 4. Imprimir los resultados detallados usando el método __str__
        print("\n" + str(trans))

    except ValueError as e:
        print(f"\nError durante el cálculo: {e}")
