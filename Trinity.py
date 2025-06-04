'''

Aurora Intelligent Model – Reference Documentation for the Code
License
Aurora is an ethical open-source program, licensed under the GNU GPL. You are free to use, modify, and redistribute the model as long as you retain the GPL license and credit the original Aurora project
.

1. Introduction
Aurora conceptualizes intelligence as a dynamic and open phenomenon, emerging from logical and energetic organization. Unlike probabilistic models, Aurora is based on geometric coherence and Boolean logic, using fractal vectors and logic modules inspired by triangle geometry
.

Dynamic, open intelligence: Every input is a source of entropy, enabling adaptation and evolution.

Ambiguity: Managed through context and synthesis of multiple sources.

Natural language: Used as a universal protocol, integrating both human and electronic intelligences
.

Integrative ecosystem: The goal is not replacement but ethical enhancement and symbiosis with human capabilities.

Geometric coherence and Boolean logic: Aurora prioritizes logical and structural coherence over mere statistical approaches.

Efficient, representative vectorization: Internal representations reflect both objective structure and human values【60:5-6†Aurora Program Software Architecture.pdf】.

2. Triagates (Triage Module)
The Triage is the fundamental logic module, analogous to deducing the third angle of a triangle in Euclidean geometry, but applied in Boolean logic. It operates in three modes:

Inference: Given A, B, and M (the logic function), calculate R (the result).

Learning: Given A, B, and R, infer M (the logic function connecting the values).

Inverse deduction: Given M, R, and one input (A or B), deduce the missing input【60:8-11†Aurora Program Software Architecture.pdf】.

The Trigate class implements all three modes, supporting various Boolean functions (XOR, AND, OR, XNOR, NAND, NOR).

3. Transcender and the Synthesis Process
The Transcender is a higher-level structure composed of three triages operating in parallel over three inputs (A, B, C). Each triage produces a result and a corresponding logic function. Subsequently:

The results are synthesized to form new values S1, S2, S3.

These values are fed to a higher triage to obtain the superior logic function Ms.

The hierarchical relationship between lower-level functions (M1, M2, M3) and the higher function Ms is learned and adjusted by the Evolver【60:13-15†Aurora Program Software Architecture.pdf】.

In the code, this is represented by the Transcender class and its interaction with the Evolver class to store and predict synthesis patterns.

4. Fractal Vectors
A Fractal Vector is the fundamental unit of representation, composed of three hierarchical layers: 3 (global), 9 (intermediate), and 27 (detailed) dimensions. Each higher layer synthesizes information from the layer below using learned logic rules. Fractal evolution enables integration of information from multiple vectors and generation of new patterns【60:17-20†Aurora Program Software Architecture.pdf】.

This is implemented in the VectorFractal class, which allows for evolving (combining) and expanding vectors at different levels of granularity.

5. Memory and Extension
AuroraMemory and Extender represent the storage and retrieval system:

Synthesized fractal vectors are stored together with their metadata (Ss, MetaM).

The extension process allows for reconstructing details from abstract layers, recovering granular information as needed【60:21-23†Aurora Program Software Architecture.pdf】.

This supports a bidirectional flow: from conceptual abstraction down to operational detail—crucial for adaptation and explainability.

6. Learning, Validation, and Storage Flow
Input and learning cycle: Each new vector is processed layer by layer, associating logic rules and metadata.

Coherence validation: Only vectors that are consistent with previously learned rules are stored, filtering out noise and redundancy.

Advantages: Incremental, autonomous learning; filtering of inconsistencies; efficiency; and full traceability of logical reasoning【60:25-26†Aurora Program Software Architecture.pdf】.

7. Glossary of Key Terms
Aurora: The intelligence system and ecosystem.

Triagate: Fundamental logic module (A, B, M → R).

Transcender: Higher-level structure; synthesis of three triages.

Evolver: Mechanism for adjusting and evolving logic across levels.

Fractal Vector: Hierarchical representation unit (3, 9, 27 dimensions).

Extender: Detail reconstruction and knowledge expansion.

MetaM: Meta-logic mapping between lower and higher layer functions.

Ss: Final synthesized value at the upper layer.

Coherence Validation: Logical consistency check for learning【60:27-28†Aurora Program Software Architecture.pdf】.

Code/Class Quick Reference
Trigate: Fundamental deduction, learning, and inference operator.

Transcender: Combination of triages for advanced logical synthesis.

VectorFractal: Fractal structure of 3, 9, and 27 dimensions.

Evolver: Meta-logic memory of synthesis patterns.

AuroraMemory / Extender: Adaptive memory and knowledge retrieval.

AuroraEcosistema: Orchestrator for the complete flow (processing, evolution, extension).

Aurora

'''


import hashlib
from copy import deepcopy

# Definición de funciones booleanas básicas (3 bits)
def bitwise_xor(a, b):
    return a ^ b

def bitwise_and(a, b):
    return a & b

def bitwise_or(a, b):
    return a | b

def bitwise_xnor(a, b):
    return ~(a ^ b) & 0b111

def bitwise_nand(a, b):
    return ~(a & b) & 0b111

def bitwise_nor(a, b):
    return ~(a | b) & 0b111

FUNCIONES_BOOL = {
    'XOR': bitwise_xor,
    'AND': bitwise_and,
    'OR': bitwise_or,
    'XNOR': bitwise_xnor,
    'NAND': bitwise_nand,
    'NOR': bitwise_nor
}

# Mapeo inverso para obtener nombres de funciones
FUNCIONES_INV = {v: k for k, v in FUNCIONES_BOOL.items()}

class Trigate:
    """Módulo básico de razonamiento basado en triángulos booleanos"""
    def __init__(self):
        self.M = None  # Función lógica actual
    
    def operar(self, A, B, M=None):
        """Operación básica del triage: R = M(A, B)"""
        funcion = FUNCIONES_BOOL[M] if M else self.M
        if not funcion:
            raise ValueError("Función lógica no definida")
        return funcion(A, B)
    
    def inferencia(self, A, B, M=None):
        """Modo 1: Calcular R dado A, B y M"""
        return self.operar(A, B, M)
    
    def aprendizaje(self, A, B, R):
        """Modo 2: Encontrar M que satisface R = M(A, B)"""
        candidatos = []
        for nombre, funcion in FUNCIONES_BOOL.items():
            if funcion(A, B) == R:
                candidatos.append(nombre)
        return candidatos
    
    def deduccion_inversa(self, M, R, entrada_conocida, es_A=True):
        """Modo 3: Encontrar entrada faltante dado M y R"""
        funcion = FUNCIONES_BOOL[M]
        soluciones = []
        for candidato in range(8):  # Todos los valores 3-bit (0-7)
            if es_A:
                resultado = funcion(entrada_conocida, candidato)
            else:
                resultado = funcion(candidato, entrada_conocida)
            if resultado == R:
                soluciones.append(candidato)
        return soluciones

class Transcender:
    """Estructura superior que combina tres triages"""
    def __init__(self, M1='XOR', M2='XOR', M3='XOR'):
        self.triage1 = Trigate()
        self.triage2 = Trigate()
        self.triage3 = Trigate()
        
        # Almacenar nombres de funciones para MetaM
        self.M1_name = M1
        self.M2_name = M2
        self.M3_name = M3
        
        # Configurar funciones lógicas iniciales
        self.triage1.M = FUNCIONES_BOOL[M1]
        self.triage2.M = FUNCIONES_BOOL[M2]
        self.triage3.M = FUNCIONES_BOOL[M3]
        
        self.evolver = Evolver()
        self.metaM_table = {}  # Tabla Ss -> MetaM
    
    def sintetizar(self, A, B, R):
        """Síntesis relacional basada en R"""
        S_bits = []
        for i in range(3):  # Procesar cada bit (0-2)
            bit_A = (A >> (2-i)) & 1
            bit_B = (B >> (2-i)) & 1
            bit_R = (R >> (2-i)) & 1
            
            if bit_R == 1:
                S_bit = bit_A ^ bit_B  # XOR
            else:
                S_bit = 1 - (bit_A ^ bit_B)  # XNOR
            S_bits.append(str(S_bit))
        
        return int(''.join(S_bits), 2)  # Convertir binario a entero
    
    def procesar(self, A, B, C):
        """Procesamiento completo de un Transcender"""
        # Paso 1: Operar triages inferiores
        R1 = self.triage1.inferencia(A, B)
        R2 = self.triage2.inferencia(B, C)
        R3 = self.triage3.inferencia(C, A)
        
        # Paso 2: Síntesis relacional
        S1 = self.sintetizar(A, B, R1)
        S2 = self.sintetizar(B, C, R2)
        S3 = self.sintetizar(C, A, R3)
        
        # Paso 3: Triage superior (aprendizaje)
        Ms_candidatas = self.triage1.aprendizaje(S1, S2, S3)
        Ms = Ms_candidatas[0] if Ms_candidatas else 'XOR'  # Selección simple
        
        # Paso 4: Calcular MetaM y Ss
        M_inferior = (self.M1_name, self.M2_name, self.M3_name)
        self.evolver.actualizar(M_inferior, Ms)
        
        # Correspondencia Ss ↔ MetaM
        Ss = self.calcular_ss(M_inferior, Ms)
        self.metaM_table[Ss] = (M_inferior, Ms)
        
        return {
            'R': (R1, R2, R3),
            'S': (S1, S2, S3),
            'Ms': Ms,
            'Ss': Ss,
            'MetaM': (M_inferior, Ms)
        }
    
    def calcular_ss(self, M_inferior, Ms):
        """Codifica (M1,M2,M3,Ms) en 3 bits (Ss) usando geometría booleana"""
        # Mapeo mejorado con codificación geométrica
        func_map = {
            'XOR': 0b00,  # 00
            'AND': 0b01,  # 01
            'OR': 0b10,   # 10
            'XNOR': 0b11, # 11
            'NAND': 0b01, # Mismo que AND por significado geométrico
            'NOR': 0b10   # Mismo que OR
        }
        
        # Codificar cada función en 2 bits
        bits = [func_map.get(m, 0b00) for m in M_inferior]  # 3 funciones × 2 bits
        bits.append(func_map.get(Ms, 0b00))
        
        # Combinar usando operación XOR fractal
        ss = 0
        for b in bits:
            ss ^= b  # Operación XOR acumulativa
            
        # Comprimir a 3 bits usando triángulo lógico
        bit1 = (ss >> 0) & 1
        bit2 = (ss >> 1) & 1
        bit3 = bit1 ^ bit2  # Tercer bit como "ángulo" lógico
        
        return (bit3 << 2) | (bit2 << 1) | bit1

class VectorFractal:
    def __init__(self, capa3=None, capa9=None, capa27=None):
        self.capa3 = capa3 or [0]*3
        self.capa9 = capa9 or [0]*9
        self.capa27 = capa27 or [0]*27
    
    def get_valor(self, capa, index):
        """Obtiene valor de una capa específica"""
        if capa == '3':
            return self.capa3[index] if index < len(self.capa3) else 0
        elif capa == '9':
            return self.capa9[index] if index < len(self.capa9) else 0
        else:  # '27'
            return self.capa27[index] if index < len(self.capa27) else 0
    
    def evolucionar(self, v2, v3):
        """Combina tres vectores en un nuevo vector fractal"""
        return VectorFractal(
            capa3=self._interactuar_capa(self.capa3, v2.capa3, v3.capa3, 3),
            capa9=self._interactuar_capa(self.capa9, v2.capa9, v3.capa9, 9),
            capa27=self._interactuar_capa(self.capa27, v2.capa27, v3.capa27, 27)
        )
    
    def _interactuar_capa(self, c1, c2, c3, size):
        """Interacción fractal de tres capas de igual dimensión"""
        nueva_capa = []
        for i in range(size):
            # Cada triplete (a,b,c) genera nuevo valor mediante transcender
            t = Transcender()
            resultado = t.procesar(
                c1[i] if i < len(c1) else 0, 
                c2[i] if i < len(c2) else 0, 
                c3[i] if i < len(c3) else 0
            )
            # Usamos Ss como valor evolucionado
            nueva_capa.append(resultado['Ss'])
        return nueva_capa

    def expandir(self, target_size):
        """Expansión adaptativa del vector"""
        if target_size <= 3:
            return self.capa3
        elif target_size <= 9:
            return self.capa3 + self.capa9
        else:
            return self.capa3 + self.capa9 + self.capa27

class Evolver:
    def __init__(self):
        self.meta_memory = {}  # {tuple(M1,M2,M3): {Ms: frequency}}
    
    def actualizar(self, M_inferior, Ms_superior):
        """Actualiza las relaciones entre funciones lógicas"""
        key = tuple(sorted(M_inferior))
        if key not in self.meta_memory:
            self.meta_memory[key] = {}
        
        if Ms_superior not in self.meta_memory[key]:
            self.meta_memory[key][Ms_superior] = 0
        
        self.meta_memory[key][Ms_superior] += 1
    
    def predecir_ms(self, M_inferior):
        """Predice Ms más probable para un conjunto dado de M"""
        key = tuple(sorted(M_inferior))
        if key not in self.meta_memory or not self.meta_memory[key]:
            return None
        
        return max(self.meta_memory[key], key=self.meta_memory[key].get)

class AuroraMemory:
    def __init__(self):
        self.vectores = {}  # Ss -> VectorFractal completo
        self.diccionario_inverso = {}  # Ss -> [vectores base]
    
    def almacenar(self, vector_fractal, Ss, entrada_base):
        """Almacena vector con metadatos"""
        self.vectores[Ss] = vector_fractal
        if Ss not in self.diccionario_inverso:
            self.diccionario_inverso[Ss] = []
        self.diccionario_inverso[Ss].append(entrada_base)
    
    def extender(self, Ss, metaM):
        """Reconstruye detalles desde representación comprimida"""
        if Ss not in self.vectores:
            return None
        
        vector = deepcopy(self.vectores[Ss])
        
        # Reconstrucción usando diccionario inverso
        if Ss in self.diccionario_inverso:
            mejor_base = min(self.diccionario_inverso[Ss],
                            key=lambda x: self.calcular_distancia(x, metaM))
            vector.capa_inferior = mejor_base[:27]
        
        return vector
    
    def calcular_distancia(self, base, metaM):
        """Métrica de similitud para reconstrucción"""
        # Implementación simplificada
        return sum(1 for m in base[3:6] if m in metaM[0])

class Extender:
    def __init__(self, memoria_vectores, diccionario_meta):
        self.memoria = memoria_vectores
        self.diccionario_meta = diccionario_meta
        self.plantillas_fractales = self._generar_plantillas_base()
    
    def _generar_plantillas_base(self):
        """Genera plantillas fractales fundamentales"""
        return {
            'triangulo': VectorFractal([1,0,1], [1,0,1,0,1,0,1,0,1], [1]*27),
            'linea': VectorFractal([0,1,0], [0,1,0,1,0,1,0,1,0], [0]*27),
            'espiral': VectorFractal([1,1,0], [1,1,0,0,1,1,0,0,1], [1,0]*13 + [1])
        }
    
    def extender(self, ss, nivel_detalle='completo', contexto=None):
        """
        Reconstruye conocimiento a múltiples niveles de detalle
        desde una representación comprimida (Ss)
        """
        # 1. Recuperar vector base y MetaM
        vector_base = next(v for v in self.memoria if self._calcular_ss(v) == ss)
        metaM = self.diccionario_meta.get(ss, {})
        
        # 2. Determinar nivel de reconstrucción
        if nivel_detalle == 'minimo':
            return self._reconstruir_minimo(vector_base, contexto)
        elif nivel_detalle == 'medio':
            return self._reconstruir_medio(vector_base, metaM, contexto)
        else:
            return self._reconstruir_completo(vector_base, metaM, contexto)
    
    def _reconstruir_minimo(self, vector, contexto):
        """Reconstrucción rápida: solo capa 3 + contexto"""
        return {
            'capa3': vector.capa3,
            'contexto': contexto,
            'tipo': self._identificar_patron(vector.capa3)
        }
    
    def _reconstruir_medio(self, vector, metaM, contexto):
        """Reconstrucción balanceada: capas 3 y 9"""
        capa9_reconstruida = []
        for i in range(0, 9, 3):
            # Usar triage inverso con MetaM
            valores = self._aplicar_triage_inverso(
                vector.capa3[i//3], 
                metaM, 
                contexto
            )
            capa9_reconstruida.extend(valores)
        
        return VectorFractal(
            capa3=vector.capa3,
            capa9=capa9_reconstruida
        )
    
    def _reconstruir_completo(self, vector, metaM, contexto):
        """Reconstrucción completa usando plantillas y MetaM"""
        patron = self._identificar_patron(vector.capa3)
        plantilla = self.plantillas_fractales.get(patron, self.plantillas_fractales['triangulo'])
        
        # Reconstrucción fractal adaptativa
        nueva_capa27 = []
        for i in range(27):
            if i < len(plantilla.capa27):
                # Aplicar transformación simplificada
                valor_base = plantilla.capa27[i]
                # Obtener valor de la capa 3 correspondiente (0-2)
                idx_capa3 = i // 9  # 27 elementos / 9 = 3 grupos
                nuevo_valor = self._aplicar_transformacion_metaM(
                    valor_base, 
                    vector.capa3[idx_capa3]
                )
                nueva_capa27.append(nuevo_valor)
            else:
                # Completar con valor neutro
                nueva_capa27.append(0)
        
        return VectorFractal(
            capa3=vector.capa3,
            capa9=vector.capa9,
            capa27=nueva_capa27
        )
    
    def _aplicar_transformacion_metaM(self, valor_base, valor_superior):
        """Transforma valores usando valor superior (versión simplificada)"""
        # Transformación booleana básica: invertir si valor_superior es 1
        return valor_base if valor_superior == 0 else 1 - valor_base
    
    def _aplicar_triage_inverso(self, valor_superior, metaM, contexto):
        """Deducción inversa usando triage y MetaM"""
        # Buscar en MetaM patrones similares
        claves_similares = [k for k in metaM.keys() if k[0] == valor_superior]
        
        if claves_similares:
            # Seleccionar el patrón más frecuente
            clave = max(claves_similares, key=lambda k: metaM[k]['frecuencia'])
            return list(clave[1:])
        
        # Si no hay coincidencia, usar patrón contextual
        if contexto == 'temporal':
            return [valor_superior, 0, 1]
        elif contexto == 'espacial':
            return [1, valor_superior, 0]
        else:
            return [valor_superior, valor_superior, valor_superior]
        '''    
         def _aplicar_transformacion_metaM(self, valor_base, valor_superior, metaM):
        """Transforma valores usando relaciones MetaM"""
        # Buscar transformación específica
        for (sup, inf), datos in metaM.items():
            if sup == valor_superior:
                # Aplicar función de transformación
                if datos['funcion'] == 'incremento':
                    return valor_base + datos['parametro']
                elif datos['funcion'] == 'rotacion':
                    return (valor_base + datos['parametro']) % 2
                elif datos['funcion'] == 'xor':
                    return valor_base ^ datos['parametro']



                    '''
        
        
    def _identificar_patron(self, capa3):
        """Clasifica patrones fundamentales en la capa 3"""
        umbral = 0.7
        if capa3[0] > umbral and capa3[1] < (1 - umbral) and capa3[2] > umbral:
            return 'triangulo'
        elif capa3[0] < (1 - umbral) and capa3[1] > umbral and capa3[2] < (1 - umbral):
            return 'linea'
        elif capa3[0] > umbral and capa3[1] > umbral and capa3[2] < (1 - umbral):
            return 'espiral'
        else:
            return 'complejo'
    
    def _calcular_ss(self, vector):
        """Calcula hash único para identificación"""
        return hash(tuple(vector.capa3 + vector.capa9 + vector.capa27))

class AuroraEcosistema:
    def __init__(self):
        self.memoria = []
        self.diccionario_meta = {}  # Ss -> MetaM
        self.contexto = "general"
        self.extender = Extender(self.memoria, self.diccionario_meta)
    
    def establecer_contexto(self, nuevo_contexto):
        """Actualiza el contexto para reconstrucciones"""
        self.contexto = nuevo_contexto
    
    def _calcular_ss(self, vector):
        """Código de identificación fractal único"""
        return hash(tuple(vector.capa3 + vector.capa9 + vector.capa27))
    
    def procesar_entrada(self, v1, v2, v3):
        """Procesa tres vectores en el ecosistema"""
        # 1. Evolución fractal
        nuevo_vector = v1.evolucionar(v2, v3)
        
        # 2. Extraer MetaM del proceso
        metaM = self._extraer_metaM(v1, v2, v3, nuevo_vector)
        
        # 3. Almacenamiento en memoria
        ss = self._calcular_ss(nuevo_vector)
        self.memoria.append(nuevo_vector)
        self.diccionario_meta[ss] = metaM
        
        return nuevo_vector, ss
    
    def _extraer_metaM(self, v1, v2, v3, resultado):
        """Extrae relaciones MetaM entre vectores originales y resultado"""
        metaM = {}
        for capa, size in [('3', 3), ('9', 9), ('27', 27)]:
            for i in range(size):
                key = (v1.get_valor(capa, i), 
                       v2.get_valor(capa, i), 
                       v3.get_valor(capa, i))
                metaM[key] = resultado.get_valor(capa, i)
        return metaM
    
    def recuperar_conocimiento(self, ss, nivel_detalle='completo'):
        """Recupera conocimiento con nivel de detalle escalable"""
        return self.extender.extender(ss, nivel_detalle, self.contexto)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Crear instancia del ecosistema Aurora
    aurora = AuroraEcosistema()
    aurora.establecer_contexto("espacial")

    # Crear vectores base
    v1 = VectorFractal([1,0,1], [1,0,1,0,1,0,1,0,1], [1]*27)
    v2 = VectorFractal([0,1,0], [0,1,0,1,0,1,0,1,0], [0]*27)
    v3 = VectorFractal([1,1,0], [1,1,0,0,1,1,0,0,1], [1,0]*13 + [1])

    # Procesamiento y evolución
    v_evolucionado, ss_target = aurora.procesar_entrada(v1, v2, v3)
    print(f"Vector evolucionado creado con SS: {ss_target}")

    # Reconstrucción con Extender
    print("\n=== Reconstrucción Mínima (Concepto) ===")
    concepto = aurora.recuperar_conocimiento(ss_target, 'minimo')
    print(concepto)

    print("\n=== Reconstrucción Media (Estructura) ===")
    estructura = aurora.recuperar_conocimiento(ss_target, 'medio')
    print(f"Capa 3: {estructura.capa3}")
    print(f"Capa 9: {estructura.capa9[:6]}...")

    print("\n=== Reconstrucción Completa (Detalle) ===")
    detalle = aurora.recuperar_conocimiento(ss_target, 'completo')
    print(f"Capa 3: {detalle.capa3}")
    print(f"Capa 9: {detalle.capa9[:3]}...")
    print(f"Capa 27: {detalle.capa27[:6]}...")

    # Cambiar contexto y reconstruir
    print("\n=== Reconstrucción con Nuevo Contexto (Temporal) ===")
    aurora.establecer_contexto("temporal")
    detalle_temporal = aurora.recuperar_conocimiento(ss_target, 'completo')
    print(f"Capa 27: {detalle_temporal.capa27[:6]}...")