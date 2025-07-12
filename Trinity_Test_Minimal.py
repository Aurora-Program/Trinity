import random
from collections import defaultdict
import time

print("AURORA_FRACTAL_SYNTHESIS: Module loaded successfully")

# ==============================================================================
#  AURORA FRACTAL SYNTHESIS SYSTEM - ARQUITECTURA CORRECTA
# ==============================================================================

class Trigate:
    """Unidad básica de razonamiento ternario"""
    def __init__(self, A=None, B=None, R=None, M=None):
        self.A, self.B, self.R = A, B, R
        self.M = M if M is not None else [0, 0, 0]

    def _xor(self, b1, b2):
        if b1 is None or b2 is None: return None
        return 1 if b1 != b2 else 0

    def _xnor(self, b1, b2):
        if b1 is None or b2 is None: return None
        return 1 if b1 == b2 else 0

    def inferir(self):
        """Calcula R basado en A, B y M"""
        if self.A is None or self.B is None:
            self.R = [None, None, None]
            return self.R

        r = [
            self._xor(self.A[0], self.B[0]) if self.M[0] == 0 else self._xnor(self.A[0], self.B[0]),
            self._xor(self.A[1], self.B[1]) if self.M[1] == 0 else self._xnor(self.A[1], self.B[1]),
            self._xor(self.A[2], self.B[2]) if self.M[2] == 0 else self._xnor(self.A[2], self.B[2])
        ]
        
        self.R = r
        return self.R

class Transcender:
    """TRANSCENDER: Síntesis de vectores fractales 3→9→27, descubre Ms"""
    def __init__(self):  
        self.transcenders = [Trigate() for _ in range(13)]
        self._setup_aurora_architecture()
        
        # Registro de Ms descubiertos por capa
        self.layer1_Ms = []  # Ms de vectores 3-bit
        self.layer2_Ms = []  # Ms de vectores 9-bit 
        self.layer3_Ms = []  # Ms de vectores 27-bit
        
    def _setup_aurora_architecture(self):
        """Configuración de 13 Transcenders según documento Aurora"""
        # 9 Transcenders Layer 1→2 (3→9 bits)
        for i in range(9):
            self.transcenders[i].M = [i % 2, (i + 1) % 2, i % 2]
        
        # 3 Transcenders Layer 2→3 (9→27 bits)
        for i in range(3):
            self.transcenders[9 + i].M = [1, 0, 1]
        
        # 1 Transcender maestro (síntesis final)
        self.transcenders[12].M = [1, 1, 0]

    def fractal_synthesis(self, input_vectors_9bit):
        """
        Síntesis fractal Aurora: 9 → 27 bits
        Cada vector de entrada es ahora de 9 bits (3 vectores de 3 bits cada uno)
        """
        print(f"\n🔄 Síntesis fractal Aurora: {len(input_vectors_9bit)} vectores 9-bit")
        
        # CAPA 1: 9-bit → 27-bit (usando 9 Transcenders)
        layer1_results = []
        layer1_Ms_discovered = []
        
        for i in range(0, len(input_vectors_9bit) - 1, 2):
            if i + 1 < len(input_vectors_9bit):
                transcender_idx = (i // 2) % 9
                transcender = self.transcenders[transcender_idx]
                
                # Tomar solo los primeros 3 bits de cada vector 9-bit para el Trigate
                vector_a = input_vectors_9bit[i][:3]
                vector_b = input_vectors_9bit[i + 1][:3]
                
                # Síntesis
                transcender.A = vector_a
                transcender.B = vector_b
                synthesis_result = transcender.inferir()
                
                # Expandir resultado a 9-bit combinando con información adicional
                expanded_result = synthesis_result + input_vectors_9bit[i][3:6] + input_vectors_9bit[i + 1][3:6]
                layer1_results.append(expanded_result)
                layer1_Ms_discovered.append(transcender.M.copy())
                
                print(f"   L1[{transcender_idx}]: {vector_a} + {vector_b} → {synthesis_result} → {expanded_result}")
        
        self.layer1_Ms.extend(layer1_Ms_discovered)
        
        # CAPA 2: 27-bit synthesis (usando 3 Transcenders)
        layer2_results = []
        layer2_Ms_discovered = []
        
        if len(layer1_results) >= 2:
            for i in range(0, len(layer1_results) - 1, 2):
                if i + 1 < len(layer1_results):
                    transcender_idx = 9 + (i // 2) % 3
                    transcender = self.transcenders[transcender_idx]
                    
                    # Síntesis usando los primeros 3 bits
                    transcender.A = layer1_results[i][:3]
                    transcender.B = layer1_results[i + 1][:3]
                    synthesis_result = transcender.inferir()
                    
                    # Crear resultado expandido de 27-bit
                    expanded_result = synthesis_result + layer1_results[i][3:] + layer1_results[i + 1][3:]
                    layer2_results.append(expanded_result)
                    layer2_Ms_discovered.append(transcender.M.copy())
                    
                    print(f"   L2[{transcender_idx}]: {layer1_results[i][:3]} + {layer1_results[i + 1][:3]} → {synthesis_result}")
        
        self.layer2_Ms.extend(layer2_Ms_discovered)
        
        # CAPA 3: Síntesis final (usando Transcender maestro)
        final_synthesis = None
        final_M = None
        
        if len(layer2_results) >= 1:
            master_transcender = self.transcenders[12]
            
            if len(layer2_results) >= 2:
                master_transcender.A = layer2_results[0][:3]
                master_transcender.B = layer2_results[1][:3]
            else:
                master_transcender.A = layer2_results[0][:3]
                master_transcender.B = [0, 0, 0]  # Vector por defecto
                
            final_synthesis = master_transcender.inferir()
            final_M = master_transcender.M.copy()
            self.layer3_Ms.append(final_M)
            
            print(f"   L3[12]: {master_transcender.A} + {master_transcender.B} → {final_synthesis}")
        
        return {
            "layer1_synthesis": layer1_results,
            "layer1_Ms": layer1_Ms_discovered,
            "layer2_synthesis": layer2_results, 
            "layer2_Ms": layer2_Ms_discovered,
            "final_synthesis": final_synthesis,
            "final_M": final_M,
            "all_Ms": {
                "layer1": self.layer1_Ms,
                "layer2": self.layer2_Ms,
                "layer3": self.layer3_Ms
            }
        }

class Evolver:
    """EVOLVER: De Ms descubre MetaM y arquetipos"""
    def __init__(self):
        self.discovered_MetaMs = []
        self.archetypal_patterns = []
        self.axiom_spaces = {}  # Diferentes espacios de axiomas
        
    def discover_MetaM_from_Ms(self, all_Ms_by_layer):
        """Descubre MetaM analizando patrones en los Ms de todas las capas"""
        print(f"\n🧬 Evolver: Descubriendo MetaM desde Ms...")
        
        # Analizar patrones en Ms de cada capa
        layer1_Ms = all_Ms_by_layer.get("layer1", [])
        layer2_Ms = all_Ms_by_layer.get("layer2", [])
        layer3_Ms = all_Ms_by_layer.get("layer3", [])
        
        print(f"   Analizando: L1={len(layer1_Ms)} Ms, L2={len(layer2_Ms)} Ms, L3={len(layer3_Ms)} Ms")
        
        # Buscar patrones arquetípicos (simplificado)
        archetypal_patterns = self.find_archetypal_patterns_simple(layer1_Ms, layer2_Ms, layer3_Ms)
        
        # Generar MetaMs basados en arquetipos (máximo 3 para ser preciso)
        discovered_MetaMs = []
        
        for i, archetype in enumerate(archetypal_patterns[:3]):  # Limitar a 3
            meta_M = self.synthesize_MetaM_from_archetype(archetype)
            discovered_MetaMs.append(meta_M)
            
            print(f"   🏛️  Arquetipo {i+1}: {archetype['pattern']} → MetaM: {meta_M}")
        
        self.discovered_MetaMs.extend(discovered_MetaMs)
        self.archetypal_patterns.extend(archetypal_patterns[:3])
        
        return {
            "MetaMs": discovered_MetaMs,
            "archetypal_patterns": archetypal_patterns[:3]
        }
    
    def find_archetypal_patterns_simple(self, layer1_Ms, layer2_Ms, layer3_Ms):
        """Encuentra patrones arquetípicos simples"""
        patterns = []
        
        # Patrón 1: M más frecuente en layer1
        if layer1_Ms:
            most_common_M = max(set(map(tuple, layer1_Ms)), key=lambda x: [tuple(m) for m in layer1_Ms].count(x))
            patterns.append({
                "type": "frequent_pattern",
                "pattern": list(most_common_M),
                "strength": 1.0,
                "description": f"Patrón más frecuente: {list(most_common_M)}"
            })
        
        # Patrón 2: Patrón por defecto para estabilidad
        patterns.append({
            "type": "default_pattern", 
            "pattern": [1, 0, 1],
            "strength": 0.5,
            "description": "Patrón por defecto"
        })
        
        print(f"   📊 Encontrados {len(patterns)} patrones arquetípicos")
        return patterns
    
    def synthesize_MetaM_from_archetype(self, archetype):
        """Sintetiza MetaM desde un patrón arquetípico"""
        return archetype["pattern"]
    
    def create_axiom_space(self, space_name, MetaMs, archetypal_patterns):
        """Crea un espacio de axiomas con reglas específicas"""
        print(f"   🌌 Creando espacio de axiomas: '{space_name}'")
        
        # Crear axiomas reales basados en MetaMs
        axioms = self.generate_axioms_from_MetaMs(MetaMs)
        
        axiom_space = {
            "name": space_name,
            "MetaMs": MetaMs,
            "archetypal_patterns": archetypal_patterns,
            "axioms": axioms,  # Reglas que deben cumplirse
            "validation_rules": self.create_validation_rules(MetaMs)
        }
        
        self.axiom_spaces[space_name] = axiom_space
        return axiom_space
    
    def generate_axioms_from_MetaMs(self, MetaMs):
        """Genera axiomas reales desde MetaMs"""
        axioms = []
        
        for i, meta_M in enumerate(MetaMs):
            # Axioma 1: Conservación de bits activos
            active_bits = sum(meta_M)
            axioms.append({
                "id": f"conservation_{i}",
                "rule": "conservation_of_active_bits",
                "value": active_bits,
                "description": f"Debe conservar {active_bits} bits activos"
            })
            
            # Axioma 2: Patrón específico
            axioms.append({
                "id": f"pattern_{i}",
                "rule": "pattern_match",
                "pattern": meta_M,
                "description": f"Debe coincidir con patrón {meta_M}"
            })
        
        return axioms
    
    def create_validation_rules(self, MetaMs):
        """Crea reglas de validación binarias"""
        return {
            "must_have_active_bits": True,
            "allowed_patterns": MetaMs,
            "min_pattern_match": 1  # Al menos 1 patrón debe coincidir
        }
    
    def validate_against_axioms(self, new_case_Ms, axiom_space):
        """Validación BINARIA: ¿Cumple los axiomas? SÍ/NO"""
        axioms = axiom_space["axioms"]
        validation_rules = axiom_space["validation_rules"]
        
        if not new_case_Ms:
            return False
        
        print(f"   🔍 Validando contra {len(axioms)} axiomas de '{axiom_space['name']}'")
        
        # Validar cada M del caso nuevo
        valid_patterns = 0
        
        for case_M in new_case_Ms:
            # Regla 1: Debe tener bits activos
            if sum(case_M) == 0:
                print(f"      ❌ FALLA: Vector nulo {case_M}")
                continue
                
            # Regla 2: Debe coincidir con al menos un patrón permitido
            pattern_matches = False
            for allowed_pattern in validation_rules["allowed_patterns"]:
                if case_M == allowed_pattern:
                    pattern_matches = True
                    print(f"      ✅ VÁLIDO: {case_M} coincide con {allowed_pattern}")
                    break
            
            if pattern_matches:
                valid_patterns += 1
            else:
                print(f"      ⚠️  PARCIAL: {case_M} no coincide exactamente")
                # Validar conservación de bits activos
                case_active_bits = sum(case_M)
                for axiom in axioms:
                    if axiom["rule"] == "conservation_of_active_bits":
                        if case_active_bits == axiom["value"]:
                            valid_patterns += 1
                            print(f"      ✅ VÁLIDO: Conserva {case_active_bits} bits activos")
                            break
        
        # Decisión final: ¿Es válido para este espacio?
        is_valid = valid_patterns >= validation_rules["min_pattern_match"]
        
        print(f"   📊 Resultado: {valid_patterns}/{len(new_case_Ms)} patrones válidos")
        print(f"   🎯 Espacio '{axiom_space['name']}': {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'}")
        
        return is_valid

# ...existing code...

class Extender:
    """EXTENDER: Despliega aprendizaje a vectores nuevos no sintetizados"""
    def __init__(self, transcender, evolver):
        self.transcender = transcender
        self.evolver = evolver
        self.deployment_rules = []
        
    def deploy_intelligence(self, new_vectors_9bit, target_space=None):
        """Despliega inteligencia aprendida a nuevos vectores"""
        print(f"\n🚀 Extender: Desplegando inteligencia a {len(new_vectors_9bit)} vectores nuevos")
        
        # 1. Realizar síntesis fractal de los nuevos vectores
        synthesis_result = self.transcender.fractal_synthesis(new_vectors_9bit)
        new_Ms = synthesis_result["all_Ms"]
        
        # 2. Buscar espacio de axiomas VÁLIDO (no "más compatible")
        valid_space = self.find_valid_axiom_space(new_Ms, target_space)
        
        if valid_space:
            print(f"   🎯 Usando espacio VÁLIDO: '{valid_space['name']}'")
            
            # 3. Aplicar MetaMs del espacio válido
            deployed_results = self.apply_space_MetaMs(synthesis_result, valid_space)
            
            return deployed_results
        
        else:
            print(f"   🆕 Creando nuevo espacio: No hay espacios válidos")
            
            # 4. Crear nuevo espacio si ninguno es válido
            new_space = self.create_new_space_for_vectors(new_Ms)
            deployed_results = self.apply_space_MetaMs(synthesis_result, new_space)
            
            return deployed_results
    
    def find_valid_axiom_space(self, new_Ms, target_space=None):
        """Encuentra un espacio de axiomas VÁLIDO (no compatible)"""
        if target_space and target_space in self.evolver.axiom_spaces:
            space = self.evolver.axiom_spaces[target_space]
            all_new_Ms = []
            for layer_Ms in new_Ms.values():
                all_new_Ms.extend(layer_Ms)
            
            if self.evolver.validate_against_axioms(all_new_Ms, space):
                return space
        
        # Probar todos los espacios hasta encontrar uno válido
        all_new_Ms = []
        for layer_Ms in new_Ms.values():
            all_new_Ms.extend(layer_Ms)
        
        if not all_new_Ms:
            return None
        
        for space_name, space in self.evolver.axiom_spaces.items():
            is_valid = self.evolver.validate_against_axioms(all_new_Ms, space)
            
            if is_valid:
                print(f"   ✅ Espacio '{space_name}': VÁLIDO")
                return space
            else:
                print(f"   ❌ Espacio '{space_name}': INVÁLIDO")
        
        # Ningún espacio es válido
        return None
    
    def apply_space_MetaMs(self, synthesis_result, axiom_space):
        """Aplica MetaMs del espacio de axiomas a los resultados de síntesis"""
        space_MetaMs = axiom_space["MetaMs"]
        
        print(f"   🔧 Aplicando {len(space_MetaMs)} MetaMs del espacio '{axiom_space['name']}'")
        
        # Aplicar cada MetaM como regla de transformación
        transformed_results = []
        
        for meta_M in space_MetaMs:
            # Usar MetaM como M en un Trigate para transformar resultado final
            transformation_trigate = Trigate()
            transformation_trigate.M = meta_M
            
            # Asegurar que los vectores no sean None
            final_synthesis = synthesis_result.get("final_synthesis", [0, 0, 0])
            layer2_synthesis = synthesis_result.get("layer2_synthesis", [[0, 0, 0]])
            
            if final_synthesis is None:
                final_synthesis = [0, 0, 0]
            
            transformation_trigate.A = final_synthesis
            transformation_trigate.B = layer2_synthesis[0][:3] if layer2_synthesis and layer2_synthesis[0] else [0, 0, 0]
            
            transformed_result = transformation_trigate.inferir()
            
            # Asegurar que el resultado no sea None
            if transformed_result is None or any(x is None for x in transformed_result):
                transformed_result = [0, 0, 0]
            
            transformed_results.append(transformed_result)
            
            print(f"      MetaM {meta_M}: {final_synthesis} → {transformed_result}")
        
        return {
            "original_synthesis": synthesis_result,
            "applied_space": axiom_space["name"],
            "MetaMs_applied": space_MetaMs,
            "transformed_results": transformed_results,
            "final_intelligence": transformed_results[-1] if transformed_results else [0, 0, 0]
        }
    
    def create_new_space_for_vectors(self, new_Ms):
        """Crea un nuevo espacio de axiomas para vectores que no cumplen ningún axioma existente"""
        space_name = f"space_{len(self.evolver.axiom_spaces) + 1}"
        
        # Descubrir MetaMs para este nuevo espacio
        evolution_result = self.evolver.discover_MetaM_from_Ms(new_Ms)
        
        # Crear espacio
        new_space = self.evolver.create_axiom_space(
            space_name,
            evolution_result["MetaMs"],
            evolution_result["archetypal_patterns"]
        )
        
        return new_space

# ELIMINAR LA SEGUNDA CLASE EXTENDER DUPLICADA
# (Las líneas 427-556 están duplicadas y causan el error)
# class Extender: ... # ELIMINAR ESTA CLASE DUPLICADA

# ==============================================================================
#  SISTEMA AURORA FRACTAL COMPLETO
# ==============================================================================
class AuroraFractalSystem:
    """Sistema Aurora completo: Transcender → Evolver → Extender"""
    def __init__(self):
        self.transcender = Transcender()
        self.evolver = Evolver()
        self.extender = Extender(self.transcender, self.evolver)
        
        # VECTORES FONÉTICOS RICOS (3 vectores de 3 dimensiones = 9 bits total)
        self.phonetic_vectors_9bit = {
            # VOCALES - Vectores más diferenciados
            'a': [1, 1, 0, 0, 1, 0, 1, 0, 0],  # [sonido][lugar][manera]
            'e': [1, 1, 1, 0, 1, 0, 1, 0, 1],  # vocal media-alta
            'i': [1, 0, 1, 0, 1, 0, 1, 1, 0],  # vocal cerrada anterior
            'o': [1, 1, 0, 0, 0, 1, 1, 0, 0],  # vocal media posterior
            'u': [1, 0, 0, 0, 0, 1, 1, 1, 0],  # vocal cerrada posterior
            
            # CONSONANTES NASALES
            'm': [1, 0, 0, 1, 0, 0, 0, 0, 1],  # nasal bilabial
            'n': [1, 0, 0, 1, 1, 0, 0, 0, 1],  # nasal dental
            'ñ': [1, 0, 1, 1, 1, 1, 0, 0, 1],  # nasal palatal
            
            # CONSONANTES LÍQUIDAS
            'l': [1, 1, 1, 1, 1, 0, 0, 1, 0],  # lateral dental
            'r': [1, 1, 1, 1, 1, 0, 0, 1, 1],  # vibrante dental
            
            # CONSONANTES OCLUSIVAS SORDAS
            'p': [0, 0, 0, 1, 0, 0, 0, 0, 0],  # oclusiva bilabial sorda
            't': [0, 0, 0, 1, 1, 0, 0, 0, 0],  # oclusiva dental sorda
            'k': [0, 0, 0, 1, 0, 1, 0, 0, 0],  # oclusiva velar sorda
            'c': [0, 0, 0, 1, 1, 0, 0, 0, 0],  # como 't'
            
            # CONSONANTES OCLUSIVAS SONORAS
            'b': [1, 0, 0, 1, 0, 0, 0, 0, 0],  # oclusiva bilabial sonora
            'd': [1, 0, 0, 1, 1, 0, 0, 0, 0],  # oclusiva dental sonora
            'g': [1, 0, 0, 1, 0, 1, 0, 0, 0],  # oclusiva velar sonora
            
            # CONSONANTES FRICATIVAS
            'f': [0, 1, 1, 1, 0, 0, 0, 1, 0],  # fricativa bilabial sorda
            's': [0, 1, 1, 1, 1, 0, 0, 1, 0],  # fricativa dental sorda
            'j': [0, 1, 1, 1, 0, 1, 0, 1, 0],  # fricativa velar sorda
            'z': [1, 1, 1, 1, 1, 0, 0, 1, 0],  # fricativa dental sonora
            'v': [1, 1, 1, 1, 0, 0, 0, 1, 0],  # fricativa bilabial sonora
        }

# ...existing code...
# ==============================================================================
#  PRUEBA DEL SISTEMA AURORA FRACTAL COMPLETO
# ==============================================================================
def run_aurora_fractal_test():
    print("\n" + "="*70)
    print("🎯 AURORA FRACTAL SYSTEM TEST")
    print("   TRANSCENDER → EVOLVER → EXTENDER")
    print("   Síntesis Fractal 9→27 + MetaM + Despliegue Inteligente")
    print("="*70)
    
    system = AuroraFractalSystem()
    
    # Corpus de entrenamiento
    training_corpus = [
        ("casa", [1, 3]),      # ca-sa
        ("mesa", [1, 3]),      # me-sa  
        ("marco", [2, 4]),     # mar-co
        ("campo", [2, 5]),     # cam-po
        ("mundo", [2, 5]),     # mun-do
        ("libro", [2, 5]),     # li-bro
        ("musica", [1, 3, 5]), # mú-si-ca
        ("numero", [1, 3, 5]), # nú-me-ro
    ]
    
    print(f"📚 Corpus de entrenamiento: {len(training_corpus)} casos")
    
    # ENTRENAMIENTO
    system.train_aurora_system(training_corpus)
    
    # ANÁLISIS DEL SISTEMA ENTRENADO
    print(f"\n📊 ANÁLISIS DEL SISTEMA ENTRENADO:")
    print(f"   Ms descubiertos L1: {len(system.transcender.layer1_Ms)}")
    print(f"   Ms descubiertos L2: {len(system.transcender.layer2_Ms)}")
    print(f"   Ms descubiertos L3: {len(system.transcender.layer3_Ms)}")
    print(f"   MetaMs descubiertos: {len(system.evolver.discovered_MetaMs)}")
    print(f"   Arquetipos encontrados: {len(system.evolver.archetypal_patterns)}")
    print(f"   Espacios de axiomas: {len(system.evolver.axiom_spaces)}")
    
    # PREDICCIONES
    print(f"\n🔮 PREDICCIONES AURORA:")
    test_words = ["grupo", "tiempo", "trabajo", "sistema"]
    
    results = []
    for word in test_words:
        syllables = system.predict_aurora_syllables(word)
        results.append({
            "word": word,
            "syllables": syllables
        })
    
    # RESULTADOS FINALES
    print(f"\n🎯 RESULTADOS FINALES:")
    for result in results:
        print(f"   '{result['word']}' → {result['syllables']}")
    
    print(f"\n✅ SISTEMA AURORA FRACTAL COMPLETADO")
    return system, results

# ==============================================================================
#  EJECUCIÓN
# ==============================================================================
print("DEBUG: About to test Aurora Fractal System...")

try:
    # Ejecutar sistema Aurora fractal completo
    aurora_system, fractal_results = run_aurora_fractal_test()
    
    print("DEBUG: ✅ Aurora Fractal System completed successfully")
    
except Exception as e:
    print(f"DEBUG: ❌ Error in Aurora Fractal System: {e}")
    import traceback
    traceback.print_exc()

print("DEBUG: Script completed successfully")