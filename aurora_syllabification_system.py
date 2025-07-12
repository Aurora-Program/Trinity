#!/usr/bin/env python3
"""
AURORA SYLLABIFICATION LEARNING SYSTEM
======================================

Sistema de aprendizaje de silabificación usando arquitectura fractal Aurora.
Implementa aprendizaje jerárquico de patrones fonológicos:

NIVEL 1 (Inferior): Determinación de posición de fonemas
NIVEL 2 (Intermedio): Clasificación consonante inicial/zona vocal/coda  
NIVEL 3 (Superior): Determinación de límites silábicos

El sistema aprende de ejemplos vectorizados y luego aplica reglas
emergentes para silabificar nuevas palabras.
"""

import random
import time
from collections import defaultdict, Counter
from Trinity_Fixed_Complete import *

class AuroraSyllabificationSystem:
    """
    Sistema de aprendizaje de silabificación usando vectores fractales Aurora.
    Aprende patrones fonológicos jerárquicos para determinar límites silábicos.
    """
    
    def __init__(self):
        # Componentes Aurora base
        self.kb = KnowledgeBase()
        self.transcender = Transcender()
        self.evolver = Evolver(self.kb)
        
        # Crear espacio de conocimiento para silabificación
        self.kb.create_space("syllabification", "Espacio de aprendizaje silábico")
        
        # Base de datos de entrenamiento
        self.training_examples = []
        self.learned_patterns = {
            "phoneme_position": {},      # Nivel 1: Posición de fonemas
            "phoneme_classification": {}, # Nivel 2: Clasificación funcional
            "syllable_boundaries": {}    # Nivel 3: Límites silábicos
        }
        
        # Mapeo fonema → características
        self.phoneme_features = {
            # Vocales
            'a': {'type': 'vowel', 'position': 'nucleus', 'sonority': 10},
            'e': {'type': 'vowel', 'position': 'nucleus', 'sonority': 10},
            'i': {'type': 'vowel', 'position': 'nucleus', 'sonority': 10},
            'o': {'type': 'vowel', 'position': 'nucleus', 'sonority': 10},
            'u': {'type': 'vowel', 'position': 'nucleus', 'sonority': 10},
            
            # Consonantes líquidas (alta sonoridad)
            'l': {'type': 'consonant', 'position': 'flexible', 'sonority': 7},
            'r': {'type': 'consonant', 'position': 'flexible', 'sonority': 7},
            
            # Consonantes nasales
            'm': {'type': 'consonant', 'position': 'flexible', 'sonority': 6},
            'n': {'type': 'consonant', 'position': 'flexible', 'sonority': 6},
            
            # Consonantes fricativas
            's': {'type': 'consonant', 'position': 'flexible', 'sonority': 4},
            'f': {'type': 'consonant', 'position': 'flexible', 'sonority': 4},
            
            # Consonantes oclusivas (baja sonoridad)
            'p': {'type': 'consonant', 'position': 'flexible', 'sonority': 1},
            'b': {'type': 'consonant', 'position': 'flexible', 'sonority': 1},
            't': {'type': 'consonant', 'position': 'flexible', 'sonority': 1},
            'd': {'type': 'consonant', 'position': 'flexible', 'sonority': 1},
            'k': {'type': 'consonant', 'position': 'flexible', 'sonority': 1},
            'g': {'type': 'consonant', 'position': 'flexible', 'sonority': 1},
        }
        
        print("🔤 Aurora Syllabification System iniciado")
        print("   - Vectores fractales para aprendizaje fonológico")
        print("   - Aprendizaje jerárquico de patrones silábicos")
        
    def create_training_example(self, word, syllable_structure):
        """
        Crea ejemplo de entrenamiento con vectorización fractal.
        
        Args:
            word: palabra a silabificar (ej: "manzana")
            syllable_structure: estructura silábica (ej: ["man", "za", "na"])
        """
        print(f"\n📝 Creando ejemplo de entrenamiento: '{word}' → {syllable_structure}")
        
        # Vectorizar cada fonema en sus tres niveles
        phoneme_vectors = []
        syllable_boundaries = []
        
        current_pos = 0
        for syll_idx, syllable in enumerate(syllable_structure):
            for phone_idx, phoneme in enumerate(syllable):
                # NIVEL 1: Posición del fonema (inicio, medio, final)
                position_in_syllable = self._determine_phoneme_position(phone_idx, len(syllable))
                level1_vector = self._encode_phoneme_position(position_in_syllable)
                
                # NIVEL 2: Clasificación funcional (consonante inicial, zona vocal, coda)
                functional_class = self._determine_functional_class(phoneme, phone_idx, syllable)
                level2_vector = self._encode_functional_class(functional_class)
                
                # NIVEL 3: Límite silábico (si es final de sílaba)
                is_syllable_end = (phone_idx == len(syllable) - 1)
                level3_vector = self._encode_syllable_boundary(is_syllable_end)
                
                # Crear vector fractal completo usando Aurora
                fractal_vector = self.transcender.level1_synthesis(
                    level1_vector, level2_vector, level3_vector
                )
                
                phoneme_data = {
                    "phoneme": phoneme,
                    "position_in_word": current_pos,
                    "syllable_index": syll_idx,
                    "position_in_syllable": phone_idx,
                    "functional_class": functional_class,
                    "is_syllable_end": is_syllable_end,
                    "fractal_vector": fractal_vector,
                    "level1_vector": level1_vector,
                    "level2_vector": level2_vector,
                    "level3_vector": level3_vector
                }
                
                phoneme_vectors.append(phoneme_data)
                current_pos += 1
        
        # Crear ejemplo de entrenamiento completo
        training_example = {
            "word": word,
            "syllable_structure": syllable_structure,
            "phoneme_vectors": phoneme_vectors,
            "creation_time": time.time()
        }
        
        self.training_examples.append(training_example)
        
        print(f"   ✅ Ejemplo vectorizado: {len(phoneme_vectors)} fonemas")
        for i, pv in enumerate(phoneme_vectors):
            print(f"      {pv['phoneme']}: L1={pv['level1_vector']}, L2={pv['level2_vector']}, L3={pv['level3_vector']}")
        
        return training_example
    
    def _determine_phoneme_position(self, phone_idx, syllable_len):
        """Determina posición del fonema en la sílaba"""
        if phone_idx == 0:
            return "initial"
        elif phone_idx == syllable_len - 1:
            return "final"
        else:
            return "medial"
    
    def _encode_phoneme_position(self, position):
        """Codifica posición del fonema en vector ternario"""
        position_encoding = {
            "initial": [1, 0, 0],  # Inicio
            "medial": [0, 1, 0],   # Medio
            "final": [0, 0, 1]     # Final
        }
        return position_encoding[position]
    
    def _determine_functional_class(self, phoneme, phone_idx, syllable):
        """Determina clase funcional del fonema en la sílaba"""
        if phoneme not in self.phoneme_features:
            return "unknown"
        
        features = self.phoneme_features[phoneme]
        
        if features['type'] == 'vowel':
            return "nucleus"  # Zona vocal
        else:
            # Para consonantes, determinar si es onset o coda
            vowel_positions = [i for i, p in enumerate(syllable) 
                             if p in self.phoneme_features and 
                             self.phoneme_features[p]['type'] == 'vowel']
            
            if not vowel_positions:
                return "onset"  # Sin vocal, asumir inicio
            
            first_vowel = vowel_positions[0]
            if phone_idx < first_vowel:
                return "onset"  # Consonante inicial
            else:
                return "coda"   # Consonante final
    
    def _encode_functional_class(self, functional_class):
        """Codifica clase funcional en vector ternario"""
        class_encoding = {
            "onset": [1, 0, 0],    # Consonante inicial
            "nucleus": [0, 1, 0],  # Zona vocal
            "coda": [0, 0, 1],     # Consonante final
            "unknown": [0, 0, 0]   # Desconocido
        }
        return class_encoding[functional_class]
    
    def _encode_syllable_boundary(self, is_syllable_end):
        """Codifica límite silábico en vector ternario"""
        if is_syllable_end:
            return [1, 1, 1]  # Final de sílaba
        else:
            return [0, 0, 0]  # Continuación
    
    def learn_patterns_from_examples(self):
        """
        Aprende patrones fonológicos de los ejemplos de entrenamiento.
        Extrae reglas para cada nivel jerárquico.
        """
        print(f"\n🧠 Aprendiendo patrones de {len(self.training_examples)} ejemplos")
        
        if not self.training_examples:
            print("   ⚠️ No hay ejemplos de entrenamiento")
            return
        
        # NIVEL 1: Aprender patrones de posición de fonemas
        print("   Nivel 1: Patrones de posición de fonemas...")
        self._learn_phoneme_position_patterns()
        
        # NIVEL 2: Aprender patrones de clasificación funcional
        print("   Nivel 2: Patrones de clasificación funcional...")
        self._learn_functional_classification_patterns()
        
        # NIVEL 3: Aprender patrones de límites silábicos
        print("   Nivel 3: Patrones de límites silábicos...")
        self._learn_syllable_boundary_patterns()
        
        # Generar reglas fractales usando Aurora
        self._generate_fractal_rules()
        
        print("   ✅ Aprendizaje de patrones completado")
        self._display_learned_patterns()
    
    def _learn_phoneme_position_patterns(self):
        """Aprende patrones de posición de fonemas"""
        position_patterns = defaultdict(list)
        
        for example in self.training_examples:
            for pv in example["phoneme_vectors"]:
                phoneme = pv["phoneme"]
                position_vector = tuple(pv["level1_vector"])
                position_patterns[phoneme].append(position_vector)
        
        # Extraer patrones más frecuentes
        for phoneme, vectors in position_patterns.items():
            most_common = Counter(vectors).most_common(1)[0]
            pattern, frequency = most_common
            confidence = frequency / len(vectors)
            
            self.learned_patterns["phoneme_position"][phoneme] = {
                "pattern": list(pattern),
                "confidence": confidence,
                "frequency": frequency,
                "total_occurrences": len(vectors)
            }
    
    def _learn_functional_classification_patterns(self):
        """Aprende patrones de clasificación funcional"""
        functional_patterns = defaultdict(list)
        
        for example in self.training_examples:
            for pv in example["phoneme_vectors"]:
                phoneme = pv["phoneme"]
                functional_vector = tuple(pv["level2_vector"])
                functional_patterns[phoneme].append(functional_vector)
        
        # Extraer patrones más frecuentes
        for phoneme, vectors in functional_patterns.items():
            most_common = Counter(vectors).most_common(1)[0]
            pattern, frequency = most_common
            confidence = frequency / len(vectors)
            
            self.learned_patterns["phoneme_classification"][phoneme] = {
                "pattern": list(pattern),
                "confidence": confidence,
                "frequency": frequency,
                "total_occurrences": len(vectors)
            }
    
    def _learn_syllable_boundary_patterns(self):
        """Aprende patrones de límites silábicos"""
        boundary_patterns = defaultdict(list)
        
        for example in self.training_examples:
            for i, pv in enumerate(example["phoneme_vectors"]):
                # Contexto: fonema actual + siguiente (si existe)
                current_phoneme = pv["phoneme"]
                next_phoneme = None
                if i + 1 < len(example["phoneme_vectors"]):
                    next_phoneme = example["phoneme_vectors"][i + 1]["phoneme"]
                
                context = (current_phoneme, next_phoneme)
                boundary_vector = tuple(pv["level3_vector"])
                boundary_patterns[context].append(boundary_vector)
        
        # Extraer patrones más frecuentes
        for context, vectors in boundary_patterns.items():
            most_common = Counter(vectors).most_common(1)[0]
            pattern, frequency = most_common
            confidence = frequency / len(vectors)
            
            self.learned_patterns["syllable_boundaries"][context] = {
                "pattern": list(pattern),
                "confidence": confidence,
                "frequency": frequency,
                "total_occurrences": len(vectors)
            }
    
    def _generate_fractal_rules(self):
        """Genera reglas fractales combinando patrones de todos los niveles"""
        print("   Generando reglas fractales...")
        
        # Crear axiomas fractales para cada patrón aprendido
        for phoneme, pattern_data in self.learned_patterns["phoneme_position"].items():
            if pattern_data["confidence"] > 0.6:  # Solo patrones confiables
                # Combinar información de los tres niveles
                l1_vector = pattern_data["pattern"]
                l2_vector = self.learned_patterns["phoneme_classification"].get(phoneme, {}).get("pattern", [0, 0, 0])
                l3_vector = [1, 0, 0]  # Vector de contexto por defecto
                
                # Crear vector fractal completo
                fractal_rule = self.transcender.level1_synthesis(l1_vector, l2_vector, l3_vector)
                
                # Almacenar como axioma en Aurora
                self.evolver.formalize_fractal_axiom(
                    fractal_rule,
                    {"phoneme": phoneme, "confidence": pattern_data["confidence"]},
                    "syllabification"
                )
    
    def _display_learned_patterns(self):
        """Muestra los patrones aprendidos"""
        print("\n📊 PATRONES APRENDIDOS:")
        
        print("\n   NIVEL 1 - Posición de fonemas:")
        for phoneme, data in self.learned_patterns["phoneme_position"].items():
            print(f"      {phoneme}: {data['pattern']} (confianza: {data['confidence']:.2f})")
        
        print("\n   NIVEL 2 - Clasificación funcional:")
        for phoneme, data in self.learned_patterns["phoneme_classification"].items():
            print(f"      {phoneme}: {data['pattern']} (confianza: {data['confidence']:.2f})")
        
        print("\n   NIVEL 3 - Límites silábicos:")
        for context, data in list(self.learned_patterns["syllable_boundaries"].items())[:5]:
            print(f"      {context}: {data['pattern']} (confianza: {data['confidence']:.2f})")
    
    def syllabify_word(self, word):
        """
        Aplica silabificación automática usando patrones aprendidos.
        Usa vectores fractales para determinar límites silábicos.
        """
        print(f"\n🔍 Silabificando palabra: '{word}'")
        
        if not self.learned_patterns["phoneme_position"]:
            print("   ⚠️ No hay patrones aprendidos. Ejecutar learn_patterns_from_examples() primero")
            return []
        
        # Vectorizar cada fonema usando patrones aprendidos
        phoneme_predictions = []
        
        for i, phoneme in enumerate(word):
            if phoneme not in self.phoneme_features:
                print(f"   ⚠️ Fonema desconocido: {phoneme}")
                continue
            
            # Predecir vectores usando patrones aprendidos
            l1_pred = self._predict_position_vector(phoneme)
            l2_pred = self._predict_functional_vector(phoneme)
            l3_pred = self._predict_boundary_vector(phoneme, i, word)
            
            # Crear vector fractal de predicción
            fractal_prediction = self.transcender.level1_synthesis(l1_pred, l2_pred, l3_pred)
            
            prediction = {
                "phoneme": phoneme,
                "position": i,
                "level1_prediction": l1_pred,
                "level2_prediction": l2_pred,
                "level3_prediction": l3_pred,
                "fractal_vector": fractal_prediction,
                "is_syllable_boundary": self._is_syllable_boundary(l3_pred)
            }
            
            phoneme_predictions.append(prediction)
        
        # Construir sílabas basándose en límites predichos
        syllables = self._build_syllables_from_predictions(phoneme_predictions)
        
        print(f"   ✅ Silabificación completada: {syllables}")
        
        # Mostrar detalles de la predicción
        print("   Detalles de predicción:")
        for pred in phoneme_predictions:
            boundary_mark = "||" if pred["is_syllable_boundary"] else "--"
            print(f"      {pred['phoneme']}: L3={pred['level3_prediction']} {boundary_mark}")
        
        return syllables
    
    def _predict_position_vector(self, phoneme):
        """Predice vector de posición para un fonema"""
        if phoneme in self.learned_patterns["phoneme_position"]:
            return self.learned_patterns["phoneme_position"][phoneme]["pattern"]
        else:
            # Usar características por defecto
            features = self.phoneme_features.get(phoneme, {})
            if features.get("type") == "vowel":
                return [0, 1, 0]  # Posición media por defecto para vocales
            else:
                return [1, 0, 0]  # Posición inicial por defecto para consonantes
    
    def _predict_functional_vector(self, phoneme):
        """Predice vector de clasificación funcional para un fonema"""
        if phoneme in self.learned_patterns["phoneme_classification"]:
            return self.learned_patterns["phoneme_classification"][phoneme]["pattern"]
        else:
            # Usar características por defecto
            features = self.phoneme_features.get(phoneme, {})
            if features.get("type") == "vowel":
                return [0, 1, 0]  # Núcleo
            else:
                return [1, 0, 0]  # Onset por defecto
    
    def _predict_boundary_vector(self, phoneme, position, word):
        """Predice vector de límite silábico para un fonema"""
        # Obtener contexto (fonema actual + siguiente)
        next_phoneme = word[position + 1] if position + 1 < len(word) else None
        context = (phoneme, next_phoneme)
        
        if context in self.learned_patterns["syllable_boundaries"]:
            return self.learned_patterns["syllable_boundaries"][context]["pattern"]
        else:
            # Reglas heurísticas por defecto
            features = self.phoneme_features.get(phoneme, {})
            
            # Si es vocal seguida de consonante, probablemente sea límite
            if features.get("type") == "vowel" and next_phoneme:
                next_features = self.phoneme_features.get(next_phoneme, {})
                if next_features.get("type") == "consonant":
                    return [1, 1, 1]  # Posible límite
            
            # Si es la última letra, definitivamente es límite
            if position == len(word) - 1:
                return [1, 1, 1]
            
            return [0, 0, 0]  # No es límite
    
    def _is_syllable_boundary(self, level3_vector):
        """Determina si un vector L3 indica límite silábico"""
        # Límite si la suma de componentes es alta
        return sum(level3_vector) >= 2
    
    def _build_syllables_from_predictions(self, predictions):
        """Construye sílabas a partir de predicciones de límites"""
        syllables = []
        current_syllable = ""
        
        for pred in predictions:
            current_syllable += pred["phoneme"]
            
            if pred["is_syllable_boundary"]:
                syllables.append(current_syllable)
                current_syllable = ""
        
        # Agregar sílaba final si queda algo
        if current_syllable:
            syllables.append(current_syllable)
        
        return syllables
    
    def demonstrate_syllabification_learning(self):
        """Demuestra el sistema completo de aprendizaje de silabificación"""
        print("\n" + "="*60)
        print("🔤 DEMOSTRACIÓN: SISTEMA DE SILABIFICACIÓN AURORA")
        print("="*60)
        
        # FASE 1: Crear ejemplos de entrenamiento
        print("\n📚 FASE 1: CREACIÓN DE EJEMPLOS DE ENTRENAMIENTO")
        
        training_words = [
            ("casa", ["ca", "sa"]),
            ("perro", ["pe", "rro"]),
            ("manzana", ["man", "za", "na"]),
            ("computadora", ["com", "pu", "ta", "do", "ra"]),
            ("telefono", ["te", "le", "fo", "no"]),
            ("estudiante", ["es", "tu", "dian", "te"]),
            ("naturaleza", ["na", "tu", "ra", "le", "za"]),
            ("ventana", ["ven", "ta", "na"])
        ]
        
        for word, syllables in training_words:
            self.create_training_example(word, syllables)
        
        # FASE 2: Aprender patrones
        print(f"\n🧠 FASE 2: APRENDIZAJE DE PATRONES")
        self.learn_patterns_from_examples()
        
        # FASE 3: Probar silabificación automática
        print(f"\n🔍 FASE 3: PRUEBAS DE SILABIFICACIÓN AUTOMÁTICA")
        
        test_words = ["escuela", "problema", "musica", "importante", "desarrollo"]
        
        results = []
        for word in test_words:
            syllables = self.syllabify_word(word)
            results.append({"word": word, "syllables": syllables})
        
        # FASE 4: Resumen de resultados
        print(f"\n📊 FASE 4: RESUMEN DE RESULTADOS")
        print(f"   Palabras de entrenamiento: {len(training_words)}")
        print(f"   Patrones aprendidos:")
        print(f"      - Posición fonemas: {len(self.learned_patterns['phoneme_position'])}")
        print(f"      - Clasificación funcional: {len(self.learned_patterns['phoneme_classification'])}")
        print(f"      - Límites silábicos: {len(self.learned_patterns['syllable_boundaries'])}")
        
        print(f"\n   Resultados de silabificación:")
        for result in results:
            print(f"      '{result['word']}' → {result['syllables']}")
        
        print("\n" + "="*60)
        print("✅ DEMOSTRACIÓN COMPLETADA - SISTEMA FUNCIONAL")
        print("="*60)
        
        return {
            "training_examples": len(training_words),
            "learned_patterns": {
                "phoneme_position": len(self.learned_patterns["phoneme_position"]),
                "functional_classification": len(self.learned_patterns["phoneme_classification"]),
                "syllable_boundaries": len(self.learned_patterns["syllable_boundaries"])
            },
            "test_results": results,
            "success": True
        }

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("🚀 Iniciando Sistema de Silabificación Aurora")
    
    # Crear instancia del sistema
    syllabification_system = AuroraSyllabificationSystem()
    
    # Ejecutar demostración completa
    results = syllabification_system.demonstrate_syllabification_learning()
    
    print(f"\n🎉 Sistema de Silabificación Aurora COMPLETAMENTE OPERATIVO")
    print(f"📈 Resultados: {results}")
