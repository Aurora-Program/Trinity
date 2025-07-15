#!/usr/bin/env python3
"""
Benchmark T4-Fractal Integrado – Aurora Trinity-3 Golden-Fibo Edition
====================================================================
Benchmark completo con integración de rotación áurea, L-Spaces y métricas avanzadas.
Implementa todas las optimizaciones identificadas para máximo rendimiento.

🔧 FEATURES:
- Pool Manager con persistencia de estado
- Rotación híbrida φ/Fibonacci 
- Métricas de cobertura y diversidad
- Validación adversarial
- Comparación baseline vs optimizado
"""

import argparse
import random
import time
import csv
import os
import logging
from typing import List, Tuple, Dict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from allcode import (
    Transcender, Evolver, Extender, KnowledgeBase, FractalTensor
)
import hashlib

# 🔧 FIX: Configurar logging para controlar spam de prints
logging.basicConfig(level=logging.WARNING)  # Solo warnings y errores
logger = logging.getLogger(__name__)

class RelatorFractal:
    """
    🌌 RELATOR FRACTAL: Análisis relacional en cascada fractal
    Compatible con Trigate y lógica booleana pura.
    """
    
    def __init__(self, chaos_seed=0.314):
        from allcode import Trigate
        self.trigate = Trigate()
        self.chaos_seed = chaos_seed
        self.phi_ratio = (5**0.5 - 1) / 2  # Golden ratio
        
    def chaotic_order(self, n, r=3.95, x0=None):
        """Genera permutación caótica determinística usando mapa logístico."""
        if x0 is None:
            x0 = self.chaos_seed
        
        x = x0
        seq = list(range(n))
        
        for i in range(n):
            x = r * x * (1 - x)  # Mapa logístico
            j = int(x * n) % n
            if i != j:
                seq[i], seq[j] = seq[j], seq[i]
        
        return seq
    
    def phi_permutation(self, n, offset=7):
        """Permutación áurea determinística con offset primo."""
        step = int(n * self.phi_ratio) or 1
        step = (step + offset) % n if step + offset < n else step
        return [(i * step) % n for i in range(n)]
    
    def participation_mask(self, vectors, min_valid_ratio=0.66):
        """Máscara de participación: solo procesa vectores con suficientes bits válidos."""
        mask = []
        for vec in vectors:
            valid_bits = sum(1 for x in vec if x is not None)
            participates = (valid_bits / len(vec)) >= min_valid_ratio
            mask.append(participates)
        return mask
    
    def relate_trio(self, trio):
        """Relaciona un trío de vectores usando Trigate synthesize."""
        try:
            # Verificar que el trío tenga exactamente 3 vectores válidos
            if len(trio) != 3 or any(v is None for v in trio):
                return None
            
            # Primera síntesis: A ⊕ B
            m1, _ = self.trigate.synthesize(trio[0], trio[1])
            if m1 is None:
                return None
                
            # Segunda síntesis: (A ⊕ B) ⊕ C  
            m_rel, _ = self.trigate.synthesize(m1, trio[2])
            return m_rel
            
        except Exception:
            return None
    
    def relate_fractal(self, ft: FractalTensor, use_chaos=True, use_phi=True):
        """
        🌌 ANÁLISIS RELACIONAL FRACTAL MULTINIVEL
        """
        levels = ['nivel_27', 'nivel_9', 'nivel_3']
        Ms_rel_stack = []
        coverage_metrics = {'processed_trios': 0, 'valid_relations': 0, 'participation_rate': 0.0}
        
        current_vectors = None
        
        # Encontrar el nivel más profundo disponible
        for level_name in levels:
            level_vectors = getattr(ft, level_name, None)
            if level_vectors and len(level_vectors) >= 3:
                current_vectors = level_vectors
                break
        
        if not current_vectors:
            return None
        
        # Procesar nivel por nivel hacia arriba
        for level_idx, level_name in enumerate(levels):
            if current_vectors is None or len(current_vectors) < 3:
                break
                
            # Aplicar máscara de participación
            participation = self.participation_mask(current_vectors)
            valid_indices = [i for i, valid in enumerate(participation) if valid]
            
            if len(valid_indices) < 3:
                break  # No hay suficientes vectores válidos
            
            # Generar orden de procesamiento
            if use_chaos and use_phi:
                # Híbrido: chaos para diversidad, phi para cobertura
                if level_idx % 2 == 0:
                    order = self.chaotic_order(len(valid_indices))
                else:
                    order = self.phi_permutation(len(valid_indices))
            elif use_chaos:
                order = self.chaotic_order(len(valid_indices))
            elif use_phi:
                order = self.phi_permutation(len(valid_indices))
            else:
                order = list(range(len(valid_indices)))  # Secuencial
            
            # Procesar en tríos
            Ms_level = []
            processed_trios = 0
            valid_relations = 0
            
            for i in range(0, len(order) - 2, 3):
                trio_indices = [valid_indices[order[i + k]] for k in range(3)]
                trio = [current_vectors[idx] for idx in trio_indices]
                
                processed_trios += 1
                m_rel = self.relate_trio(trio)
                
                if m_rel is not None:
                    Ms_level.append(m_rel)
                    valid_relations += 1
            
            # Actualizar métricas
            coverage_metrics['processed_trios'] += processed_trios
            coverage_metrics['valid_relations'] += valid_relations
            coverage_metrics['participation_rate'] = len(valid_indices) / len(current_vectors)
            
            if Ms_level:
                Ms_rel_stack.append(Ms_level)
                current_vectors = Ms_level  # Subir al siguiente nivel
            else:
                break  # No se pudieron generar relaciones válidas
        
        # Calcular resultado final
        M_rel_emergent = None
        if Ms_rel_stack:
            # El vector emergente es el primer elemento del nivel más alto
            M_rel_emergent = Ms_rel_stack[-1][0] if Ms_rel_stack[-1] else None
        
        return {
            'Ms_rel_levels': Ms_rel_stack,
            'M_rel_emergent': M_rel_emergent,
            'coverage_metrics': coverage_metrics,
            'chaos_used': use_chaos,
            'phi_used': use_phi
        }

class AuroraIntegratedSystem:
    """
    🔧 Sistema Aurora completamente integrado con rotación áurea.
    Maneja instancias únicas y persistencia de estado.
    """
    
    def __init__(self, seed: int, enable_rotation: bool = True, 
                 enable_persistence: bool = True, verbose: bool = False,
                 enable_fractal_relator: bool = False):
        random.seed(seed)
        self.seed = seed
        self.enable_rotation = enable_rotation
        self.enable_persistence = enable_persistence
        self.verbose = verbose
        
        # Crear instancias únicas del stack Aurora
        self.kb = KnowledgeBase()
        self.evolver = Evolver()
        
        # 🔧 FIX: Pasar verbose flag al evolver para controlar logs
        if hasattr(self.evolver, 'verbose'):
            self.evolver.verbose = verbose
        
        self.extender = Extender(self.kb, self.evolver)
        self.transcender = Transcender()
        
        # Integrar rotación áurea si está habilitada
        if enable_rotation:
            try:
                from utils_golden import integrate_golden_rotation
                self.tensor_pool = integrate_golden_rotation(
                    self, enable_persistence=enable_persistence
                )
                self.rotation_enabled = True
                
                # Cargar estado previo si existe
                if enable_persistence:
                    state_file = f"rotor_state_seed_{seed}.pkl"
                    self.tensor_pool.load_pool_state(state_file)
                    
            except ImportError:
                self.tensor_pool = None
                self.rotation_enabled = False
                if verbose:
                    print("[WARNING] utils_golden no disponible - rotación deshabilitada")
        else:
            self.tensor_pool = None
            self.rotation_enabled = False
        
        # 🌌 RELATOR FRACTAL: Nuevo módulo experimental
        self.enable_fractal_relator = enable_fractal_relator
        if enable_fractal_relator:
            self.relator_fractal = RelatorFractal(chaos_seed=seed * 0.314)
            if verbose:
                print("✅ Relator Fractal activado con caos controlado")
        else:
            self.relator_fractal = None
        
        # Métricas de tracking
        self.training_entries = 0
        self.coherence_violations = 0
        self.lut_operations = 0
        self.tensor_processing_attempts = 0
        
        # Silenciar warnings
        import warnings
        warnings.filterwarnings("ignore", message="MetaM cascade")

    def _canonicalize_ms(self, ms: List[int]) -> Tuple[int, int, int]:
        """🔧 FRACTAL FIX: Pre-fold usando rotación áurea para normalizar Ms."""
        if not ms or len(ms) != 3:
            return (0, 0, 0)
        
        # Usar suma como índice para rotación áurea determinística
        golden_ratio = 1.618033988749895
        phi_inv = 1 / golden_ratio
        
        # Calcular índice de rotación usando proporción áurea
        sum_ms = sum(ms) if all(x is not None for x in ms) else 0
        rot_index = int((sum_ms * phi_inv) * 3) % 3
        
        # Rotación circular determinística
        normalized = ms[rot_index:] + ms[:rot_index]
        return tuple(int(x) if x is not None else 0 for x in normalized)

    def _quantize_ms(self, ms: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """🔧 FRACTAL FIX: Cuantización en malla φ de 4 posiciones para reducir colisiones."""
        golden_ratio = 1.618033988749895
        phi_inv = 1 / golden_ratio
        
        quantized = []
        for i, bit in enumerate(ms):
            # 🔧 FIX: Malla de 4 posiciones → 2 bits más precisos
            phi_offset = int((i * phi_inv) * 4) % 4
            quantized_4bit = (bit + phi_offset) % 4
            # Convertir a 2 bits: 0,1,2,3 → 0,0,1,1 → preserva granularidad
            quantized_bit = quantized_4bit >> 1
            quantized.append(quantized_bit)
        
        return tuple(quantized)

    def _ms_slot_id(self, ms_raw: List[int]) -> str:
        """🔧 FRACTAL FIX: Genera ID de slot fractal único y determinístico."""
        # Paso 1: Canonicalizar (pre-fold fractal)
        ms_canon = self._canonicalize_ms(ms_raw)
        
        # Paso 2: Cuantizar en malla áurea
        ms_q = self._quantize_ms(ms_canon)
        
        # Paso 3: Generar slot ID compacto
        return f"{ms_q[0]}{ms_q[1]}{ms_q[2]}"

    def _generate_fibonacci_slots(self, base_slot_id: str) -> List[str]:
        """🔧 FRACTAL FIX: Genera variaciones usando secuencia Fibonacci para sub-slots."""
        if len(base_slot_id) != 3:
            return []
        
        fib_sequence = [1, 1, 2, 3, 5, 8]
        variations = []
        
        for fib in fib_sequence[:3]:  # Solo primeros 3 números Fibonacci
            varied_slot = ""
            for i, bit_char in enumerate(base_slot_id):
                bit = int(bit_char)
                # Aplicar rotación Fibonacci
                rotated_bit = (bit + fib + i) % 2
                varied_slot += str(rotated_bit)
            variations.append(varied_slot)
        
        return variations

    def ingest_fractal_tensor(self, ft: FractalTensor, space_id: str = "default"):
        """🔧 FRACTAL FIX: Ingesta con normalización fractal y slots jerárquicos."""
        self.tensor_processing_attempts += 1
        
        try:
            # Añadir al pool manager si está disponible
            if self.tensor_pool:
                self.tensor_pool.add_tensor(ft)
                
                # Forzar rotación periódicamente para activar métricas
                if self.tensor_processing_attempts % 20 == 0:
                    _ = self.tensor_pool.get_tensor_trio("arquetipo")
            
            # Procesamiento estándar con transcender
            results = self.transcender.compute_fractal(ft)
            
            if results and 'final' in results:
                final_result = results['final']
                
                # Validación más robusta antes de guardar
                m_emergent = final_result.get('M_emergent', [])
                meta_m = final_result.get('MetaM', [])
                
                # Solo guardar si AMBOS vectores están completos (sin None)
                if (m_emergent and len(m_emergent) == 3 and all(x is not None for x in m_emergent) and
                    meta_m and len(meta_m) == 3 and all(x is not None for x in meta_m)):
                    
                    # 🔧 FRACTAL FIX: Generar slot fractal normalizado
                    slot_id = self._ms_slot_id(m_emergent)
                    
                    # Detectar patrones y almacenar
                    patterns = self._extract_patterns(results)
                    
                    # 🔧 FRACTAL FIX: Almacenar con slot fractal como clave primaria
                    combined_id = f"{space_id}_{slot_id}"
                    
                    self.kb.add_entry(
                        A=final_result['A'],
                        B=final_result['B'],
                        C=final_result['C'],
                        M_emergent=m_emergent,
                        MetaM=meta_m,
                        R_validos=[patterns],
                        space_id=space_id,
                        transcender_id=combined_id
                    )
                    self.training_entries += 1
                    
                    if self.verbose:
                        logger.debug(f"Stored in slot {slot_id}: M_emergent={m_emergent}, MetaM={meta_m}")
                else:
                    self.coherence_violations += 1
                    if self.verbose:
                        logger.debug(f"Rejected: M_emergent={m_emergent}, MetaM={meta_m}")
                    
        except Exception as e:
            self.coherence_violations += 1
            if self.verbose:
                logger.warning(f"Ingestion failed: {e}")

    def _extract_patterns(self, fractal_results: Dict) -> List:
        """
        Extrae los patrones MetaM de los diferentes niveles de la síntesis fractal
        para almacenarlos como parte de la entrada en la Knowledge Base.
        """
        patterns = []
        # Iterar sobre los niveles de la síntesis fractal de más abstracto a más concreto
        for level_name in ['nivel_3', 'nivel_9', 'nivel_27']:
            # Verificar si el nivel existe y tiene resultados
            if level_name in fractal_results and fractal_results[level_name]:
                # Extraer el MetaM del primer resultado de ese nivel
                # Es el MetaM más representativo de esa capa de síntesis
                level_meta = fractal_results[level_name][0].get('MetaM', [0, 0, 0])
                patterns.append([level_name, level_meta])
        
        # Si no se extrajo ningún patrón, devolver un valor por defecto
        # para evitar errores en la Knowledge Base.
        if not patterns:
            return [[0, 0, 0]]
            
        return patterns

    def complete_fractal_enhanced(self, ft_masked: FractalTensor, 
                                 target_ft: FractalTensor,
                                 enable_parameter_sweep: bool = False) -> Dict:
        """🔧 FRACTAL ARCHETYPE: Reconstrucción con Relator Fractal opcional."""
        try:
            start_time = time.time()
            
            # Contadores para medir uso real de KB
            kb_queries = 0
            kb_hits = 0
            reconstruction_method = "none"
            
            # 🌌 FASE 0: ANÁLISIS RELACIONAL FRACTAL (si está habilitado)
            relational_context = None
            if self.enable_fractal_relator and self.relator_fractal:
                try:
                    relational_analysis = self.relator_fractal.relate_fractal(ft_masked)
                    if relational_analysis and relational_analysis['M_rel_emergent']:
                        relational_context = {
                            'M_rel_emergent': relational_analysis['M_rel_emergent'],
                            'coverage_metrics': relational_analysis['coverage_metrics'],
                            'relational_levels': len(relational_analysis['Ms_rel_levels'])
                        }
                        if self.verbose:
                            coverage = relational_analysis['coverage_metrics']
                            print(f"🌌 Relator Fractal: {coverage['valid_relations']}/{coverage['processed_trios']} relaciones válidas")
                
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Relator Fractal failed: {e}")
                    relational_context = None
            
            # Calcular métricas incluyendo lookup fractal jerárquico
            metrics = self._compute_learning_metrics(
                ft_masked, target_ft, ft_masked, 
                reconstruction_method, time.time() - start_time,
                kb_queries, kb_hits, None, None
            )
            
            # 🔧 FIX: Añadir métricas de calibración
            metrics['best_hit_weight'] = 0.0
            metrics['parameter_sweep_enabled'] = enable_parameter_sweep
            
            # 🌌 AÑADIR MÉTRICAS DEL RELATOR FRACTAL
            if relational_context:
                metrics['relational_coverage'] = relational_context['coverage_metrics']['participation_rate']
                metrics['relational_levels'] = relational_context['relational_levels']
                metrics['relational_valid_ratio'] = (
                    relational_context['coverage_metrics']['valid_relations'] / 
                    max(1, relational_context['coverage_metrics']['processed_trios'])
                )
            else:
                metrics['relational_coverage'] = 0.0
                metrics['relational_levels'] = 0
                metrics['relational_valid_ratio'] = 0.0
            
            return metrics
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Reconstruction failed: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                'accuracy_lvl3': 0.0,
                'accuracy_lvl9': 0.0, 
                'accuracy_lvl27': 0.0,
                'reconstruction_method': 'exception_fallback',
                'kb_queries': 0,
                'kb_hits': 0,
                'kb_hit_ratio': 0.0,
                'learning_signal': False,
                'best_hit_weight': 0.0,
                'parameter_sweep_enabled': False,
                'relational_coverage': 0.0,
                'relational_levels': 0,
                'relational_valid_ratio': 0.0,
                'error': str(e)
            }

    def _compute_learning_metrics(self, reconstructed: FractalTensor, target: FractalTensor,
                                 masked: FractalTensor, method: str, duration: float,
                                 kb_queries: int, kb_hits: float, target_ms: List, 
                                 target_metam: List) -> Dict:
        """🔧 AUDIT: Métricas que miden aprendizaje real."""
        # Métricas básicas
        metrics = {
            'accuracy_lvl3': 0.0,
            'accuracy_lvl9': 0.0,
            'accuracy_lvl27': 0.0,
            'reconstruction_method': method,
            'duration': duration
        }
        
        # 🔧 AUDIT: Métricas de aprendizaje con manejo robusto
        metrics.update({
            'kb_queries': kb_queries,
            'kb_hits': kb_hits,
            'kb_hit_ratio': kb_hits / max(1, kb_queries),
            'learning_signal': kb_hits > 0 and method not in ['kb_miss_no_fallback', 'exception_fallback'],
            'target_ms_valid': target_ms is not None and all(x is not None for x in target_ms) if target_ms else False,
            'target_metam_valid': target_metam is not None and all(x is not None for x in target_metam) if target_metam else False
        })
        
        # Accuracy por nivel (solo si hubo reconstrucción válida)
        for level_name in ['nivel_3', 'nivel_9', 'nivel_27']:
            acc = self._compute_level_accuracy(
                getattr(reconstructed, level_name, None),
                getattr(target, level_name, None),
                getattr(masked, level_name, None)
            )
            metrics[f'accuracy_{level_name.replace("nivel_", "lvl")}'] = acc
        
        return metrics

    def _compute_level_accuracy(self, reconstructed_level, target_level, masked_level) -> float:
        """🔧 FIX: Calcula accuracy para un nivel específico con manejo robusto."""
        if not all([reconstructed_level, target_level, masked_level]):
            return 0.0
        
        correct = 0
        total = 0
        
        try:
            for rec_vec, tar_vec, mas_vec in zip(reconstructed_level, target_level, masked_level):
                for rec_val, tar_val, mas_val in zip(rec_vec, tar_vec, mas_vec):
                    if mas_val is None:  # Solo evaluar posiciones enmascaradas
                        total += 1
                        if rec_val is not None and tar_val is not None:
                            # Convertir a enteros para comparación
                            rec_int = int(rec_val) if isinstance(rec_val, (int, float)) else 0
                            tar_int = int(tar_val) if isinstance(tar_val, (int, float)) else 0
                            if rec_int == tar_int:
                                correct += 1
        except Exception as e:
            if self.verbose:
                logger.debug(f"Error computing accuracy: {e}")
            return 0.0
        
        return correct / total if total > 0 else 0.0

# 🔧 FRACTAL FIX: Funciones coherentes mejoradas para generar familias de reglas
def generate_coherent_tensor(A: List = None, B: List = None, seed: int = None) -> FractalTensor:
    """🔧 AUDIT: Genera tensor coherente siguiendo lógica Aurora canónica."""
    if seed is not None:
        local_random = random.Random(seed)
    else:
        local_random = random
    
    # Generar A, B aleatorios si no se proporcionan
    if A is None:
        A = [local_random.randint(0, 1) for _ in range(3)]
    if B is None:
        B = [local_random.randint(0, 1) for _ in range(3)]
    
    # Aplicar lógica Aurora: R = A XOR B (lógica canónica)
    R = [a ^ b for a, b in zip(A, B)]
    
    # C derivado de A, B para garantizar coherencia
    C = [(a + b) % 2 for a, b in zip(A, B)]
    
    # Construir tensor coherente
    nivel_3 = [A, B, C]
    
    # Expandir a niveles superiores con coherencia
    nivel_9 = []
    for i in range(9):
        base_vec = nivel_3[i % 3].copy()
        # Añadir variación controlada
        variation = [(base_vec[j] + i) % 2 for j in range(3)]
        nivel_9.append(variation)
    
    nivel_27 = []
    for i in range(27):
        base_vec = nivel_9[i % 9].copy()
        variation = [(base_vec[j] + (i // 9)) % 2 for j in range(3)]
        nivel_27.append(variation)
    
    return FractalTensor(nivel_3=nivel_3, nivel_9=nivel_9, nivel_27=nivel_27)

def generate_family_tensors(family_seed: int, family_size: int = 5) -> List[FractalTensor]:
    """🔧 FRACTAL FIX: Genera familia con variaciones controladas en A, B y C."""
    local_random = random.Random(family_seed)
    family = []
    
    # Generar familia base con mismo patrón lógico fractal
    base_A = [local_random.randint(0, 1) for _ in range(3)]
    base_B = [local_random.randint(0, 1) for _ in range(3)]
    base_C = [(a + b) % 2 for a, b in zip(base_A, base_B)]  # C coherente con A,B
    
    for i in range(family_size):
        # 🔧 FRACTAL FIX: Variaciones controladas que preservan firma semántica
        var_A = base_A.copy()
        var_B = base_B.copy()
        var_C = base_C.copy()
        
        if i > 0:
            # Aplicar transformaciones fractales que preservan el slot
            golden_offset = int((i * 0.618033988749895) * 3) % 3
            
            # 🔧 FIX: Variación en A con compensación
            original_sum_A = sum(var_A) % 2
            var_A[golden_offset] = 1 - var_A[golden_offset]
            
            # Compensar para mantener el slot fractal
            if sum(var_A) % 2 != original_sum_A:
                next_pos = (golden_offset + 1) % 3
                var_A[next_pos] = 1 - var_A[next_pos]
            
            # 🔧 FIX: Pequeña variación controlada en B para evitar over-fit
            if i % 3 == 0:  # Solo cada 3 elementos
                var_B[golden_offset] = (var_B[golden_offset] + 1) % 2
            
            # 🔧 FIX: Recalcular C para mantener coherencia
            var_C = [(a + b) % 2 for a, b in zip(var_A, var_B)]
        
        ft = generate_coherent_tensor(var_A, var_B, seed=family_seed + i)
        # 🔧 FIX: Sobrescribir nivel_3 con variaciones controladas
        ft.nivel_3 = [var_A, var_B, var_C]
        family.append(ft)
    
    return family

def create_tensor_families(samples_per_family: int, seed: int) -> Dict[str, List[FractalTensor]]:
    """
    🔧 SOLUCIÓN DEFINITIVA: Genera y VERIFICA familias de tensores para GARANTIZAR
    la diversidad lógica necesaria para la síntesis fractal.
    """
    print("🔧 Creando y VERIFICANDO familias de tensores con diversidad lógica garantizada...")
    local_random = random.Random(seed)
    transcender_verifier = Transcender()
    families = {"arithmetic": [], "geometric": [], "cyclic": []}
    
    # Almacenar las firmas Ms únicas para asegurar diversidad
    unique_ms_signatures = set()

    for family_name in families.keys():
        family_tensors = []
        family_ms_signature = None
        attempts = 0
        max_attempts = 1000  # Evitar bucle infinito
        
        while len(family_tensors) < samples_per_family and attempts < max_attempts:
            attempts += 1
            # Generar un tensor candidato aleatorio
            A = [local_random.randint(0, 1) for _ in range(3)]
            B = [local_random.randint(0, 1) for _ in range(3)]
            C = [local_random.randint(0, 1) for _ in range(3)]
            ft = FractalTensor(nivel_3=[A, B, C])
            
            # Verificar su firma lógica
            try:
                res = transcender_verifier.compute(A, B, C)
                ms = tuple(res.get('M_emergent', [])) if res else None
            except:
                continue
            
            if not ms or None in ms or ms == (0,0,0):
                continue # Omitir tensores con firmas triviales o inválidas

            # Si es la primera muestra, establece la firma de la familia
            if not family_ms_signature:
                if ms not in unique_ms_signatures:
                    family_ms_signature = ms
                    unique_ms_signatures.add(ms)
                    family_tensors.append(ft)
            # Si no, asegurarse de que las siguientes muestras convergen a la misma firma
            elif ms == family_ms_signature:
                family_tensors.append(ft)
        
        families[family_name] = family_tensors
        print(f"  ✅ Familia '{family_name}' creada con firma Ms: {family_ms_signature} ({len(family_tensors)} tensores)")

    print("✅ Todas las familias de tensores diversas han sido generadas y verificadas.")
    return families

def run_integrated_benchmark(seed: int = 42, n_train: int = 500, n_test: int = 150,
                            compare_modes: bool = True, save_metrics: bool = True,
                            verbose: bool = False, use_coherent_dataset: bool = True,
                            enable_parameter_sweep: bool = False):
    """
    🔧 FRACTAL SYNTHESIS: Benchmark con síntesis fractal optimizada y parámetros configurables.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        print("🔧 Iniciando Benchmark T4-Fractal con Síntesis Jerárquica Optimizada...")
    else:
        logger.setLevel(logging.WARNING)
        print("🔧 Iniciando Benchmark T4-Fractal con Síntesis Jerárquica Optimizada...")
    
    results_data = []
    
    # Configuraciones incluyendo síntesis fractal optimizada
    configs = [
        ("no_training", False, False, 0, False),        # Sin entrenamiento
        ("level1_only", False, False, 0.3, True),      # Solo Nivel 1 (ejemplos base)
        ("fractal_synthesis", True, True, 1.0, True),   # Síntesis Fractal multinivel completa
        ("ablation_no_fallback", True, True, 1.0, False), # Test puro sin heurísticas
        ("parameter_sweep", True, True, 1.0, True)      # 🔧 FIX: Con calibración de parámetros
    ] if compare_modes else [("fractal_synthesis", True, True, 1.0, True)]
    
    for config_name, enable_rotation, enable_persistence, training_ratio, allow_fallback in configs:
        print(f"\n🔧 Ejecutando configuración: {config_name}")
        system = AuroraIntegratedSystem(
            seed=seed, 
            enable_rotation=enable_rotation,
            enable_persistence=enable_persistence,
            verbose=verbose,
            enable_fractal_relator=args.fractal_relator
        )
        system.allow_heuristic_fallback = allow_fallback

        actual_n_train = int(n_train * training_ratio)
        ms_sequence = []
        meta_sequence = []
        
        if actual_n_train > 0:
            print(f"🔧 FASE 1: Aprendizaje de Ejemplos Base (Nivel 1)")
            samples_per_family = max(1, actual_n_train // 3)
            tensor_families = create_tensor_families(samples_per_family, seed)
            successful_ingests = 0

            for family_name, tensors in tensor_families.items():
                for ft in tensors:
                    entries_before = system.training_entries
                    system.ingest_fractal_tensor(ft, family_name)
                    if system.training_entries > entries_before:
                        successful_ingests += 1
        else:
            successful_ingests = 0
            tensor_families = {} # No families for no_training case

        # --- FASE 3: EVALUACIÓN ---
        print(f"🔧 FASE 3: Evaluación con {n_test} tensores")
        
        test_families = create_tensor_families(n_test // 3, seed + 1000)
        test_tensors = []
        for family_tensors in test_families.values():
            test_tensors.extend(family_tensors[:n_test//3])
        
        test_metrics = []
        kb_metrics = {'total_queries': 0, 'total_hits': 0, 'learning_signals': 0}
        reconstruction_methods = {}
        
        dynamic_model = None # Placeholder for dynamic model logic
        
        for i, clean_ft in enumerate(test_tensors[:n_test]):
            masked_ft = mask_tensor(clean_ft, 'nivel_3', 0.20, seed=seed+i)
            try:
                # Pasar el modelo dinámico en el contexto si existe
                contexto = {}
                if dynamic_model:
                    contexto["dynamic_prior"] = dynamic_model
                    if ms_sequence:
                        contexto["last_Ms"] = ms_sequence[-1]

                metrics = system.complete_fractal_enhanced(
                    masked_ft, clean_ft, 
                    enable_parameter_sweep=(config_name == "parameter_sweep") and enable_parameter_sweep
                )
                metrics['config'] = config_name
                metrics['test_id'] = i
                test_metrics.append(metrics)
                kb_metrics['total_queries'] += metrics.get('kb_queries', 0)
                kb_metrics['total_hits'] += metrics.get('kb_hits', 0)
                if metrics.get('learning_signal', False):
                    kb_metrics['learning_signals'] += 1
                method = metrics.get('reconstruction_method', 'unknown')
                reconstruction_methods[method] = reconstruction_methods.get(method, 0) + 1
            except Exception as e:
                if verbose:
                    logger.error(f"Test {i} failed: {e}")
                test_metrics.append({
                    'accuracy_lvl3': 0.0,
                    'reconstruction_method': 'test_exception',
                    'kb_queries': 0, 'kb_hits': 0, 'learning_signal': False,
                    'config': config_name, 'test_id': i
                })
        
        # Calcular promedios con métricas de aprendizaje
        n_valid_tests = len(test_metrics)
        if n_valid_tests == 0:
            continue  # Se añade "continue" para evitar bloque vacío
        avg_metrics = {
            'config': config_name,
            'training_ratio': training_ratio,
            'actual_n_train': actual_n_train,
            'training_entries': system.training_entries,
            'training_success_rate': successful_ingests / max(1, actual_n_train),
            'kb_total_queries': kb_metrics['total_queries'],
            'kb_total_hits': kb_metrics['total_hits'],
            'kb_global_hit_ratio': kb_metrics['total_hits'] / max(1, kb_metrics['total_queries']),
            'learning_signals_pct': kb_metrics['learning_signals'] / max(1, n_valid_tests),
            'avg_accuracy_lvl3': sum(m.get('accuracy_lvl3', 0) for m in test_metrics) / n_valid_tests,
            'reconstruction_methods': reconstruction_methods,
            'coherent_dataset': use_coherent_dataset,
            'allow_fallback': allow_fallback
        }
        
        results_data.append(avg_metrics)
        
        # 🔧 FRACTAL SYNTHESIS: Reporte centrado en síntesis jerárquica optimizada
        print(f"\n📊 RESULTADOS SÍNTESIS FRACTAL OPTIMIZADA {config_name.upper()}:")
        print(f"  Training entries  : {system.training_entries}")
        print(f"  KB hit ratio      : {avg_metrics['kb_global_hit_ratio']:.3f}")
        print(f"  Learning signals  : {avg_metrics['learning_signals_pct']:.3f}")
        print(f"  Accuracy lvl3     : {avg_metrics['avg_accuracy_lvl3']:.3f}")
        print(f"  Methods used      : {reconstruction_methods}")
        
        # 🔧 FIX: Reporte de calibración si está habilitado
        use_param_sweep = (config_name == "parameter_sweep") and enable_parameter_sweep
        if use_param_sweep:
            avg_hit_weight = sum(m.get('best_hit_weight', 0) for m in test_metrics) / max(1, len(test_metrics))
            print(f"  Avg hit weight    : {avg_hit_weight:.3f}")
        
        # system.save_state() # Descomentar si se implementa

    # Guardar métricas
    if save_metrics:
        filename = f'benchmark_fractal_hierarchical_optimized_coherent_{use_coherent_dataset}_seed_{seed}.csv'
        with open(filename, 'w', newline='') as csvfile:
            if results_data:
                writer = csv.DictWriter(csvfile, fieldnames=results_data[0].keys())
                writer.writeheader()
                writer.writerows(results_data)
        print(f"\n💾 Métricas guardadas en {filename}")
    
    # Análisis de curva de aprendizaje fractal
    if compare_modes and len(results_data) > 1:
        print(f"\n🔍 ANÁLISIS DE APRENDIZAJE FRACTAL JERÁRQUICO OPTIMIZADO:")
        print(f"{'Config':<20} {'KB Hit Ratio':<12} {'Learning %':<12} {'Accuracy':<10}")
        print("-" * 60)
        for result in results_data:
            print(f"{result['config']:<20} {result['kb_global_hit_ratio']:<12.3f} "
                  f"{result['learning_signals_pct']:<12.3f} {result['avg_accuracy_lvl3']:<10.3f}")
    
    return results_data

# 🔧 MISSING FUNCTIONS: Funciones auxiliares para masking y testing
def random_tensor(depth_prob=(1.0, 0.3, 0.05), value_range=(0, 1)):
    """🔧 FIX: Genera tensor fractal aleatorio (re-añadido para --random-dataset)."""
    def rand_vec(): 
        return [random.randint(*value_range) for _ in range(3)]
    
    lvl3 = [rand_vec() for _ in range(3)]
    lvl9 = [rand_vec() for _ in range(9)] if random.random() < depth_prob[1] else None
    lvl27 = [rand_vec() for _ in range(27)] if random.random() < depth_prob[2] else None
    
    return FractalTensor(nivel_3=lvl3, nivel_9=lvl9, nivel_27=nivel_27)

def mask_tensor(ft: FractalTensor, level: str = 'nivel_3', ratio: float = 0.25, seed: int = None) -> FractalTensor:
    """🔧 AUDIT FIX: Enmascara tensor con semilla local para reproducibilidad."""
    import copy
    
    if seed is not None:
        local_random = random.Random(seed)
    else:
        local_random = random
    
    ft_masked = copy.deepcopy(ft)
    target = getattr(ft_masked, level)
    
    if target is None:
        return ft_masked
    
    total_bits = len(target) * 3
    k = max(1, int(total_bits * ratio))
    
    for _ in range(k):
        vec_idx = local_random.randint(0, len(target) - 1)
        bit_idx = local_random.randint(0, 2)
        target[vec_idx][bit_idx] = None
    
    return ft_masked

def quick_test_tensors(system, n_tensors=5):
    """🔧 FIX: Función para test rápido de ingesta de tensores densos."""
    print(f"🔧 Quick test con {n_tensors} tensores densos...")
    
    for i in range(n_tensors):
        # Tensor garantizado con nivel_27
        ft = random_tensor(depth_prob=(1.0, 1.0, 1.0))
        system.ingest_fractal_tensor(ft, "test")
        
        if i % 2 == 0 and system.tensor_pool:
            _ = system.tensor_pool.get_tensor_trio("arquetipo")
    
    print(f"Resultado: {system.training_entries}/{system.tensor_processing_attempts} tensores ingresados")
    
    if system.tensor_pool:
        try:
            metrics = system.tensor_pool.get_rotation_metrics()
            print(f"Métricas de rotación: {metrics}")
        except:
            print("Métricas de rotación: No disponibles")
    
    return system.training_entries > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora T4-Fractal Integrated Benchmark - OPTIMIZED Edition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="Ablation study completo")
    parser.add_argument("--save-metrics", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", help="Modo verbose para debugging")
    parser.add_argument("--coherent", action="store_true", default=True, help="Usar dataset coherente")
    parser.add_argument("--random-dataset", action="store_true", help="Usar dataset aleatorio (comparación)")
    parser.add_argument("--quick", action="store_true", help="Test rápido de ingesta")
    parser.add_argument("--parameter-sweep", action="store_true", help="Habilitar calibración de parámetros")
    
    args = parser.parse_args()
    
    # Opción de test rápido
    if args.quick:
        print("🔧 Modo quick test activado")
        system = AuroraIntegratedSystem(seed=args.seed, verbose=True)
        success = quick_test_tensors(system, 10)
        if success:
            print("✅ Quick test PASSED - sistema funcionando")
            # Test de reconstrucción básica con coherent tensor
            clean_ft = generate_coherent_tensor(seed=args.seed)
            masked_ft = mask_tensor(clean_ft, 'nivel_3', 0.25, seed=args.seed)
            metrics = system.complete_fractal_enhanced(masked_ft, clean_ft)
            print(f"📊 Test reconstruction: accuracy={metrics.get('accuracy_lvl3', 0):.3f}")
            print(f"📊 KB queries: {metrics.get('kb_queries', 0)}")
            print(f"📊 KB hits: {metrics.get('kb_hits', 0)}")
        else:
            print("❌ Quick test FAILED - revisar configuración")
        exit()
    
    use_coherent = args.coherent and not args.random_dataset
    
    start_total = time.time()
    results = run_integrated_benchmark(
        seed=args.seed,
        n_train=args.train,
        n_test=args.test,
        compare_modes=args.compare,
        save_metrics=args.save_metrics,
        verbose=args.verbose,
        use_coherent_dataset=use_coherent,
        enable_parameter_sweep=args.parameter_sweep
    )
    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
    print("🎯 Benchmark T4-Fractal OPTIMIZADO completado.")
    
    # Resumen ejecutivo
    if results:
        best_result = max(results, key=lambda x: x.get('kb_global_hit_ratio', 0))
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  {best_result['config']} - KB Hit Ratio: {best_result['kb_global_hit_ratio']:.3f}")
        print(f"  Learning Signals: {best_result['learning_signals_pct']:.3f}")
        print(f"  Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")
        
        # Validar curva de aprendizaje
        no_training = next((r for r in results if r['config'] == 'no_training'), None)
        full_training = next((r for r in results if r['config'] in ['fractal_synthesis', 'parameter_sweep']), None)
        
        if no_training and full_training:
            learning_improvement = full_training['kb_global_hit_ratio'] - no_training['kb_global_hit_ratio']
            print(f"\n📈 EVIDENCIA DE APRENDIZAJE FRACTAL OPTIMIZADO:")
            print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")
            if learning_improvement > 0.1:
                print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")
            else:
                print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")
            n_train=args.train,
            n_test=args.test,
            compare_modes=args.compare,
            save_metrics=args.save_metrics,
            verbose=args.verbose,
            use_coherent_dataset=use_coherent,
            enable_parameter_sweep=args.parameter_sweep
      

        # Restaurar la clase original
        globals()['AuroraIntegratedSystem'] = original_AuroraIntegratedSystem
    
    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
    print("🎯 Benchmark T4-Fractal OPTIMIZADO completado.")
    
    # Resumen ejecutivo
    if results:
        best_result = max(results, key=lambda x: x.get('kb_global_hit_ratio', 0))
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  {best_result['config']} - KB Hit Ratio: {best_result['kb_global_hit_ratio']:.3f}")
        print(f"  Learning Signals: {best_result['learning_signals_pct']:.3f}")
        print(f"  Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")
        
        # Validar curva de aprendizaje
        no_training = next((r for r in results if r['config'] == 'no_training'), None)
        full_training = next((r for r in results if r['config'] in ['fractal_synthesis', 'parameter_sweep']), None)
        
        if no_training and full_training:
            learning_improvement = full_training['kb_global_hit_ratio'] - no_training['kb_global_hit_ratio']
            print(f"\n📈 EVIDENCIA DE APRENDIZAJE FRACTAL OPTIMIZADO:")
            print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")
            if learning_improvement > 0.1:
                print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")
            else:
                print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")
            print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")
            if learning_improvement > 0.1:
                print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")
            else:
                print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")
                if metrics.get('learning_signal', False):
                    kb_metrics['learning_signals'] += 1
                method = metrics.get('reconstruction_method', 'unknown')
                reconstruction_methods[method] = reconstruction_methods.get(method, 0) + 1
         
                if verbose:
                    logger.error(f"Test {i} failed: {e}")
                test_metrics.append({
                    'accuracy_lvl3': 0.0,
                    'reconstruction_method': 'test_exception',
                    'kb_queries': 0, 'kb_hits': 0, 'learning_signal': False,
                    'config': config_name, 'test_id': i
                })
        
        # Calcular promedios con métricas de aprendizaje
        n_valid_tests = len(test_metrics)
        if n_valid_tests == 0:
     
        avg_metrics = {
            'config': config_name,
            'training_ratio': training_ratio,
            'actual_n_train': actual_n_train,
            'training_entries': system.training_entries,
            'training_success_rate': successful_ingests / max(1, actual_n_train),
            'kb_total_queries': kb_metrics['total_queries'],
            'kb_total_hits': kb_metrics['total_hits'],
            'kb_global_hit_ratio': kb_metrics['total_hits'] / max(1, kb_metrics['total_queries']),
            'learning_signals_pct': kb_metrics['learning_signals'] / max(1, n_valid_tests),
            'avg_accuracy_lvl3': sum(m.get('accuracy_lvl3', 0) for m in test_metrics) / n_valid_tests,
            'reconstruction_methods': reconstruction_methods,
            'coherent_dataset': use_coherent_dataset,
            'allow_fallback': allow_fallback
        }
        
        results_data.append(avg_metrics)
        
        # 🔧 FRACTAL SYNTHESIS: Reporte centrado en síntesis jerárquica optimizada
        print(f"\n📊 RESULTADOS SÍNTESIS FRACTAL OPTIMIZADA {config_name.upper()}:")
        print(f"  Training entries  : {system.training_entries}")
        print(f"  KB hit ratio      : {avg_metrics['kb_global_hit_ratio']:.3f}")
        print(f"  Learning signals  : {avg_metrics['learning_signals_pct']:.3f}")
        print(f"  Accuracy lvl3     : {avg_metrics['avg_accuracy_lvl3']:.3f}")
        print(f"  Methods used      : {reconstruction_methods}")
        
        # 🔧 FIX: Reporte de calibración si está habilitado
        if use_param_sweep:
            avg_hit_weight = sum(m.get('best_hit_weight', 0) for m in test_metrics) / max(1, len(test_metrics))
            print(f"  Avg hit weight    : {avg_hit_weight:.3f}")
        
        system.save_state()

    # Guardar métricas
    if save_metrics:
        filename = f'benchmark_fractal_hierarchical_optimized_coherent_{use_coherent_dataset}_seed_{seed}.csv'
        with open(filename, 'w', newline='') as csvfile:
            if results_data:
                writer = csv.DictWriter(csvfile, fieldnames=results_data[0].keys())
                writer.writeheader()
                writer.writerows(results_data)
        print(f"\n💾 Métricas guardadas en {filename}")
    
    # Análisis de curva de aprendizaje fractal
    if compare_modes and len(results_data) > 1:
        print(f"\n🔍 ANÁLISIS DE APRENDIZAJE FRACTAL JERÁRQUICO OPTIMIZADO:")
        print(f"{'Config':<20} {'KB Hit Ratio':<12} {'Learning %':<12} {'Accuracy':<10}")
        print("-" * 60)
        for result in results_data:
            print(f"{result['config']:<20} {result['kb_global_hit_ratio']:<12.3f} "
                  f"{result['learning_signals_pct']:<12.3f} {result['avg_accuracy_lvl3']:<10.3f}")
    
    return results_data

# 🔧 MISSING FUNCTIONS: Funciones auxiliares para masking y testing
def random_tensor(depth_prob=(1.0, 0.3, 0.05), value_range=(0, 1)):
    """🔧 FIX: Genera tensor fractal aleatorio (re-añadido para --random-dataset)."""
    def rand_vec(): 
        return [random.randint(*value_range) for _ in range(3)]
    
    lvl3 = [rand_vec() for _ in range(3)]
    lvl9 = [rand_vec() for _ in range(9)] if random.random() < depth_prob[1] else None
    lvl27 = [rand_vec() for _ in range(27)] if random.random() < depth_prob[2] else None
    
    return FractalTensor(nivel_3=lvl3, nivel_9=lvl9, nivel_27=nivel_27)

def mask_tensor(ft: FractalTensor, level: str = 'nivel_3', ratio: float = 0.25, seed: int = None) -> FractalTensor:
    """🔧 AUDIT FIX: Enmascara tensor con semilla local para reproducibilidad."""
    import copy
    
    if seed is not None:
        local_random = random.Random(seed)
    else:
        local_random = random
    
    ft_masked = copy.deepcopy(ft)
    target = getattr(ft_masked, level)
    
    if target is None:
        return ft_masked
    
    total_bits = len(target) * 3
    k = max(1, int(total_bits * ratio))
    
    for _ in range(k):
        vec_idx = local_random.randint(0, len(target) - 1)
        bit_idx = local_random.randint(0, 2)
        target[vec_idx][bit_idx] = None
    
    return ft_masked

def quick_test_tensors(system, n_tensors=5):
    """🔧 FIX: Función para test rápido de ingesta de tensores densos."""
    print(f"🔧 Quick test con {n_tensors} tensores densos...")
    
    for i in range(n_tensors):
        # Tensor garantizado con nivel_27
        ft = random_tensor(depth_prob=(1.0, 1.0, 1.0))
        system.ingest_fractal_tensor(ft, "test")
        
        if i % 2 == 0 and system.tensor_pool:
            _ = system.tensor_pool.get_tensor_trio("arquetipo")
    
    print(f"Resultado: {system.training_entries}/{system.tensor_processing_attempts} tensores ingresados")
    
    if system.tensor_pool:
        try:
            metrics = system.tensor_pool.get_rotation_metrics()
            print(f"Métricas de rotación: {metrics}")
        except:
            print("Métricas de rotación: No disponibles")
    
    return system.training_entries > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora T4-Fractal Integrated Benchmark - OPTIMIZED Edition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="Ablation study completo")
    parser.add_argument("--save-metrics", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", help="Modo verbose para debugging")
    parser.add_argument("--coherent", action="store_true", default=True, help="Usar dataset coherente")
    parser.add_argument("--random-dataset", action="store_true", help="Usar dataset aleatorio (comparación)")
    parser.add_argument("--quick", action="store_true", help="Test rápido de ingesta")
    parser.add_argument("--parameter-sweep", action="store_true", help="Habilitar calibración de parámetros")
    parser.add_argument("--fractal-relator", action="store_true", help="Habilitar Relator Fractal experimental")
    parser.add_argument("--chaos-only", action="store_true", help="Usar solo permutación caótica (no phi)")
    parser.add_argument("--phi-only", action="store_true", help="Usar solo permutación áurea (no caos)")
    
    args = parser.parse_args()
    
    # Configurar permutaciones del Relator Fractal
    if args.chaos_only and args.phi_only:
        print("⚠️ No se pueden usar --chaos-only y --phi-only al mismo tiempo")
        exit(1)
    
    # Opción de test rápido
    if args.quick:
        print("🔧 Modo quick test activado")
        system = AuroraIntegratedSystem(
            seed=args.seed, 
            verbose=True, 
            enable_fractal_relator=args.fractal_relator
        )
        system._extend_evolver_dynamics()  # Asegurar extensión dinámica
        success = quick_test_tensors(system, 10)
        if success:
            print("✅ Quick test PASSED - sistema funcionando")
            # Test de reconstrucción básica con coherent tensor
            clean_ft = generate_coherent_tensor(seed=args.seed)
            masked_ft = mask_tensor(clean_ft, 'nivel_3', 0.25, seed=args.seed)
            metrics = system.complete_fractal_enhanced(masked_ft, clean_ft)
            print(f"📊 Test reconstruction: accuracy={metrics.get('accuracy_lvl3', 0):.3f}")
            print(f"📊 KB queries: {metrics.get('kb_queries', 0)}")
            print(f"📊 KB hits: {metrics.get('kb_hits', 0)}")
            if args.fractal_relator:
                print(f"🌌 Relational coverage: {metrics.get('relational_coverage', 0):.3f}")
                print(f"🌌 Relational levels: {metrics.get('relational_levels', 0)}")
        else:
            print("❌ Quick test FAILED - revisar configuración")
        exit()
    
    use_coherent = args.coherent and not args.random_dataset
    
    # 🌌 CONFIGURAR SISTEMA CON RELATOR FRACTAL
    def create_system_with_relator(seed, enable_rotation, enable_persistence, verbose):
        return AuroraIntegratedSystem(
            seed=seed, 
            enable_rotation=enable_rotation,
            enable_persistence=enable_persistence,
            verbose=verbose,
            enable_fractal_relator=args.fractal_relator
        )
    
    # Modificar la función run_integrated_benchmark para usar el factory
    start_total = time.time()
    
    # Inyectar el factory en el scope global temporalmente
    original_AuroraIntegratedSystem = globals()['AuroraIntegratedSystem']
    globals()['AuroraIntegratedSystem'] = create_system_with_relator
    
    try:
        results = run_integrated_benchmark(
            seed=args.seed,
            n_train=args.train,
            n_test=args.test,
            compare_modes=args.compare,
            save_metrics=args.save_metrics,
            verbose=args.verbose,
            use_coherent_dataset=use_coherent,
            enable_parameter_sweep=args.parameter_sweep
        )
    finally:
        # Restaurar la clase original
        globals()['AuroraIntegratedSystem'] = original_AuroraIntegratedSystem
    
    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
    print("🎯 Benchmark T4-Fractal OPTIMIZADO completado.")
    
    # Resumen ejecutivo
    if results:
        best_result = max(results, key=lambda x: x.get('kb_global_hit_ratio', 0))
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  {best_result['config']} - KB Hit Ratio: {best_result['kb_global_hit_ratio']:.3f}")
        print(f"  Learning Signals: {best_result['learning_signals_pct']:.3f}")
        print(f"  Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")
        
        # 🌌 MOSTRAR MÉTRICAS DEL RELATOR FRACTAL si está habilitado
        if args.fractal_relator:
            print(f"\n🌌 MÉTRICAS RELATOR FRACTAL:")
            print(f"  Relational Coverage: {best_result.get('relational_coverage', 0):.3f}")
            print(f"  Relational Levels: {best_result.get('relational_levels', 0)}")
            print(f"  Valid Relations: {best_result.get('relational_valid_ratio', 0):.3f}")
        
        # Validar curva de aprendizaje
        no_training = next((r for r in results if r['config'] == 'no_training'), None)
        full_training = next((r for r in results if r['config'] in ['fractal_synthesis', 'parameter_sweep']), None)
        
        if no_training and full_training:
            learning_improvement = full_training['kb_global_hit_ratio'] - no_training['kb_global_hit_ratio']
            print(f"\n📈 EVIDENCIA DE APRENDIZAJE FRACTAL OPTIMIZADO:")
            print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")
            if learning_improvement > 0.1:
                print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")
            else:
                print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")
                metrics['test_id'] = i
                test_metrics.append(metrics)
                kb_metrics['total_queries'] += metrics.get('kb_queries', 0)
                kb_metrics['total_hits'] += metrics.get('kb_hits', 0)
                if metrics.get('learning_signal', False):
                    kb_metrics['learning_signals'] += 1
                method = metrics.get('reconstruction_method', 'unknown')
                reconstruction_methods[method] = reconstruction_methods.get(method, 0) + 1

        
        # Calcular promedios con métricas de aprendizaje
        n_valid_tests = len(test_metrics)
        if n_valid_tests == 0:

            return None
        avg_metrics = {
            'config': config_name,
            'training_ratio': training_ratio,
            'actual_n_train': actual_n_train,
            'training_entries': system.training_entries,
            'training_success_rate': successful_ingests / max(1, actual_n_train),
            'kb_total_queries': kb_metrics['total_queries'],
            'kb_total_hits': kb_metrics['total_hits'],
            'kb_global_hit_ratio': kb_metrics['total_hits'] / max(1, kb_metrics['total_queries']),
            'learning_signals_pct': kb_metrics['learning_signals'] / max(1, n_valid_tests),
            'avg_accuracy_lvl3': sum(m.get('accuracy_lvl3', 0) for m in test_metrics) / n_valid_tests,
            'reconstruction_methods': reconstruction_methods,
            'coherent_dataset': use_coherent_dataset,
            'allow_fallback': allow_fallback
        }
        
        results_data.append(avg_metrics)
        
        # 🔧 FRACTAL SYNTHESIS: Reporte centrado en síntesis jerárquica optimizada
        print(f"\n📊 RESULTADOS SÍNTESIS FRACTAL OPTIMIZADA {config_name.upper()}:")
        print(f"  Training entries  : {system.training_entries}")
        print(f"  KB hit ratio      : {avg_metrics['kb_global_hit_ratio']:.3f}")
        print(f"  Learning signals  : {avg_metrics['learning_signals_pct']:.3f}")
        print(f"  Accuracy lvl3     : {avg_metrics['avg_accuracy_lvl3']:.3f}")
        print(f"  Methods used      : {reconstruction_methods}")
        
        # 🔧 FIX: Reporte de calibración si está habilitado
        if use_param_sweep:
            avg_hit_weight = sum(m.get('best_hit_weight', 0) for m in test_metrics) / max(1, len(test_metrics))
            print(f"  Avg hit weight    : {avg_hit_weight:.3f}")
        
        system.save_state()

    # Guardar métricas
    if save_metrics:
        filename = f'benchmark_fractal_hierarchical_optimized_coherent_{use_coherent_dataset}_seed_{seed}.csv'
        with open(filename, 'w', newline='') as csvfile:
            if results_data:
                writer = csv.DictWriter(csvfile, fieldnames=results_data[0].keys())
                writer.writeheader()
                writer.writerows(results_data)
        print(f"\n💾 Métricas guardadas en {filename}")
    
    # Análisis de curva de aprendizaje fractal
    if compare_modes and len(results_data) > 1:
        print(f"\n🔍 ANÁLISIS DE APRENDIZAJE FRACTAL JERÁRQUICO OPTIMIZADO:")
        print(f"{'Config':<20} {'KB Hit Ratio':<12} {'Learning %':<12} {'Accuracy':<10}")
        print("-" * 60)
        for result in results_data:
            print(f"{result['config']:<20} {result['kb_global_hit_ratio']:<12.3f} "
                  f"{result['learning_signals_pct']:<12.3f} {result['avg_accuracy_lvl3']:<10.3f}")
    
    return results_data

# 🔧 MISSING FUNCTIONS: Funciones auxiliares para masking y testing
def random_tensor(depth_prob=(1.0, 0.3, 0.05), value_range=(0, 1)):
    """🔧 FIX: Genera tensor fractal aleatorio (re-añadido para --random-dataset)."""
    def rand_vec(): 
        return [random.randint(*value_range) for _ in range(3)]
    
    lvl3 = [rand_vec() for _ in range(3)]
    lvl9 = [rand_vec() for _ in range(9)] if random.random() < depth_prob[1] else None
    lvl27 = [rand_vec() for _ in range(27)] if random.random() < depth_prob[2] else None
    
    return FractalTensor(nivel_3=lvl3, nivel_9=lvl9, nivel_27=lvl27)

def mask_tensor(ft: FractalTensor, level: str = 'nivel_3', ratio: float = 0.25, seed: int = None) -> FractalTensor:
    """🔧 AUDIT FIX: Enmascara tensor con semilla local para reproducibilidad."""
    import copy
    
    if seed is not None:
        local_random = random.Random(seed)
    else:
        local_random = random
    
    ft_masked = copy.deepcopy(ft)
    target = getattr(ft_masked, level)
    
    if target is None:
        return ft_masked
    
    total_bits = len(target) * 3
    k = max(1, int(total_bits * ratio))
    
    for _ in range(k):
        vec_idx = local_random.randint(0, len(target) - 1)
        bit_idx = local_random.randint(0, 2)
        target[vec_idx][bit_idx] = None
    
    return ft_masked

def quick_test_tensors(system, n_tensors=5):
    """🔧 FIX: Función para test rápido de ingesta de tensores densos."""
    print(f"🔧 Quick test con {n_tensors} tensores densos...")
    
    for i in range(n_tensors):
        # Tensor garantizado con nivel_27
        ft = random_tensor(depth_prob=(1.0, 1.0, 1.0))
        system.ingest_fractal_tensor(ft, "test")
        
        if i % 2 == 0 and system.tensor_pool:
            _ = system.tensor_pool.get_tensor_trio("arquetipo")
    
    print(f"Resultado: {system.training_entries}/{system.tensor_processing_attempts} tensores ingresados")
    
    if system.tensor_pool:
        try:
            metrics = system.tensor_pool.get_rotation_metrics()
            print(f"Métricas de rotación: {metrics}")
        except:
            print("Métricas de rotación: No disponibles")
    
    return system.training_entries > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora T4-Fractal Integrated Benchmark - OPTIMIZED Edition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=50)  # 🔧 FIX: Valor más pequeño por defecto
    parser.add_argument("--test", type=int, default=20)   # 🔧 FIX: Valor más pequeño por defecto
    parser.add_argument("--compare", action="store_true", help="Ablation study completo")
    parser.add_argument("--save-metrics", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", help="Modo verbose para debugging")
    parser.add_argument("--coherent", action="store_true", default=True, help="Usar dataset coherente")
    parser.add_argument("--random-dataset", action="store_true", help="Usar dataset aleatorio (comparación)")
    parser.add_argument("--quick", action="store_true", help="Test rápido de ingesta")
    parser.add_argument("--parameter-sweep", action="store_true", help="Habilitar calibración de parámetros")
    
    args = parser.parse_args()
    
    # Opción de test rápido
    if args.quick:
        print("🔧 Modo quick test activado")
        system = AuroraIntegratedSystem(seed=args.seed, verbose=True)
        success = quick_test_tensors(system, 10)
        if success:
            print("✅ Quick test PASSED - sistema funcionando")
            # Test de reconstrucción básica con coherent tensor
            clean_ft = generate_coherent_tensor(seed=args.seed)
            masked_ft = mask_tensor(clean_ft, 'nivel_3', 0.25, seed=args.seed)
            metrics = system.complete_fractal_enhanced(masked_ft, clean_ft)
            print(f"📊 Test reconstruction: accuracy={metrics.get('accuracy_lvl3', 0):.3f}")
            print(f"📊 KB queries: {metrics.get('kb_queries', 0)}")
            print(f"📊 KB hits: {metrics.get('kb_hits', 0)}")
        else:
            print("❌ Quick test FAILED - revisar configuración")
        exit()
    
    use_coherent = args.coherent and not args.random_dataset
    
    start_total = time.time()
    results = run_integrated_benchmark(
        seed=args.seed,
        n_train=args.train,
        n_test=args.test,
        compare_modes=args.compare,
        save_metrics=args.save_metrics,
        verbose=args.verbose,
        use_coherent_dataset=use_coherent,
        enable_parameter_sweep=args.parameter_sweep
    )
    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
    print("🎯 Benchmark T4-Fractal OPTIMIZADO completado.")
    
    # Resumen ejecutivo
    if results:
        best_result = max(results, key=lambda x: x.get('kb_global_hit_ratio', 0))
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  {best_result['config']} - KB Hit Ratio: {best_result['kb_global_hit_ratio']:.3f}")
        print(f"  Learning Signals: {best_result['learning_signals_pct']:.3f}")
        print(f"  Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")
        
        # Validar curva de aprendizaje
        no_training = next((r for r in results if r['config'] == 'no_training'), None)
        full_training = next((r for r in results if r['config'] in ['fractal_synthesis', 'parameter_sweep']), None)
        
        if no_training and full_training:
            learning_improvement = full_training['kb_global_hit_ratio'] - no_training['kb_global_hit_ratio']
            print(f"\n📈 EVIDENCIA DE APRENDIZAJE FRACTAL OPTIMIZADO:")
            print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")
            if learning_improvement > 0.1:
                print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")
            else:
                print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")
        verbose=args.verbose,
        use_coherent_dataset=use_coherent,
        enable_parameter_sweep=args.parameter_sweep

    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
    print("🎯 Benchmark T4-Fractal OPTIMIZADO completado.")
    
    # Resumen ejecutivo
    if results:
        best_result = max(results, key=lambda x: x.get('kb_global_hit_ratio', 0))





        print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")           
    else:
        print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")            
    if learning_improvement > 0.1:            
        print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")            
        print(f"\n📈 EVIDENCIA DE APRENDIZAJE FRACTAL OPTIMIZADO:")            
        learning_improvement = full_training['kb_global_hit_ratio'] - no_training['kb_global_hit_ratio']        
        if no_training and full_training:                
            full_training = next((r for r in results if r['config'] in ['fractal_synthesis', 'parameter_sweep']), None)        
            no_training = next((r for r in results if r['config'] == 'no_training'), None)        # Validar curva de aprendizaje                
            print(f"  Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")        
            print(f"  Learning Signals: {best_result['learning_signals_pct']:.3f}")        
            print(f"  {best_result['config']} - KB Hit Ratio: {best_result['kb_global_hit_ratio']:.3f}")        
            print(f"\n🏆 MEJOR CONFIGURACIÓN:")        
            print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  {best_result['config']} - KB Hit Ratio: {best_result['kb_global_hit_ratio']:.3f}")
        print(f"  Learning Signals: {best_result['learning_signals_pct']:.3f}")
        print(f"  Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")
        
        # Validar curva de aprendizaje
        no_training = next((r for r in results if r['config'] == 'no_training'), None)
        full_training = next((r for r in results if r['config'] in ['fractal_synthesis', 'parameter_sweep']), None)
        
        if no_training and full_training:
            learning_improvement = full_training['kb_global_hit_ratio'] - no_training['kb_global_hit_ratio']
            print(f"\n📈 EVIDENCIA DE APRENDIZAJE FRACTAL OPTIMIZADO:")
            print(f"  Mejora KB Hit Ratio: +{learning_improvement:.3f}")
            if learning_improvement > 0.1:
                print("  ✅ APRENDIZAJE FRACTAL DETECTADO - Sistema mejora con entrenamiento optimizado")
            else:
                print("  ⚠️  APRENDIZAJE FRACTAL DÉBIL - Revisar normalización Ms→slots optimizada")
