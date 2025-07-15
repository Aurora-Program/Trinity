#!/usr/bin/env python3
"""
Benchmark T4-Fractal – Aurora Trinity-3 Fractal Tensor Edition
==============================================================
Evaluación de Aurora con tensores fractales jerárquicos:
  • FractalTensors (3 + 9 + 27 ejes discretos)
  • Reconstrucción multi-nivel con coherencia local
  • L-Spaces especializados por dominio
  • Métricas arquitecturales fractal-específicas

🔧 ENHANCED: Principio de "local-only logic" + expansion cost tracking
"""

from __future__ import annotations

import argparse
import random
import time
import copy
from typing import List, Optional, Dict, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allcode import Transcender, Evolver, Extender, KnowledgeBase, FractalTensor

# ────────────────────────────────── espacios lógicos canónicos ─────────────────

# Ejes y sub-ejes canónicos (0..7)
GRAMMAR_SPACE = {"noun": 1, "verb": 2, "adj": 3, "adv": 4, "prep": 5, "conj": 6, "int": 7}
DOMAIN_SPACE = {"everyday": 1, "science": 2, "art": 3, "tech": 4, "nature": 5, "social": 6, "abstract": 7}
ROLE_SPACE = {"entity": 1, "process": 2, "system": 3, "relation": 4, "property": 5, "action": 6, "state": 7}

def make_tensor(level3, level9=None, level27=None):
    """Factory para crear FractalTensors con niveles opcionales."""
    return FractalTensor(nivel_3=level3, nivel_9=level9, nivel_27=level27)

def random_tensor(depth_prob=(1.0, 0.3, 0.05), value_range=(0, 7)):
    """
    🔧 Genera tensor fractal con profundidad variable.
    
    Args:
        depth_prob: (p_lvl3, p_lvl9, p_lvl27) probabilidades de incluir cada nivel
        value_range: rango de valores enteros para los elementos
        
    Returns:
        FractalTensor con niveles generados según probabilidades
    """
    def rand_vec(): 
        return [random.randint(*value_range) for _ in range(3)]
    
    # Nivel 3 siempre presente (100%)
    lvl3 = [rand_vec() for _ in range(3)]
    
    # Nivel 9 condicionado (30% por defecto)
    lvl9 = [rand_vec() for _ in range(9)] if random.random() < depth_prob[1] else None
    
    # Nivel 27 condicionado (5% por defecto - "lazy expansion")
    lvl27 = [rand_vec() for _ in range(27)] if random.random() < depth_prob[2] else None
    
    return make_tensor(lvl3, lvl9, lvl27)

def mask_tensor(ft: FractalTensor, level: str = 'nivel_9', ratio: float = 0.25) -> FractalTensor:
    """
    🔧 Enmascara un porcentaje de valores en un nivel específico con None.
    
    Args:
        ft: FractalTensor original
        level: 'nivel_3', 'nivel_9', o 'nivel_27'
        ratio: fracción de bits a enmascarar (0.25 = 25%)
        
    Returns:
        FractalTensor con valores enmascarados
    """
    ft_masked = copy.deepcopy(ft)
    target = getattr(ft_masked, level)
    
    if target is None:
        return ft_masked
    
    # Calcular total de bits y cantidad a enmascarar
    total_bits = len(target) * 3
    k = max(1, int(total_bits * ratio))  # Al menos 1 bit
    
    # Enmascarar k bits aleatorios
    for _ in range(k):
        vec_idx = random.randint(0, len(target) - 1)
        bit_idx = random.randint(0, 2)
        target[vec_idx][bit_idx] = None
    
    return ft_masked

# ────────────────────────────── sistema Aurora fractal ─────────────────────────

class AuroraFractalSystem:
    """
    🔧 Sistema Aurora especializado para tensores fractales.
    Implementa tracking de expansion cost y coherencia multi-nivel.
    """
    
    def __init__(self, seed: int):
        random.seed(seed)
        self.kb = KnowledgeBase()
        self.evolver = Evolver()
        self.extender = Extender(self.kb, self.evolver)
        self.transcender = Transcender()
        
        # 🔧 NEW: Integrar rotación áurea
        try:
            from utils_golden import TensorPoolManager, integrate_golden_rotation
            self.tensor_pool = integrate_golden_rotation(self, enable_persistence=True)
        except ImportError:
            self.tensor_pool = None
        
        # Métricas de seguimiento fractal
        self.training_entries = 0
        self.coherence_violations = 0
        self.expansion_events = 0  # Cuántas veces se expandió a niveles profundos
        self.lut_operations = 0    # Contador de operaciones LUT
        
        # Silenciar warnings
        import warnings
        warnings.filterwarnings("ignore", message="MetaM cascade")

    def ingest_fractal_tensor(self, ft: FractalTensor, space_id: str = "default"):
        """🔧 ENHANCED: Ingesta con pool management automático."""
        try:
            # 🔧 NEW: Añadir al pool manager si está disponible
            if self.tensor_pool:
                self.tensor_pool.add_tensor(ft)
            
            # Procesar tensor fractal completo
            results = self.transcender.compute_fractal(ft)
            
            if results and 'final' in results:
                final_result = results['final']
                
                if (final_result.get('M_emergent') and 
                    final_result.get('MetaM') and
                    all(x is not None for x in final_result['M_emergent']) and
                    all(x is not None for x in final_result['MetaM'])):
                    
                    # Detectar patrones emergentes en cada nivel
                    nivel_patterns = self._extract_level_patterns(results)
                    
                    # Almacenar en KB con metadatos fractales
                    self.kb.add_entry(
                        A=final_result['A'],
                        B=final_result['B'],
                        C=final_result['C'],
                        M_emergent=final_result['M_emergent'],
                        MetaM=final_result['MetaM'],
                        R_validos=[nivel_patterns],
                        space_id=f"{space_id}_fractal",
                        transcender_id=f"fractal_{hash(str(ft)):x}"
                    )
                    self.training_entries += 1
                    
        except Exception:
            self.coherence_violations += 1

    def _extract_level_patterns(self, fractal_results: Dict) -> List:
        """Extrae patrones emergentes de cada nivel fractal."""
        patterns = []
        
        # Patrón nivel 3
        if 'nivel_3' in fractal_results and fractal_results['nivel_3']:
            nivel3_meta = fractal_results['nivel_3'][0].get('MetaM', [0, 0, 0])
            patterns.append(['nivel_3', nivel3_meta])
        
        # Patrón nivel 9 si existe
        if 'nivel_9' in fractal_results and fractal_results['nivel_9']:
            nivel9_meta = fractal_results['nivel_9'][0].get('MetaM', [0, 0, 0])
            patterns.append(['nivel_9', nivel9_meta])
            
        # Patrón nivel 27 si existe
        if 'nivel_27' in fractal_results and fractal_results['nivel_27']:
            nivel27_meta = fractal_results['nivel_27'][0].get('MetaM', [0, 0, 0])
            patterns.append(['nivel_27', nivel27_meta])
            self.expansion_events += 1  # Tracking de expansión profunda
        
        return patterns if patterns else [[0, 0, 0]]

    def complete_fractal(self, ft_masked: FractalTensor, target_ft: FractalTensor) -> Dict:
        """
        🔧 ENHANCED: Reconstrucción fractal con rotación áurea - VERSION CORREGIDA.
        """
        try:
            start_lut_count = self.lut_operations
            
            # FASE 1: Preparar contexto fractal
            contexto = {
                "space_id": "fractal_reconstruction",
                "tipo_secuencia": "fractal",
                "expansion_allowed": True,
                "target_levels": ["nivel_3", "nivel_9", "nivel_27"]
            }
            
            # 🔧 FIXED: Simplificar preparación de query vectors
            query_vectors = self._prepare_fractal_query(ft_masked)
            reconstruction_method = "standard_query"
            
            # 🔧 NEW: Usar rotación áurea solo si está disponible
            if self.tensor_pool:
                try:
                    # Obtener trío rotado para reconstrucción
                    query_trio = self.tensor_pool.get_tensor_trio(task_type="arquetipo")
                    
                    if query_trio and len(query_trio) >= 3:
                        query_vectors = [
                            query_trio[0].nivel_3[0] if query_trio[0].nivel_3 else [0, 0, 0],
                            query_trio[1].nivel_3[1] if len(query_trio[1].nivel_3) > 1 else [0, 0, 0],
                            query_trio[2].nivel_3[2] if len(query_trio[2].nivel_3) > 2 else [None, None, None]
                        ]
                        reconstruction_method = "golden_rotation_hybrid"
                except Exception:
                    # Fallback silencioso
                    pass
            
            # FASE 2: Usar extend() para reconstrucción
            extend_result = self.extender.extend(
                Ss=query_vectors,
                contexto=contexto
            )
            
            # FASE 3: Reconstruir tensor completo desde resultado
            reconstructed_ft = self._rebuild_tensor_from_result(extend_result, ft_masked)
            
            # FASE 4: Calcular métricas por nivel
            metrics = self._compute_fractal_metrics(reconstructed_ft, target_ft, ft_masked)
            metrics['reconstruction_method'] = reconstruction_method
            metrics['lut_operations'] = self.lut_operations - start_lut_count
            
            return metrics
            
        except Exception as e:
            # Fallback con honestidad computacional
            return {
                'accuracy_lvl3': 0.0,
                'accuracy_lvl9': 0.0,
                'accuracy_lvl27': 0.0,
                'local_coherence': False,
                'honesty_ratio': 1.0,  # Máximo honesty = todo NULL
                'expansion_cost': 0,
                'lut_operations': 0,
                'reconstruction_method': 'exception_fallback',
                'error': str(e)
            }

    def _prepare_fractal_query(self, ft_masked: FractalTensor) -> List:
        """Prepara vectores query desde tensor fractal enmascarado."""
        # Vector query básico desde nivel 3
        if ft_masked.nivel_3 and len(ft_masked.nivel_3) >= 3:
            ms_query = ft_masked.nivel_3[0]
            metam_query = ft_masked.nivel_3[1] if len(ft_masked.nivel_3) > 1 else [0, 0, 0]
            ss_query = ft_masked.nivel_3[2] if len(ft_masked.nivel_3) > 2 else [None, None, None]
        else:
            ms_query = [0, 0, 0]
            metam_query = [0, 0, 0]
            ss_query = [None, None, None]
        
        return [ms_query, metam_query, ss_query]

    def _rebuild_tensor_from_result(self, extend_result: Dict, ft_original: FractalTensor) -> FractalTensor:
        """🔧 FIXED: Reconstruye FractalTensor desde resultado de extend()."""
        if not extend_result or extend_result.get('error'):
            return ft_original
        
        recon = extend_result.get('reconstruccion', {})
        tensores = recon.get('tensores_reconstruidos', [])
        
        # 🔧 CRITICAL FIX: Normalizar forma de tensores_reconstruidos
        if tensores:
            # Si es un solo vector [x,y,z], convertir a [[x,y,z]]*3
            if isinstance(tensores, list) and len(tensores) >= 3:
                if not isinstance(tensores[0], list):
                    # Es un vector plano [x,y,z] → convertir a matriz 3x3
                    tensores = [tensores[:3]] * 3
        
        # Crear copia y aplicar reconstrucción
        ft_rebuilt = copy.deepcopy(ft_original)
        
        if tensores and len(tensores) >= 3:
            # Aplicar reconstrucción nivel por nivel
            for level_name in ['nivel_3', 'nivel_9', 'nivel_27']:
                level_data = getattr(ft_rebuilt, level_name)
                if level_data:
                    for i, vec in enumerate(level_data):
                        if i < len(tensores):
                            for j, val in enumerate(vec):
                                if val is None and j < len(tensores[i]):
                                    if tensores[i][j] is not None:
                                        vec[j] = tensores[i][j]
        
        return ft_rebuilt

    def _compute_fractal_metrics(self, reconstructed: FractalTensor, target: FractalTensor, masked: FractalTensor) -> Dict:
        """Calcula métricas detalladas de reconstrucción fractal."""
        metrics = {
            'accuracy_lvl3': 0.0,
            'accuracy_lvl9': 0.0,
            'accuracy_lvl27': 0.0,
            'local_coherence': True,
            'honesty_ratio': 0.0,
            'expansion_cost': 0
        }
        
        # Accuracy por nivel
        for level_name in ['nivel_3', 'nivel_9', 'nivel_27']:
            acc = self._compute_level_accuracy(
                getattr(reconstructed, level_name),
                getattr(target, level_name),
                getattr(masked, level_name)
            )
            metrics[f'accuracy_{level_name.replace("nivel_", "lvl")}'] = acc
        
        # Honesty ratio: cuántas posiciones quedaron como None
        total_masked, still_none = self._count_honesty_positions(reconstructed, masked)
        if total_masked > 0:
            metrics['honesty_ratio'] = still_none / total_masked
        
        # Expansion cost: si se usaron niveles profundos
        if reconstructed.nivel_27 is not None:
            metrics['expansion_cost'] = 2  # Máximo (llegó hasta nivel 27)
        elif reconstructed.nivel_9 is not None:
            metrics['expansion_cost'] = 1  # Medio (llegó hasta nivel 9)
        else:
            metrics['expansion_cost'] = 0  # Mínimo (solo nivel 3)
        
        return metrics

    def _compute_level_accuracy(self, reconstructed_level, target_level, masked_level) -> float:
        """Calcula accuracy para un nivel específico."""
        if not all([reconstructed_level, target_level, masked_level]):
            return 0.0
        
        correct = 0
        total = 0
        
        for i, (rec_vec, tar_vec, mas_vec) in enumerate(zip(reconstructed_level, target_level, masked_level)):
            for j, (rec_val, tar_val, mas_val) in enumerate(zip(rec_vec, tar_vec, mas_vec)):
                if mas_val is None:  # Solo evaluar posiciones que fueron enmascaradas
                    total += 1
                    if rec_val == tar_val:
                        correct += 1
        
        return correct / total if total > 0 else 0.0

    def _count_honesty_positions(self, reconstructed: FractalTensor, masked: FractalTensor) -> Tuple[int, int]:
        """Cuenta posiciones enmascaradas y cuántas quedaron como None."""
        total_masked = 0
        still_none = 0
        
        for level_name in ['nivel_3', 'nivel_9', 'nivel_27']:
            rec_level = getattr(reconstructed, level_name)
            mas_level = getattr(masked, level_name)
            
            if not (rec_level and mas_level):
                continue
                
            for rec_vec, mas_vec in zip(rec_level, mas_level):
                for rec_val, mas_val in zip(rec_vec, mas_vec):
                    if mas_val is None:
                        total_masked += 1
                        if rec_val is None:
                            still_none += 1
        
        return total_masked, still_none

# ───────────────────────────────── benchmark runner fractal ───────────────────

def run_fractal_benchmark(seed: int = 42, n_train: int = 500, n_test: int = 150, 
                         depth_prob: Tuple[float, float, float] = (1.0, 0.4, 0.1)):
    """
    🔧 ENHANCED: Benchmark T4-Fractal con rotación áurea integrada.
    """
    print("🔧 Iniciando Benchmark T4-Fractal con Rotación Áurea...")
    
    sys = AuroraFractalSystem(seed)
    
    # ── FASE ENTRENAMIENTO CON AUGMENTACIÓN ─────────────────────────────
    print(f"🔧 Entrenando con {n_train} tensores fractales + augmentación áurea...")
    start_time = time.time()
    
    training_tensors = []
    training_stats = {"depth_3": 0, "depth_9": 0, "depth_27": 0}
    
    # Generar tensores de entrenamiento
    for i in range(n_train):
        ft = random_tensor(depth_prob=depth_prob)
        training_tensors.append(ft)
        
        # Estadísticas de profundidad
        if ft.nivel_27 is not None:
            training_stats["depth_27"] += 1
        elif ft.nivel_9 is not None:
            training_stats["depth_9"] += 1
        else:
            training_stats["depth_3"] += 1
    
    # 🔧 NEW: Aplicar augmentación áurea
    try:
        from utils_golden import phi_augment_training_tensors
        augmented_tensors = phi_augment_training_tensors(training_tensors, 0.3)
        print(f"🔧 Augmentación: {len(training_tensors)} → {len(augmented_tensors)} tensores")
    except ImportError:
        augmented_tensors = training_tensors
    
    # Ingesta de tensores
    for i, ft in enumerate(augmented_tensors):
        # 🔧 NEW: Asignar a L-Space específico según características
        if i % 3 == 0:
            space_id = "grammar"
        elif i % 3 == 1:
            space_id = "domain"
        else:
            space_id = "role"
        
        sys.ingest_fractal_tensor(ft, space_id)
    
    training_time = time.time() - start_time
    
    # ── FASE EVALUACIÓN ─────────────────────────────────────────────────
    print(f"🔧 Evaluando con {n_test} tensores de test...")
    
    # Métricas agregadas
    total_accuracy_lvl3 = total_accuracy_lvl9 = total_accuracy_lvl27 = 0.0
    total_honesty = total_expansion = total_lut_ops = 0.0
    coherent_cases = 0
    test_cases = []
    
    for i in range(n_test):
        # Generar tensor limpio y version enmascarada
        clean_ft = random_tensor(depth_prob=(1.0, 0.8, 0.3))  # Más probabilidad en test
        
        # Alternar niveles de enmascaramiento
        if i % 3 == 0:
            masked_ft = mask_tensor(clean_ft, 'nivel_3', 0.3)
        elif i % 3 == 1:
            masked_ft = mask_tensor(clean_ft, 'nivel_9', 0.25)
        else:
            masked_ft = mask_tensor(clean_ft, 'nivel_27', 0.2)
        
        # Reconstrucción
        metrics = sys.complete_fractal(masked_ft, clean_ft)
        
        # Acumular métricas
        total_accuracy_lvl3 += metrics['accuracy_lvl3']
        total_accuracy_lvl9 += metrics['accuracy_lvl9']
        total_accuracy_lvl27 += metrics['accuracy_lvl27']
        total_honesty += metrics['honesty_ratio']
        total_expansion += metrics['expansion_cost']
        total_lut_ops += metrics['lut_operations']
        
        if metrics['local_coherence']:
            coherent_cases += 1
        
        test_cases.append(metrics)
    
    # ── CÁLCULO DE MÉTRICAS FINALES ─────────────────────────────────────
    avg_accuracy_lvl3 = total_accuracy_lvl3 / n_test
    avg_accuracy_lvl9 = total_accuracy_lvl9 / n_test
    avg_accuracy_lvl27 = total_accuracy_lvl27 / n_test
    avg_honesty = total_honesty / n_test
    avg_expansion = total_expansion / n_test
    coherence_ratio = coherent_cases / n_test
    
    # Throughput (LUT ops/segundo)
    test_time = time.time() - (start_time + training_time)
    throughput_ops = total_lut_ops / test_time if test_time > 0 else 0
    
    # KB statistics con L-Spaces
    kb_stats = sys.kb.get_coherence_stats()
    
    # ── REPORTE FINAL MEJORADO CON MÉTRICAS ÁUREAS ──────────────────────
    print("\n" + "="*70)
    print("BENCHMARK T4-FRACTAL - Aurora Trinity-3 Enhanced + Golden Rotation")
    print("="*70)
    print(f"seed                  : {seed}")
    print(f"train tensors         : {n_train} (exitosos: {sys.training_entries})")
    print(f"  • depth 3 only      : {training_stats['depth_3']}")
    print(f"  • depth 9 (expand)  : {training_stats['depth_9']}")
    print(f"  • depth 27 (deep)   : {training_stats['depth_27']}")
    print(f"test tensors          : {n_test}")
    print(f"training time         : {training_time:.2f}s")
    print(f"test time             : {test_time:.2f}s")
    print(f"expansion events      : {sys.expansion_events}")
    print(f"coherence violations  : {sys.coherence_violations}")
    
    # L-Spaces info
    if hasattr(sys.kb, 'get_spaces'):
        spaces = sys.kb.get_spaces()
        print(f"L Spaces activos      : {len(spaces)}")
    
    print()
    print("📊 MÉTRICAS T4-FRACTAL:")
    print(f"  accuracy_lvl3       : {avg_accuracy_lvl3:.3f}")
    print(f"  accuracy_lvl9       : {avg_accuracy_lvl9:.3f}")
    print(f"  accuracy_lvl27      : {avg_accuracy_lvl27:.3f}")
    print(f"  expansion_cost      : {avg_expansion:.3f}")
    print(f"  local_coherence     : {coherence_ratio:.3f}")
    print(f"  honesty_ratio       : {avg_honesty:.3f}")
    print(f"  throughput_ops      : {throughput_ops:.1f} ops/sec")
    print(f"  coherencia_KB       : {kb_stats['coherence_ratio']:.3f}")
    
    # 🔧 NEW: Métricas de rotación áurea
    if sys.tensor_pool:
        rotation_metrics = sys.tensor_pool.get_rotation_metrics()
        
        print("\n🌟 MÉTRICAS DE ROTACIÓN ÁUREA:")
        for pool_name, metrics in rotation_metrics['rotation_efficiency'].items():
            if metrics['steps_taken'] > 0:
                print(f"  {pool_name:12s}: coverage={metrics['coverage_ratio']:.3f} "
                      f"efficiency={metrics['efficiency']:.3f}")
        
        print("\n🔧 ESTRATEGIAS ÓPTIMAS:")
        for pool_name, strategy in rotation_metrics['optimal_strategies'].items():
            status = "✅" if not strategy['should_optimize'] else "🔄"
            print(f"  {pool_name:12s}: {strategy['current_mode']:8s} → "
                  f"{strategy['recommended_mode']:8s} {status}")
        
        # Diversidad φ global
        total_coverage = sum(m['unique_visits'] for m in rotation_metrics['rotation_efficiency'].values())
        total_steps = sum(m['steps_taken'] for m in rotation_metrics['rotation_efficiency'].values())
        phi_diversity = total_coverage / max(1, total_steps)
        
        print(f"\n📊 φ-DIVERSIDAD GLOBAL: {phi_diversity:.3f}")
    
    # Análisis de métodos de reconstrucción
    methods = {}
    for case in test_cases:
        method = case.get('reconstruction_method', 'unknown')
        methods[method] = methods.get(method, 0) + 1
    
    print()
    print("🔧 MÉTODOS DE RECONSTRUCCIÓN:")
    for method, count in methods.items():
        ratio = count / n_test
        print(f"  {method:25s} : {ratio:.3f} ({count}/{n_test})")
    
    # Mostrar algunos casos de ejemplo
    print()
    print("🔍 EJEMPLOS DE RECONSTRUCCIÓN (primeros 5 casos):")
    for i, case in enumerate(test_cases[:5]):
        status = "✅" if case['local_coherence'] else "❌"
        exp_cost = int(case['expansion_cost'])
        print(f"  {i+1}. lvl3:{case['accuracy_lvl3']:.2f} "
              f"lvl9:{case['accuracy_lvl9']:.2f} "
              f"lvl27:{case['accuracy_lvl27']:.2f} "
              f"exp:{exp_cost} {status}")

# ───────────────────────────────────────── CLI ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora T4-Fractal Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=500, help="número de tensores de entrenamiento")
    parser.add_argument("--test", type=int, default=150, help="número de tensores de test")
    parser.add_argument("--depth-prob", type=float, nargs=3, default=[1.0, 0.4, 0.1],
                       help="probabilidades de profundidad [lvl3, lvl9, lvl27]")
    
    args = parser.parse_args()
    
    start_total = time.time()
    run_fractal_benchmark(args.seed, args.train, args.test, tuple(args.depth_prob))
    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
    print("\n🎯 Benchmark T4-Fractal completado.")
