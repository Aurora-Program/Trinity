#!/usr/bin/env python3
"""
Benchmark T4-Fractal Golden-Fibo Edition - PRAGMATIC VERSION
=============================================================
Benchmark limpio con Relator Fractal y rotación áurea.
Optimizado para rendimiento y debugging efectivo.
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

# 🔧 LOGGING: Solo una configuración
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RelatorFractal:
    """🌌 RELATOR FRACTAL: Análisis relacional en cascada fractal"""
    
    def __init__(self, chaos_seed=0.314):
        from allcode import Trigate
        self.trigate = Trigate()
        self.chaos_seed = chaos_seed
        self.phi_ratio = (5**0.5 - 1) / 2
        
    def chaotic_order(self, n, r=3.95, x0=None):
        """Genera permutación caótica determinística usando mapa logístico."""
        if x0 is None:
            x0 = self.chaos_seed
        
        x = x0
        seq = list(range(n))
        
        for i in range(n):
            x = r * x * (1 - x)
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
            if len(trio) != 3 or any(v is None for v in trio):
                return None
            
            m1, _ = self.trigate.synthesize(trio[0], trio[1])
            if m1 is None:
                return None
                
            m_rel, _ = self.trigate.synthesize(m1, trio[2])
            return m_rel
            
        except Exception:
            return None
    
    def relate_fractal(self, ft: FractalTensor, use_chaos=True, use_phi=True):
        """🌌 ANÁLISIS RELACIONAL FRACTAL MULTINIVEL"""
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
                
            participation = self.participation_mask(current_vectors)
            valid_indices = [i for i, valid in enumerate(participation) if valid]
            
            if len(valid_indices) < 3:
                break
            
            # Generar orden de procesamiento
            if use_chaos and use_phi:
                if level_idx % 2 == 0:
                    order = self.chaotic_order(len(valid_indices))
                else:
                    order = self.phi_permutation(len(valid_indices))
            elif use_chaos:
                order = self.chaotic_order(len(valid_indices))
            elif use_phi:
                order = self.phi_permutation(len(valid_indices))
            else:
                order = list(range(len(valid_indices)))
            
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
            
            coverage_metrics['processed_trios'] += processed_trios
            coverage_metrics['valid_relations'] += valid_relations
            coverage_metrics['participation_rate'] = len(valid_indices) / len(current_vectors)
            
            if Ms_level:
                Ms_rel_stack.append(Ms_level)
                current_vectors = Ms_level
            else:
                break
        
        M_rel_emergent = None
        if Ms_rel_stack:
            M_rel_emergent = Ms_rel_stack[-1][0] if Ms_rel_stack[-1] else None
        
        return {
            'Ms_rel_levels': Ms_rel_stack,
            'M_rel_emergent': M_rel_emergent,
            'coverage_metrics': coverage_metrics,
            'chaos_used': use_chaos,
            'phi_used': use_phi
        }

class AuroraIntegratedSystem:
    """🔧 Sistema Aurora completamente integrado"""
    
    def __init__(self, seed: int, enable_rotation: bool = True, 
                 enable_persistence: bool = True, verbose: bool = False,
                 enable_fractal_relator: bool = False):
        random.seed(seed)
        self.seed = seed
        self.enable_rotation = enable_rotation
        self.enable_persistence = enable_persistence
        self.verbose = verbose
        
        # Stack Aurora
        self.kb = KnowledgeBase()
        self.evolver = Evolver()
        self.extender = Extender(self.kb, self.evolver)
        self.transcender = Transcender()
        
        # 🔧 POOL MANAGER: Safe import con fallback
        if enable_rotation:
            try:
                from utils_golden import integrate_golden_rotation
                self.tensor_pool = integrate_golden_rotation(self, enable_persistence=enable_persistence)
                self.rotation_enabled = True
                
                if enable_persistence:
                    state_file = f"rotor_state_seed_{seed}.pkl"
                    try:
                        self.tensor_pool.load_pool_state(state_file)
                    except FileNotFoundError:
                        if verbose:
                            logger.info(f"No pool state found: {state_file}")
                    
            except ImportError as e:
                self.tensor_pool = None
                self.rotation_enabled = False
                if verbose:
                    logger.warning(f"utils_golden unavailable: {e}")
        else:
            self.tensor_pool = None
            self.rotation_enabled = False
        
        # 🌌 RELATOR FRACTAL
        self.enable_fractal_relator = enable_fractal_relator
        if enable_fractal_relator:
            self.relator_fractal = RelatorFractal(chaos_seed=seed * 0.314)
            if verbose:
                logger.info("✅ Relator Fractal activado")
        else:
            self.relator_fractal = None
        
        # Métricas
        self.training_entries = 0
        self.coherence_violations = 0
        self.tensor_processing_attempts = 0

    def _canonicalize_ms(self, ms: List[int]) -> Tuple[int, int, int]:
        """🔧 Pre-fold usando rotación áurea para normalizar Ms."""
        if not ms or len(ms) != 3:
            return (0, 0, 0)
        
        phi_inv = 1 / 1.618033988749895
        sum_ms = sum(ms) if all(x is not None for x in ms) else 0
        rot_index = int((sum_ms * phi_inv) * 3) % 3
        
        normalized = ms[rot_index:] + ms[:rot_index]
        return tuple(int(x) if x is not None else 0 for x in normalized)

    def _ms_slot_id(self, ms_raw: List[int]) -> str:
        """🔧 Genera ID de slot fractal único."""
        ms_canon = self._canonicalize_ms(ms_raw)
        return f"{ms_canon[0]}{ms_canon[1]}{ms_canon[2]}"

    def ingest_fractal_tensor(self, ft: FractalTensor, space_id: str = "default"):
        """🔧 Ingesta con normalización fractal."""
        self.tensor_processing_attempts += 1
        
        try:
            if self.tensor_pool:
                self.tensor_pool.add_tensor(ft)
            
            results = self.transcender.compute_fractal(ft)
            
            if results and 'final' in results:
                final_result = results['final']
                m_emergent = final_result.get('M_emergent', [])
                meta_m = final_result.get('MetaM', [])
                
                if (m_emergent and len(m_emergent) == 3 and all(x is not None for x in m_emergent) and
                    meta_m and len(meta_m) == 3 and all(x is not None for x in meta_m)):
                    
                    slot_id = self._ms_slot_id(m_emergent)
                    combined_id = f"{space_id}_{slot_id}"
                    
                    self.kb.add_entry(
                        A=final_result['A'],
                        B=final_result['B'],
                        C=final_result['C'],
                        M_emergent=m_emergent,
                        MetaM=meta_m,
                        R_validos=[],
                        space_id=space_id,
                        transcender_id=combined_id
                    )
                    self.training_entries += 1
                    
                    if self.verbose:
                        logger.debug(f"Stored slot {slot_id}: Ms={m_emergent}")
                else:
                    self.coherence_violations += 1
                    
        except Exception as e:
            self.coherence_violations += 1
            if self.verbose:
                logger.warning(f"Ingestion failed: {e}")

    def complete_fractal_enhanced(self, ft_masked: FractalTensor, 
                                 target_ft: FractalTensor,
                                 enable_parameter_sweep: bool = False) -> Dict:
        """🔧 Reconstrucción con Relator Fractal opcional."""
        try:
            start_time = time.time()
            
            # 🌌 ANÁLISIS RELACIONAL FRACTAL
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
                            logger.info(f"🌌 Relator: {coverage['valid_relations']}/{coverage['processed_trios']} valid")
                
                except Exception as e:
                    if self.verbose:
                        logger.debug(f"Relator failed: {e}")
                    relational_context = None
            
            # Métricas básicas
            metrics = {
                'accuracy_lvl3': 0.0,
                'reconstruction_method': 'basic_fractal',
                'duration': time.time() - start_time,
                'kb_queries': 0,
                'kb_hits': 0,
                'learning_signal': False
            }
            
            # 🌌 MÉTRICAS DEL RELATOR FRACTAL
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
            
            return {
                'accuracy_lvl3': 0.0,
                'reconstruction_method': 'exception_fallback',
                'relational_coverage': 0.0,
                'relational_levels': 0,
                'relational_valid_ratio': 0.0,
                'error': str(e)
            }

# 🔧 UTILS: Funciones unificadas sin duplicación
def generate_coherent_tensor(A: List = None, B: List = None, seed: int = None) -> FractalTensor:
    """Genera tensor coherente siguiendo lógica Aurora."""
    if seed is not None:
        local_random = random.Random(seed)
    else:
        local_random = random
    
    if A is None:
        A = [local_random.randint(0, 1) for _ in range(3)]
    if B is None:
        B = [local_random.randint(0, 1) for _ in range(3)]
    
    C = [(a + b) % 2 for a, b in zip(A, B)]
    nivel_3 = [A, B, C]
    
    # Expandir a niveles superiores
    nivel_9 = []
    for i in range(9):
        base_vec = nivel_3[i % 3].copy()
        variation = [(base_vec[j] + i) % 2 for j in range(3)]
        nivel_9.append(variation)
    
    nivel_27 = []
    for i in range(27):
        base_vec = nivel_9[i % 9].copy()
        variation = [(base_vec[j] + (i // 9)) % 2 for j in range(3)]
        nivel_27.append(variation)
    
    return FractalTensor(nivel_3=nivel_3, nivel_9=nivel_9, nivel_27=nivel_27)

def mask_tensor(ft: FractalTensor, level: str = 'nivel_3', ratio: float = 0.25, seed: int = None) -> FractalTensor:
    """Enmascara tensor con semilla local."""
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

def create_tensor_families(samples_per_family: int, seed: int) -> Dict[str, List[FractalTensor]]:
    """Genera familias de tensores con diversidad lógica garantizada."""
    print("🔧 Creando familias de tensores...")
    local_random = random.Random(seed)
    transcender_verifier = Transcender()
    families = {"arithmetic": [], "geometric": [], "cyclic": []}
    
    unique_ms_signatures = set()

    for family_name in families.keys():
        family_tensors = []
        family_ms_signature = None
        attempts = 0
        max_attempts = 1000
        
        while len(family_tensors) < samples_per_family and attempts < max_attempts:
            attempts += 1
            A = [local_random.randint(0, 1) for _ in range(3)]
            B = [local_random.randint(0, 1) for _ in range(3)]
            C = [local_random.randint(0, 1) for _ in range(3)]
            ft = FractalTensor(nivel_3=[A, B, C])
            
            try:
                res = transcender_verifier.compute(A, B, C)
                ms = tuple(res.get('M_emergent', [])) if res else None
            except:
                continue
            
            if not ms or None in ms or ms == (0,0,0):
                continue

            if not family_ms_signature:
                if ms not in unique_ms_signatures:
                    family_ms_signature = ms
                    unique_ms_signatures.add(ms)
                    family_tensors.append(ft)
            elif ms == family_ms_signature:
                family_tensors.append(ft)
        
        families[family_name] = family_tensors
        print(f"  ✅ {family_name}: {family_ms_signature} ({len(family_tensors)} tensores)")

    return families

def run_benchmark(seed: int = 42, n_train: int = 500, n_test: int = 150,
                 compare_modes: bool = True, save_metrics: bool = True,
                 verbose: bool = False, enable_fractal_relator: bool = False,
                 enable_parameter_sweep: bool = False):
    """🔧 Benchmark principal con métricas limpias."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        print("🔧 Benchmark T4-Fractal Golden-Fibo iniciado...")
    else:
        print("🔧 Benchmark T4-Fractal Golden-Fibo iniciado...")
    
    results_data = []
    
    # Configuraciones
    configs = [
        ("no_training", False, False, 0, False),
        ("level1_only", False, False, 0.3, True),
        ("fractal_synthesis", True, True, 1.0, True),
        ("parameter_sweep", True, True, 1.0, True)
    ] if compare_modes else [("fractal_synthesis", True, True, 1.0, True)]
    
    for config_name, enable_rotation, enable_persistence, training_ratio, allow_fallback in configs:
        print(f"\n🔧 Config: {config_name}")
        
        system = AuroraIntegratedSystem(
            seed=seed, 
            enable_rotation=enable_rotation,
            enable_persistence=enable_persistence,
            verbose=verbose,
            enable_fractal_relator=enable_fractal_relator
        )
        
        actual_n_train = int(n_train * training_ratio)
        
        # FASE 1: Entrenamiento
        if actual_n_train > 0:
            print(f"🔧 Entrenando con {actual_n_train} tensores...")
            samples_per_family = max(1, actual_n_train // 3)
            tensor_families = create_tensor_families(samples_per_family, seed)
            successful_ingests = 0

            for family_name, tensors in tensor_families.items():
                for ft in tensors:
                    entries_before = system.training_entries
                    system.ingest_fractal_tensor(ft, family_name)
                    if system.training_entries > entries_before:
                        successful_ingests += 1
            
            print(f"✅ Ingested: {successful_ingests}/{actual_n_train}")
        
        # FASE 2: Testing
        print(f"🔧 Testing con {n_test} tensores...")
        test_metrics = []
        
        for i in range(n_test):
            clean_ft = generate_coherent_tensor(seed=seed + i + 1000)
            masked_ft = mask_tensor(clean_ft, 'nivel_3', 0.20, seed=seed+i)
            
            try:
                metrics = system.complete_fractal_enhanced(
                    masked_ft, clean_ft, 
                    enable_parameter_sweep=(config_name == "parameter_sweep") and enable_parameter_sweep
                )
                metrics['config'] = config_name
                test_metrics.append(metrics)
            except Exception as e:
                if verbose:
                    logger.error(f"Test {i} failed: {e}")
                test_metrics.append({
                    'accuracy_lvl3': 0.0,
                    'config': config_name,
                    'relational_coverage': 0.0
                })
        
        # Promedios
        n_valid_tests = len(test_metrics)
        if n_valid_tests == 0:
            continue
            
        avg_metrics = {
            'config': config_name,
            'training_entries': system.training_entries,
            'avg_accuracy_lvl3': sum(m.get('accuracy_lvl3', 0) for m in test_metrics) / n_valid_tests,
            'avg_relational_coverage': sum(m.get('relational_coverage', 0) for m in test_metrics) / n_valid_tests,
            'avg_relational_levels': sum(m.get('relational_levels', 0) for m in test_metrics) / n_valid_tests
        }
        
        results_data.append(avg_metrics)
        
        print(f"📊 {config_name.upper()}:")
        print(f"  Training entries: {system.training_entries}")
        print(f"  Accuracy lvl3: {avg_metrics['avg_accuracy_lvl3']:.3f}")
        if enable_fractal_relator:
            print(f"  Relational coverage: {avg_metrics['avg_relational_coverage']:.3f}")
            print(f"  Relational levels: {avg_metrics['avg_relational_levels']:.1f}")
    
    # 🔧 GUARDAR: Con timestamp para evitar sobrescritura
    if save_metrics:
        timestamp = int(time.time())
        filename = f'data/bench_gf_s{seed}_t{timestamp}.csv'
        os.makedirs('data', exist_ok=True)
        
        with open(filename, 'w', newline='') as csvfile:
            if results_data:
                writer = csv.DictWriter(csvfile, fieldnames=results_data[0].keys())
                writer.writeheader()
                writer.writerows(results_data)
        print(f"\n💾 Métricas guardadas: {filename}")
    
    return results_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora T4-Fractal Golden-Fibo Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--compare", action="store_true", help="Comparar múltiples configs")
    parser.add_argument("--save-metrics", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fractal-relator", action="store_true", help="Activar Relator Fractal")
    parser.add_argument("--parameter-sweep", action="store_true", help="Calibración de parámetros")
    parser.add_argument("--chaos-only", action="store_true", help="Solo permutación caótica")
    parser.add_argument("--phi-only", action="store_true", help="Solo permutación áurea")
    
    args = parser.parse_args()
    
    # Validación
    if args.chaos_only and args.phi_only:
        print("⚠️ No se pueden usar --chaos-only y --phi-only al mismo tiempo")
        exit(1)
    
    start_time = time.time()
    
    results = run_benchmark(
        seed=args.seed,
        n_train=args.train,
        n_test=args.test,
        compare_modes=args.compare,
        save_metrics=args.save_metrics,
        verbose=args.verbose,
        enable_fractal_relator=args.fractal_relator,
        enable_parameter_sweep=args.parameter_sweep
    )
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Duración total: {total_time:.2f}s")
    print("🎯 Benchmark completado.")
    
    if results:
        best_result = max(results, key=lambda x: x.get('avg_accuracy_lvl3', 0))
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  {best_result['config']} - Accuracy: {best_result['avg_accuracy_lvl3']:.3f}")
        
        if args.fractal_relator:
            print(f"  Relational Coverage: {best_result.get('avg_relational_coverage', 0):.3f}")
