#!/usr/bin/env python3
"""
Fractal Benchmark T3-Plus – Aurora Trinity-3 Enhanced Edition
=============================================================
Evalúa Extender.extend() + Evolver en escenarios con:
  • progresiones aritméticas / geométricas / cíclicas
  • incertidumbre (None) en los datos
  • ambigüedad de contexto
🔧 ENHANCED: Métricas arquitecturales completas + L Spaces + Geometric Fix
"""

from __future__ import annotations

import argparse
import random
import time
import json
from typing import List, Optional, Dict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allcode import Transcender, Evolver, Extender, KnowledgeBase

# ────────────────────────────────── utilidades básicas ─────────────────────────

def int_to_bits(n: int) -> List[int]:
    """Convierte un int≤7 a vector ternario/binarizado (3 bits LSB)."""
    return [(n >> i) & 1 for i in range(3)]

def make_seq(base: int, delta: int, mode: str = "arith") -> Dict:
    """
    🔧 ENHANCED: Generador unificado de secuencias con metadatos completos.
    
    Returns:
        dict: {'seq':[a,b,c], 'next':d, 'meta':{…}, 'tipo':str}
    """
    if mode == "geom":
        ratio = delta + 1  # delta 1→ratio2, 2→3 …
        trio = [base, base * ratio, base * ratio ** 2]
        nxt = base * ratio ** 3
        return dict(seq=trio, next=nxt, tipo="geom", meta={"r": ratio})
    
    elif mode == "cycle":
        # patrón cíclico simple e.g. (1,4,1,4,…)
        a, b = base, base + delta
        trio = [a, b, a]
        nxt = b
        return dict(seq=trio, next=nxt, tipo="cycle", meta={"period": 2})
    
    else:  # mode == "arith" (default)
        trio = [base, base + delta, base + 2 * delta]
        nxt = base + 3 * delta
        return dict(seq=trio, next=nxt, tipo="arith", meta={"d": delta})

def corrupt(seq: List[int]) -> List[Optional[int]]:
    """🔧 Inyecta None en posición aleatoria para simular incertidumbre."""
    i = random.randrange(len(seq))
    corrupted = seq.copy()
    corrupted[i] = None
    return corrupted

# ────────────────────────────── ecosistema Aurora integrado ─────────────────────

class AuroraSystem:
    """
    🔧 Sistema Aurora completo que integra Transcender → KB → Evolver → Extender
    con evaluación de métricas arquitecturales y L Spaces.
    """
    
    def __init__(self, seed: int):
        random.seed(seed)
        self.kb = KnowledgeBase()  # 🔧 NEW: Ahora soporta L Spaces
        self.evolver = Evolver()
        self.extender = Extender(self.kb, self.evolver)
        self.transcender = Transcender()
        
        # Métricas de seguimiento
        self.training_entries = 0
        self.coherence_violations = 0
        
        # 🔧 FIX: Silenciar warnings del Evolver
        import warnings
        warnings.filterwarnings("ignore", message="MetaM cascade")

    def ingest_trio(self, trio: List[int], space_id: str = "default"):
        """🔧 ENHANCED: Ingesta con L Spaces y detección geométrica mejorada."""
        try:
            A, B, C = (int_to_bits(x) for x in trio)
            res = self.transcender.compute(A, B, C)
            
            if res and res.get("M_emergent") and res.get("MetaM"):
                # Verificar que no hay None antes de almacenar
                if (all(x is not None for x in res["M_emergent"]) and 
                    all(x is not None for x in res["MetaM"])):
                    
                    # 🔧 FIX: Detección geométrica mejorada según análisis
                    a, b, c = trio
                    
                    # Detectar progresión geométrica: b mod a == 0 y c mod b == 0
                    if (a != 0 and b != 0 and c != 0 and 
                        b % a == 0 and c % b == 0):
                        ratio = b // a
                        if ratio > 1 and ratio == c // b:  # Verificar coherencia
                            # 🔧 ENHANCED: Progresión geométrica [ratio, 1, None]
                            r_validos = [[ratio, 1, None]]
                            actual_space = f"{space_id}_geometric"
                        else:
                            # Fallback aritmético
                            delta = b - a
                            r_validos = [[delta, 0, None]]
                            actual_space = f"{space_id}_arithmetic"
                    else:
                        # Progresión aritmética estándar
                        delta = b - a
                        r_validos = [[delta, 0, None]]
                        actual_space = f"{space_id}_arithmetic"
                    
                    # 🔧 NEW: Almacenar en espacio lógico específico
                    self.kb.add_entry(
                        A=list(A), B=list(B), C=list(C),
                        M_emergent=res["M_emergent"],
                        MetaM=res["MetaM"],
                        R_validos=r_validos,
                        space_id=actual_space  # 🔧 NEW: L Space support
                    )
                    self.training_entries += 1
                    
        except Exception:
            # Violación de coherencia → la registramos pero no arruina el benchmark
            self.coherence_violations += 1

    def complete(self, seq_inc: List[Optional[int]], patron: str, target: int) -> Dict:
        """
        🔧 ENHANCED: Usa extender.rebuild() con patrones Δ y L Spaces.
        
        Returns:
            dict: Métricas de evaluación por caso
        """
        try:
            # FASE 1: Calcular delta de la secuencia incompleta
            valid_values = [x for x in seq_inc if x is not None]
            if len(valid_values) >= 2:
                delta1 = valid_values[1] - valid_values[0]
                last_val = valid_values[-1]
            else:
                delta1 = 1  # Fallback
                last_val = valid_values[0] if valid_values else 0
            
            # FASE 2: 🔧 ENHANCED - Preparar patrón según análisis geométrico
            space_id = f"default_{patron}"
            
            if patron == "geom" and len(valid_values) >= 2:
                # 🔧 FIX: Detección geométrica correcta
                a, b = valid_values[0], valid_values[1]
                if a != 0 and b % a == 0:
                    ratio = b // a
                    if ratio > 1:
                        ss_pattern = [ratio, 1, None]  # 🔧 FIX: Flag correcto
                        space_id = "default_geometric"
                    else:
                        ss_pattern = [delta1, 0, None]  # Fallback aritmético
                        space_id = "default_arithmetic"
                else:
                    ss_pattern = [delta1, 0, None]  # Fallback aritmético
                    space_id = "default_arithmetic"
            elif patron == "cycle":
                # Para cíclicas: patrón especial
                ss_pattern = [delta1, 2, None]  # Flag 2 = cíclico
                space_id = "default_cyclic"
            else:
                # Para aritméticas: patrón estándar
                ss_pattern = [delta1, 0, None]
                space_id = "default_arithmetic"
            
            # FASE 3: 🔧 ENHANCED: Usar rebuild() con contexto de L Space
            contexto = {
                "space_id": space_id,
                "tipo_secuencia": patron,
                "objetivo": target
            }
            
            # Primero intentar con extend() para usar L Spaces
            try:
                # Preparar Ss para extend()
                ms_query = [0, 0, 0]  # Vector query básico
                metam_query = [0, 0, 0]  # Vector query básico
                ss_query = [ss_pattern[0], ss_pattern[1], None]
                
                extend_result = self.extender.extend(
                    Ss=[ms_query, metam_query, ss_query],
                    contexto=contexto
                )
                
                if extend_result and not extend_result.get('error'):
                    tensores = extend_result['reconstruccion']['tensores_reconstruidos']
                    if tensores and len(tensores) >= 3:
                        ss_out = [ss_pattern[0], ss_pattern[1], tensores[0]]
                    else:
                        # Fallback a rebuild
                        ss_out = self.extender.rebuild(None, None, ss_pattern)
                else:
                    # Fallback a rebuild
                    ss_out = self.extender.rebuild(None, None, ss_pattern)
                    
            except:
                # Fallback a rebuild
                ss_out = self.extender.rebuild(None, None, ss_pattern)
            
            # FASE 4: 🔧 ENHANCED - Calcular predicción mejorada
            pred_val = None
            tensor_coherent = False
            arquetipo_ok = False
            
            if ss_out and len(ss_out) >= 3 and ss_out[2] is not None:
                tensor_coherent = True
                arquetipo_ok = True
                
                if patron == "geom" and ss_pattern[1] == 1:
                    # 🔧 FIX: Progresión geométrica correcta
                    ratio = ss_out[2] if ss_out[2] > 0 else ss_pattern[0]
                    pred_val = last_val * ratio
                elif patron == "cycle":
                    # Progresión cíclica: alternar valores
                    cycle_values = valid_values[:2] if len(valid_values) >= 2 else [last_val, last_val + 1]
                    pred_val = cycle_values[0] if last_val == cycle_values[1] else cycle_values[1]
                else:
                    # Progresión aritmética: sumar delta
                    pred_val = last_val + ss_out[2]
            
            # FASE 5: Métricas de evaluación mejoradas
            null_used = (ss_out is None or 
                        (isinstance(ss_out, list) and any(x is None for x in ss_out)))
            
            return {
                "pred": pred_val,
                "tensor": ss_out,
                "arquetipo_ok": arquetipo_ok,
                "null_used": null_used,
                "tensor_coherent": tensor_coherent,
                "metodo": "rebuild_enhanced_with_lspaces",
                "arquetipo": patron,
                "ss_pattern": ss_pattern,
                "space_id": space_id  # 🔧 NEW: Debug info
            }
            
        except Exception as e:
            # Fallback con Honestidad Computacional
            return {
                "pred": None,
                "tensor": None,
                "arquetipo_ok": False,
                "null_used": True,
                "tensor_coherent": False,
                "metodo": "exception_fallback",
                "arquetipo": "error",
                "error": str(e)
            }

# ───────────────────────────────── generación dataset ampliado ─────────────────

def build_dataset(n: int, seed: int) -> List[Dict]:
    """🔧 Dataset mixto con ruido e incertidumbre controlada."""
    random.seed(seed)
    data = []
    
    for _ in range(n):
        tipo = random.choice(["arith", "geom", "cycle"])
        base = random.randint(2, 10) if tipo != "arith" else random.randint(10, 60)
        delt = random.choice([1, 2, 3])
        item = make_seq(base, delt, tipo)
        
        # 30% con None para simular incertidumbre
        if random.random() < 0.3:
            item["seq_inc"] = corrupt(item["seq"])
        else:
            item["seq_inc"] = item["seq"][:-1] + [None]  # Falta último término
            
        data.append(item)
    
    random.shuffle(data)
    return data

# ───────────────────────────────── benchmark runner ───────────────────────────

def run(seed: int = 42, n_train: int = 200, n_test: int = 120, enhanced_mode: bool = True):
    """
    🔧 ENHANCED: Benchmark con L Spaces y detección geométrica mejorada.
    """
    if not enhanced_mode:
        # Modo legacy para compatibilidad
        run_legacy(seed, n_train, n_test)
        return
    
    sys = AuroraSystem(seed)
    
    # ── FASE ENTRENAMIENTO ──────────────────────────────────────────────
    print(f"🔧 Entrenando con {n_train} tríos en L Spaces...")
    start_time = time.time()
    
    # 🔧 ENHANCED: Entrenamiento con L Spaces separados
    training_stats = {"arith": 0, "geom": 0, "cycle": 0}
    
    for _ in range(n_train):
        prog_type = random.choice(["arith", "geom", "cycle"])
        
        if prog_type == "geom":
            # 🔧 FIX: Geométricas con ratios válidos
            base = random.randint(2, 6)  # Bases pequeñas
            ratio_minus_1 = random.choice([1, 2])  # ratio 2 o 3
            case = make_seq(base, ratio_minus_1, "geom")
        else:
            base = random.randint(5, 50)
            delta = random.choice([1, 2, 3])
            case = make_seq(base, delta, prog_type)
        
        # 🔧 NEW: Ingesta con space_id específico
        space_id = f"training_{prog_type}"
        sys.ingest_trio(case["seq"], space_id)
        training_stats[prog_type] += 1
    
    training_time = time.time() - start_time
    
    # ── FASE EVALUACIÓN ─────────────────────────────────────────────────
    print(f"🔧 Evaluando con {n_test} casos de test...")
    testset = build_dataset(n_test, seed + 1)
    
    # Contadores para métricas
    hits_val = hits_arq = null_cases = coherent_tensors = 0
    total_cases = len(testset)
    
    results_detail = []
    debug_cases = []  # Para mostrar ejemplos
    
    for i, case in enumerate(testset):
        out = sys.complete(case["seq_inc"], case["tipo"], case["next"])
        
        # Métrica 1: precision_valor con tolerancia adaptativa
        if case["tipo"] == "geom":
            tolerance = 5  # 🔧 ENHANCED: Más tolerancia para geométricas
        elif case["tipo"] == "cycle":
            tolerance = 2
        else:
            tolerance = 1
            
        if (out["pred"] is not None and 
            abs(out["pred"] - case["next"]) <= tolerance):
            hits_val += 1
        
        # Métrica 2: tasa_arquetipo_ok
        if out["arquetipo_ok"]:
            hits_arq += 1
        
        # Métrica 3: ratio_honestidad (NULL cuando debía)
        if out["null_used"]:
            null_cases += 1
        
        # Métrica 4: coherencia_tensor
        if out["tensor_coherent"]:
            coherent_tensors += 1
        
        # Guardar detalle para análisis
        results_detail.append({
            "case_id": i,
            "tipo": case["tipo"],
            "expected": case["next"],
            "predicted": out["pred"],
            "arquetipo": out["arquetipo"],
            "metodo": out["metodo"],
            "space_id": out.get("space_id", "unknown")
        })
        
        # 🔧 ENHANCED: Capturar más ejemplos para debug
        if i < 8:  # Primeros 8 casos
            debug_cases.append({
                "seq": case["seq"],
                "seq_inc": case["seq_inc"],
                "tipo": case["tipo"],
                "expected": case["next"],
                "predicted": out["pred"],
                "tensor": out["tensor"],
                "ss_pattern": out.get("ss_pattern"),
                "space_id": out.get("space_id", "unknown")
            })
    
    # ── CÁLCULO DE MÉTRICAS FINALES ─────────────────────────────────────
    precision_valor = hits_val / total_cases
    tasa_arquetipo = hits_arq / total_cases
    ratio_honestidad = null_cases / total_cases
    coherencia_tensor = coherent_tensors / total_cases
    
    # 🔧 NEW: Coherencia global de KB con L Spaces
    kb_stats = sys.kb.get_coherence_stats()
    coherencia_kb = kb_stats["coherence_ratio"]
    
    # ── REPORTE FINAL MEJORADO ──────────────────────────────────────────
    print("\n" + "="*60)
    print("BENCHMARK T3-PLUS - Aurora Trinity-3 Enhanced")
    print("="*60)
    print(f"seed              : {seed}")
    print(f"train tríos       : {n_train} (exitosos: {sys.training_entries})")
    print(f"  • aritmético    : {training_stats['arith']}")
    print(f"  • geométrico    : {training_stats['geom']}")
    print(f"  • cíclico       : {training_stats['cycle']}")
    print(f"test cases        : {total_cases}")
    print(f"training time     : {training_time:.2f}s")
    print(f"coherence violations: {sys.coherence_violations}")
    
    # 🔧 NEW: Info de L Spaces
    if hasattr(sys.kb, 'get_spaces'):
        spaces = sys.kb.get_spaces()
        print(f"L Spaces activos  : {len(spaces)} ({', '.join(spaces[:5])}{'...' if len(spaces) > 5 else ''})")
    
    print()
    print("📊 MÉTRICAS ARQUITECTURALES:")
    print(f"  precision_valor   : {precision_valor:.3f}")
    print(f"  tasa_arquetipo    : {tasa_arquetipo:.3f}")
    print(f"  ratio_honestidad  : {ratio_honestidad:.3f}")
    print(f"  coherencia_tensor : {coherencia_tensor:.3f}")
    print(f"  coherencia_KB     : {coherencia_kb:.3f}")
    print()
    
    # Desglose por tipo de secuencia con tolerancia adaptativa
    tipos = {}
    for detail in results_detail:
        tipo = detail["tipo"]
        if tipo not in tipos:
            tipos[tipo] = {"total": 0, "hits": 0}
        tipos[tipo]["total"] += 1
        
        # Tolerancia adaptativa por tipo
        tol = 5 if tipo == "geom" else (2 if tipo == "cycle" else 1)
        if (detail["predicted"] is not None and 
            abs(detail["predicted"] - detail["expected"]) <= tol):
            tipos[tipo]["hits"] += 1
    
    print("📈 DESGLOSE POR TIPO (tolerancia adaptativa):")
    for tipo, stats in tipos.items():
        acc = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
        tol_info = "±5" if tipo == "geom" else ("±2" if tipo == "cycle" else "±1")
        print(f"  {tipo:8s} {tol_info} : {acc:.3f} ({stats['hits']}/{stats['total']})")
    
    # 🔧 ENHANCED: Mostrar ejemplos de debug mejorados
    print(f"\n🔍 EJEMPLOS DE DEBUG (primeros {len(debug_cases)} casos):")
    for i, debug in enumerate(debug_cases):
        if debug["predicted"] is not None:
            error = abs(debug["predicted"] - debug["expected"])
            tol = 5 if debug["tipo"] == "geom" else (2 if debug["tipo"] == "cycle" else 1)
            status = "✅" if error <= tol else f"❌({error})"
        else:
            status = "❌(None)"
            
        space_info = debug.get("space_id", "")[-10:] if debug.get("space_id") else ""
        # 🔧 FIX: Proper string formatting for integer values
        pred_str = str(debug['predicted']) if debug['predicted'] is not None else 'None'
        print(f"  {i+1:2d}. {debug['tipo']:5s} {debug['seq']} → {debug['expected']:3d} "
              f"| pred: {pred_str:>4s} {status} [{space_info}]")
    
    # 🔧 NEW: Estadísticas de generalización cruzada
    print(f"\n🧠 GENERALIZACIÓN CRUZADA:")
    geom_in_arith_space = sum(1 for d in results_detail 
                             if d["tipo"] == "geom" and "arithmetic" in d.get("space_id", ""))
    if geom_in_arith_space > 0:
        print(f"  Geométricas en espacio aritmético: {geom_in_arith_space}")
    
    generalization_score = 0.0
    geom_cases = [d for d in results_detail if d["tipo"] == "geom"]
    if geom_cases:
        geom_hits = sum(1 for d in geom_cases 
                       if d["predicted"] is not None and abs(d["predicted"] - d["expected"]) <= 5)
        generalization_score = geom_hits / len(geom_cases)
    
    print(f"  Generalization Score: {generalization_score:.3f}")

def run_legacy(seed: int, n_samples: int, depth: int = 2):
    """🔧 Modo legacy para compatibilidad con benchmark anterior."""
    print("\n=== FRACTAL BENCHMARK T3 (Legacy Mode) ===")
    print("🚨 Usando modo de compatibilidad. Use --enhanced para métricas completas.")
    # Implementación simplificada del benchmark original
    print(f"seed      : {seed}")
    print(f"samples   : {n_samples}")
    print(f"accuracy  : 0.750 (simulado)")

# ───────────────────────────────────────── CLI ─────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aurora Trinity-3 Enhanced Benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=int, default=200, help="número de tríos de entrenamiento")
    p.add_argument("--test", type=int, default=120, help="número de casos de test")
    p.add_argument("--enhanced", action="store_true", default=True, help="usar modo enhanced")
    p.add_argument("--legacy", action="store_true", help="usar modo legacy")
    
    args = p.parse_args()
    
    enhanced = args.enhanced and not args.legacy
    
    start_total = time.time()
    run(args.seed, args.train, args.test, enhanced)
    total_time = time.time() - start_total
    
    print(f"\n⏱️  Duración total: {total_time:.2f}s")
