#!/usr/bin/env python3
"""
EVALUACIÓN DE RENDIMIENTO Y BENCHMARKS PARA TRINITY
==================================================
Mide el rendimiento, escalabilidad y eficiencia de la librería Trinity.
"""

import time
import sys
import os
import statistics
from typing import List, Dict, Any

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Trinity import Trigate, Transcender, KnowledgeBase, Evolver, Extender

class TrinityBenchmark:
    """Clase para ejecutar benchmarks de Trinity"""
    
    def __init__(self):
        self.results = {}
    
    def time_function(self, func, *args, iterations=1000):
        """Mide el tiempo de ejecución de una función"""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_trigate_operations(self):
        """Benchmark de operaciones básicas de Trigate"""
        print("\n🔧 BENCHMARK: Operaciones Trigate")
        print("-" * 50)
        
        trigate = Trigate([1,0,1], [0,1,0], [1,1,0], [0,1,1])
        
        # Test inferir()
        stats = self.time_function(trigate.inferir, iterations=10000)
        print(f"inferir() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['trigate_inferir'] = stats
        
        # Test aprender()
        trigate.R = [1,0,1]  # Establecer R para aprender
        stats = self.time_function(trigate.aprender, iterations=10000)
        print(f"aprender() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['trigate_aprender'] = stats
        
        # Test sintesis_S()
        stats = self.time_function(trigate.sintesis_S, iterations=10000)
        print(f"sintesis_S() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['trigate_sintesis'] = stats
    
    def benchmark_transcender_processing(self):
        """Benchmark de procesamiento en Transcender"""
        print("\n🚀 BENCHMARK: Procesamiento Transcender")
        print("-" * 50)
        
        transcender = Transcender()
        inputs = ([1,0,1], [0,1,0], [1,1,1])
        
        # Test procesar()
        stats = self.time_function(lambda: transcender.procesar(*inputs), iterations=1000)
        print(f"procesar() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['transcender_procesar'] = stats
        
        # Test level1_synthesis()
        stats = self.time_function(lambda: transcender.level1_synthesis(*inputs), iterations=100)
        print(f"level1_synthesis() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['transcender_level1'] = stats
    
    def benchmark_knowledge_base_operations(self):
        """Benchmark de operaciones en KnowledgeBase"""
        print("\n💾 BENCHMARK: Operaciones KnowledgeBase")
        print("-" * 50)
        
        kb = KnowledgeBase()
        
        # Test store_axiom() con diferentes tamaños
        axiom_data = {
            "Ms": [1,0,1],
            "MetaM": [[0,1,1], [1,0,1], [0,0,0], [1,1,0]],
            "Ss": [0,1,0],
            "inputs": {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]}
        }
        
        # Llenar la base de conocimiento
        for i in range(100):
            kb.store_axiom("default", [i%2, (i+1)%2, i%2], 
                          axiom_data["MetaM"], axiom_data["Ss"], axiom_data["inputs"])
        
        # Test get_axiom_by_ms()
        stats = self.time_function(lambda: kb.get_axiom_by_ms("default", [1,0,1]), iterations=10000)
        print(f"get_axiom_by_ms() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['kb_get_axiom'] = stats
        
        # Test get_axioms_in_space()
        stats = self.time_function(lambda: kb.get_axioms_in_space("default"), iterations=1000)
        print(f"get_axioms_in_space() - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['kb_get_space'] = stats
    
    def benchmark_scalability(self):
        """Test de escalabilidad con diferentes tamaños de datos"""
        print("\n📈 BENCHMARK: Escalabilidad")
        print("-" * 50)
        
        sizes = [10, 50, 100, 500, 1000]
        
        for size in sizes:
            kb = KnowledgeBase()
            evolver = Evolver(kb)
            
            # Llenar con datos de prueba
            start_time = time.perf_counter()
            for i in range(size):
                data = {
                    "inputs": {"A": [i%2, (i+1)%2, i%2], "B": [1,0,1], "C": [0,1,0]},
                    "outputs": {
                        "Ms": [i%2, (i+1)%2, i%2],
                        "Ss": [0,1,0],
                        "MetaM": [[0,1,1], [1,0,1], [0,0,0], [1,1,0]]
                    }
                }
                evolver.formalize_axiom(data, "default")
            
            fill_time = time.perf_counter() - start_time
              # Test búsqueda
            start_time = time.perf_counter()
            for i in range(min(100, size)):
                kb.get_axiom_by_ms("default", [i%2, (i+1)%2, i%2])
            search_time = time.perf_counter() - start_time
            
            print(f"Tamaño {size:4d}: Llenado {fill_time:.4f}s, Búsqueda {search_time:.4f}s")
    
    def benchmark_memory_usage(self):
        """Benchmark de uso de memoria (simplificado)"""
        print("\n🧠 BENCHMARK: Uso de Memoria")
        print("-" * 50)
        
        kb = KnowledgeBase()
        transcender = Transcender()
        evolver = Evolver(kb)
        
        # Crear muchos vectores fractales y medir aproximadamente
        start_objects = len(kb.spaces["default"]["axiom_registry"])
        
        for i in range(100):
            fv = transcender.level1_synthesis([i%2, (i+1)%2, i%2], [1,0,1], [0,1,0])
            evolver.formalize_fractal_axiom(fv, {"test": i}, "default")
        
        end_objects = len(kb.spaces["default"]["axiom_registry"])
        print(f"Objetos creados: {end_objects - start_objects}")
        print("Uso de memoria medido de forma aproximada")
    
    def benchmark_full_pipeline(self):
        """Benchmark del pipeline completo"""
        print("\n🔄 BENCHMARK: Pipeline Completo")
        print("-" * 50)
        
        def full_pipeline():
            # Setup
            kb = KnowledgeBase()
            transcender = Transcender()
            evolver = Evolver(kb)
            extender = Extender()
            
            # Síntesis
            Ms, Ss, MetaM = transcender.procesar([1,0,1], [0,1,0], [1,1,1])
            
            # Formalización
            evolver.formalize_axiom(transcender.last_run_data, "default")
            
            # Reconstrucción
            guide_package = evolver.generate_guide_package("default")
            extender.load_guide_package(guide_package)
            reconstructed = extender.reconstruct(Ms)
            
            return reconstructed
        
        stats = self.time_function(full_pipeline, iterations=100)
        print(f"Pipeline completo - Media: {stats['mean']*1000:.4f}ms, Std: {stats['std']*1000:.4f}ms")
        self.results['full_pipeline'] = stats
    
    def generate_report(self):
        """Genera un reporte de todos los benchmarks"""
        print("\n" + "="*60)
        print("📊 REPORTE FINAL DE RENDIMIENTO")
        print("="*60)
        
        # Clasificar resultados por velocidad
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['mean'])
        
        print("\n🏆 OPERACIONES MÁS RÁPIDAS:")
        for name, stats in sorted_results[:3]:
            print(f"  {name}: {stats['mean']*1000:.4f}ms promedio")
        
        print("\n⚠️  OPERACIONES MÁS LENTAS:")
        for name, stats in sorted_results[-3:]:
            print(f"  {name}: {stats['mean']*1000:.4f}ms promedio")
        
        # Calcular estadísticas generales
        all_times = [stats['mean'] for stats in self.results.values()]
        print(f"\n📈 ESTADÍSTICAS GENERALES:")
        print(f"  Tiempo promedio general: {statistics.mean(all_times)*1000:.4f}ms")
        print(f"  Desviación estándar: {statistics.stdev(all_times)*1000:.4f}ms")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        if self.results.get('transcender_level1', {}).get('mean', 0) > 0.01:
            print("  - Considerar optimizar level1_synthesis() para mejor rendimiento")
        if self.results.get('kb_get_space', {}).get('mean', 0) > 0.001:
            print("  - Implementar caché para get_axioms_in_space()")
        
        print("\n✅ Evaluación de rendimiento completada")
    
    def run_all_benchmarks(self):
        """Ejecuta todos los benchmarks"""
        print("🚀 INICIANDO EVALUACIÓN COMPLETA DE RENDIMIENTO")
        print("="*60)
        
        self.benchmark_trigate_operations()
        self.benchmark_transcender_processing()
        self.benchmark_knowledge_base_operations()
        self.benchmark_scalability()
        self.benchmark_full_pipeline()
        
        # Solo ejecutar benchmark de memoria si memory_profiler está disponible
        try:
            self.benchmark_memory_usage()
        except Exception as e:
            print(f"\n⚠️  Benchmark de memoria saltado: {e}")
        
        self.generate_report()


if __name__ == "__main__":
    benchmark = TrinityBenchmark()
    benchmark.run_all_benchmarks()
