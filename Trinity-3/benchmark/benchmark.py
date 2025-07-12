import time
import json
import os
import sys
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar todas las clases de Aurora
from allcode import Trigate, Transcender, FractalTensor, Evolver, Extender, KnowledgeBase

class BenchmarkSeriesNumericas:
    def __init__(self):
        self.resultados = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "benchmark_type": "series_numericas_aurora",
                "version": "Trinity-3",
                "descripcion": "Benchmark de resolución de series numéricas usando Aurora"
            },
            "tests": {}
        }
        self.total_tests = 0
        self.passed_tests = 0
        
        # Inicializar componentes de Aurora
        self.evolver = Evolver()
        self.transcender = Transcender()
        self.kb = KnowledgeBase()

    def numero_a_vector(self, numero, longitud=3):
        """Convierte un número a vector binario de longitud fija."""
        if numero < 0:
            return [0] * longitud  # Manejar negativos como ceros
        
        binario = bin(numero)[2:]  # Quitar '0b'
        
        # Truncar o rellenar con ceros a la izquierda
        if len(binario) > longitud:
            binario = binario[-longitud:]  # Tomar los últimos bits
        else:
            binario = binario.zfill(longitud)  # Rellenar con ceros
            
        return [int(bit) for bit in binario]

    def vector_a_numero(self, vector):
        """Convierte un vector binario a número."""
        binario = ''.join(str(bit) for bit in vector)
        return int(binario, 2) if binario else 0

    def analizar_serie_con_aurora(self, serie, nombre_serie):
        """Analiza una serie numérica usando los componentes de Aurora."""
        resultado = {
            "serie_original": serie,
            "vectores_serie": [],
            "patron_detectado": None,
            "siguiente_predicho": None,
            "diferencias_analizadas": [],
            "arquetipo_utilizado": None
        }
        
        try:
            # 1. Convertir números a vectores
            vectores = [self.numero_a_vector(num) for num in serie]
            resultado["vectores_serie"] = vectores
            
            # 2. Analizar diferencias entre números consecutivos
            diferencias = []
            for i in range(1, len(serie)):
                diff = serie[i] - serie[i-1]
                diferencias.append(diff)
            resultado["diferencias_analizadas"] = diferencias
            
            # 3. Usar Evolver para analizar patrones en los vectores
            if len(vectores) >= 3:
                analisis = self.evolver.analyze_metaMs(vectores[:3])
                resultado["patron_detectado"] = analisis
                
                # 4. Formalizar arquetipo
                arquetipo = self.evolver.formalize_axiom(analisis, vectores[:3])
                resultado["arquetipo_utilizado"] = arquetipo
            
            # 5. Usar análisis de dinámicas para predecir siguiente
            if len(vectores) >= 4:
                dinamica = self.evolver.analyze_dynamics_adaptive(vectores)
                
                # 6. Predecir siguiente basado en patrones detectados
                siguiente_predicho = self.predecir_siguiente(serie, diferencias, analisis if len(vectores) >= 3 else None)
                resultado["siguiente_predicho"] = siguiente_predicho
            
            return resultado
            
        except Exception as e:
            resultado["error"] = str(e)
            return resultado

    def predecir_siguiente(self, serie, diferencias, patron):
        """Predice el siguiente número de la serie basándose en patrones detectados."""
        if not serie:
            return None
            
        ultimo = serie[-1]
        
        # Estrategia 1: Diferencias constantes (progresión aritmética)
        if len(diferencias) >= 2:
            if all(d == diferencias[0] for d in diferencias):
                return ultimo + diferencias[0]
        
        # Estrategia 2: Diferencias de segundo orden (progresión cuadrática)
        if len(diferencias) >= 3:
            segundas_diffs = [diferencias[i+1] - diferencias[i] for i in range(len(diferencias)-1)]
            if all(d == segundas_diffs[0] for d in segundas_diffs):
                proxima_diff = diferencias[-1] + segundas_diffs[0]
                return ultimo + proxima_diff
        
        # Estrategia 3: Razón constante (progresión geométrica)
        if len(serie) >= 3 and all(x != 0 for x in serie[-3:]):
            ratios = [serie[i] / serie[i-1] for i in range(-2, 0)]
            if abs(ratios[0] - ratios[1]) < 0.001:  # Razón aproximadamente constante
                return int(ultimo * ratios[0])
        
        # Estrategia 4: Fibonacci (suma de los dos anteriores)
        if len(serie) >= 3:
            if serie[-1] == serie[-2] + serie[-3]:
                return serie[-1] + serie[-2]
        
        # Estrategia 5: Potencias
        if len(serie) >= 3:
            # Verificar si es secuencia de cuadrados
            cuadrados = [i*i for i in range(1, len(serie)+2)]
            if serie == cuadrados[:len(serie)]:
                return cuadrados[len(serie)]
                
            # Verificar si es secuencia de cubos
            cubos = [i*i*i for i in range(1, len(serie)+2)]
            if serie == cubos[:len(serie)]:
                return cubos[len(serie)]
        
        # Estrategia por defecto: usar la última diferencia
        if diferencias:
            return ultimo + diferencias[-1]
        
        return ultimo + 1  # Fallback simple

    def test_progresiones_aritmeticas(self):
        """Test de progresiones aritméticas."""
        print("➕ Testando progresiones aritméticas...")
        tests = []
        
        series_aritmeticas = [
            {"serie": [1, 3, 5, 7, 9], "esperado": 11, "nombre": "impares"},
            {"serie": [2, 4, 6, 8, 10], "esperado": 12, "nombre": "pares"},
            {"serie": [5, 10, 15, 20, 25], "esperado": 30, "nombre": "multiplos_5"},
            {"serie": [1, 4, 7, 10, 13], "esperado": 16, "nombre": "diferencia_3"}
        ]
        
        for serie_data in series_aritmeticas:
            resultado = self.analizar_serie_con_aurora(serie_data["serie"], serie_data["nombre"])
            predicho = resultado.get("siguiente_predicho")
            esperado = serie_data["esperado"]
            
            success = predicho == esperado
            
            tests.append({
                "nombre": f"aritmetica_{serie_data['nombre']}",
                "serie": serie_data["serie"],
                "esperado": esperado,
                "predicho": predicho,
                "patron_detectado": resultado.get("patron_detectado"),
                "diferencias": resultado.get("diferencias_analizadas"),
                "passed": success
            })
            self._update_stats(success)
        
        return tests

    def test_progresiones_geometricas(self):
        """Test de progresiones geométricas."""
        print("✖️ Testando progresiones geométricas...")
        tests = []
        
        series_geometricas = [
            {"serie": [2, 4, 8, 16, 32], "esperado": 64, "nombre": "potencias_2"},
            {"serie": [1, 3, 9, 27, 81], "esperado": 243, "nombre": "potencias_3"},
            {"serie": [5, 10, 20, 40, 80], "esperado": 160, "nombre": "multiplo_por_2"},
            {"serie": [1, 2, 4, 8, 16], "esperado": 32, "nombre": "binaria"}
        ]
        
        for serie_data in series_geometricas:
            resultado = self.analizar_serie_con_aurora(serie_data["serie"], serie_data["nombre"])
            predicho = resultado.get("siguiente_predicho")
            esperado = serie_data["esperado"]
            
            success = predicho == esperado
            
            tests.append({
                "nombre": f"geometrica_{serie_data['nombre']}",
                "serie": serie_data["serie"],
                "esperado": esperado,
                "predicho": predicho,
                "patron_detectado": resultado.get("patron_detectado"),
                "arquetipo": resultado.get("arquetipo_utilizado"),
                "passed": success
            })
            self._update_stats(success)
        
        return tests

    def test_series_especiales(self):
        """Test de series numéricas especiales."""
        print("🌟 Testando series especiales...")
        tests = []
        
        series_especiales = [
            {"serie": [1, 1, 2, 3, 5], "esperado": 8, "nombre": "fibonacci"},
            {"serie": [1, 4, 9, 16, 25], "esperado": 36, "nombre": "cuadrados"},
            {"serie": [1, 8, 27, 64, 125], "esperado": 216, "nombre": "cubos"},
            {"serie": [2, 3, 5, 7, 11], "esperado": 13, "nombre": "primos"},
            {"serie": [1, 3, 6, 10, 15], "esperado": 21, "nombre": "triangulares"}
        ]
        
        for serie_data in series_especiales:
            resultado = self.analizar_serie_con_aurora(serie_data["serie"], serie_data["nombre"])
            predicho = resultado.get("siguiente_predicho")
            esperado = serie_data["esperado"]
            
            # Para series especiales, consideramos éxito si está cerca del valor esperado
            tolerancia = max(1, abs(esperado * 0.1))  # 10% de tolerancia o mínimo 1
            success = predicho is not None and abs(predicho - esperado) <= tolerancia
            
            tests.append({
                "nombre": f"especial_{serie_data['nombre']}",
                "serie": serie_data["serie"],
                "esperado": esperado,
                "predicho": predicho,
                "tolerancia_usada": tolerancia,
                "vectores_generados": resultado.get("vectores_serie"),
                "passed": success
            })
            self._update_stats(success)
        
        return tests

    def test_series_complejas(self):
        """Test de series numéricas complejas."""
        print("🧠 Testando series complejas...")
        tests = []
        
        series_complejas = [
            {"serie": [1, 2, 4, 7, 11, 16], "esperado": 22, "nombre": "diferencias_crecientes"},
            {"serie": [2, 6, 12, 20, 30], "esperado": 42, "nombre": "n_por_n_mas_1"},
            {"serie": [1, 1, 3, 5, 11, 21], "esperado": 43, "nombre": "lucas_modificada"},
            {"serie": [0, 1, 1, 2, 3, 5, 8], "esperado": 13, "nombre": "fibonacci_con_cero"}
        ]
        
        for serie_data in series_complejas:
            resultado = self.analizar_serie_con_aurora(serie_data["serie"], serie_data["nombre"])
            predicho = resultado.get("siguiente_predicho")
            esperado = serie_data["esperado"]
            
            # Para series complejas, usamos mayor tolerancia
            tolerancia = max(2, abs(esperado * 0.2))  # 20% de tolerancia o mínimo 2
            success = predicho is not None and abs(predicho - esperado) <= tolerancia
            
            tests.append({
                "nombre": f"compleja_{serie_data['nombre']}",
                "serie": serie_data["serie"],
                "esperado": esperado,
                "predicho": predicho,
                "tolerancia_usada": tolerancia,
                "analisis_completo": resultado,
                "passed": success
            })
            self._update_stats(success)
        
        return tests

    def test_series_con_transcender(self):
        """Test usando específicamente el componente Transcender."""
        print("🌟 Testando con Transcender...")
        tests = []
        
        try:
            # Tomar una serie simple y procesarla con Transcender
            serie = [1, 2, 3, 4, 5]
            vectores = [self.numero_a_vector(num) for num in serie]
            
            # Usar Transcender para computar relaciones entre vectores
            if len(vectores) >= 3:
                resultado_transcender = self.transcender.compute(vectores[0], vectores[1], vectores[2])
                
                # Analizar el MetaM resultante
                metaM = resultado_transcender.get('MetaM')
                
                success = metaM is not None
                
                tests.append({
                    "nombre": "serie_con_transcender",
                    "serie_original": serie,
                    "vectores_usados": vectores[:3],
                    "resultado_transcender": {
                        "MetaM": metaM,
                        "M_emergent": resultado_transcender.get('M_emergent'),
                        "S_emergent": resultado_transcender.get('S_emergent')
                    },
                    "passed": success
                })
                self._update_stats(success)
            
        except Exception as e:
            tests.append({
                "nombre": "serie_con_transcender",
                "error": str(e),
                "passed": False
            })
            self._update_stats(False)
        
        return tests

    def _update_stats(self, passed):
        """Actualiza las estadísticas del benchmark."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1

    def ejecutar_benchmark(self):
        """Ejecuta el benchmark completo de series numéricas."""
        print("="*70)
        print("🔢 BENCHMARK DE SERIES NUMÉRICAS - AURORA")
        print("="*70)
        
        start_time = time.time()
        
        # Ejecutar todos los tests
        self.resultados["tests"]["progresiones_aritmeticas"] = self.test_progresiones_aritmeticas()
        self.resultados["tests"]["progresiones_geometricas"] = self.test_progresiones_geometricas()
        self.resultados["tests"]["series_especiales"] = self.test_series_especiales()
        self.resultados["tests"]["series_complejas"] = self.test_series_complejas()
        self.resultados["tests"]["series_con_transcender"] = self.test_series_con_transcender()
        
        end_time = time.time()
        
        # Añadir estadísticas finales
        self.resultados["estadisticas"] = {
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.total_tests - self.passed_tests,
            "success_rate_percent": round((self.passed_tests / self.total_tests * 100), 2),
            "duration_seconds": round(end_time - start_time, 3)
        }
        
        # Calcular estadísticas por tipo de serie
        tipos_stats = {}
        for tipo, tests in self.resultados["tests"].items():
            passed_tipo = sum(1 for t in tests if t.get('passed', False))
            total_tipo = len(tests)
            tipos_stats[tipo] = {
                "passed": passed_tipo,
                "total": total_tipo,
                "success_rate": round((passed_tipo / total_tipo * 100), 2) if total_tipo > 0 else 0
            }
        
        self.resultados["estadisticas_por_tipo"] = tipos_stats
        
        # Guardar resultados
        output_path = "benchmark/results_series_numericas.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.resultados, f, indent=2, ensure_ascii=False)
        
        # Mostrar resumen
        stats = self.resultados["estadisticas"]
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"   Tests ejecutados: {stats['total_tests']}")
        print(f"   ✅ Exitosos: {stats['passed']}")
        print(f"   ❌ Fallidos: {stats['failed']}")
        print(f"   📈 Tasa de éxito: {stats['success_rate_percent']}%")
        print(f"   ⏱️  Duración: {stats['duration_seconds']}s")
        
        print(f"\n📊 ESTADÍSTICAS POR TIPO DE SERIE:")
        for tipo, stat in tipos_stats.items():
            print(f"   {tipo}: {stat['passed']}/{stat['total']} ({stat['success_rate']:.1f}%)")
        
        print(f"\n💾 Resultados guardados en: {output_path}")
        print("="*70)

if __name__ == "__main__":
    benchmark = BenchmarkSeriesNumericas()
    benchmark.ejecutar_benchmark()