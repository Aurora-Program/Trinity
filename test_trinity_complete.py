#!/usr/bin/env python3
"""
TEST SUITE COMPLETA PARA TRINITY
=====================================
Evalúa todas las funcionalidades de la librería Trinity de forma sistemática.
"""

import unittest
import time
import sys
import os

# Añadir el directorio actual al path para importar Trinity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Trinity import Trigate, Transcender, KnowledgeBase, Evolver, Extender

class TestTrigate(unittest.TestCase):
    """Tests para la clase Trigate"""
    
    def setUp(self):
        """Configuración para cada test"""
        self.trigate = Trigate()
    
    def test_init(self):
        """Test de inicialización"""
        t = Trigate([1,0,1], [0,1,0], [1,1,0], [0,1,1])
        self.assertEqual(t.A, [1,0,1])
        self.assertEqual(t.B, [0,1,0])
        self.assertEqual(t.R, [1,1,0])
        self.assertEqual(t.M, [0,1,1])
    
    def test_xor_basic(self):
        """Test de operación XOR básica"""
        self.assertEqual(self.trigate._xor(0, 0), 0)
        self.assertEqual(self.trigate._xor(0, 1), 1)
        self.assertEqual(self.trigate._xor(1, 0), 1)
        self.assertEqual(self.trigate._xor(1, 1), 0)
    
    def test_xor_with_none(self):
        """Test de XOR con valores None"""
        self.assertIsNone(self.trigate._xor(None, 0))
        self.assertIsNone(self.trigate._xor(1, None))
        self.assertIsNone(self.trigate._xor(None, None))
    
    def test_xnor_basic(self):
        """Test de operación XNOR básica"""
        self.assertEqual(self.trigate._xnor(0, 0), 1)
        self.assertEqual(self.trigate._xnor(0, 1), 0)
        self.assertEqual(self.trigate._xnor(1, 0), 0)
        self.assertEqual(self.trigate._xnor(1, 1), 1)
    
    def test_validate_valid_input(self):
        """Test de validación con entrada válida"""
        try:
            self.trigate._validate([0, 1, None], "test")
            self.trigate._validate([1, 1, 1], "test")
        except ValueError:
            self.fail("_validate() lanzó ValueError con entrada válida")
    
    def test_validate_invalid_input(self):
        """Test de validación con entradas inválidas"""
        with self.assertRaises(ValueError):
            self.trigate._validate([0, 1], "test")  # Muy corto
        
        with self.assertRaises(ValueError):
            self.trigate._validate([0, 1, 2], "test")  # Valor inválido
        
        with self.assertRaises(ValueError):
            self.trigate._validate("invalid", "test")  # No es lista
    
    def test_inferir(self):
        """Test de inferencia"""
        self.trigate.A = [1, 0, 1]
        self.trigate.B = [0, 1, 0]
        self.trigate.M = [0, 1, 1]
        
        result = self.trigate.inferir()
        self.assertEqual(len(result), 3)
        self.assertIn(result[0], [0, 1, None])
    
    def test_aprender(self):
        """Test de aprendizaje"""
        self.trigate.A = [1, 0, 1]
        self.trigate.B = [0, 1, 0]
        self.trigate.R = [0, 1, 1]
        
        result = self.trigate.aprender()
        self.assertEqual(len(result), 3)
        self.assertTrue(all(x in [0, 1, None] for x in result))
    
    def test_sintesis_s(self):
        """Test de síntesis S"""
        self.trigate.A = [1, 0, 1]
        self.trigate.B = [0, 1, 0]
        self.trigate.R = [0, 1, 1]
        
        result = self.trigate.sintesis_S()
        self.assertEqual(len(result), 3)


class TestTranscender(unittest.TestCase):
    """Tests para la clase Transcender"""
    
    def setUp(self):
        self.transcender = Transcender()
    
    def test_init(self):
        """Test de inicialización"""
        self.assertIsInstance(self.transcender._TG1, Trigate)
        self.assertIsInstance(self.transcender._TG2, Trigate)
        self.assertIsInstance(self.transcender._TG3, Trigate)
        self.assertIsInstance(self.transcender._TG_S, Trigate)
    
    def test_procesar(self):
        """Test de procesamiento básico"""
        InA = [1, 0, 1]
        InB = [0, 1, 0]
        InC = [1, 1, 1]
        
        Ms, Ss, MetaM = self.transcender.procesar(InA, InB, InC)
        
        # Verificar que se devuelven listas del tamaño correcto
        self.assertEqual(len(Ms), 3)
        self.assertEqual(len(Ss), 3)
        self.assertEqual(len(MetaM), 4)  # [M1, M2, M3, Ms]
        
        # Verificar que last_run_data se guarda correctamente
        self.assertIn("inputs", self.transcender.last_run_data)
        self.assertIn("outputs", self.transcender.last_run_data)
    
    def test_level1_synthesis(self):
        """Test de síntesis fractal nivel 1"""
        A = [1, 0, 1]
        B = [0, 1, 0]
        C = [1, 1, 1]
        
        result = self.transcender.level1_synthesis(A, B, C)
        
        self.assertIn("layer1", result)
        self.assertIn("layer2", result)
        self.assertIn("layer3", result)
        
        # Verificar dimensiones correctas
        self.assertEqual(len(result["layer1"]), 3)
        self.assertEqual(len(result["layer2"]), 9)
        self.assertEqual(len(result["layer3"]), 27)


class TestKnowledgeBase(unittest.TestCase):
    """Tests para la clase KnowledgeBase"""
    
    def setUp(self):
        self.kb = KnowledgeBase()
    
    def test_init(self):
        """Test de inicialización"""
        self.assertIn("default", self.kb.spaces)
        self.assertEqual(self.kb.spaces["default"]["description"], "Espacio lógico predeterminado")
    
    def test_create_space(self):
        """Test de creación de espacios"""
        result = self.kb.create_space("test_space", "Espacio de prueba")
        self.assertTrue(result)
        self.assertIn("test_space", self.kb.spaces)
        
        # Intentar crear el mismo espacio otra vez
        result2 = self.kb.create_space("test_space", "Otra descripción")
        self.assertFalse(result2)
    
    def test_delete_space(self):
        """Test de eliminación de espacios"""
        self.kb.create_space("temp_space", "Temporal")
        result = self.kb.delete_space("temp_space")
        self.assertTrue(result)
        self.assertNotIn("temp_space", self.kb.spaces)
        
        # No se puede eliminar el espacio default
        result2 = self.kb.delete_space("default")
        self.assertFalse(result2)
    
    def test_store_axiom(self):
        """Test de almacenamiento de axiomas"""
        Ms = [1, 0, 1]
        MetaM = [[0,1,1], [1,0,1], [0,0,0], [1,1,0]]
        Ss = [0, 1, 0]
        inputs = {"A": [1,0,1], "B": [0,1,0], "C": [1,1,1]}
        
        result = self.kb.store_axiom("default", Ms, MetaM, Ss, inputs)
        self.assertTrue(result)
        
        # Verificar que se almacenó correctamente
        stored = self.kb.get_axiom_by_ms("default", Ms)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["MetaM"], MetaM)


class TestEvolver(unittest.TestCase):
    """Tests para la clase Evolver"""
    
    def setUp(self):
        self.kb = KnowledgeBase()
        self.evolver = Evolver(self.kb)
    
    def test_init(self):
        """Test de inicialización"""
        self.assertEqual(self.evolver.kb, self.kb)
        self.assertIsNone(self.evolver.relational_map)
    
    def test_classify_null(self):
        """Test de clasificación de NULL"""
        # NULL en capa abstracta
        result = self.evolver.classify_null([1,1,1], (0, 1))
        self.assertEqual(result, 'N_u')
        
        # NULL con contexto positivo
        result2 = self.evolver.classify_null([1,0,1], (1, 1, 2))
        self.assertEqual(result2, 'N_i')
    
    def test_detect_fractal_pattern(self):
        """Test de detección de patrones fractales"""
        # Patrón unitario
        self.assertEqual(self.evolver.detect_fractal_pattern([1,1,1]), "unitary")
        
        # Patrón uniforme
        self.assertEqual(self.evolver.detect_fractal_pattern([0,0,0]), "uniform")
        
        # Patrón complejo
        self.assertEqual(self.evolver.detect_fractal_pattern([1,0,1]), "complex")


class TestExtender(unittest.TestCase):
    """Tests para la clase Extender"""
    
    def setUp(self):
        self.extender = Extender()
    
    def test_init(self):
        """Test de inicialización"""
        self.assertIsNone(self.extender.guide_package)
        self.assertIsInstance(self.extender.transcender, Transcender)
    
    def test_load_guide_package(self):
        """Test de carga de paquete de guías"""
        package = {"axiom_registry": {}, "space": "test"}
        self.extender.load_guide_package(package)
        self.assertEqual(self.extender.guide_package, package)


class TestIntegration(unittest.TestCase):
    """Tests de integración del sistema completo"""
    
    def setUp(self):
        self.kb = KnowledgeBase()
        self.transcender = Transcender()
        self.evolver = Evolver(self.kb)
        self.extender = Extender()
    
    def test_full_pipeline(self):
        """Test del pipeline completo: síntesis → formalización → reconstrucción"""
        # 1. Síntesis
        InA = [1, 0, 1]
        InB = [0, 1, 0]
        InC = [1, 1, 1]
        
        Ms, Ss, MetaM = self.transcender.procesar(InA, InB, InC)
        
        # 2. Formalización
        self.evolver.formalize_axiom(self.transcender.last_run_data, "default")
        
        # 3. Reconstrucción
        guide_package = self.evolver.generate_guide_package("default")
        self.extender.load_guide_package(guide_package)
        
        reconstructed = self.extender.reconstruct(Ms)
        self.assertIsNotNone(reconstructed)
        
        # Verificar que se reconstruyeron las entradas originales
        self.assertEqual(reconstructed["InA"], InA)
        self.assertEqual(reconstructed["InB"], InB)
        self.assertEqual(reconstructed["InC"], InC)


class TestPerformance(unittest.TestCase):
    """Tests de rendimiento"""
    
    def test_trigate_performance(self):
        """Test de rendimiento de Trigate"""
        trigate = Trigate([1,0,1], [0,1,0], None, [0,1,1])
        
        start_time = time.time()
        for _ in range(1000):
            trigate.inferir()
        end_time = time.time()
        
        duration = end_time - start_time
        self.assertLess(duration, 1.0, "Trigate.inferir() demasiado lento")
        print(f"1000 inferencias en {duration:.4f} segundos")
    
    def test_transcender_performance(self):
        """Test de rendimiento de Transcender"""
        transcender = Transcender()
        
        start_time = time.time()
        for _ in range(100):
            transcender.procesar([1,0,1], [0,1,0], [1,1,1])
        end_time = time.time()
        
        duration = end_time - start_time
        self.assertLess(duration, 5.0, "Transcender.procesar() demasiado lento")
        print(f"100 procesamientos en {duration:.4f} segundos")


if __name__ == "__main__":
    print("="*60)
    print("EJECUTANDO SUITE COMPLETA DE TESTS PARA TRINITY")
    print("="*60)
    
    # Ejecutar todos los tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print("TESTS COMPLETADOS")
    print("="*60)
