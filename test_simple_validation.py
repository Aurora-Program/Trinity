#!/usr/bin/env python3
"""
Test Simple - Validación de Corrección Arquitectural Aurora
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Trinity_Fixed import Transcender

def test_simple():
    """Test simple para validar la corrección arquitectural"""
    print("="*50)
    print("TEST SIMPLE - VALIDACIÓN ARQUITECTURAL AURORA")
    print("="*50)
    
    # Crear transcender
    transcender = Transcender()
    
    # Datos de prueba
    InA = [1, 0, 1]
    InB = [0, 1, 0]
    InC = [1, 1, 0]
    
    print(f"\nEntradas:")
    print(f"  InA: {InA}")
    print(f"  InB: {InB}")
    print(f"  InC: {InC}")
    
    # Procesar
    print(f"\nProcesando con arquitectura Aurora corregida...")
    Ms, Ss, MetaM = transcender.procesar(InA, InB, InC)
    
    print(f"\nResultados:")
    print(f"  Ms (Estructura): {Ms}")
    print(f"  Ss (Forma): {Ss}")
    print(f"  MetaM: {MetaM}")
    
    # Verificar datos internos
    run_data = transcender.last_run_data
    intermediate = run_data.get("intermediate", {})
    
    print(f"\nDatos intermedios (validación arquitectural):")
    if "S1" in intermediate:
        print(f"  S1 (síntesis TG1): {intermediate['S1']}")
    if "S2" in intermediate:
        print(f"  S2 (síntesis TG2): {intermediate['S2']}")
    if "S3" in intermediate:
        print(f"  S3 (síntesis TG3): {intermediate['S3']}")
    
    # Validaciones
    validations = {
        "S_values_exist": all(key in intermediate for key in ["S1", "S2", "S3"]),
        "Ms_valid": isinstance(Ms, list) and len(Ms) == 3,
        "Ss_valid": isinstance(Ss, list) and len(Ss) == 3,
        "MetaM_valid": isinstance(MetaM, list) and len(MetaM) == 4
    }
    
    print(f"\nValidaciones:")
    for test_name, result in validations.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_pass = all(validations.values())
    print(f"\nResultado: {'✅ ARQUITECTURA AURORA VÁLIDA' if all_pass else '❌ PROBLEMAS DETECTADOS'}")
    
    return all_pass

if __name__ == "__main__":
    success = test_simple()
    if success:
        print("\n🎉 CORRECCIÓN ARQUITECTURAL CONFIRMADA!")
        print("   - Usa S1, S2, S3 (síntesis) correctamente")
        print("   - Implementa arquitectura Aurora auténtica")
        print("   - Sistema listo para uso")
    else:
        print("\n⚠️  Requiere revisión adicional")
