#!/usr/bin/env python3
"""
Test de verificación de importación
"""

from Trinity_Fixed import Transcender
import inspect
import os

def check_import():
    print("VERIFICANDO IMPORTACIÓN")
    print("="*40)
    
    # Verificar la fuente del archivo
    transcender = Transcender()
    source_file = inspect.getfile(transcender.__class__)
    print(f"Archivo fuente de Transcender: {source_file}")
    
    # Verificar el método procesar
    procesar_method = getattr(transcender, 'procesar')
    procesar_source = inspect.getsource(procesar_method)
    
    print(f"\nCódigo del método procesar:")
    print("-" * 40)
    print(procesar_source[:500] + "..." if len(procesar_source) > 500 else procesar_source)
    
    # Test rápido
    print(f"\nTest rápido:")
    Ms, Ss, MetaM = transcender.procesar([1,0,1], [0,1,0], [1,1,0])
    print(f"Ms: {Ms}")
    
    print(f"\nKeys en last_run_data:")
    print(f"Keys: {list(transcender.last_run_data.keys())}")
    
    if "intermediate" in transcender.last_run_data:
        print("✅ VERSIÓN CORREGIDA")
        intermediate = transcender.last_run_data["intermediate"]
        print(f"S1: {intermediate.get('S1', 'NOT_FOUND')}")
        print(f"S2: {intermediate.get('S2', 'NOT_FOUND')}")
        print(f"S3: {intermediate.get('S3', 'NOT_FOUND')}")
    elif "logic" in transcender.last_run_data:
        print("❌ VERSIÓN ANTIGUA")
    else:
        print("⚠️  VERSIÓN DESCONOCIDA")

if __name__ == "__main__":
    check_import()
