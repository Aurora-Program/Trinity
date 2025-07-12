#!/usr/bin/env python3
"""
Test de Diagnóstico - Investigar problema con valores S
"""

from Trinity_Fixed import Transcender

def diagnose():
    print("DIAGNÓSTICO: Investigando problema con valores S")
    print("="*50)
    
    # Crear transcender
    transcender = Transcender()
    
    # Datos de prueba
    InA = [1, 0, 1]
    InB = [0, 1, 0] 
    InC = [1, 1, 0]
    
    print(f"Procesando: InA={InA}, InB={InB}, InC={InC}")
    
    try:
        Ms, Ss, MetaM = transcender.procesar(InA, InB, InC)
        
        print(f"\nResultados obtenidos:")
        print(f"Ms: {Ms}")
        print(f"Ss: {Ss}")
        
        print(f"\nDatos almacenados en last_run_data:")
        run_data = transcender.last_run_data
        print(f"Keys disponibles: {list(run_data.keys())}")
        
        if "intermediate" in run_data:
            intermediate = run_data["intermediate"]
            print(f"Keys en intermediate: {list(intermediate.keys())}")
            
            for key in ["S1", "S2", "S3"]:
                if key in intermediate:
                    print(f"{key}: {intermediate[key]}")
                else:
                    print(f"{key}: NO ENCONTRADO")
        else:
            print("NO HAY DATOS INTERMEDIATE")
            
        # Verificar directamente los Trigates
        print(f"\nVerificación directa de Trigates:")
        print(f"TG1 - A: {transcender._TG1.A}, B: {transcender._TG1.B}, R: {transcender._TG1.R}")
        print(f"TG2 - A: {transcender._TG2.A}, B: {transcender._TG2.B}, R: {transcender._TG2.R}")
        print(f"TG3 - A: {transcender._TG3.A}, B: {transcender._TG3.B}, R: {transcender._TG3.R}")
        
        # Intentar calcular S directamente
        print(f"\nCalculando S directamente:")
        if transcender._TG1.A and transcender._TG1.B and transcender._TG1.R:
            S1_direct = transcender._TG1.sintesis_S()
            print(f"S1 directo: {S1_direct}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
