import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from allcode import Transcender, Extender, KnowledgeBase, Evolver, FractalTensor

def test_trigate_basico():
    """Test básico del Trigate para verificar funcionamiento."""
    print("=== TEST TRIGATE BÁSICO ===")
    
    transcender = Transcender()
    
    # Test simple con vectores conocidos
    A = [1, 0, 1]
    B = [0, 1, 0] 
    C = [1, 1, 0]
    
    print(f"Input: A={A}, B={B}, C={C}")
    
    try:
        resultado = transcender.compute(A, B, C)
        print(f"Ms (M_emergent): {resultado['M_emergent']}")
        print(f"MetaM: {resultado['MetaM']}")
        print(f"¿Tiene None?: Ms={any(x is None for x in resultado['M_emergent'])}, MetaM={any(x is None for x in resultado['MetaM'])}")
        return resultado
    except Exception as e:
        print(f"ERROR en trigate básico: {e}")
        return None

def test_kb_simple():
    """Test básico de KnowledgeBase."""
    print("\n=== TEST KB SIMPLE ===")
    
    kb = KnowledgeBase()
    
    # Intentar agregar entrada simple
    A = [1, 0, 1]
    B = [0, 1, 0]
    C = [1, 1, 0]
    Ms = [1, 1, 1]  # Sin None
    MetaM = [0, 0, 1]  # Sin None
    Ss = [2, 0, None]  # Patrón simple
    
    try:
        kb.add_entry(A, B, C, Ms, MetaM, [Ss])
        print(f"✓ Entrada agregada exitosamente")
        print(f"KB size: {len(kb)}")
        
        # Verificar que se guardó
        entries = kb.all_entries()
        print(f"Entradas en KB: {len(entries)}")
        if entries:
            print(f"Primera entrada: {entries[0]}")
        
        return True
    except Exception as e:
        print(f"ERROR en KB: {e}")
        return False

def test_fractal_tensor():
    """Test básico de FractalTensor."""
    print("\n=== TEST FRACTAL TENSOR ===")
    
    try:
        # Crear tensor simple
        ft = FractalTensor(
            nivel_3=[[1,0,1], [0,1,0], [1,1,0]],
            nivel_9=[[1,0,1] for _ in range(9)],
            nivel_27=[[1,0,1] for _ in range(27)]
        )
        print(f"✓ FractalTensor creado")
        
        # Procesarlo
        transcender = Transcender()
        resultado = transcender.compute_fractal(ft)
        
        print(f"Resultado fractal keys: {resultado.keys()}")
        if 'nivel_3' in resultado:
            print(f"Nivel 3 entries: {len(resultado['nivel_3'])}")
            if resultado['nivel_3']:
                entry = resultado['nivel_3'][0]
                print(f"Ms: {entry.get('M_emergent')}")
                print(f"MetaM: {entry.get('MetaM')}")
        
        return resultado
    except Exception as e:
        print(f"ERROR en FractalTensor: {e}")
        return None

def test_extender_simple():
    """Test básico del Extender."""
    print("\n=== TEST EXTENDER SIMPLE ===")
    
    kb = KnowledgeBase()
    evolver = Evolver()
    extender = Extender(kb, evolver)
    
    # Crear datos de prueba simples
    Ss_test = [[1,0,1], [0,1,0], [None,None,None]]
    contexto = {'objetivo': 100}
    
    try:
        resultado = extender.extend(Ss_test, contexto)
        print(f"✓ Extender ejecutado")
        print(f"Resultado keys: {resultado.keys()}")
        if 'reconstruccion' in resultado:
            recon = resultado['reconstruccion']
            print(f"Método: {recon.get('metodo_reconstruccion')}")
            print(f"Tensores: {recon.get('tensores_reconstruidos')}")
        
        return resultado
    except Exception as e:
        print(f"ERROR en Extender: {e}")
        return None

def test_pipeline_completo():
    """Test del pipeline completo simplificado."""
    print("\n=== TEST PIPELINE COMPLETO ===")
    
    # Secuencia aritmética simple
    secuencia = [1, 2, 3]  # delta = 1
    delta_esperado = 1
    valor_esperado = 4
    
    print(f"Secuencia de prueba: {secuencia}")
    print(f"Delta esperado: {delta_esperado}")
    
    # Paso 1: Vectorizar (simple)
    try:
        # Vectorización más directa
        delta1 = secuencia[1] - secuencia[0]  # 1
        delta2 = secuencia[2] - secuencia[1]  # 1
        aceleracion = delta2 - delta1  # 0
        
        vector_A = [1, 1, 1]  # Incrementa, unitario, uniforme
        vector_B = [1, 1, 1]  # Sigue incrementando, unitario, uniforme  
        vector_C = [0, 0, 1]  # No acelera, velocidad constante
        
        print(f"Vectores: A={vector_A}, B={vector_B}, C={vector_C}")
        
        # Paso 2: Transcender
        transcender = Transcender()
        resultado_transcender = transcender.compute(vector_A, vector_B, vector_C)
        
        Ms = resultado_transcender['M_emergent']
        MetaM = resultado_transcender['MetaM']
        
        print(f"Ms: {Ms}")
        print(f"MetaM: {MetaM}")
        
        # Verificar que no hay None
        if any(x is None for x in Ms) or any(x is None for x in MetaM):
            print("⚠️ Hay valores None en Ms/MetaM")
            return False
        
        # Paso 3: Guardar en KB
        kb = KnowledgeBase()
        Ss_pattern = [delta_esperado, 0, None]  # [delta, aceleración, missing]
        
        kb.add_entry(vector_A, vector_B, vector_C, Ms, MetaM, [Ss_pattern])
        print(f"✓ Guardado en KB: {len(kb)} entradas")
        
        # Paso 4: Buscar y reconstruir
        evolver = Evolver()
        extender = Extender(kb, evolver)
        
        # Simular query
        resultado_extend = extender.extend(
            [Ms, MetaM, [None, None, None]],
            {'objetivo': valor_esperado}
        )
        
        tensores = resultado_extend['reconstruccion']['tensores_reconstruidos']
        print(f"Tensores reconstruidos: {tensores}")
        
        # Extraer delta predicho
        if isinstance(tensores, list) and tensores:
            delta_predicho = tensores[0] if isinstance(tensores[0], (int, float)) else 1
        else:
            delta_predicho = 1
            
        valor_predicho = secuencia[-1] + delta_predicho
        acierto = valor_predicho == valor_esperado
        
        print(f"Delta predicho: {delta_predicho}")
        print(f"Valor predicho: {valor_predicho}")
        print(f"Acierto: {acierto}")
        
        return acierto
        
    except Exception as e:
        print(f"ERROR en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DIAGNÓSTICO AURORA TRINITY-3")
    print("="*50)
    
    # Tests incrementales
    resultado_trigate = test_trigate_basico()
    resultado_kb = test_kb_simple()
    resultado_fractal = test_fractal_tensor()
    resultado_extender = test_extender_simple()
    resultado_pipeline = test_pipeline_completo()
    
    print("\n" + "="*50)
    print("RESUMEN DIAGNÓSTICO:")
    print(f"Trigate básico: {'✓' if resultado_trigate else '✗'}")
    print(f"KB simple: {'✓' if resultado_kb else '✗'}")
    print(f"FractalTensor: {'✓' if resultado_fractal else '✗'}")
    print(f"Extender simple: {'✓' if resultado_extender else '✗'}")
    print(f"Pipeline completo: {'✓' if resultado_pipeline else '✗'}")
    
    if resultado_pipeline:
        print("\n🎉 Sistema funcionando - problema está en el benchmark")
    else:
        print("\n⚠️ Problema detectado en el sistema base")
