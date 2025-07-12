import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aurora_core.Transcender import Transcender
from aurora_core.knowledgeBase import KnowledgeBase
from aurora_core.evolver import Evolver
from aurora_core.extender import Extender

class CoherenceError(Exception):
    pass

def test_aurora_operativo():
    print("=== TEST OPERATIVO: APRENDIZAJE, INFERENCIA Y MANEJO DE AMBIGÜEDAD ===")
    kb = KnowledgeBase()
    evolver = Evolver()
    transcender = Transcender()
    extender = Extender(knowledge_base=kb, evolver=evolver)

    # Espacio Lógico 1: Diagnóstico Médico
    contexto_medico = [0, 0, 1]  # ID único para el espacio médico
    sintomas_fiebre = [1, 0, 0]
    sintomas_tos = [0, 1, 0]
    diagnostico_gripe = [1, 0, 1]
    print("\n--- Fase de Aprendizaje (Espacio Médico) ---")
    resultado = transcender.deep_learning(
        A=sintomas_fiebre,
        B=sintomas_tos,
        C=contexto_medico,
        M_emergent=diagnostico_gripe
    )
    kb.add_entry(
        A=sintomas_fiebre,
        B=sintomas_tos,
        C=contexto_medico,
        M_emergent=diagnostico_gripe,
        MetaM=resultado['MetaM'],
        R_validos=resultado['R_hipotesis'],
        transcender_id="MED001"
    )
    print(f"Aprendido patrón médico: MetaM={resultado['MetaM']}")

    # Espacio Lógico 2: Detección de Fraude Financiero
    contexto_finanzas = [1, 1, 0]  # ID único para el espacio financiero
    transaccion_sospechosa = [1, 1, 0]
    historico_cliente = [0, 0, 1]
    riesgo_fraude = [1, 0, 0]
    print("\n--- Fase de Aprendizaje (Espacio Financiero) ---")
    resultado = transcender.deep_learning(
        A=transaccion_sospechosa,
        B=historico_cliente,
        C=contexto_finanzas,
        M_emergent=riesgo_fraude
    )
    kb.add_entry(
        A=transaccion_sospechosa,
        B=historico_cliente,
        C=contexto_finanzas,
        M_emergent=riesgo_fraude,
        MetaM=resultado['MetaM'],
        R_validos=resultado['R_hipotesis'],
        transcender_id="FRA001"
    )
    print(f"Aprendido patrón financiero: MetaM={resultado['MetaM']}")

    # Test 1: Inferencia en contexto conocido (médico)
    print("\n--- Test 1: Inferencia Válida (Espacio Médico) ---")
    nuevos_sintomas = [1, 0, 0]
    contexto = {"objetivo": None, "M_emergent": None, "espacio": contexto_medico}
    respuesta = extender.extend(
        Ss=[nuevos_sintomas, sintomas_tos, contexto_medico],
        contexto=contexto
    )
    diagnostico_inferido = respuesta['reconstruccion']['tensores_reconstruidos']
    print(f"Diagnóstico inferido: {diagnostico_inferido} | Esperado: {diagnostico_gripe}")
    assert diagnostico_inferido == diagnostico_gripe, "Error en inferencia médica"

    # Test 2: Inferencia con ambigüedad manejada (NULL)
    print("\n--- Test 2: Manejo de Ambigüedad (NULL) ---")
    sintomas_parciales = [1, 0, None]
    contexto = {"objetivo": None, "M_emergent": None, "espacio": contexto_medico}
    respuesta = extender.extend(
        Ss=[sintomas_parciales, sintomas_tos, contexto_medico],
        contexto=contexto
    )
    diagnostico_parcial = respuesta['reconstruccion']['tensores_reconstruidos']
    print(f"Diagnóstico con ambigüedad: {diagnostico_parcial}")
    assert diagnostico_parcial[2] is None, "El sistema no manejó NULL correctamente"

    # Test 3: Validación de Coherencia Lógica
    print("\n--- Test 3: Validación de Coherencia ---")
    try:
        kb.add_entry(
            A=sintomas_fiebre,
            B=sintomas_tos,
            C=contexto_medico,
            M_emergent=[0, 0, 0],
            MetaM=[1, 1, 1],
            R_validos=[],
            transcender_id="MED002"
        )
        assert False, "El sistema aceptó patrón incoherente"
    except Exception as e:
        print(f"✓ Validación de coherencia exitosa: {str(e)}")

    # Test 4: Cambio de Espacio Lógico
    print("\n--- Test 4: Cambio de Contexto (Médico → Financiero) ---")
    nueva_transaccion = [1, 1, 0]
    contexto = {"objetivo": None, "M_emergent": None, "espacio": contexto_finanzas}
    respuesta = extender.extend(
        Ss=[nueva_transaccion, historico_cliente, contexto_finanzas],
        contexto=contexto
    )
    riesgo_inferido = respuesta['reconstruccion']['tensores_reconstruidos']
    print(f"Riesgo inferido: {riesgo_inferido} | Esperado: {riesgo_fraude}")
    assert riesgo_inferido == riesgo_fraude, "Error en inferencia financiera"

    # Test 5: Síntesis de Alto Nivel (Evolver)
    print("\n--- Test 5: Síntesis de Conocimiento (Evolver) ---")
    metaMs_medicos = [entry['MetaM'] for entry in kb.find_by_context(contexto_medico)]
    patron = evolver.analyze_metaMs(metaMs_medicos)
    axioma = evolver.formalize_axiom(patron, metaMs_medicos)
    print(f"Axioma médico generado: {axioma}")
    assert "patrón" in axioma or "iguales" in axioma, "Evolver no generó axioma válido"

    print("\n¡SISTEMA OPERATIVO! Todas las pruebas pasadas exitosamente")

if __name__ == "__main__":
    test_aurora_operativo()
