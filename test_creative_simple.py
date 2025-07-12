#!/usr/bin/env python3
"""
Test simple del motor de inteligencia creativa
"""

from trinity_creative_complete import CreativeReasoningEngine

print("🚀 Iniciando test del motor de inteligencia creativa")

# Crear instancia del motor
creative_engine = CreativeReasoningEngine()

print("✅ Motor creado exitosamente")

# Test de generación de hipótesis
print("\n🎯 Test: Generación de hipótesis creativas")
hypotheses = creative_engine.creative_hypothesis_generation("inteligencia", "philosophy")

print(f"✅ Generadas {len(hypotheses)} hipótesis")

# Test de chat simple
print("\n💬 Test: Chat inteligente")
response = creative_engine.creative_chat_generation("¿Qué es la creatividad?")

print(f"✅ Respuesta generada: {response['response'][:100]}...")

print("\n🎉 TEST COMPLETADO EXITOSAMENTE")
