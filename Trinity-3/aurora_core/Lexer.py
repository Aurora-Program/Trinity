# aurora_core/Lexer.py

import re

class Lexer:
    def __init__(self):
        # Reglas simples para sufijos comunes en español (verbos, sustantivos, adjetivos)
        # Esto se puede expandir enormemente.
        self.reglas_morfemas = {
            'verbos': {
                'ar': ['o', 'as', 'a', 'amos', 'áis', 'an', 'é', 'aste', 'ó', 'amos', 'asteis', 'aron', 'aba', 'abas', 'ábamos', 'abais', 'aban', 'aré', 'arás', 'ará', 'aremos', 'aréis', 'arán', 'aría', 'arías', 'aríamos', 'aríais', 'arían', 'ando', 'ado'],
                'er': ['o', 'es', 'e', 'emos', 'éis', 'en', 'í', 'iste', 'ió', 'imos', 'isteis', 'ieron', 'ía', 'ías', 'íamos', 'íais', 'ían', 'eré', 'erás', 'erá', 'eremos', 'eréis', 'erán', 'ería', 'erías', 'eríamos', 'eríais', 'erían', 'iendo', 'ido'],
                'ir': ['o', 'es', 'e', 'imos', 'ís', 'en', 'í', 'iste', 'ió', 'imos', 'isteis', 'ieron', 'ía', 'ías', 'íamos', 'íais', 'ían', 'iré', 'irás', 'irá', 'iremos', 'iréis', 'irán', 'iría', 'irías', 'iríamos', 'iríais', 'irían', 'iendo', 'ido']
            },
            'sustantivos': {
                'plural': ['s', 'es'],
                'femenino': ['a', 'ina', 'isa', 'triz']
            },
            'adjetivos': {
                'plural': ['s', 'es'],
                'femenino': ['a']
            }
        }

    def analizar_palabra(self, palabra):
        """
        Analiza una palabra para extraer su lexema y morfemas.
        Este es un enfoque simplificado. Un sistema real usaría un análisis más sofisticado.
        """
        palabra_original = palabra.lower()

        # 1. Manejar verbos
        for base_sufijo, terminaciones in self.reglas_morfemas['verbos'].items():
            for terminacion in sorted(terminaciones, key=len, reverse=True):
                if palabra_original.endswith(terminacion):
                    # Caso especial para evitar falsos positivos como "canto" -> "cant" (bien) vs "anto" -> "ant" (mal)
                    if palabra_original.endswith(base_sufijo + terminacion):
                        lexema = palabra_original[:-len(terminacion)]
                        return {'lexema': lexema, 'morfemas': [terminacion], 'tipo': 'verbo'}

        # 2. Manejar sustantivos y adjetivos (plural y género)
        # Esta lógica es más compleja y puede requerir un enfoque iterativo
        morfemas_encontrados = []
        lexema_potencial = palabra_original

        # Quitar plural
        for sufijo_plural in sorted(self.reglas_morfemas['sustantivos']['plural'], key=len, reverse=True):
            if lexema_potencial.endswith(sufijo_plural):
                lexema_potencial = lexema_potencial[:-len(sufijo_plural)]
                morfemas_encontrados.append(sufijo_plural)
                break # Asumimos un solo morfema de plural

        # Quitar género
        for sufijo_fem in sorted(self.reglas_morfemas['sustantivos']['femenino'], key=len, reverse=True):
             if lexema_potencial.endswith(sufijo_fem):
                lexema_potencial = lexema_potencial[:-len(sufijo_fem)]
                morfemas_encontrados.append(sufijo_fem)
                break

        if morfemas_encontrados:
            return {'lexema': lexema_potencial, 'morfemas': list(reversed(morfemas_encontrados)), 'tipo': 'nombre/adjetivo'}

        # Si no se encontraron morfemas, la palabra es el lexema
        return {'lexema': palabra_original, 'morfemas': [], 'tipo': 'desconocido'}

# Ejemplo de uso
if __name__ == '__main__':
    lexer = Lexer()
    palabras = ["corriendo", "casas", "gato", "gatas", "hablaremos", "actriz", "comimos"]
    for p in palabras:
        analisis = lexer.analizar_palabra(p)
        print(f"Palabra: {p} -> Lexema: '{analisis['lexema']}', Morfemas: {analisis['morfemas']}")
