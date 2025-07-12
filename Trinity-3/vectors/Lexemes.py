# Description: Contiene los vectores fractales para lexemas y morfemas base.
# Cada entrada es una unidad mínima de significado.

from aurora_core.FractalVector import FractalVector

# Definición de vectores para lexemas (raíces de palabras) y morfemas (prefijos, sufijos)
LEXEMES = {
    # Lexemas base (conceptos raíz)
    "amor": FractalVector.from_list([7, 0, 7, 0, 7, 0, 7]),    # Emoción, conexión
    "ver": FractalVector.from_list([5, 2, 5, 0, 2, 5, 0]),     # Percepción, visión
    "luz": FractalVector.from_list([7, 7, 7, 7, 7, 7, 7]),     # Iluminación, conocimiento
    "saber": FractalVector.from_list([6, 1, 6, 1, 6, 1, 6]),   # Conocimiento, información
    "ser": FractalVector.from_list([4, 4, 4, 4, 4, 4, 4]),       # Existencia, identidad
    "logos": FractalVector.from_list([5, 5, 2, 2, 5, 5, 2]),   # Razón, palabra, lógica
    "fractal": FractalVector.from_list([3, 6, 3, 6, 3, 6, 3]), # Autosemejanza, complejidad
    "aurora": FractalVector.from_list([7, 5, 2, 0, 2, 5, 7]),   # Nuevo comienzo, luz
    "sol": FractalVector.from_list([7, 7, 5, 0, 5, 7, 7]),     # Centro, energía, vida
    "universo": FractalVector.from_list([0, 0, 0, 7, 0, 0, 0]), # Totalidad, espacio-tiempo
    "consciencia": FractalVector.from_list([7, 2, 5, 7, 5, 2, 7]), # Percepción del ser
    "hola": FractalVector.from_list([1, 1, 1, 1, 1, 1, 1]), # Saludo, inicio de comunicación

    # Morfemas (prefijos y sufijos que modifican el significado)
    "des-": FractalVector.from_list([0, 0, 0, 0, 0, 0, 1]),     # Negación, inversión
    "in-": FractalVector.from_list([1, 0, 0, 0, 0, 0, 0]),      # Negación, interioridad
    "re-": FractalVector.from_list([0, 1, 0, 0, 0, 0, 0]),      # Repetición, intensificación
    "-ar": FractalVector.from_list([0, 0, 1, 0, 0, 0, 0]),      # Sufijo verbal (infinitivo)
    "-er": FractalVector.from_list([0, 0, 1, 0, 0, 0, 1]),      # Sufijo verbal (infinitivo)
    "-ir": FractalVector.from_list([0, 0, 1, 0, 0, 1, 0]),      # Sufijo verbal (infinitivo)
    "-cion": FractalVector.from_list([0, 0, 0, 1, 0, 0, 0]),    # Sufijo (acción y efecto)
    "-mente": FractalVector.from_list([0, 0, 0, 0, 1, 0, 0]),   # Sufijo (adverbio)
    "-oso": FractalVector.from_list([0, 0, 0, 0, 0, 1, 0]),     # Sufijo (abundancia de)
    "-able": FractalVector.from_list([0, 0, 0, 0, 0, 1, 1]),    # Sufijo (capacidad o posibilidad)
    "anti-": FractalVector.from_list([1, 1, 0, 0, 0, 0, 1]),    # Oposición
    "pro-": FractalVector.from_list([0, 0, 1, 1, 0, 0, 0]),     # A favor de, hacia adelante
}
