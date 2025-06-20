#  Trinity Library: Fractal

> Bilingual README | README bilingüe (English & Español)

## 🇬🇧 Introduction (English)

**Trinity** is a Python library that implements the core reasoning engine of the **Aurora Model**, designed around **fractal processing principles** and the **golden ratio (φ ≈ 1.618)**. Trinity is not an application—it is the foundation of intelligent computation.

###  Key Features
- **Trigate**: Atomic unit for ternary logic reasoning, learning, and inference with NULL propagation
- **Transcender**: Synthesizes fractal knowledge vectors through hierarchical processing
- **KnowledgeBase**: Manages axiom storage across multiple knowledge spaces with coherence validation
- **Evolver**: Detects patterns, formalizes axioms, and generates reconstruction guides
- **Extender**: Enables deep interpretability through fractal reconstruction and reverse engineering



##  System Estructure  / Estructura del Sistema

```text
inputs → Trigate → Transcender → KnowledgeBase
           ↓           ↓            ↓
       inference → fractal     → Evolver
                   synthesis      ↓
                                Extender
                                  ↓
                              reconstruction
```

---


###  Example
```python
from Trinity import Trigate, Transcender, KnowledgeBase, Evolver, Extender

# Basic ternary logic operations
trigate = Trigate([1,0,1], [0,1,0], [1,1,0], [0,1,1])
result = trigate.inferir()  # Infer from inputs
trigate.aprender()  # Learn from current state

# Fractal synthesis
transcender = Transcender()
fractal_vector = transcender.level1_synthesis([1,0,1], [0,1,0], [1,1,1])

# Knowledge management
kb = KnowledgeBase()
evolver = Evolver(kb)
evolver.formalize_fractal_axiom(fractal_vector, {"context": "test"}, "default")

# Reconstruction
extender = Extender()
guide_package = evolver.generate_guide_package("default")
extender.load_guide_package(guide_package)
reconstructed = extender.reconstruct([1,0,1])
```

## 🇪🇸 Introducción (Español)

**Trinity** es una librería Python que implementa el motor de razonamiento del **Modelo Aurora**, basado en **principios fractales** y la **proporción áurea (φ ≈ 1.618)**. Trinity no es una app, es la base lógica de la inteligencia.

###  Características Principales
- **Trigate**: Unidad atómica para razonamiento de lógica ternaria, aprendizaje e inferencia con propagación de NULL
- **Transcender**: Sintetiza vectores de conocimiento fractal a través de procesamiento jerárquico
- **KnowledgeBase**: Gestiona almacenamiento de axiomas en múltiples espacios de conocimiento con validación de coherencia
- **Evolver**: Detecta patrones, formaliza axiomas y genera guías de reconstrucción
- **Extender**: Permite interpretabilidad profunda a través de reconstrucción fractal e ingeniería inversa

### Ejemplo
```python
from Trinity import Trigate, Transcender, KnowledgeBase, Evolver, Extender

# Operaciones básicas de lógica ternaria
trigate = Trigate([1,0,1], [0,1,0], [1,1,0], [0,1,1])
resultado = trigate.inferir()  # Inferir desde entradas
trigate.aprender()  # Aprender del estado actual

# Síntesis fractal
transcender = Transcender()
vector_fractal = transcender.level1_synthesis([1,0,1], [0,1,0], [1,1,1])

# Gestión de conocimiento
kb = KnowledgeBase()
evolver = Evolver(kb)
evolver.formalize_fractal_axiom(vector_fractal, {"contexto": "prueba"}, "default")

# Reconstrucción
extender = Extender()
paquete_guia = evolver.generate_guide_package("default")
extender.load_guide_package(paquete_guia)
reconstruido = extender.reconstruct([1,0,1])
```

---

##  Use Cases / Casos de Uso

-  AI interpretability / Interpretabilidad en IA
-  Conscious robotics / Robótica consciente
-  Transparent diagnosis / Diagnóstico médico explicable
-  NLP & semantic reasoning / PLN y razonamiento semántico

---

##  License / Licencia

Released under GPLv3.  
Publicado bajo GPLv3.

> “True intelligence does not only solve problems—it understands and explains its solutions.”  
> "La verdadera inteligencia no solo resuelve problemas, sino que comprende y explica sus soluciones." – Aurora Principle
