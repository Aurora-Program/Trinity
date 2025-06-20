# Trinity Library API Documentation

## Overview

Trinity is a symbolic AI library implementing fractal/triangular logic with ternary operations. It provides hierarchical synthesis and knowledge management through five core classes.

## Core Classes

### 1. Trigate

The atomic unit for ternary logic operations with NULL propagation support.

#### Constructor
```python
Trigate(A=None, B=None, R=None, M=None)
```
- **A, B**: Input vectors (lists of 3 trits: 0, 1, or None)
- **R**: Result vector 
- **M**: Meta-information vector

#### Methods

##### `inferir() -> list`
Performs inference using XOR operations between inputs A and B.
- **Returns**: List of 3 trits representing the inferred result
- **Example**: `trigate.inferir()` → `[1, 0, None]`

##### `aprender() -> bool`
Updates meta-information M based on current inputs and result R.
- **Returns**: Boolean indicating learning success
- **Requires**: R must be set before calling

##### `sintesis_S() -> list`
Performs synthesis using XNOR operations between inputs A and B.
- **Returns**: List of 3 trits representing synthesized result

##### `validar_entrada(entrada, nombre) -> None`
Validates input format (internal method).
- **Raises**: ValueError if input is not a valid trit list

---

### 2. Transcender

Handles fractal synthesis through hierarchical processing.

#### Constructor
```python
Transcender()
```

#### Methods

##### `procesar(A, B, C) -> dict`
Main processing method that generates fractal vectors.
- **Parameters**: Three trit lists (A, B, C)
- **Returns**: Dictionary with fractal vector representation
- **Example**:
```python
result = transcender.procesar([1,0,1], [0,1,0], [1,1,1])
# Returns: {"layer1": [...], "layer2": [...], "layer3": [...]}
```

##### `level1_synthesis(A, B, C) -> dict`
Creates hierarchical fractal synthesis with 3 layers.
- **Layer 1**: Direct XOR operations
- **Layer 2**: Pairwise combinations  
- **Layer 3**: Detailed 9-element expansion
- **Returns**: Complete fractal vector structure

---

### 3. KnowledgeBase

Manages axiom storage and retrieval across multiple knowledge spaces.

#### Constructor
```python
KnowledgeBase()
```

#### Methods

##### `create_space(space_name, description="") -> bool`
Creates a new knowledge space.
- **space_name**: String identifier for the space
- **description**: Optional description
- **Returns**: True if created successfully

##### `store_axiom(space_name, Ms, MetaM, Ss, original_inputs) -> bool`
Stores an axiom in the specified space.
- **Ms**: Main synthesis vector (3 trits)
- **MetaM**: Meta-information matrix
- **Ss**: Secondary synthesis vector
- **original_inputs**: Dictionary of original input data
- **Returns**: True if stored successfully

##### `get_axiom_by_ms(space_name, Ms) -> dict`
Retrieves axiom by its Ms vector.
- **Returns**: Axiom dictionary or None if not found

##### `get_axioms_in_space(space_name) -> dict`
Returns all axioms in a space.
- **Returns**: Dictionary of all axioms in the space

##### `validate_fractal_coherence(space_name, fractal_vector, metam_rep) -> bool`
Validates coherence across all hierarchical levels.
- **Returns**: True if coherent, False otherwise

##### `list_spaces() -> list`
Returns list of available knowledge spaces.

##### `space_stats(space_name) -> dict`
Returns statistics for a knowledge space.
- **Returns**: Dictionary with description and axiom count

---

### 4. Evolver

Detects patterns and formalizes knowledge with dynamic context management.

#### Constructor
```python
Evolver(knowledge_base)
```
- **knowledge_base**: KnowledgeBase instance

#### Methods

##### `detect_fractal_pattern(vector) -> str`
Analyzes fractal vector patterns.
- **vector**: List of 3 values to analyze
- **Returns**: Pattern type ("unitary", "uniform", "complex")

##### `formalize_fractal_axiom(fractal_vector, context, space_name="default") -> bool`
Formalizes a fractal vector as an axiom.
- **fractal_vector**: Result from Transcender processing
- **context**: Dictionary with contextual information
- **space_name**: Target knowledge space
- **Returns**: True if formalized successfully

##### `generate_guide_package(space_name) -> dict`
Creates reconstruction guide package.
- **Returns**: Dictionary with axiom registry and metadata for reconstruction

---

### 5. Extender

Enables deep interpretability through fractal reconstruction.

#### Constructor
```python
Extender()
```

#### Methods

##### `load_guide_package(package) -> None`
Loads guide package from Evolver.
- **package**: Guide package dictionary

##### `reconstruct(target_ms) -> list`
Reconstructs original inputs from Ms vector.
- **target_ms**: Target Ms vector to reconstruct
- **Returns**: Original inputs list or None if not found

##### `reconstruct_fractal(target_fractal_vector, space_name="default") -> dict`
Reconstructs complete fractal vector from abstract representation.
- **Returns**: Detailed reconstruction information

---

## Data Formats

### Trit Lists
- Format: `[0, 1, None]` (3 elements)
- Values: 0, 1, or None (representing uncertainty/NULL)

### Fractal Vectors
```python
{
    "layer1": [t1, t2, t3],           # 3 trits
    "layer2": [[...], [...], [...]],  # 3x3 matrix  
    "layer3": [...]                   # 9-element list
}
```

### Axioms
```python
{
    "MetaM": [...],                   # Meta-information
    "Ss": [t1, t2, t3],              # Secondary synthesis
    "original_inputs": {...}          # Original input data
}
```

## Usage Patterns

### Basic Inference
```python
from Trinity import Trigate

trigate = Trigate([1,0,1], [0,1,0])
result = trigate.inferir()
```

### Complete Pipeline
```python
from Trinity import *

# 1. Create components
trigate = Trigate([1,0,1], [0,1,0], [1,1,0], [0,1,1])
transcender = Transcender()
kb = KnowledgeBase()
evolver = Evolver(kb)
extender = Extender()

# 2. Process and synthesize
fractal_vector = transcender.level1_synthesis([1,0,1], [0,1,0], [1,1,1])

# 3. Store knowledge
evolver.formalize_fractal_axiom(fractal_vector, {"context": "test"})

# 4. Reconstruct
guide_package = evolver.generate_guide_package("default")
extender.load_guide_package(guide_package)
reconstructed = extender.reconstruct([1,0,1])
```

### Knowledge Space Management
```python
kb = KnowledgeBase()
kb.create_space("medical", "Medical diagnosis knowledge")
kb.create_space("robotics", "Robotics reasoning patterns")

# Store axioms in different spaces
evolver = Evolver(kb)
evolver.formalize_fractal_axiom(fractal_vector, context, "medical")

# Query spaces
print(kb.list_spaces())
print(kb.space_stats("medical"))
```

## Performance Characteristics

Based on benchmarks:
- **Trigate operations**: ~0.002ms average
- **Transcender synthesis**: ~0.018ms average  
- **KnowledgeBase queries**: ~0.001ms average
- **Full pipeline**: ~0.56ms average
- **Scalability**: Linear growth up to 1000+ objects

## Error Handling

The library includes validation for:
- Trit format validation (3-element lists with valid values)
- Knowledge space existence
- Axiom coherence across fractal layers
- NULL propagation in ternary operations

Common exceptions:
- `ValueError`: Invalid trit format or missing required data
- `Exception`: Missing guide packages or space configuration errors
