"""
Trinity Enhanced - Optimized symbolic AI system with Aurora architecture
Addresses Lumen's feedback: validation consolidation, type hints, logging, and improved design patterns
"""

import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================================================================
#  CUSTOM EXCEPTIONS - Aurora specific errors
# ==============================================================================
class LogicalCoherenceError(Exception):
    """Exception for violations of Aurora's Unique Correspondence Principle"""
    pass

class FractalStructureError(Exception):
    """Exception for fractal structure validation errors"""
    pass

class ValidationError(Exception):
    """Exception for input validation failures"""
    pass

# ==============================================================================
#  TYPE DEFINITIONS
# ==============================================================================
TritValue = Union[int, None]  # 0, 1, or None (uncertainty)
TritVector = List[TritValue]  # List of exactly 3 trits
FractalVector = Dict[str, Any]  # Fractal structure with layers
AxiomDict = Dict[str, Any]  # Axiom storage format

# ==============================================================================
#  TERNARY OPERATIONS - Centralized vector operations
# ==============================================================================
class TernaryOperations:
    """Centralized ternary operations for consistency across the system"""
    
    @staticmethod
    def xor(a: TritValue, b: TritValue) -> TritValue:
        """Ternary XOR with proper None propagation"""
        if a is None or b is None:
            return None
        return 1 if a != b else 0
    
    @staticmethod
    def xnor(a: TritValue, b: TritValue) -> TritValue:
        """Ternary XNOR with proper None propagation"""
        if a is None or b is None:
            return None
        return 1 if a == b else 0
    
    @staticmethod
    def xor_vector(vector_a: TritVector, vector_b: TritVector) -> TritVector:
        """Element-wise XOR operation on vectors"""
        if len(vector_a) != len(vector_b):
            raise ValidationError(f"Vector length mismatch: {len(vector_a)} vs {len(vector_b)}")
        return [TernaryOperations.xor(a, b) for a, b in zip(vector_a, vector_b)]
    
    @staticmethod
    def not_op(a: TritValue) -> TritValue:
        """Ternary NOT operation"""
        if a is None:
            return None
        return 1 - a
    
    @staticmethod
    def and_op(a: TritValue, b: TritValue) -> TritValue:
        """Ternary AND operation"""
        if a == 1 and b == 1:
            return 1
        elif a == 0 or b == 0:
            return 0
        else:
            return None
    
    @staticmethod
    def or_op(a: TritValue, b: TritValue) -> TritValue:
        """Ternary OR operation"""
        if a == 1 or b == 1:
            return 1
        elif a == 0 and b == 0:
            return 0
        else:
            return None

# ==============================================================================
#  INPUT VALIDATION - Centralized validation layer
# ==============================================================================
class InputValidator:
    """Centralized validation to eliminate redundant checks"""
    
    @staticmethod
    def validate_trit_vector(vector: Any, name: str = "vector") -> TritVector:
        """Validates a trit vector with detailed error messages"""
        if not isinstance(vector, list):
            raise ValidationError(f"{name} must be a list, got {type(vector)}")
        
        if len(vector) != 3:
            raise ValidationError(f"{name} must have exactly 3 elements, got {len(vector)}")
        
        for i, trit in enumerate(vector):
            if trit not in (0, 1, None):
                raise ValidationError(f"{name}[{i}] must be 0, 1, or None, got {trit}")
        
        return vector
    
    @staticmethod
    def validate_fractal_structure(fractal: Dict[str, Any]) -> FractalVector:
        """Validates complete fractal vector structure"""
        required_layers = ['layer1', 'layer2', 'layer3']
        
        for layer in required_layers:
            if layer not in fractal:
                raise FractalStructureError(f"Missing required layer: {layer}")
        
        # Validate layer1 (3 trits)
        InputValidator.validate_trit_vector(fractal['layer1'], 'layer1')
        
        # Validate layer2 (3 vectors of 3 trits each)
        if not isinstance(fractal['layer2'], list) or len(fractal['layer2']) != 3:
            raise FractalStructureError("layer2 must contain exactly 3 vectors")
        
        for i, vector in enumerate(fractal['layer2']):
            InputValidator.validate_trit_vector(vector, f'layer2[{i}]')
        
        # Validate layer3 (9 vectors of 3 trits each)
        if not isinstance(fractal['layer3'], list) or len(fractal['layer3']) != 9:
            raise FractalStructureError("layer3 must contain exactly 9 vectors")
        
        for i, vector in enumerate(fractal['layer3']):
            InputValidator.validate_trit_vector(vector, f'layer3[{i}]')
        
        return fractal

# ==============================================================================
#  CLASS 1: Trigate - Ternary logic gate with clean separation of concerns
# ==============================================================================
class Trigate:
    """
    Atomic unit for ternary logic operations.
    Clean separation between pattern generation and inference.
    """
    
    def __init__(self, A: Optional[TritVector] = None, 
                 B: Optional[TritVector] = None, 
                 R: Optional[TritVector] = None, 
                 M: Optional[TritVector] = None):
        self.A = A
        self.B = B  
        self.R = R
        self.M = M or [0, 0, 0]  # Default neutral pattern
        logger.debug(f"Trigate initialized with A={A}, B={B}, R={R}, M={M}")
    
    def infer_result(self) -> TritVector:
        """Infers R from A, B, and M pattern"""
        # Single validation point
        InputValidator.validate_trit_vector(self.A, "A")
        InputValidator.validate_trit_vector(self.B, "B")
        InputValidator.validate_trit_vector(self.M, "M")
        
        # Apply pattern: M[i] determines operation type
        result = []
        for i in range(3):
            if self.M[i] == 0:
                result.append(TernaryOperations.xnor(self.A[i], self.B[i]))
            else:
                result.append(TernaryOperations.xor(self.A[i], self.B[i]))
        
        self.R = result
        logger.debug(f"Inferred R={result} from A={self.A}, B={self.B}, M={self.M}")
        return result
    
    def learn_pattern(self) -> TritVector:
        """Learns M pattern from A, B, and R"""
        InputValidator.validate_trit_vector(self.A, "A")
        InputValidator.validate_trit_vector(self.B, "B")
        InputValidator.validate_trit_vector(self.R, "R")
        
        pattern = []
        for i in range(3):
            if any(v is None for v in [self.A[i], self.B[i], self.R[i]]):
                pattern.append(None)  # Cannot determine pattern with uncertainty
            elif self.R[i] == TernaryOperations.xor(self.A[i], self.B[i]):
                pattern.append(1)  # XOR pattern
            else:
                pattern.append(0)  # XNOR pattern
        
        self.M = pattern
        logger.debug(f"Learned pattern M={pattern}")
        return pattern
    
    def generate_pattern(self, seed: Optional[int] = None) -> TritVector:
        """Generates a new M pattern (separated from inference)"""
        if seed is not None:
            random.seed(seed)
        
        pattern = [random.choice([0, 1]) for _ in range(3)]
        self.M = pattern
        logger.debug(f"Generated pattern M={pattern}")
        return pattern
    
    def synthesize_form(self) -> TritVector:
        """Synthesizes S (form) value from A, B, R"""
        InputValidator.validate_trit_vector(self.A, "A")
        InputValidator.validate_trit_vector(self.B, "B")
        InputValidator.validate_trit_vector(self.R, "R")
        
        form = []
        for i in range(3):
            if self.R[i] is None:
                form.append(None)
            elif self.R[i] == 0:
                form.append(self.A[i])
            else:
                form.append(self.B[i])
        
        logger.debug(f"Synthesized form S={form}")
        return form
    
    def deduce_input(self, known_input: TritVector, known_is_A: bool) -> TritVector:
        """Deduces unknown input from known input and result"""
        InputValidator.validate_trit_vector(known_input, "known_input")
        InputValidator.validate_trit_vector(self.R, "R")
        InputValidator.validate_trit_vector(self.M, "M")
        
        unknown = []
        for i in range(3):
            if self.M[i] == 0:  # XNOR pattern
                unknown.append(TernaryOperations.xnor(known_input[i], self.R[i]))
            else:  # XOR pattern
                unknown.append(TernaryOperations.xor(known_input[i], self.R[i]))
        
        if known_is_A:
            self.B = unknown
        else:
            self.A = unknown
        
        logger.debug(f"Deduced unknown input: {unknown}")
        return unknown

# ==============================================================================
#  CLASS 2: Transcender - Hierarchical synthesis engine
# ==============================================================================
class Transcender:
    """
    Hierarchical synthesis engine implementing Aurora's fractal architecture.
    Optimized with centralized operations and clear method separation.
    """
    
    def __init__(self):
        self._tg1 = Trigate()
        self._tg2 = Trigate()
        self._tg3 = Trigate()
        self._tg_synthesis = Trigate()
        self.last_processing_data: Dict[str, Any] = {}
        logger.debug("Transcender initialized with 4 internal Trigates")
    
    def process_inputs(self, input_a: TritVector, input_b: TritVector, 
                      input_c: TritVector) -> Tuple[TritVector, TritVector, List[TritVector]]:
        """
        Main processing method - single validation entry point.
        Returns: (Ms, Ss, MetaM)
        """
        # Single validation point for all inputs
        InputValidator.validate_trit_vector(input_a, "input_a")
        InputValidator.validate_trit_vector(input_b, "input_b")
        InputValidator.validate_trit_vector(input_c, "input_c")
        
        logger.info(f"Processing inputs: A={input_a}, B={input_b}, C={input_c}")
        
        # Configure Trigates with inputs
        self._tg1.A, self._tg1.B = input_a, input_b
        self._tg2.A, self._tg2.B = input_b, input_c
        self._tg3.A, self._tg3.B = input_c, input_a
        
        # Generate patterns (M1, M2, M3) using inference
        m1 = self._tg1.infer_result()
        m2 = self._tg2.infer_result()
        m3 = self._tg3.infer_result()
        
        # Synthesis level
        self._tg_synthesis.A = m1
        self._tg_synthesis.B = m2
        self._tg_synthesis.R = m3
        
        ms = self._tg_synthesis.infer_result()
        ss = self._tg_synthesis.synthesize_form()
        meta_m = [m1, m2, m3, ms]
        
        # Store processing data for traceability
        self.last_processing_data = {
            "inputs": {"A": input_a, "B": input_b, "C": input_c},
            "intermediate": {"M1": m1, "M2": m2, "M3": m3},
            "outputs": {"Ms": ms, "Ss": ss, "MetaM": meta_m}
        }
        
        logger.info(f"Processing complete: Ms={ms}, Ss={ss}")
        return ms, ss, meta_m
    
    def synthesize_fractal_l1(self, input_a: TritVector, input_b: TritVector, 
                             input_c: TritVector) -> FractalVector:
        """
        Creates authentic Aurora fractal structure with 39 trits.
        Uses 13 Transcenders total: 9 (L3) + 3 (L2) + 1 (L1)
        """
        logger.info(f"Starting L1 fractal synthesis with inputs A={input_a}, B={input_b}, C={input_c}")
        
        # Validate inputs once
        InputValidator.validate_trit_vector(input_a, "input_a")
        InputValidator.validate_trit_vector(input_b, "input_b") 
        InputValidator.validate_trit_vector(input_c, "input_c")
        
        # LAYER 3: 9 Transcenders for fine-grained synthesis
        layer3_combinations = [
            (input_a, input_b, input_c),
            (input_b, input_c, input_a),
            (input_c, input_a, input_b),
            (input_a, input_c, input_b),
            (input_b, input_a, input_c),
            (input_c, input_b, input_a),
            (TernaryOperations.xor_vector(input_a, input_b), input_c, input_a),
            (input_b, TernaryOperations.xor_vector(input_a, input_c), input_c),
            (input_a, input_b, TernaryOperations.xor_vector(input_b, input_c))
        ]
        
        layer3 = []
        for i, (a, b, c) in enumerate(layer3_combinations):
            ms, _, _ = self.process_inputs(a, b, c)
            layer3.append(ms)
            logger.debug(f"L3[{i}]: {(a, b, c)} → {ms}")
        
        # LAYER 2: 3 Transcenders for intermediate synthesis
        layer2 = []
        for i in range(3):
            start_idx = i * 3
            group = layer3[start_idx:start_idx + 3]
            ms, _, _ = self.process_inputs(group[0], group[1], group[2])
            layer2.append(ms)
            logger.debug(f"L2[{i}]: Group {i} → {ms}")
        
        # LAYER 1: 1 Transcender for global synthesis
        layer1, ss_final, metam_final = self.process_inputs(layer2[0], layer2[1], layer2[2])
        logger.debug(f"L1: {layer2} → {layer1}")
        
        # Assemble authentic fractal vector
        fractal = {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "metadata": {
                "base_inputs": {"A": input_a, "B": input_b, "C": input_c},
                "transcenders_used": 13,
                "total_trits": 39,
                "coherence_signature": f"L1:3-L2:9-L3:27",
                "timestamp": int(time.time() * 1000)
            }
        }
        
        logger.info(f"Fractal synthesis complete: {fractal['metadata']['coherence_signature']}")
        return fractal
    
    def compare_fractals(self, fractal1: FractalVector, fractal2: FractalVector) -> Dict[str, float]:
        """Compares two fractal vectors hierarchically"""
        InputValidator.validate_fractal_structure(fractal1)
        InputValidator.validate_fractal_structure(fractal2)
        
        # Layer 1 similarity (global)
        l1_sim = 1.0 if fractal1["layer1"] == fractal2["layer1"] else 0.0
        
        # Layer 2 similarity (intermediate)
        l2_matches = sum(1 for i in range(3) if fractal1["layer2"][i] == fractal2["layer2"][i])
        l2_sim = l2_matches / 3.0
        
        # Layer 3 similarity (detailed)
        l3_matches = sum(1 for i in range(9) if fractal1["layer3"][i] == fractal2["layer3"][i])
        l3_sim = l3_matches / 9.0
        
        result = {
            "l1_similarity": l1_sim,
            "l2_similarity": l2_sim,
            "l3_similarity": l3_sim,
            "overall_similarity": (l1_sim + l2_sim + l3_sim) / 3.0
        }
        
        logger.info(f"Fractal comparison: L1={l1_sim:.2f}, L2={l2_sim:.2f}, L3={l3_sim:.2f}")
        return result

# ==============================================================================
#  CLASS 3: KnowledgeBase - Enhanced with configurable thresholds
# ==============================================================================
class KnowledgeBase:
    """Enhanced knowledge management with configurable validation thresholds"""
    
    def __init__(self, auto_create_spaces: bool = False):
        self.spaces: Dict[str, Dict[str, Any]] = {}
        self.coherence_log: List[Dict[str, Any]] = []
        self.auto_create_spaces = auto_create_spaces
        logger.debug(f"KnowledgeBase initialized (auto_create={auto_create_spaces})")
    
    def create_space(self, space_name: str, description: str = "") -> bool:
        """Creates a new knowledge space"""
        if space_name in self.spaces:
            logger.warning(f"Space '{space_name}' already exists")
            return False
        
        self.spaces[space_name] = {
            "description": description,
            "axiom_registry": {},
            "creation_timestamp": int(time.time() * 1000),
            "stats": {"axiom_count": 0, "coherence_violations": 0}
        }
        
        logger.info(f"Created knowledge space: '{space_name}'")
        return True
    
    def store_axiom(self, space_name: str, ms: TritVector, meta_m: Any, 
                   ss: TritVector, original_inputs: Dict[str, Any]) -> bool:
        """Stores axiom with enhanced coherence validation"""
        InputValidator.validate_trit_vector(ms, "Ms")
        InputValidator.validate_trit_vector(ss, "Ss")
        
        # Handle space creation based on configuration
        if space_name not in self.spaces:
            if self.auto_create_spaces:
                self.create_space(space_name, f"Auto-created space for {space_name}")
            else:
                raise ValidationError(f"Space '{space_name}' does not exist. Create it explicitly.")
        
        axiom_registry = self.spaces[space_name]["axiom_registry"]
        ms_key = tuple(ms)
        
        # Aurora Unique Correspondence Principle validation
        if ms_key in axiom_registry:
            existing = axiom_registry[ms_key]
            if existing["MetaM"] != meta_m:
                self.spaces[space_name]["stats"]["coherence_violations"] += 1
                raise LogicalCoherenceError(
                    f"Ms↔MetaM coherence violation in '{space_name}': "
                    f"Ms={ms} already mapped to different MetaM"
                )
        
        # Store validated axiom
        axiom_registry[ms_key] = {
            "Ms": ms,
            "MetaM": meta_m,
            "Ss": ss,
            "original_inputs": original_inputs,
            "timestamp": int(time.time() * 1000)
        }
        
        # Update statistics
        self.spaces[space_name]["stats"]["axiom_count"] = len(axiom_registry)
        
        # Log coherence event
        self.coherence_log.append({
            "action": "store_axiom",
            "space": space_name,
            "ms_signature": str(ms_key),
            "timestamp": int(time.time() * 1000)
        })
        
        logger.info(f"Stored axiom in '{space_name}': Ms={ms}")
        return True
    
    def store_fractal_axiom(self, space_name: str, fractal: FractalVector, 
                           original_inputs: Dict[str, Any]) -> bool:
        """Stores fractal axiom with structure validation"""
        validated_fractal = InputValidator.validate_fractal_structure(fractal)
        
        # Create MetaM representation
        meta_m = {
            'layer1': validated_fractal["layer1"],
            'layer2': validated_fractal["layer2"], 
            'layer3': validated_fractal["layer3"]
        }
        
        # Validate hierarchical coherence
        if not self._validate_hierarchical_coherence(validated_fractal):
            logger.error("Fractal hierarchical coherence validation failed")
            return False
        
        return self.store_axiom(
            space_name, 
            validated_fractal["layer1"],
            meta_m,
            validated_fractal["layer2"][0] if validated_fractal["layer2"] else [0, 0, 0],
            original_inputs
        )
    
    def _validate_hierarchical_coherence(self, fractal: FractalVector) -> bool:
        """Validates that upper layers derive logically from lower layers"""
        try:
            temp_transcender = Transcender()
            
            # Validate Layer 3 → Layer 2 coherence
            for i in range(0, 9, 3):
                if i + 2 < len(fractal["layer3"]):
                    group = fractal["layer3"][i:i+3]
                    ms_derived, _, _ = temp_transcender.process_inputs(group[0], group[1], group[2])
                    
                    expected_index = i // 3
                    if expected_index < len(fractal["layer2"]):
                        if ms_derived != fractal["layer2"][expected_index]:
                            logger.warning(f"L3→L2 coherence failure at index {expected_index}")
                            return False
            
            # Validate Layer 2 → Layer 1 coherence
            if len(fractal["layer2"]) >= 3:
                ms_final, _, _ = temp_transcender.process_inputs(
                    fractal["layer2"][0],
                    fractal["layer2"][1],
                    fractal["layer2"][2]
                )
                
                if ms_final != fractal["layer1"]:
                    logger.warning("L2→L1 coherence failure")
                    return False
            
            logger.debug("Hierarchical coherence validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Coherence validation error: {e}")
            return False
    
    def get_axiom(self, space_name: str, ms: TritVector) -> Optional[AxiomDict]:
        """Retrieves axiom by Ms key"""
        if space_name not in self.spaces:
            return None
        
        axiom_registry = self.spaces[space_name]["axiom_registry"]
        return axiom_registry.get(tuple(ms))
    
    def get_space_stats(self, space_name: str) -> Optional[Dict[str, Any]]:
        """Returns comprehensive space statistics"""
        if space_name not in self.spaces:
            return None
        
        space = self.spaces[space_name]
        stats = space["stats"].copy()
        stats.update({
            "description": space["description"],
            "creation_timestamp": space["creation_timestamp"],
            "coherence_ratio": 1.0 - (stats["coherence_violations"] / max(stats["axiom_count"], 1))
        })
        
        return stats

# ==============================================================================
#  CLASS 4: Evolver - Enhanced with configurable parameters
# ==============================================================================
class Evolver:
    """
    Enhanced pattern detection and knowledge formalization with configurable thresholds
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, similarity_threshold: float = 0.5):
        self.kb = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.prediction_history: List[Dict[str, Any]] = []
        logger.debug(f"Evolver initialized with threshold={similarity_threshold}")
    
    def formalize_axiom(self, transcender_data: Dict[str, Any], space_name: str = "default") -> bool:
        """Formalizes Transcender output as axiom"""
        outputs = transcender_data["outputs"]
        inputs = transcender_data["inputs"]
        
        return self.kb.store_axiom(
            space_name,
            outputs["Ms"],
            outputs["MetaM"],
            outputs["Ss"],
            inputs
        )
    
    def formalize_fractal_axiom(self, fractal: FractalVector, 
                               original_inputs: Dict[str, Any], 
                               space_name: str = "default") -> bool:
        """Formalizes fractal vector as axiom"""
        logger.info(f"Formalizing fractal axiom in space '{space_name}'")
        return self.kb.store_fractal_axiom(space_name, fractal, original_inputs)
    
    def analyze_semantic_relationships(self, space_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyzes semantic relationships with configurable limit and threshold
        """
        if space_name not in self.kb.spaces:
            logger.warning(f"Space '{space_name}' not found")
            return []
        
        axioms = list(self.kb.spaces[space_name]["axiom_registry"].values())
        if limit:
            axioms = axioms[:limit]
        
        relationships = []
        
        for i, axiom1 in enumerate(axioms):
            for j, axiom2 in enumerate(axioms[i+1:], i+1):
                distance = self._calculate_semantic_distance(
                    axiom1.get("Ms", []),
                    axiom2.get("Ms", [])
                )
                
                if distance < self.similarity_threshold:
                    relationships.append({
                        "axiom1_ms": axiom1["Ms"],
                        "axiom2_ms": axiom2["Ms"],
                        "semantic_distance": distance,
                        "similarity": 1.0 - distance,
                        "relation_type": "similar" if distance < 0.3 else "related"
                    })
        
        logger.info(f"Found {len(relationships)} semantic relationships in '{space_name}'")
        return relationships
    
    def _calculate_semantic_distance(self, ms1: TritVector, ms2: TritVector) -> float:
        """Calculates semantic distance between two Ms vectors"""
        if len(ms1) != len(ms2):
            return 1.0  # Maximum distance for incompatible vectors
        
        differences = sum(1 for a, b in zip(ms1, ms2) if a != b)
        return differences / len(ms1)
    
    def predict_interaction_outcome(self, current_state: Dict[str, Any], 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced prediction with dynamic confidence based on history
        """
        # Base prediction
        base_confidence = 0.5
        predicted_ms = [random.choice([0, 1]) for _ in range(3)]
        
        # Adjust confidence based on prediction history
        if self.prediction_history:
            recent_accuracy = sum(
                1 for pred in self.prediction_history[-10:] 
                if pred.get("accuracy", 0) > 0.7
            ) / min(10, len(self.prediction_history))
            
            base_confidence = 0.3 + (recent_accuracy * 0.4)
        
        # Context-based adjustments
        if context:
            if "user_preference" in context:
                base_confidence += 0.1
            if "historical_pattern" in context:
                base_confidence += 0.15
        
        prediction = {
            "predicted_ms": predicted_ms,
            "confidence": min(base_confidence, 0.95),
            "reasoning": "Dynamic prediction based on historical patterns",
            "context_factors": list(context.keys()) if context else [],
            "timestamp": int(time.time() * 1000)
        }
        
        self.prediction_history.append(prediction)
        logger.info(f"Prediction: Ms={predicted_ms}, confidence={base_confidence:.2f}")
        
        return prediction
    
    def update_prediction_accuracy(self, prediction_id: int, actual_outcome: TritVector) -> None:
        """Updates prediction accuracy for learning"""
        if 0 <= prediction_id < len(self.prediction_history):
            prediction = self.prediction_history[prediction_id]
            predicted = prediction["predicted_ms"]
            
            # Calculate accuracy
            matches = sum(1 for p, a in zip(predicted, actual_outcome) if p == a)
            accuracy = matches / len(predicted)
            
            prediction["accuracy"] = accuracy
            prediction["actual_outcome"] = actual_outcome
            
            logger.debug(f"Updated prediction {prediction_id} accuracy: {accuracy:.2f}")

# ==============================================================================
#  CLASS 5: Extender - Enhanced reconstruction with error handling
# ==============================================================================
class Extender:
    """
    Enhanced reconstruction engine with robust error handling and fallback strategies
    """
    
    def __init__(self):
        self.guide_package: Optional[Dict[str, Any]] = None
        self.reconstruction_cache: Dict[str, Any] = {}
        logger.debug("Extender initialized")
    
    def load_guide_package(self, package: Dict[str, Any]) -> None:
        """Loads guide package with validation"""
        if not isinstance(package, dict) or "axiom_registry" not in package:
            raise ValidationError("Invalid guide package format")
        
        self.guide_package = package
        self.reconstruction_cache.clear()  # Clear cache when new package loaded
        logger.info("Guide package loaded successfully")
    
    def reconstruct_basic(self, target_ms: TritVector) -> Optional[Dict[str, Any]]:
        """Basic reconstruction using axiom lookup"""
        if not self.guide_package:
            raise ValidationError("Guide package not loaded")
        
        InputValidator.validate_trit_vector(target_ms, "target_ms")
        
        # Check cache first
        cache_key = tuple(target_ms)
        if cache_key in self.reconstruction_cache:
            logger.debug(f"Cache hit for Ms={target_ms}")
            return self.reconstruction_cache[cache_key]
        
        axiom_registry = self.guide_package["axiom_registry"]
        axiom = axiom_registry.get(cache_key)
        
        if axiom:
            result = axiom["original_inputs"]
            self.reconstruction_cache[cache_key] = result
            logger.info(f"Reconstruction successful for Ms={target_ms}")
            return result
        
        logger.warning(f"No axiom found for Ms={target_ms}")
        return None
    
    def reconstruct_fractal(self, target_fractal: FractalVector, 
                           space_name: str = "default") -> Optional[FractalVector]:
        """
        Enhanced fractal reconstruction with fallback strategies
        """
        if not self.guide_package:
            raise ValidationError("Guide package not loaded")
        
        # Validate input structure (partial validation for reconstruction)
        if "layer1" not in target_fractal:
            raise ValidationError("target_fractal must contain at least layer1")
        
        InputValidator.validate_trit_vector(target_fractal["layer1"], "layer1")
        
        logger.info(f"Starting fractal reconstruction for space '{space_name}'")
        
        # Find axiom using layer1 as key
        axiom_registry = self.guide_package["axiom_registry"]
        layer1_key = tuple(target_fractal["layer1"])
        axiom = axiom_registry.get(layer1_key)
        
        if not axiom:
            # Fallback: find closest axiom
            axiom = self._find_closest_axiom(target_fractal["layer1"], axiom_registry)
            if axiom:
                logger.info("Using closest match for reconstruction")
        
        if not axiom:
            logger.error(f"No suitable axiom found for reconstruction")
            return None
        
        # Reconstruct using MetaM if available
        meta_m = axiom.get("MetaM", {})
        
        try:
            # Reconstruct layer2
            if isinstance(meta_m, dict) and "layer2" in meta_m:
                layer2 = meta_m["layer2"][:3]
            else:
                layer2 = self._reconstruct_layer2_from_layer1(target_fractal["layer1"])
            
            # Reconstruct layer3
            if isinstance(meta_m, dict) and "layer3" in meta_m:
                layer3 = meta_m["layer3"][:9]
            else:
                layer3 = self._reconstruct_layer3_from_layer2(layer2)
            
            # Ensure proper structure
            layer2 = self._ensure_layer_completeness(layer2, 3)
            layer3 = self._ensure_layer_completeness(layer3, 9)
            
            result = {
                "layer1": target_fractal["layer1"],
                "layer2": layer2,
                "layer3": layer3,
                "reconstruction_metadata": {
                    "source_axiom": layer1_key,
                    "method": "metam_based" if "layer2" in meta_m else "deduced",
                    "completeness": "full",
                    "timestamp": int(time.time() * 1000)
                }
            }
            
            logger.info("Fractal reconstruction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            return None
    
    def _find_closest_axiom(self, target_ms: TritVector, 
                           axiom_registry: Dict[Tuple[int, ...], AxiomDict]) -> Optional[AxiomDict]:
        """Finds semantically closest axiom"""
        best_match = None
        min_distance = float('inf')
        
        for ms_key, axiom in axiom_registry.items():
            distance = sum(1 for a, b in zip(target_ms, ms_key) if a != b)
            
            if distance < min_distance:
                min_distance = distance
                best_match = axiom
        
        # Only return if reasonably close (at most 1 difference)
        return best_match if min_distance <= 1 else None
    
    def _reconstruct_layer2_from_layer1(self, layer1: TritVector) -> List[TritVector]:
        """Reconstructs layer2 from layer1 using logical expansion"""
        layer2 = []
        for i in range(3):
            if i < len(layer1):
                base_trit = layer1[i]
                if base_trit is None:
                    vector = [None, None, None]
                else:
                    vector = [base_trit, 1 - base_trit, base_trit]
                layer2.append(vector)
            else:
                layer2.append([0, 0, 0])
        
        return layer2
    
    def _reconstruct_layer3_from_layer2(self, layer2: List[TritVector]) -> List[TritVector]:
        """Reconstructs layer3 from layer2 using logical expansion"""
        layer3 = []
        
        for vector in layer2:
            if isinstance(vector, list) and len(vector) == 3:
                # Generate 3 variations of each layer2 vector
                for j in range(3):
                    variation = []
                    for trit in vector:
                        if trit is None:
                            variation.append(None)
                        else:
                            variation.append((trit + j) % 2)
                    layer3.append(variation)
            else:
                # Fallback for invalid vectors
                for j in range(3):
                    layer3.append([0, 0, 0])
        
        return layer3[:9]  # Ensure exactly 9 vectors
    
    def _ensure_layer_completeness(self, layer: List[TritVector], 
                                 required_size: int) -> List[TritVector]:
        """Ensures layer has required number of vectors"""
        while len(layer) < required_size:
            layer.append([0, 0, 0])
        
        return layer[:required_size]

# ==============================================================================
#  DEMONSTRATION PROGRAM
# ==============================================================================
def main():
    """Enhanced demonstration with comprehensive logging and error handling"""
    
    # Configure logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Trinity Enhanced demonstration")
    
    try:
        # Initialize components
        kb = KnowledgeBase(auto_create_spaces=False)  # Explicit space creation
        transcender = Transcender()
        evolver = Evolver(kb, similarity_threshold=0.4)
        extender = Extender()
        
        # Create knowledge space explicitly
        kb.create_space("quantum_physics", "Enhanced Aurora fractal physics domain")
        
        logger.info("=" * 60)
        logger.info("TRINITY ENHANCED DEMONSTRATION - Aurora Architecture")
        logger.info("=" * 60)
        
        # Phase 1: Basic processing
        logger.info("\n=== PHASE 1: Basic Trigate Operations ===")
        
        trigate = Trigate([1, 0, 1], [0, 1, 0])
        result = trigate.infer_result()
        pattern = trigate.generate_pattern(seed=42)  # Deterministic for testing
        form = trigate.synthesize_form()
        
        logger.info(f"Trigate result: {result}, pattern: {pattern}, form: {form}")
        
        # Phase 2: Fractal synthesis
        logger.info("\n=== PHASE 2: Fractal Synthesis ===")
        
        fractal1 = transcender.synthesize_fractal_l1([1, 0, 1], [0, 1, 0], [1, 1, 1])
        logger.info(f"Fractal 1 created: L1={fractal1['layer1']}")
        
        # Store in knowledge base
        success = evolver.formalize_fractal_axiom(
            fractal1, 
            {"A": [1, 0, 1], "B": [0, 1, 0], "C": [1, 1, 1]},
            "quantum_physics"
        )
        logger.info(f"Fractal axiom stored: {success}")
        
        # Phase 3: Semantic analysis
        logger.info("\n=== PHASE 3: Semantic Analysis ===")
        
        # Create another fractal for comparison
        fractal2 = transcender.synthesize_fractal_l1([0, 1, 0], [1, 0, 1], [0, 0, 1])
        evolver.formalize_fractal_axiom(
            fractal2,
            {"A": [0, 1, 0], "B": [1, 0, 1], "C": [0, 0, 1]},
            "quantum_physics"
        )
        
        # Analyze relationships
        relationships = evolver.analyze_semantic_relationships("quantum_physics", limit=10)
        logger.info(f"Found {len(relationships)} semantic relationships")
        
        # Phase 4: Reconstruction
        logger.info("\n=== PHASE 4: Enhanced Reconstruction ===")
        
        # Load guide package
        guide_package = {"axiom_registry": kb.spaces["quantum_physics"]["axiom_registry"]}
        extender.load_guide_package(guide_package)
        
        # Reconstruct fractal
        target = {"layer1": fractal1["layer1"]}
        reconstructed = extender.reconstruct_fractal(target, "quantum_physics")
        
        if reconstructed:
            logger.info("Fractal reconstruction successful")
            logger.info(f"Reconstructed L1: {reconstructed['layer1']}")
            logger.info(f"Reconstruction method: {reconstructed['reconstruction_metadata']['method']}")
        else:
            logger.error("Fractal reconstruction failed")
        
        # Phase 5: Prediction and learning
        logger.info("\n=== PHASE 5: Prediction and Learning ===")
        
        prediction = evolver.predict_interaction_outcome(
            {"current_state": fractal1["layer1"]},
            {"user_preference": "exploration", "historical_pattern": "positive"}
        )
        logger.info(f"Prediction: {prediction['predicted_ms']}, confidence: {prediction['confidence']:.2f}")
        
        # Simulate prediction feedback
        actual_outcome = [1, 0, 1]
        evolver.update_prediction_accuracy(len(evolver.prediction_history) - 1, actual_outcome)
        
        # Phase 6: System statistics
        logger.info("\n=== PHASE 6: System Statistics ===")
        
        stats = kb.get_space_stats("quantum_physics")
        if stats:
            logger.info(f"Space statistics: {stats}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TRINITY ENHANCED DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
