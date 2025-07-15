"""
Utils Golden - Stub para Pool Manager de rotación áurea
======================================================
Implementación mínima para no romper imports.
TODO: Implementar lógica real del tensor ring buffer.
"""

class DummyTensorPool:
    """Pool Manager stub - reemplazar con implementación real."""
    
    def __init__(self, system=None, enable_persistence=True):
        self.system = system
        self.enable_persistence = enable_persistence
        self.tensors = []
        
    def add_tensor(self, tensor):
        """Añade tensor al pool (stub)."""
        self.tensors.append(tensor)
        if len(self.tensors) > 100:  # Límite simple
            self.tensors.pop(0)
    
    def get_tensor_trio(self, query_type="arquetipo"):
        """Devuelve trío de tensores (stub)."""
        if len(self.tensors) >= 3:
            return self.tensors[-3:]
        return None
    
    def load_pool_state(self, filepath):
        """Carga estado del pool (stub)."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
                self.tensors = state.get('tensors', [])
        except FileNotFoundError:
            pass  # Normal en primera ejecución
        except Exception as e:
            print(f"Warning: Could not load pool state: {e}")
    
    def save_pool_state(self, filepath):
        """Guarda estado del pool (stub)."""
        try:
            import pickle
            state = {'tensors': self.tensors[-50:]}  # Solo últimos 50
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Warning: Could not save pool state: {e}")
    
    def get_rotation_metrics(self):
        """Métricas de rotación (stub)."""
        return {
            'pool_size': len(self.tensors),
            'rotation_efficiency': {'stub': {'steps_taken': 1, 'unique_visits': 1, 'efficiency': 1.0, 'coverage_ratio': 1.0}}
        }

def integrate_golden_rotation(system, enable_persistence=True):
    """Factory function para crear pool manager."""
    return DummyTensorPool(system, enable_persistence)
    size = len(seq)
    offset = golden_index(steps, size)
    
    # Rotación circular optimizada
    return seq[offset:] + seq[:offset]

def golden_sequence(start: int, length: int) -> List[int]:
    """
    🔧 Genera secuencia de índices usando rotación áurea.
    
    Útil para crear agendas de exploración quasi-uniforme.
    
    Args:
        start: Índice inicial
        length: Longitud de la secuencia deseada
        
    Returns:
        Lista de índices distribuidos áureamente
    """
    return [golden_index(start + i, length) for i in range(length)]

def phi_shift_vector(vector: List[Union[int, float]], amplitude: float = 1.0) -> List[float]:
    """
    🔧 Aplica desplazamiento áureo a vector numérico.
    
    Útil para perturbar ligeramente tensores manteniendo proporciones.
    
    Args:
        vector: Vector numérico de entrada
        amplitude: Amplitud del desplazamiento
        
    Returns:
        Vector con desplazamiento áureo aplicado
    """
    if not vector:
        return []
    
    phi_offsets = [modf(i * PHI_INVERSE)[0] * amplitude for i in range(len(vector))]
    return [v + offset for v, offset in zip(vector, phi_offsets)]

def golden_hash(data: str, mod: int = 1000000) -> int:
    """
    🔧 Hash determinista usando proporción áurea.
    
    Genera distribución más uniforme que hash() estándar.
    
    Args:
        data: String a hashear
        mod: Módulo para el resultado
        
    Returns:
        Hash áureo en rango [0, mod)
    """
    hash_sum = sum(ord(c) * PHI for c in data)
    frac, _ = modf(hash_sum)
    return int(frac * mod)

def rotate_tensor_levels(tensor_dict: dict, steps: int = 1) -> dict:
    """
    🔧 Rota todos los niveles de un tensor fractal usando φ.
    
    Preserva estructura interna pero cambia perspectiva de exploración.
    
    Args:
        tensor_dict: Dict con 'nivel_3', 'nivel_9', 'nivel_27'
        steps: Pasos de rotación
        
    Returns:
        Nuevo dict con niveles rotados
    """
    result = {}
    
    for level_name, level_data in tensor_dict.items():
        if isinstance(level_data, list) and level_data:
            # Rotar cada nivel con offset áureo diferente
            level_steps = golden_index(steps, len(level_data))
            result[level_name] = golden_rotate(level_data, level_steps)
        else:
            result[level_name] = level_data
    
    return result

class GoldenRotationTracker:
    """
    🔧 Tracker para métricas de rotación áurea en benchmarks.
    """
    
    def __init__(self):
        self.rotation_history = []
        self.diversity_scores = []
        self.stability_measures = []
    
    def record_rotation(self, step: int, method: str, result_quality: float):
        """Registra una rotación y su efectividad."""
        golden_pos = golden_index(step, 360)  # Posición en "círculo" áureo
        
        self.rotation_history.append({
            'step': step,
            'golden_position': golden_pos,
            'method': method,
            'quality': result_quality
        })
    
    def compute_phi_diversity(self) -> float:
        """
        Calcula diversidad de métodos/arquetipos únicos generados.
        
        Returns:
            Ratio de diversidad [0, 1]
        """
        if not self.rotation_history:
            return 0.0
        
        unique_methods = set(entry['method'] for entry in self.rotation_history)
        total_rotations = len(self.rotation_history)
        
        return len(unique_methods) / total_rotations
    
    def compute_golden_stability(self, window: int = 10) -> float:
        """
        Calcula estabilidad usando ventana deslizante.
        
        Args:
            window: Tamaño de ventana para análisis
            
        Returns:
            Medida de estabilidad [0, 1]
        """
        if len(self.rotation_history) < window:
            return 1.0  # Estable por defecto si hay pocos datos
        
        # Calcular varianza de quality en ventanas
        qualities = [entry['quality'] for entry in self.rotation_history[-window:]]
        
        if not qualities:
            return 1.0
        
        mean_quality = sum(qualities) / len(qualities)
        variance = sum((q - mean_quality) ** 2 for q in qualities) / len(qualities)
        
        # Convertir varianza a estabilidad [0, 1]
        stability = 1.0 / (1.0 + variance)
        return stability
    
    def get_stats(self) -> dict:
        """Retorna estadísticas completas del tracker."""
        return {
            'total_rotations': len(self.rotation_history),
            'phi_diversity': self.compute_phi_diversity(),
            'golden_stability': self.compute_golden_stability(),
            'rotation_coverage': len(set(r['golden_position'] for r in self.rotation_history))
        }

# Funciones auxiliares para integración rápida
def quick_golden_rotate(data, seed: int = 42):
    """Rotación áurea rápida para uso en benchmarks."""
    return golden_rotate(data, golden_index(seed, len(data) if data else 1))

def phi_augment_dataset(dataset: List, augmentation_ratio: float = 0.2):
    """
    🔧 Data augmentation usando rotación áurea.
    
    Duplica parte del dataset aplicando rotaciones φ-deterministas.
    """
    augmented = dataset.copy()
    n_augment = int(len(dataset) * augmentation_ratio)
    
    for i in range(n_augment):
        original_item = dataset[i % len(dataset)]
        
        # Aplicar rotación áurea si el item tiene estructura de lista
        if isinstance(original_item, dict):
            rotated_item = original_item.copy()
            for key, value in original_item.items():
                if isinstance(value, list) and len(value) > 1:
                    rotated_item[key] = golden_rotate(value, i + 1)
            augmented.append(rotated_item)
        elif isinstance(original_item, list):
            rotated_item = golden_rotate(original_item, i + 1)
            augmented.append(rotated_item)
    
    return augmented

def fib(n: int) -> int:
    """
    🔧 Secuencia de Fibonacci optimizada para rotaciones.
    
    Args:
        n: Índice en la secuencia (0-indexed)
        
    Returns:
        n-ésimo número de Fibonacci
    """
    if n <= 1:
        return 1
    
    # 🔧 OPTIMIZACIÓN: Límite extendido para pools grandes
    # Fibonacci hasta F(20) = 6765 para pools de cientos de tensores
    if n > 20:
        n = n % 20  # Ciclar para evitar overflow pero con rango mayor
    
    # Calculadora rápida usando iteración
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class TensorRotor:
    """
    🔧 Rotor de tensores Aurora con estrategias híbridas de exploración.
    
    Implementa tres modos de rotación:
    - phi: Golden-Step para cobertura uniforme (arquetipos/axiomas)
    - fibonacci: Saltos jerárquicos para dinámicas multi-escala
    - hybrid: Alternancia φ/Fibo para exploración completa
    """
    
    def __init__(self, N: int, mode: str = "hybrid", start_k: int = 0):
        """
        Args:
            N: Tamaño del pool de tensores
            mode: 'phi', 'fibonacci', 'hybrid'
            start_k: Índice inicial
        """
        self.N = max(1, N)  # Evitar división por cero
        self.k = start_k % self.N
        self.i = 0  # Contador de pasos
        self.mode = mode
        
        # 🔧 FIXED: Mejor cálculo de phi_step para pools pequeños
        if self.N <= 5:
            self.phi_step = max(2, round(PHI_INVERSE * self.N))
        else:
            self.phi_step = max(1, round(PHI_INVERSE * self.N))
        
        # Historial para métricas
        self.rotation_history = []
        self.coverage_set = set()
        self.coverage_set.add(self.k)  # Incluir posición inicial
        
    def next(self) -> int:
        """
        🔧 Calcula siguiente índice según estrategia de rotación.
        
        Returns:
            Próximo índice en [0, N)
        """
        old_k = self.k
        
        if self.mode == "phi":
            # Golden-Step: cobertura quasi-uniforme
            self.k = (self.k + self.phi_step) % self.N
            
        elif self.mode == "fibonacci":
            # 🔧 ENHANCED: Fibonacci con límite extendido
            fib_step = fib(self.i % 16)  # Límite a F(16) = 987
            self.k = (self.k + fib_step) % self.N
            
        elif self.mode == "hybrid":
            # Híbrido: alternar φ y Fibonacci
            if self.i % 2 == 0:
                # Pasos pares → Golden-Step (cobertura uniforme)
                self.k = (self.k + self.phi_step) % self.N
            else:
                # Pasos impares → Fibonacci (dinámicas jerárquicas)
                fib_step = fib((self.i // 2) % 16)  # Límite extendido
                self.k = (self.k + fib_step) % self.N
        else:
            # Modo lineal simple como fallback
            self.k = (self.k + 1) % self.N
        
        # Tracking para métricas
        self.i += 1
        self.rotation_history.append({
            'step': self.i,
            'from_k': old_k,
            'to_k': self.k,
            'method': self._get_step_method()
        })
        self.coverage_set.add(self.k)
        
        return self.k
    
    def _get_step_method(self) -> str:
        """Identifica el método usado en el último paso."""
        if self.mode == "phi":
            return "golden_step"
        elif self.mode == "fibonacci":
            return f"fib_{fib((self.i-1) % 12)}"
        elif self.mode == "hybrid":
            if (self.i-1) % 2 == 0:
                return "hybrid_phi"
            else:
                return f"hybrid_fib_{fib(((self.i-1) // 2) % 12)}"
        return "linear"
    
    def get_coverage_stats(self) -> dict:
        """
        🔧 Estadísticas de cobertura y eficiencia del rotor.
        
        Returns:
            dict: Métricas de exploración
        """
        total_steps = self.i
        unique_indices = len(self.coverage_set)
        
        return {
            'total_steps': total_steps,
            'unique_indices_visited': unique_indices,
            'coverage_ratio': unique_indices / self.N if self.N > 0 else 0.0,
            'efficiency': unique_indices / total_steps if total_steps > 0 else 0.0,
            'current_k': self.k,
            'mode': self.mode,
            'steps_to_full_coverage': self._estimate_full_coverage()
        }
    
    def _estimate_full_coverage(self) -> int:
        """Estima pasos necesarios para cubrir todos los índices."""
        if self.mode == "phi":
            # Golden ratio garantiza cobertura completa en ~N pasos
            return max(1, self.N - len(self.coverage_set))
        elif self.mode == "fibonacci":
            # Fibonacci puede ser menos eficiente pero más dinámico
            return max(1, int(self.N * 1.5) - self.i)
        else:  # hybrid
            # Híbrido debería ser intermedio
            return max(1, int(self.N * 1.2) - self.i)
    
    def reset(self, new_k: int = 0):
        """Reinicia el rotor desde nueva posición."""
        self.k = new_k % self.N
        self.i = 0
        self.rotation_history.clear()
        self.coverage_set.clear()
        self.coverage_set.add(self.k)
    
    def get_trio(self) -> list:
        """
        🔧 Genera trío de índices para síntesis fractal.
        
        Returns:
            Lista de 3 índices distribuidos según estrategia
        """
        indices = [self.k]  # Comenzar desde posición actual
        
        # Generar 2 índices adicionales
        for _ in range(2):
            indices.append(self.next())
        
        return indices
    
    def get_quinteto(self) -> list:
        """
        🔧 Genera quinteto de índices para análisis arquetípico complejo.
        
        Returns:
            Lista de 5 índices para análisis multi-tensorial
        """
        indices = [self.k]  # Comenzar desde posición actual
        
        # Generar 4 índices adicionales
        for _ in range(4):
            indices.append(self.next())
        
        return indices
    
    def persist_state(self, filepath: str):
        """
        🔧 NEW: Persiste estado del rotor en disco usando pickle.
        """
        import pickle
        
        state = {
            'N': self.N,
            'k': self.k,
            'i': self.i,
            'mode': self.mode,
            'phi_step': self.phi_step,
            'coverage_set': list(self.coverage_set),
            'rotation_history': self.rotation_history[-100:]  # Solo últimas 100
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, filepath: str):
        """
        🔧 NEW: Carga estado del rotor desde disco usando pickle.
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        rotor = cls(state['N'], state['mode'], state['k'])
        rotor.i = state['i']
        rotor.phi_step = state['phi_step']
        rotor.coverage_set = set(state['coverage_set'])
        rotor.rotation_history = state['rotation_history']
        
        return rotor

class TensorPoolManager:
    """
    🔧 Gestor de pools de tensores con rotación estratificada.
    
    Mantiene pools separados por profundidad y aplica rotación
    específica según la tarea (arquetipos, dinámicas, relatores).
    """
    
    def __init__(self):
        self.pools = {
            'deep27': [],      # Tensores con nivel_27 completo
            'mid9': [],        # Tensores con nivel_9 pero sin nivel_27
            'shallow3': [],    # Tensores solo con nivel_3
            'mixed': []        # Pool mixto para tareas globales
        }
        
        self.rotors = {
            'deep27': TensorRotor(1, mode="phi"),      # Se actualizará cuando se llene
            'mid9': TensorRotor(1, mode="fibonacci"),
            'shallow3': TensorRotor(1, mode="hybrid"),
            'mixed': TensorRotor(1, mode="hybrid")
        }
        
        # Métricas de uso
        self.access_stats = {pool: 0 for pool in self.pools.keys()}
    
    def add_tensor(self, tensor: 'FractalTensor'):
        """
        🔧 Añade tensor al pool apropiado según su profundidad.
        
        Args:
            tensor: FractalTensor a clasificar y almacenar
        """
        # 🔧 FIX: Validación robusta de estructura de tensor
        has_27 = False
        has_9 = False
        
        try:
            # Verificar nivel_27: debe ser lista de vectores
            if (hasattr(tensor, 'nivel_27') and tensor.nivel_27 and 
                isinstance(tensor.nivel_27, list) and len(tensor.nivel_27) > 0):
                # Verificar que es lista de listas y tiene contenido válido
                if all(isinstance(vec, list) for vec in tensor.nivel_27):
                    has_27 = any(any(x is not None for x in vec) for vec in tensor.nivel_27)
        except (AttributeError, TypeError, ValueError):
            has_27 = False
        
        try:
            # Verificar nivel_9: debe ser lista de vectores
            if (hasattr(tensor, 'nivel_9') and tensor.nivel_9 and 
                isinstance(tensor.nivel_9, list) and len(tensor.nivel_9) > 0):
                # Verificar que es lista de listas y tiene contenido válido
                if all(isinstance(vec, list) for vec in tensor.nivel_9):
                    has_9 = any(any(x is not None for x in vec) for vec in tensor.nivel_9)
        except (AttributeError, TypeError, ValueError):
            has_9 = False
        
        # Clasificar por profundidad
        if has_27:
            pool_name = 'deep27'
        elif has_9:
            pool_name = 'mid9'
        else:
            pool_name = 'shallow3'
        
        # Añadir a pool específico y mixto
        self.pools[pool_name].append(tensor)
        self.pools['mixed'].append(tensor)
        
        # 🔧 FIX: Actualizar rotors solo si el pool creció
        for pool, tensors in self.pools.items():
            if tensors:
                old_mode = self.rotors[pool].mode if pool in self.rotors else "hybrid"
                self.rotors[pool] = TensorRotor(len(tensors), mode=old_mode)
    
    def get_tensor_trio(self, task_type: str = "arquetipo", explain: bool = False):
        """
        🔧 ENHANCED: Obtiene trío con explicación opcional y estrategia mejorada.
        
        Args:
            task_type: 'arquetipo', 'dinamica', 'relator', 'axioma'
            explain: Si True, retorna (trio, explanation)
            
        Returns:
            Lista de 3 FractalTensors (+ explicación si explain=True)
        """
        # 🔧 ENHANCED: Mapeo más específico de estrategias
        task_to_pool = {
            'arquetipo': 'mixed',      # Diversidad máxima - cobertura global
            'dinamica': 'shallow3',    # Rápido, gradual - zoom progresivo
            'relator': 'mid9',         # Medio rango - conectar clústeres
            'axioma': 'deep27'         # Máxima profundidad - saltos grandes
        }
        
        pool_name = task_to_pool.get(task_type, 'mixed')
        pool = self.pools[pool_name]
        
        if len(pool) < 3:
            # Fallback inteligente
            fallback_order = ['mixed', 'shallow3', 'mid9', 'deep27']
            for fallback_pool in fallback_order:
                if len(self.pools[fallback_pool]) >= 3:
                    pool_name = fallback_pool
                    pool = self.pools[fallback_pool]
                    break
        
        if len(pool) < 3:
            if explain:
                return [], {'error': 'Insufficient tensors in all pools'}
            return []
        
        # Obtener índices rotados
        rotor = self.rotors[pool_name]
        coverage_before = rotor.get_coverage_stats()
        indices = rotor.get_trio()
        
        # Mapear índices a tensores
        trio = [pool[i] for i in indices]
        
        # Actualizar estadísticas
        self.access_stats[pool_name] += 1
        
        if explain:
            explanation = {
                'task_type': task_type,
                'pool_used': pool_name,
                'rotor_mode': rotor.mode,
                'indices_selected': indices,
                'coverage_before': coverage_before,
                'coverage_after': rotor.get_coverage_stats(),
                'phi_step': rotor.phi_step if rotor.mode in ["phi", "hybrid"] else None,
                'pool_size': len(pool),
                'access_count': self.access_stats[pool_name]
            }
            return trio, explanation
        
        return trio
    
    def get_rotation_metrics(self) -> dict:
        """
        🔧 NEW: Métricas completas de rotación para benchmarks.
        
        Returns:
            dict: Métricas detalladas de todos los rotors
        """
        metrics = {
            'pools_summary': {},
            'rotation_efficiency': {},
            'coverage_analysis': {},
            'optimal_strategies': {}
        }
        
        total_accesses = sum(self.access_stats.values())
        
        for pool_name, pool in self.pools.items():
            if not pool:
                continue
                
            rotor = self.rotors[pool_name]
            coverage_stats = rotor.get_coverage_stats()
            access_ratio = self.access_stats[pool_name] / max(1, total_accesses)
            
            metrics['pools_summary'][pool_name] = {
                'size': len(pool),
                'mode': rotor.mode,
                'accesses': self.access_stats[pool_name],
                'access_ratio': access_ratio
            }
            
            metrics['rotation_efficiency'][pool_name] = {
                'coverage_ratio': coverage_stats['coverage_ratio'],
                'efficiency': coverage_stats['efficiency'],
                'steps_taken': coverage_stats['total_steps'],
                'unique_visits': coverage_stats['unique_indices_visited']
            }
            
            # Análisis de estrategia óptima
            if access_ratio > 0.4:
                optimal_mode = "phi"  # Alta demanda = eficiencia
            elif access_ratio < 0.1:
                optimal_mode = "fibonacci"  # Baja demanda = diversidad
            else:
                optimal_mode = "hybrid"  # Balance
                
            metrics['optimal_strategies'][pool_name] = {
                'current_mode': rotor.mode,
                'recommended_mode': optimal_mode,
                'should_optimize': rotor.mode != optimal_mode
            }
        
        return metrics

    # ------------------------------------------------------------------
    #  ╭──────────────────────────────────────────────────────────╮
    #  │  A P I   L E G A C Y   y   P E R S I S T E N C I A       │
    #  ╰──────────────────────────────────────────────────────────╯

    # 1)  get_pool_stats  ──   usado por código antiguo y tests
    def get_pool_stats(self) -> dict:
        """
        Mantiene compatibilidad con la versión anterior; internamente
        delega en `get_rotation_metrics()` y añade un resumen compacto.
        """
        brief = {
            "pool_sizes": {name: len(self.pools[name]) for name in self.pools},
            "access_stats": self.access_stats.copy(),
            "rotor_coverage": {}
        }

        for name, rotor in self.rotors.items():
            if self.pools[name]:                      # solo si hay tensores
                brief["rotor_coverage"][name] = rotor.get_coverage_stats()

        return {
            "brief": brief,
            "detailed": self.get_rotation_metrics()
        }

    # 2)  save_pool_state  ──   persistencia liviana en disco
    def save_pool_state(self, filepath: str):
        """
        Guarda únicamente el estado esencial de cada rotor y las métricas
        de acceso.  Ideal para re-cargar la cobertura entre ejecuciones.
        """
        import pickle, os
        state = {
            "rotors": {
                name: {
                    "N": r.N,
                    "k": r.k,
                    "i": r.i,
                    "mode": r.mode,
                    "phi_step": r.phi_step,
                    "coverage_set": list(r.coverage_set),
                    "rotation_history": r.rotation_history[-100:]  # trim
                }
                for name, r in self.rotors.items()
            },
            "access_stats": self.access_stats
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    # 3)  load_pool_state  ──   recubre el estado guardado
    def load_pool_state(self, filepath: str):
        import pickle, os
        if not os.path.exists(filepath):
            return                               # primera ejecución
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # reconstruir rotors
        for name, data in state.get("rotors", {}).items():
            self.rotors[name] = TensorRotor(
                data["N"], data["mode"], data["k"]
            )
            r = self.rotors[name]
            r.i = data["i"]
            r.phi_step = data["phi_step"]
            r.coverage_set = set(data["coverage_set"]) or {r.k}
            r.rotation_history = data["rotation_history"]

        # restaurar métricas de uso
        self.access_stats.update(state.get("access_stats", {}))

    def optimize_rotors(self):
        """
        🔧 NEW: Optimiza estrategias de rotores basándose en patrones de uso.
        """
        metrics = self.get_rotation_metrics()
        optimizations_applied = 0
        
        for pool_name, strategy_info in metrics['optimal_strategies'].items():
            if strategy_info['should_optimize'] and self.pools[pool_name]:
                current_rotor = self.rotors[pool_name]
                recommended_mode = strategy_info['recommended_mode']
                
                # Crear nuevo rotor con modo optimizado
                new_rotor = TensorRotor(
                    len(self.pools[pool_name]), 
                    mode=recommended_mode,
                    start_k=current_rotor.k  # Mantener posición actual
                )
                
                # Transferir historial relevante
                new_rotor.coverage_set = current_rotor.coverage_set.copy()
                new_rotor.i = current_rotor.i
                
                self.rotors[pool_name] = new_rotor
                optimizations_applied += 1
                
                print(f"[POOL_MANAGER] Optimized {pool_name}: {current_rotor.mode} → {recommended_mode}")
        
        return optimizations_applied

    def save_pool_state(self, filepath: str):
        """
        🔧 NEW: Persiste estado completo del pool manager.
        """
        import pickle
        
        state_data = {
            'access_stats': self.access_stats,
            'rotor_states': {}
        }
        
        # Guardar estado de cada rotor
        for pool_name, rotor in self.rotors.items():
            if self.pools[pool_name]:  # Solo si hay tensores
                state_data['rotor_states'][pool_name] = {
                    'N': rotor.N,
                    'k': rotor.k,
                    'i': rotor.i,
                    'mode': rotor.mode,
                    'phi_step': rotor.phi_step,
                    'coverage_set': list(rotor.coverage_set)
                }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
    
    def load_pool_state(self, filepath: str):
        """
        🔧 NEW: Carga estado del pool manager desde disco.
        """
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            # Restaurar estadísticas de acceso
            if 'access_stats' in state_data:
                self.access_stats.update(state_data['access_stats'])
            
            # Restaurar estado de rotors
            if 'rotor_states' in state_data:
                for pool_name, rotor_state in state_data['rotor_states'].items():
                    if pool_name in self.rotors and self.pools[pool_name]:
                        rotor = self.rotors[pool_name]
                        rotor.k = rotor_state['k']
                        rotor.i = rotor_state['i']
                        rotor.phi_step = rotor_state['phi_step']
                        rotor.coverage_set = set(rotor_state['coverage_set'])
        
        except Exception as e:
            print(f"[WARNING] Could not load pool state: {e}")

# 🔧 NEW: Función de integración rápida para benchmarks
def integrate_golden_rotation(aurora_system, enable_persistence: bool = False):
    """
    🔧 Integra rotación áurea en sistema Aurora existente.
    
    Args:
        aurora_system: Instancia de AuroraSystem/AuroraFractalSystem
        enable_persistence: Si True, habilita persistencia de estado
        
    Returns:
        TensorPoolManager configurado
    """
    # Crear pool manager
    pool_manager = TensorPoolManager()
    
    # Integrar en el sistema
    if hasattr(aurora_system, 'tensor_pool'):
        aurora_system.tensor_pool = pool_manager
    else:
        setattr(aurora_system, 'tensor_pool', pool_manager)
    
    # Configurar persistencia si se solicita
    if enable_persistence:
        import os
        state_dir = os.path.join(os.getcwd(), 'rotor_states')
        os.makedirs(state_dir, exist_ok=True)
        pool_manager.state_dir = state_dir
    
    return pool_manager

def phi_augment_training_tensors(tensor_list: list, augmentation_ratio: float = 0.5):
    """
    🔧 NEW: Augmentación específica para tensores de entrenamiento.
    
    Args:
        tensor_list: Lista de FractalTensors
        augmentation_ratio: Fracción a aumentar
        
    Returns:
        Lista aumentada con rotaciones áureas
    """
    augmented = tensor_list.copy()
    n_augment = int(len(tensor_list) * augmentation_ratio)
    
    for i in range(n_augment):
        original_tensor = tensor_list[i % len(tensor_list)]
        
        # Aplicar rotación áurea a niveles del tensor
        augmented_tensor = type(original_tensor)()  # Crear nueva instancia
        
        # Rotar cada nivel con offset diferente
        for level_name in ['nivel_3', 'nivel_9', 'nivel_27']:
            if hasattr(original_tensor, level_name):
                level_data = getattr(original_tensor, level_name)
                if level_data and isinstance(level_data, list):
                    level_steps = golden_index(i + 1, len(level_data))
                    rotated_level = golden_rotate(level_data, level_steps)
                    setattr(augmented_tensor, level_name, rotated_level)
                else:
                    setattr(augmented_tensor, level_name, level_data)
        
        # Marcar como augmentado
        augmented_tensor._is_augmented = True
        augmented_tensor._original_id = getattr(original_tensor, 'id', i)
        
        augmented.append(augmented_tensor)
    
    return augmented

if __name__ == "__main__":
    # Tests básicos
    print("🔧 Testing Golden Rotation Utils")
    
    # Test 1: Golden index distribution
    test_size = 10
    indices = [golden_index(i, test_size) for i in range(test_size * 2)]
    print(f"Golden indices: {indices[:test_size]}")
    
    # Test 2: Rotation
    test_list = [1, 2, 3, 4, 5]
    rotated = golden_rotate(test_list, 3)
    print(f"Original: {test_list}")
    print(f"Rotated:  {rotated}")
    
    # Test 3: Tracker
    tracker = GoldenRotationTracker()
    for i in range(5):
        tracker.record_rotation(i, f"method_{i}", 0.8 + i * 0.05)
    
    stats = tracker.get_stats()
    print(f"Tracker stats: {stats}")
    
    # Test 4: TensorRotor
    rotor = TensorRotor(10, mode="hybrid")
    print(f"Initial rotor state: {rotor.get_coverage_stats()}")
    
    for _ in range(15):
        next_index = rotor.next()
        print(f"Next index: {next_index}")
    
    print(f"Final rotor state: {rotor.get_coverage_stats()}")
    
    # Test 5: TensorPoolManager
    manager = TensorPoolManager()
    print(f"Initial pool stats: {manager.get_pool_stats()}")
    
    # 🔧 FIX: Crear DummyTensor con estructura correcta
    class DummyTensor:
        def __init__(self, tensor_id, depth="shallow"):
            self.id = tensor_id
            
            if depth == "deep":
                # Tensor con todos los niveles
                self.nivel_3 = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
                self.nivel_9 = [[i, 0, 1] for i in range(9)]
                self.nivel_27 = [[i, j, 1] for i in range(3) for j in range(9)]
            elif depth == "mid":
                # Tensor con nivel_3 y nivel_9
                self.nivel_3 = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
                self.nivel_9 = [[i, 0, 1] for i in range(9)]
                self.nivel_27 = None
            else:  # shallow
                # Tensor solo con nivel_3
                self.nivel_3 = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
                self.nivel_9 = None
                self.nivel_27 = None
    
    # Añadir tensores de diferentes profundidades
    depths = ["shallow", "mid", "deep"]
    for i in range(6):
        depth = depths[i % 3]
        tensor = DummyTensor(i, depth)
        manager.add_tensor(tensor)
        print(f"Added tensor {i} with depth {depth}")
    
    print(f"Pool stats after adding tensors: {manager.get_pool_stats()}")
    
    # Probar acceso a trios de tensores
    print("\n🔧 Testing tensor trio access:")
    for task in ["arquetipo", "dinamica", "relator", "axioma"]:
        trio = manager.get_tensor_trio(task)
        if trio:
            print(f"Trio for {task}: {[t.id for t in trio]}")
        else:
            print(f"No trio available for {task}")
    
    # Probar optimización de rotors
    print(f"\n🔧 Testing rotor optimization:")
    manager.optimize_rotors()
    final_stats = manager.get_pool_stats()
    print(f"Final pool stats: {final_stats}")
    
    # Test 6: Fibonacci sequence
    print(f"\n🔧 Testing Fibonacci sequence:")
    fib_seq = [fib(i) for i in range(12)]
    print(f"Fibonacci sequence (0-11): {fib_seq}")
    
    # Test 7: Data augmentation
    print(f"\n🔧 Testing data augmentation:")
    test_dataset = [
        {"seq": [1, 2, 3], "type": "arith"},
        {"seq": [2, 4, 8], "type": "geom"},
        {"seq": [1, 1, 2], "type": "fib"}
    ]
    
    augmented = phi_augment_dataset(test_dataset, 0.5)
    print(f"Original dataset size: {len(test_dataset)}")
    print(f"Augmented dataset size: {len(augmented)}")
    
    for i, item in enumerate(augmented):
        if i < len(test_dataset):
            print(f"Original {i}: {item}")
        else:
            print(f"Augmented {i}: {item}")
    
    print("✅ Golden rotation tests completed")
