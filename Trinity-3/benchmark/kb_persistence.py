"""
Módulo de persistencia para Knowledge Base de Aurora Trinity-3.
Maneja serialización/deserialización segura de reglas KB.
"""

import json
import pickle
import gzip
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

def sanitize_for_json(obj: Any) -> Any:
    """Convierte objetos no-JSON a formato serializable."""
    if obj is None:
        return None
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return sanitize_for_json(obj.__dict__)
    else:
        # Fallback: convertir a string
        return str(obj)

def save_kb_json(kb, filepath: str, metadata: Dict = None):
    """Guarda KB en formato JSON legible."""
    try:
        # Extraer todas las entradas
        entries = kb.all_entries() if hasattr(kb, 'all_entries') else []
        
        # Sanitizar para JSON
        clean_entries = [sanitize_for_json(entry) for entry in entries]
        
        # Preparar estructura final
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "Trinity-3",
                "total_entries": len(clean_entries),
                "format": "JSON",
                **(metadata or {})
            },
            "entries": clean_entries
        }
        
        # Guardar
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 KB guardada en JSON: {path} ({len(clean_entries)} entradas)")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando KB JSON: {e}")
        return False

def save_kb_pickle(kb, filepath: str, compressed: bool = True):
    """Guarda KB en formato pickle (con compresión opcional)."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if compressed:
            with gzip.open(path, 'wb') as f:
                pickle.dump(kb, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(kb, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        size_kb = path.stat().st_size // 1024
        print(f"💾 KB guardada en pickle: {path} ({size_kb} KB)")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando KB pickle: {e}")
        return False

def load_kb_json(filepath: str):
    """Carga KB desde formato JSON."""
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No existe: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        entries = data.get('entries', [])
        metadata = data.get('metadata', {})
        
        print(f"📖 KB cargada desde JSON: {len(entries)} entradas")
        print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")
        
        return entries, metadata
        
    except Exception as e:
        print(f"❌ Error cargando KB JSON: {e}")
        return None, None

def load_kb_pickle(filepath: str, compressed: bool = True):
    """Carga KB desde formato pickle."""
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No existe: {path}")
            
        if compressed:
            with gzip.open(path, 'rb') as f:
                kb = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                kb = pickle.load(f)
                
        print(f"📖 KB cargada desde pickle: {path}")
        return kb
        
    except Exception as e:
        print(f"❌ Error cargando KB pickle: {e}")
        return None

# Función de conveniencia
def save_kb_auto(kb, base_name: str = "kb_aurora", save_json: bool = True, save_pickle: bool = True):
    """Guarda KB en múltiples formatos automáticamente."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    if save_json:
        json_path = f"{base_name}_{timestamp}.json"
        success = save_kb_json(kb, json_path, {"auto_save": True})
        results.append(("JSON", json_path, success))
        
    if save_pickle:
        pickle_path = f"{base_name}_{timestamp}.pkl.gz"
        success = save_kb_pickle(kb, pickle_path, compressed=True)
        results.append(("Pickle", pickle_path, success))
        
    return results

if __name__ == "__main__":
    print("🧪 Test de persistencia KB")
    
    # Simular KB para testing
    class MockKB:
        def all_entries(self):
            return [
                {"M_emergent": [1, 0, 0], "MetaM": [1, 1, 0], "R_validos": [[1, 0, None]]},
                {"M_emergent": [0, 1, 0], "MetaM": [1, 1, 1], "R_validos": [[2, 0, None]]},
            ]
    
    kb = MockKB()
    results = save_kb_auto(kb)
    
    for format_type, path, success in results:
        print(f"   {format_type}: {'✅' if success else '❌'} {path}")
