# ==============================================================================
#  CLASE 3: KnowledgeBase (Memoria Activa del Sistema) - Sin cambios
# ==============================================================================
class KnowledgeBase:
    """
    Almacena el conocimiento validado del sistema organizado en espacios lógicos.
    Cada espacio representa un dominio de conocimiento independiente (médico, financiero, etc.)
    con sus propias reglas y correspondencias.
    """
    def __init__(self):
        # Estructura principal: diccionario de espacios
        # Cada espacio contiene su registro de axiomas y metadatos
        self.spaces = {
            "default": {
                "description": "Espacio lógico predeterminado",
                "axiom_registry": {}
            }
        }
    
    def create_space(self, name, description=""):
        """Crea un nuevo espacio lógico si no existe"""
        if name in self.spaces:
            print(f"Advertencia: El espacio '{name}' ya existe")
            return False
        
        self.spaces[name] = {
            "description": description,
            "axiom_registry": {}
        }
        print(f"Espacio '{name}' creado: {description}")
        return True
    
    def delete_space(self, name):
        """Elimina un espacio lógico existente"""
        if name not in self.spaces:
            print(f"Error: El espacio '{name}' no existe")
            return False
        
        if name == "default":
            print("Error: No se puede eliminar el espacio 'default'")
            return False
        
        del self.spaces[name]
        print(f"Espacio '{name}' eliminado")
        return True
    
    def store_axiom(self, space_name, Ms, MetaM, Ss, original_inputs):
        """
        Almacena un nuevo axioma en un espacio lógico específico.
        Verifica coherencia según el principio de correspondencia única.
        """
        # Validar existencia del espacio
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return False
        
        space = self.spaces[space_name]
        ms_key = tuple(Ms)
        
        # Verificar correspondencia única (Ms <-> MetaM)
        existing_axiom = space["axiom_registry"].get(ms_key)
        if existing_axiom and existing_axiom["MetaM"] != MetaM:
            print(f"ALERTA: Incoherencia en '{space_name}' para Ms={Ms}")
            print(f"  MetaM existente: {existing_axiom['MetaM']}")
            print(f"  MetaM nuevo:     {MetaM}")
            return False
        
        # Almacenar nuevo axioma
        space["axiom_registry"][ms_key] = {
            "MetaM": MetaM, 
            "Ss": Ss,
            "original_inputs": original_inputs
        }
        print(f"Axioma almacenado en '{space_name}' para Ms={Ms}")
        return True
    
    def get_axiom_by_ms(self, space_name, Ms):
        """Recupera un axioma de un espacio específico usando Ms como clave"""
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return None
        
        return self.spaces[space_name]["axiom_registry"].get(tuple(Ms))
    
    def get_axioms_in_space(self, space_name):
        """
        Devuelve el diccionario de axiomas de un espacio específico.
        """
        if space_name not in self.spaces:
            print(f"Error: Espacio '{space_name}' no encontrado")
            return {}
        return self.spaces[space_name]["axiom_registry"]
    
    def list_spaces(self):
        """Devuelve lista de espacios disponibles"""
        return list(self.spaces.keys())
    
    def space_stats(self, space_name):
        """Devuelve estadísticas de un espacio"""
        if space_name not in self.spaces:
            return None
        
        space = self.spaces[space_name]
        return {
            "description": space["description"],
            "axiom_count": len(space["axiom_registry"])
        }