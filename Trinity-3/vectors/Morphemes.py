# vectors/Morphemes.py

# Este diccionario contendrá los vectores para morfemas comunes del español.
# La idea es que estos vectores se SUMEN al vector del lexema para modificarlo.
# Por ejemplo, un morfema de plural podría incrementar la dimensión de "cantidad".
# Un morfema de pasado podría afectar la dimensión de "tiempo".

# NOTA: Estos vectores son ilustrativos. Un sistema completo requeriría un
# diseño y ajuste cuidadoso basado en los principios de Aurora.

MORPHEMES = {
    # Morfemas de GÉNERO (afectan Arquetipo)
    'a': [ # Femenino
        0,0,0, 0,0,0, 1,0,0, # Arquetipo: añade cualidad "femenina"
        0,0,0, 0,0,0, 0,0,0, # Relator
        0,0,0, 0,0,0, 0,0,0  # Dinámica
    ],

    # Morfemas de NÚMERO (afectan Relator)
    's': [ # Plural
        0,0,0, 0,0,0, 0,0,0, # Arquetipo
        1,0,0, 0,0,0, 0,0,0, # Relator: Dimensión de Cantidad/Multiplicidad
        0,0,0, 0,0,0, 0,0,0  # Dinámica
    ],
    'es': [ # Plural
        0,0,0, 0,0,0, 0,0,0,
        1,0,0, 0,0,0, 0,0,0, # Igual que 's'
        0,0,0, 0,0,0, 0,0,0
    ],

    # Morfemas de TIEMPO/ASPECTO VERBAL (afectan Dinámica)
    # Usaremos las últimas 3 componentes para [Pasado, Presente/Continuo, Futuro]
    'ando': [ # Gerundio (acción en progreso)
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,1,0
    ],
    'iendo': [ # Gerundio
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,1,0
    ],
    'ado': [ # Participio (pasado, estado resultante)
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 1,0,0
    ],
    'ido': [ # Participio
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 1,0,0
    ],
    'é': [ # Pretérito (pasado)
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 2,0,0 # más fuerte que el participio
    ],
    'aste': [ # Pretérito (pasado)
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 2,0,0
    ],
    'ó': [ # Pretérito (pasado)
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 2,0,0
    ],
    'aremos': [ # Futuro
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,0,
        0,0,0, 0,0,0, 0,0,2
    ],
}
