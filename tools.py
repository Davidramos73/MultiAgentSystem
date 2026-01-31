"""
=============================================================================
HERRAMIENTAS (TOOLS)
=============================================================================

Las herramientas son funciones que el agente puede llamar para realizar
acciones en el mundo real. El agente NO ejecuta código directamente,
solo decide QUÉ herramienta usar y con qué parámetros.

Conceptos clave:
- Cada herramienta tiene un NOMBRE, DESCRIPCIÓN y PARÁMETROS
- El LLM lee las descripciones para decidir cuál usar
- Nosotros ejecutamos la herramienta y devolvemos el resultado al LLM
"""

import json
from datetime import datetime

# =============================================================================
# DEFINICIÓN DE HERRAMIENTAS
# =============================================================================
# Este es el "catálogo" que le mostramos al LLM.
# El formato sigue el estándar de OpenAI para function calling.

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "calculadora",
            "description": "Realiza operaciones matemáticas básicas. Usa esta herramienta cuando el usuario pida calcular algo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operacion": {
                        "type": "string",
                        "enum": ["sumar", "restar", "multiplicar", "dividir"],
                        "description": "La operación matemática a realizar"
                    },
                    "a": {
                        "type": "number",
                        "description": "Primer número"
                    },
                    "b": {
                        "type": "number",
                        "description": "Segundo número"
                    }
                },
                "required": ["operacion", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_hora",
            "description": "Obtiene la fecha y hora actual. Usa esta herramienta cuando pregunten qué hora es o qué día es.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_agente_experto",
            "description": "Consulta a un agente especializado para tareas complejas. Úsalo cuando necesites ayuda experta en un tema específico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tipo_experto": {
                        "type": "string",
                        "enum": ["matematicas", "escritura", "codigo", "cocina"],
                        "description": "Tipo de experto a consultar: matematicas, escritura, codigo, cocina"
                    },
                    "consulta": {
                        "type": "string",
                        "description": "La pregunta o tarea para el experto"
                    }
                },
                "required": ["tipo_experto", "consulta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_calorias",
            "description": "Calcula las calorías aproximadas de una receta o lista de ingredientes. Usa esta herramienta para dar información nutricional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredientes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nombre": {"type": "string", "description": "Nombre del ingrediente"},
                                "cantidad_gramos": {"type": "number", "description": "Cantidad en gramos"}
                            },
                            "required": ["nombre", "cantidad_gramos"]
                        },
                        "description": "Lista de ingredientes con sus cantidades en gramos"
                    },
                    "num_porciones": {
                        "type": "number",
                        "description": "Número de porciones en las que se divide la receta"
                    }
                },
                "required": ["ingredientes", "num_porciones"]
            }
        }
    }
]


# =============================================================================
# IMPLEMENTACIÓN DE HERRAMIENTAS
# =============================================================================
# Aquí están las funciones reales que se ejecutan cuando el agente
# decide usar una herramienta.

def calculadora(operacion: str, a: float, b: float) -> str:
    """Ejecuta una operación matemática básica."""
    operaciones = {
        "sumar": lambda x, y: x + y,
        "restar": lambda x, y: x - y,
        "multiplicar": lambda x, y: x * y,
        "dividir": lambda x, y: x / y if y != 0 else "Error: división por cero"
    }

    if operacion not in operaciones:
        return f"Error: operación '{operacion}' no reconocida"

    resultado = operaciones[operacion](a, b)
    return f"El resultado de {a} {operacion} {b} = {resultado}"


def obtener_hora() -> str:
    """Devuelve la fecha y hora actual."""
    ahora = datetime.now()
    return f"Fecha y hora actual: {ahora.strftime('%Y-%m-%d %H:%M:%S')}"


def calcular_calorias(ingredientes: list[dict], num_porciones: int) -> str:
    """
    Calcula las calorías aproximadas de una receta.

    Base de datos simplificada de calorías por 100g.
    En un sistema real, esto se conectaría a una API nutricional.
    """
    # Base de datos simplificada (calorías por 100g)
    calorias_por_100g = {
        # Proteínas
        "pollo": 165, "pechuga de pollo": 165, "carne": 250, "carne de res": 250,
        "cerdo": 242, "pescado": 120, "salmon": 208, "atun": 130, "huevo": 155,
        "tofu": 76, "jamon": 145,
        # Carbohidratos
        "arroz": 130, "pasta": 131, "pan": 265, "papa": 77, "patata": 77,
        "quinoa": 120, "avena": 389, "tortilla": 218, "harina": 364,
        # Vegetales
        "tomate": 18, "cebolla": 40, "ajo": 149, "zanahoria": 41,
        "brocoli": 34, "espinaca": 23, "lechuga": 15, "pimiento": 31,
        "champiñon": 22, "calabacin": 17, "berenjena": 25,
        # Lácteos
        "leche": 42, "queso": 402, "crema": 340, "yogur": 59, "mantequilla": 717,
        # Aceites y grasas
        "aceite": 884, "aceite de oliva": 884, "mayonesa": 680,
        # Legumbres
        "frijoles": 347, "lentejas": 116, "garbanzos": 164,
        # Frutas
        "manzana": 52, "platano": 89, "naranja": 47, "limon": 29,
        # Otros
        "azucar": 387, "sal": 0, "pimienta": 251, "chocolate": 546,
    }

    total_calorias = 0
    detalles = []

    for ing in ingredientes:
        nombre = ing["nombre"].lower()
        gramos = ing["cantidad_gramos"]

        # Buscar coincidencia parcial en la base de datos
        calorias_base = None
        for key, cal in calorias_por_100g.items():
            if key in nombre or nombre in key:
                calorias_base = cal
                break

        if calorias_base is None:
            calorias_base = 100  # Valor por defecto si no se encuentra
            detalles.append(f"  - {nombre}: {gramos}g → ~{int(gramos * calorias_base / 100)} kcal (estimado)")
        else:
            calorias_ing = gramos * calorias_base / 100
            total_calorias += calorias_ing
            detalles.append(f"  - {nombre}: {gramos}g → {int(calorias_ing)} kcal")

    calorias_por_porcion = total_calorias / num_porciones if num_porciones > 0 else total_calorias

    resultado = f"""📊 ANÁLISIS NUTRICIONAL
{'='*40}
Desglose por ingrediente:
{chr(10).join(detalles)}

{'='*40}
🔥 Calorías totales: {int(total_calorias)} kcal
🍽️  Porciones: {num_porciones}
📍 Calorías por porción: {int(calorias_por_porcion)} kcal
"""
    return resultado


# =============================================================================
# DISPATCHER DE HERRAMIENTAS
# =============================================================================
# Esta función recibe el nombre de la herramienta y sus argumentos,
# y ejecuta la función correspondiente.

def ejecutar_herramienta(nombre: str, argumentos: dict) -> str:
    """
    Ejecuta una herramienta por su nombre.

    Args:
        nombre: Nombre de la herramienta (debe coincidir con TOOLS_DEFINITION)
        argumentos: Diccionario con los parámetros de la herramienta

    Returns:
        Resultado de la ejecución como string
    """
    print(f"  🔧 Ejecutando herramienta: {nombre}")
    print(f"     Argumentos: {argumentos}")

    if nombre == "calculadora":
        return calculadora(**argumentos)

    elif nombre == "obtener_hora":
        return obtener_hora()

    elif nombre == "calcular_calorias":
        return calcular_calorias(**argumentos)

    elif nombre == "consultar_agente_experto":
        # Esta herramienta es especial: llama a otro agente
        # La implementación está en agents.py
        return None  # Señal de que necesitamos llamar a un sub-agente

    else:
        return f"Error: herramienta '{nombre}' no encontrada"
