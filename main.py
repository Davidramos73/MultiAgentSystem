"""
=============================================================================
SISTEMA MULTI-AGENTE BÁSICO - PUNTO DE ENTRADA
=============================================================================

Este es un ejemplo didáctico de un sistema con:
- Un agente orquestador que decide qué hacer
- Herramientas (funciones) que el agente puede usar
- Sub-agentes especializados a los que puede delegar

Cómo funciona:
1. El usuario hace una pregunta
2. El orquestador analiza y decide:
   - ¿Puedo responder directamente?
   - ¿Necesito usar una herramienta (calculadora, hora)?
   - ¿Necesito consultar a un experto?
3. Ejecuta las acciones necesarias
4. Formula y devuelve la respuesta

Para ejecutar:
    python main.py

Requisitos:
    - pip install openai python-dotenv google-generativeai
    - Crear archivo .env con la configuración (ver .env.example)

Configuración:
    LLM_PROVIDER=openai  # o "gemini"
    OPENAI_API_KEY=...   # si usas OpenAI
    GOOGLE_API_KEY=...   # si usas Gemini
"""

import os
from dotenv import load_dotenv
from agents import AgenteOrquestador
from llm_client import crear_cliente, MODELOS_DISPONIBLES


def obtener_cliente():
    """
    Crea el cliente LLM basándose en la configuración del .env

    Returns:
        Tuple de (cliente, nombre_proveedor)
    """
    proveedor = os.getenv("LLM_PROVIDER", "openai").lower()

    if proveedor == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: No se encontró OPENAI_API_KEY")
            print("   Crea un archivo .env con: OPENAI_API_KEY=tu-api-key")
            return None, None
    elif proveedor == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Error: No se encontró GOOGLE_API_KEY")
            print("   Crea un archivo .env con: GOOGLE_API_KEY=tu-api-key")
            return None, None
    else:
        print(f"❌ Error: Proveedor '{proveedor}' no soportado")
        print(f"   Opciones: openai, gemini")
        return None, None

    return crear_cliente(proveedor, api_key), proveedor


def main():
    # Cargar variables de entorno desde .env
    load_dotenv()

    # Crear cliente según configuración
    client, proveedor = obtener_cliente()
    if not client:
        return

    # Crear el agente orquestador
    orquestador = AgenteOrquestador(client)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SISTEMA MULTI-AGENTE DIDÁCTICO                     ║
╠══════════════════════════════════════════════════════════════╣
║  Proveedor LLM: {proveedor.upper():<44} ║
║  Modelos disponibles: {', '.join(MODELOS_DISPONIBLES[proveedor][:2]):<38} ║
╠══════════════════════════════════════════════════════════════╣
║  El orquestador puede:                                       ║
║  • Usar la calculadora (suma, resta, etc.)                   ║
║  • Obtener la hora actual                                    ║
║  • Calcular calorías de recetas                              ║
║  • Consultar expertos (matemáticas, escritura, código,       ║
║    cocina)                                                   ║
║                                                              ║
║  Ejemplos de preguntas:                                      ║
║  • "¿Cuánto es 25 * 4?"                                      ║
║  • "¿Qué hora es?"                                           ║
║  • "Dame una receta de pasta"                                ║
║  • "Explícame qué es una derivada"                           ║
║  • "¿Cómo hago un bucle for en Python?"                      ║
║                                                              ║
║  Escribe 'salir' para terminar                               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Loop principal de interacción
    while True:
        print("\n" + "─" * 60)
        mensaje = input("📝 Tu pregunta: ").strip()

        if mensaje.lower() in ["salir", "exit", "quit"]:
            print("\n👋 ¡Hasta luego!")
            break

        if not mensaje:
            continue

        try:
            # Procesar el mensaje con el orquestador
            respuesta = orquestador.procesar(mensaje, verbose=True)
        except Exception as e:
            print(f"\n❌ Error: {e}")


def demo():
    """
    Función de demostración con ejemplos predefinidos.
    Útil para entender el flujo sin interacción manual.
    """
    load_dotenv()

    client, proveedor = obtener_cliente()
    if not client:
        return

    orquestador = AgenteOrquestador(client)

    ejemplos = [
        "¿Cuánto es 15 + 27?",
        "¿Qué hora es?",
        "Necesito que me expliques qué es la recursión en programación",
        "Ayúdame a redactar un email profesional para pedir vacaciones",
    ]

    print("\n🎬 MODO DEMOSTRACIÓN")
    print("=" * 60)

    for ejemplo in ejemplos:
        print(f"\n\n{'🔸' * 30}")
        orquestador.procesar(ejemplo, verbose=True)
        print(f"{'🔸' * 30}")
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
