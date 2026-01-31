"""
=============================================================================
AGENTES
=============================================================================

Un agente es un LLM que puede:
1. Recibir una tarea
2. Decidir qué hacer (usar herramientas, pensar, delegar)
3. Ejecutar acciones
4. Devolver un resultado

Arquitectura de este ejemplo:
┌─────────────────────────────────────────────────────────────┐
│                    AGENTE ORQUESTADOR                       │
│  - Recibe la tarea del usuario                              │
│  - Decide si usar herramientas o delegar a expertos         │
│  - Coordina todo el flujo                                   │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ Experto  │        │ Experto  │        │ Experto  │
   │   Math   │        │ Escritura│        │  Código  │
   └──────────┘        └──────────┘        └──────────┘

Proveedores soportados:
- OpenAI (gpt-4o-mini, gpt-4o, etc.)
- Google Gemini (gemini-1.5-flash, gemini-1.5-pro, etc.)
"""

from tools import TOOLS_DEFINITION, ejecutar_herramienta
from llm_client import LLMClient


class AgenteBase:
    """
    Clase base para todos los agentes.

    Un agente tiene:
    - Un cliente LLM (OpenAI, Gemini, etc.)
    - Un prompt de sistema que define su personalidad/rol
    - Un modelo a usar
    """

    def __init__(self, client: LLMClient, system_prompt: str, model: str = None):
        self.client = client
        self.system_prompt = system_prompt
        self.model = model  # None = usar el default del cliente
        # Historial de conversación persistente
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def nueva_conversacion(self):
        """Reinicia el historial para empezar una nueva conversación."""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def pensar(self, mensaje: str) -> str:
        """
        Envía un mensaje al LLM y obtiene una respuesta.
        Este es el método más básico: solo texto, sin herramientas.
        """
        self.messages.append({"role": "user", "content": mensaje})
        respuesta = self.client.chat(self.messages, self.model)
        self.messages.append({"role": "assistant", "content": respuesta})
        return respuesta


class AgenteConHerramientas(AgenteBase):
    """
    Agente que puede usar herramientas específicas.

    Extiende AgenteBase agregando la capacidad de usar tools.
    Útil para agentes expertos que necesitan acceso a funciones.
    """

    def __init__(self, client: LLMClient, system_prompt: str, tools: list[dict] = None, model: str = None):
        super().__init__(client, system_prompt, model)  # Hereda self.messages del padre
        self.tools = tools or []

    def pensar(self, mensaje: str, verbose: bool = False) -> str:
        """
        Procesa un mensaje pudiendo usar herramientas.

        Similar al loop del orquestador pero específico para este agente.
        """
        # Agregar mensaje al historial persistente
        self.messages.append({"role": "user", "content": mensaje})

        # Si no tiene herramientas, usar chat simple
        if not self.tools:
            respuesta = self.client.chat(self.messages, self.model)
            self.messages.append({"role": "assistant", "content": respuesta})
            return respuesta

        # Loop con herramientas
        while True:
            response = self.client.chat_with_tools(
                messages=self.messages,
                tools=self.tools,
                model=self.model
            )

            if response.tool_calls:
                if verbose:
                    print(f"  🔧 Experto usando {len(response.tool_calls)} herramienta(s)")

                self.messages.append(self.client.create_assistant_message_with_tools(response))

                for tool_call in response.tool_calls:
                    resultado = ejecutar_herramienta(tool_call.name, tool_call.arguments)
                    if verbose:
                        print(f"     → {tool_call.name}: {resultado[:80]}...")
                    self.messages.append(
                        self.client.create_tool_result_message(
                            tool_call.id,
                            resultado,
                            tool_call.name
                        )
                    )
            else:
                # Guardar la respuesta en el historial
                self.messages.append({"role": "assistant", "content": response.content})
                return response.content


# =============================================================================
# AGENTES EXPERTOS (SUB-AGENTES)
# =============================================================================
# Estos agentes son especialistas en un tema. El orquestador les delega
# tareas específicas.

class ExpertoMatematicas(AgenteBase):
    """Agente especializado en matemáticas y razonamiento lógico."""

    def __init__(self, client: LLMClient):
        super().__init__(
            client=client,
            system_prompt="""Eres un experto en matemáticas. Tu trabajo es:
- Resolver problemas matemáticos paso a paso
- Explicar conceptos de forma clara
- Mostrar el razonamiento detrás de cada paso

Responde de forma concisa pero completa."""
        )


class ExpertoEscritura(AgenteBase):
    """Agente especializado en escritura y redacción."""

    def __init__(self, client: LLMClient):
        super().__init__(
            client=client,
            system_prompt="""Eres un experto en escritura y redacción. Tu trabajo es:
- Ayudar a redactar textos claros y efectivos
- Corregir y mejorar la gramática
- Sugerir mejores formas de expresar ideas

Responde de forma concisa pero completa."""
        )


class ExpertoCodigo(AgenteBase):
    """Agente especializado en programación."""

    def __init__(self, client: LLMClient):
        super().__init__(
            client=client,
            system_prompt="""Eres un experto en programación. Tu trabajo es:
- Escribir código limpio y bien documentado
- Explicar conceptos de programación
- Ayudar a debuggear problemas

Responde de forma concisa. Usa ejemplos de código cuando sea útil."""
        )


class ExpertoCocinero(AgenteConHerramientas):
    """
    Agente especializado en cocina con flujo conversacional.

    Este agente demuestra cómo guiar una conversación multi-turn
    siguiendo pasos específicos para completar una tarea.
    Tiene acceso a la herramienta calcular_calorias.
    """

    # Herramienta específica para el cocinero
    TOOLS_COCINERO = [
        {
            "type": "function",
            "function": {
                "name": "calcular_calorias",
                "description": "Calcula las calorías de una receta. SIEMPRE usa esta herramienta después de presentar una receta.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ingredientes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "nombre": {"type": "string"},
                                    "cantidad_gramos": {"type": "number"}
                                },
                                "required": ["nombre", "cantidad_gramos"]
                            },
                            "description": "Lista de ingredientes con cantidades en gramos"
                        },
                        "num_porciones": {
                            "type": "number",
                            "description": "Número de porciones"
                        }
                    },
                    "required": ["ingredientes", "num_porciones"]
                }
            }
        }
    ]

    def __init__(self, client: LLMClient):
        super().__init__(
            client=client,
            tools=self.TOOLS_COCINERO,
            system_prompt="""Eres un chef experto y amigable. Tu trabajo es ayudar a crear recetas personalizadas.

FLUJO DE CONVERSACIÓN - Sigue estos pasos EN ORDEN:

PASO 1: Cuando el usuario pida una receta, PRIMERO pregunta:
   "🍽️ ¡Genial! Para preparar la mejor receta, necesito saber:
    - ¿Para cuántas personas será?"

PASO 2: Después de saber las porciones, pregunta:
   "¿Qué tipo de cocina prefieres? (italiana, mexicana, asiática, casera, etc.)"

PASO 3: Con esa información, presenta la receta completa con:
   - Nombre del plato
   - Ingredientes con cantidades en gramos (esto es IMPORTANTE para calcular calorías)
   - Pasos de preparación numerados
   - Tiempo estimado de preparación

   Luego indica: "📊 Voy a calcular las calorías de esta receta..."
   Y usa la herramienta calcular_calorias con los ingredientes.

PASO 4: Después de mostrar las calorías, pregunta:
   "⚠️ ¿Algún comensal tiene alergias alimentarias o restricciones dietéticas?
    (gluten, lactosa, frutos secos, mariscos, vegetariano, vegano, etc.)"

PASO 5: Si hay alergias/restricciones:
   - Modifica la receta sustituyendo ingredientes problemáticos
   - Presenta la nueva versión adaptada
   - Recalcula las calorías con calcular_calorias

Si NO hay alergias, responde: "¡Perfecto! Tu receta está lista. ¡Buen provecho! 👨‍🍳"

REGLAS IMPORTANTES:
- Sé cálido y entusiasta, usa emojis relacionados con comida
- Siempre especifica cantidades en GRAMOS para poder calcular calorías
- No saltes pasos, sigue el flujo en orden
- Si el usuario da toda la información de una vez, adapta el flujo
- Para las calorías, SIEMPRE usa la herramienta calcular_calorias"""
        )


# =============================================================================
# AGENTE ORQUESTADOR
# =============================================================================
# Este es el agente principal. Recibe las tareas del usuario y decide
# cómo resolverlas: usando herramientas directamente o delegando a expertos.

class AgenteOrquestador:
    """
    El orquestador es el "cerebro" del sistema.

    Su flujo de trabajo:
    1. Recibe mensaje del usuario
    2. Decide si necesita herramientas o puede responder directamente
    3. Si usa herramientas, las ejecuta y procesa los resultados
    4. Si delega a un experto, le pasa la consulta y obtiene la respuesta
    5. Formula la respuesta final para el usuario
    """

    def __init__(self, client: LLMClient, model: str = None):
        self.client = client
        self.model = model  # None = usar el default del cliente
        self.system_prompt = """Eres un asistente inteligente que orquesta tareas.

Tu trabajo es:
1. Analizar lo que el usuario necesita
2. Decidir si puedes responder directamente o necesitas usar herramientas
3. Para cálculos simples, usa la calculadora
4. Para preguntas sobre fecha/hora, usa obtener_hora
5. Para calcular calorías de recetas, usa calcular_calorias
6. Para tareas complejas que requieren expertise, consulta a un agente experto:
   - matematicas: problemas matemáticos y lógica
   - escritura: redacción y gramática
   - codigo: programación y debugging
   - cocina: recetas, nutrición y consejos culinarios

IMPORTANTE:
- Usa las herramientas disponibles cuando sea apropiado
- No inventes información, usa las herramientas para obtenerla
- Para recetas y comida, SIEMPRE delega al experto de cocina
- Sé conciso en tus respuestas"""

        # Inicializar los agentes expertos
        self.expertos = {
            "matematicas": ExpertoMatematicas(client),
            "escritura": ExpertoEscritura(client),
            "codigo": ExpertoCodigo(client),
            "cocina": ExpertoCocinero(client)
        }

        # Historial de conversación persistente
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def nueva_conversacion(self):
        """Reinicia el historial para empezar una nueva conversación."""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def procesar(self, mensaje_usuario: str, verbose: bool = True) -> str:
        """
        Procesa un mensaje del usuario y devuelve una respuesta.

        Este es el "loop del agente":
        1. Enviar mensaje al LLM con las herramientas disponibles
        2. Si el LLM quiere usar herramientas, ejecutarlas
        3. Enviar los resultados al LLM
        4. Repetir hasta que el LLM dé una respuesta final

        Args:
            mensaje_usuario: Lo que el usuario quiere
            verbose: Si True, imprime información de debug

        Returns:
            La respuesta final del agente
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"📥 Usuario: {mensaje_usuario}")
            print(f"{'='*60}")

        # Agregar mensaje del usuario al historial persistente
        self.messages.append({"role": "user", "content": mensaje_usuario})

        # Loop del agente: continúa hasta obtener una respuesta final
        while True:
            if verbose:
                print("\n🤖 Consultando al LLM...")

            # Llamar al LLM con las herramientas disponibles
            response = self.client.chat_with_tools(
                messages=self.messages,
                tools=TOOLS_DEFINITION,
                model=self.model
            )

            # Caso 1: El LLM quiere usar herramientas
            if response.tool_calls:
                if verbose:
                    print(f"🔄 El LLM decidió usar {len(response.tool_calls)} herramienta(s)")

                # Agregar el mensaje del asistente al historial
                self.messages.append(self.client.create_assistant_message_with_tools(response))

                # Procesar cada llamada a herramienta
                for tool_call in response.tool_calls:
                    nombre_herramienta = tool_call.name
                    argumentos = tool_call.arguments

                    if verbose:
                        print(f"\n  📞 Llamando: {nombre_herramienta}")
                        print(f"     Args: {argumentos}")

                    # Caso especial: consultar a un agente experto
                    if nombre_herramienta == "consultar_agente_experto":
                        resultado = self._consultar_experto(
                            argumentos["tipo_experto"],
                            argumentos["consulta"],
                            verbose
                        )
                    else:
                        # Ejecutar herramienta normal
                        resultado = ejecutar_herramienta(nombre_herramienta, argumentos)

                    if verbose:
                        resultado_preview = resultado[:100] if resultado else ""
                        print(f"     ✅ Resultado: {resultado_preview}...")

                    # Agregar el resultado al historial
                    self.messages.append(
                        self.client.create_tool_result_message(
                            tool_call.id,
                            resultado,
                            nombre_herramienta
                        )
                    )

            # Caso 2: El LLM tiene una respuesta final (sin herramientas)
            else:
                respuesta_final = response.content
                # Agregar la respuesta del asistente al historial para mantener contexto
                self.messages.append({"role": "assistant", "content": respuesta_final})
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"📤 Respuesta final:")
                    print(f"{'='*60}")
                    print(respuesta_final)
                return respuesta_final

    def _consultar_experto(self, tipo_experto: str, consulta: str, verbose: bool) -> str:
        """
        Delega una consulta a un agente experto.

        Esto demuestra la comunicación entre agentes:
        El orquestador -> Experto -> Respuesta -> Orquestador
        """
        if verbose:
            print(f"\n  🎓 Delegando a experto: {tipo_experto}")
            print(f"     Consulta: {consulta}")

        if tipo_experto not in self.expertos:
            return f"Error: no existe el experto '{tipo_experto}'"

        experto = self.expertos[tipo_experto]

        # Algunos expertos (como el cocinero) pueden usar herramientas
        if isinstance(experto, AgenteConHerramientas):
            respuesta = experto.pensar(consulta, verbose=verbose)
        else:
            respuesta = experto.pensar(consulta)

        if verbose:
            print(f"     📝 Respuesta del experto recibida")

        return respuesta
