"""
=============================================================================
CAPA DE ABSTRACCIÓN PARA CLIENTES LLM
=============================================================================

Esta capa permite usar diferentes proveedores de LLM (OpenAI, Gemini, etc.)
de forma intercambiable. El usuario elige cuál usar mediante configuración.

Arquitectura:
┌─────────────────────────────────────────────────────────────┐
│                      LLMClient (ABC)                        │
│  - chat(): Envía mensajes y obtiene respuesta               │
│  - chat_with_tools(): Chat con function calling             │
└─────────────────────────────────────────────────────────────┘
              ▲                           ▲
              │                           │
     ┌────────┴────────┐         ┌────────┴────────┐
     │  OpenAIClient   │         │  GeminiClient   │
     └─────────────────┘         └─────────────────┘

Uso:
    client = crear_cliente("openai", api_key="...")
    # o
    client = crear_cliente("gemini", api_key="...")
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Representa una llamada a herramienta solicitada por el LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Respuesta unificada del LLM."""
    content: Optional[str]  # Texto de respuesta (puede ser None si hay tool_calls)
    tool_calls: list[ToolCall]  # Lista de herramientas a ejecutar
    raw_message: object  # Mensaje original del proveedor (para el historial)


class LLMClient(ABC):
    """
    Clase base abstracta para clientes de LLM.

    Define la interfaz común que deben implementar todos los proveedores.
    """

    @abstractmethod
    def chat(self, messages: list[dict], model: str) -> str:
        """
        Envía mensajes al LLM y obtiene una respuesta de texto.

        Args:
            messages: Lista de mensajes [{"role": "user", "content": "..."}]
            model: Nombre del modelo a usar

        Returns:
            Respuesta de texto del LLM
        """
        pass

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str
    ) -> LLMResponse:
        """
        Envía mensajes al LLM con herramientas disponibles.

        Args:
            messages: Lista de mensajes
            tools: Definición de herramientas (formato OpenAI)
            model: Nombre del modelo a usar

        Returns:
            LLMResponse con contenido y/o llamadas a herramientas
        """
        pass

    @abstractmethod
    def create_tool_result_message(self, tool_call_id: str, result: str, tool_name: str = None) -> dict:
        """
        Crea un mensaje con el resultado de una herramienta.

        Args:
            tool_call_id: ID de la llamada a herramienta
            result: Resultado de la ejecución
            tool_name: Nombre de la herramienta (requerido para algunos proveedores)

        Returns:
            Mensaje formateado para el proveedor
        """
        pass

    @abstractmethod
    def create_assistant_message_with_tools(self, response: 'LLMResponse') -> dict:
        """
        Crea un mensaje del asistente que incluye llamadas a herramientas.

        Args:
            response: La respuesta del LLM con tool_calls

        Returns:
            Mensaje formateado para agregar al historial
        """
        pass


# =============================================================================
# IMPLEMENTACIÓN OPENAI
# =============================================================================

class OpenAIClient(LLMClient):
    """Cliente para la API de OpenAI."""

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.default_model = "gpt-4o-mini"

    def chat(self, messages: list[dict], model: str = None) -> str:
        model = model or self.default_model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str = None
    ) -> LLMResponse:
        model = model or self.default_model
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        tool_calls = []
        if assistant_message.tool_calls:
            import json
            for tc in assistant_message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LLMResponse(
            content=assistant_message.content,
            tool_calls=tool_calls,
            raw_message=assistant_message
        )

    def create_tool_result_message(self, tool_call_id: str, result: str, tool_name: str = None) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }

    def create_assistant_message_with_tools(self, response: LLMResponse) -> dict:
        # Para OpenAI, podemos usar el mensaje raw directamente
        return response.raw_message


# =============================================================================
# IMPLEMENTACIÓN GEMINI (usando google.genai - nuevo SDK)
# =============================================================================

class GeminiClient(LLMClient):
    """Cliente para la API de Google Gemini usando el nuevo SDK google.genai."""

    def __init__(self, api_key: str):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.default_model = "gemini-2.5-flash"
        self._tool_call_counter = 0

    def _build_contents(self, messages: list[dict]) -> tuple[str, list]:
        """
        Convierte mensajes al formato del nuevo SDK de Gemini.

        Returns:
            Tuple de (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(self.types.Content(
                    role="user",
                    parts=[self.types.Part.from_text(text=content)]
                ))
            elif role == "assistant" or role == "model":
                if content:
                    contents.append(self.types.Content(
                        role="model",
                        parts=[self.types.Part.from_text(text=content)]
                    ))
            elif role == "tool":
                # Resultado de función
                contents.append(self.types.Content(
                    role="user",
                    parts=[self.types.Part.from_function_response(
                        name=msg.get("name", "tool"),
                        response={"result": content}
                    )]
                ))
            elif role == "model_function_call":
                # Llamada a función del modelo (para el historial)
                parts = []
                for fc in msg.get("function_calls", []):
                    parts.append(self.types.Part.from_function_call(
                        name=fc["name"],
                        args=fc["args"]
                    ))
                contents.append(self.types.Content(role="model", parts=parts))

        return system_instruction, contents

    def _convert_tools_to_gemini(self, tools: list[dict]) -> list:
        """Convierte definición de herramientas de OpenAI a Gemini."""
        gemini_tools = []

        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                gemini_tools.append(self.types.FunctionDeclaration(
                    name=func["name"],
                    description=func["description"],
                    parameters=func.get("parameters", {})
                ))

        return gemini_tools

    def chat(self, messages: list[dict], model: str = None) -> str:
        model_name = model or self.default_model
        system_instruction, contents = self._build_contents(messages)

        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction
        ) if system_instruction else None

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        return response.text

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str = None
    ) -> LLMResponse:
        model_name = model or self.default_model
        system_instruction, contents = self._build_contents(messages)
        gemini_tools = self._convert_tools_to_gemini(tools)

        config = self.types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[self.types.Tool(function_declarations=gemini_tools)]
        )

        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        # Procesar la respuesta
        tool_calls = []
        content = None

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    self._tool_call_counter += 1
                    tool_calls.append(ToolCall(
                        id=f"gemini_call_{self._tool_call_counter}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {}
                    ))
                elif part.text:
                    content = part.text

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_message=response
        )

    def create_tool_result_message(self, tool_call_id: str, result: str, tool_name: str = None) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
            "name": tool_name or "tool"
        }

    def create_assistant_message_with_tools(self, response: LLMResponse) -> dict:
        # Para Gemini, creamos un mensaje especial para el historial
        return {
            "role": "model_function_call",
            "function_calls": [
                {"name": tc.name, "args": tc.arguments}
                for tc in response.tool_calls
            ]
        }


# =============================================================================
# FACTORY PARA CREAR CLIENTES
# =============================================================================

def crear_cliente(proveedor: str, api_key: str) -> LLMClient:
    """
    Crea un cliente LLM según el proveedor especificado.

    Args:
        proveedor: "openai" o "gemini"
        api_key: API key del proveedor

    Returns:
        Instancia del cliente correspondiente

    Raises:
        ValueError: Si el proveedor no está soportado
    """
    proveedores = {
        "openai": OpenAIClient,
        "gemini": GeminiClient
    }

    if proveedor.lower() not in proveedores:
        raise ValueError(
            f"Proveedor '{proveedor}' no soportado. "
            f"Opciones: {list(proveedores.keys())}"
        )

    return proveedores[proveedor.lower()](api_key)


# Modelos disponibles por proveedor (para referencia)
MODELOS_DISPONIBLES = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "gemini": ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
}
