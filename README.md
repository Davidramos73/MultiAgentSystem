# Sistema Multi-Agente Básico (Didáctico)

Un ejemplo educativo para aprender los conceptos fundamentales de agentes con IA.

## Conceptos Clave

### 1. ¿Qué es un Agente?

Un **agente** es un programa que usa un LLM (como GPT) para:
- Recibir una tarea
- **Decidir** qué acciones tomar
- Ejecutar esas acciones
- Devolver un resultado

La diferencia con un chatbot normal es que el agente puede **actuar** en el mundo (usar herramientas, llamar APIs, etc.).

### 2. ¿Qué son las Herramientas (Tools)?

Las herramientas son **funciones** que el agente puede usar. Por ejemplo:
- Calculadora
- Obtener la hora
- Buscar en internet
- Leer/escribir archivos

El LLM **NO ejecuta código**. Solo decide qué herramienta usar y con qué parámetros. Nosotros ejecutamos la herramienta y le devolvemos el resultado.

### 3. ¿Qué es Function Calling?

Es la capacidad del LLM para decir: "Quiero usar la función X con estos parámetros".

```
Usuario: "¿Cuánto es 5 + 3?"

LLM decide: usar calculadora(operacion="sumar", a=5, b=3)

Nosotros ejecutamos: calculadora("sumar", 5, 3) → "8"

LLM responde: "El resultado de 5 + 3 es 8"
```

### 4. ¿Qué es un Sistema Multi-Agente?

Es cuando tienes **varios agentes** que colaboran:

```
┌─────────────────────────────────────┐
│        AGENTE ORQUESTADOR           │
│  (decide qué hacer y a quién pedir) │
└─────────────────────────────────────┘
         │         │         │         │
         ▼         ▼         ▼         ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Experto │ │Experto │ │Experto │ │Experto │
   │  Math  │ │Escritura│ │ Código │ │ Cocina │
   └────────┘ └────────┘ └────────┘ └────────┘
```

### 5. Historial Persistente

Los agentes mantienen un **historial de conversación** que persiste entre mensajes. Esto permite conversaciones multi-turno donde el agente recuerda el contexto:

```
Usuario: "Dame una receta de canelones"
Experto: "¿Para cuántas personas?"
Usuario: "Para 10"                      ← El experto recuerda que hablaban de canelones
Experto: "¿Qué tipo de cocina?"
Usuario: "Italiana"                     ← El experto recuerda todo el contexto
Experto: [Da la receta completa]
```

## Estructura del Proyecto

```
Orchestrator/
├── main.py          # Punto de entrada, loop de interacción
├── agents.py        # Definición de agentes (orquestador + expertos)
├── tools.py         # Herramientas disponibles (calculadora, hora, calorías)
├── llm_client.py    # Capa de abstracción para LLMs (OpenAI, Gemini)
├── requirements.txt # Dependencias
├── .env.example     # Ejemplo de configuración
└── README.md        # Este archivo
```

### Descripción de Archivos

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Loop principal de chat, carga configuración del `.env` |
| `agents.py` | `AgenteOrquestador` + 4 expertos (matemáticas, escritura, código, cocina) |
| `tools.py` | Funciones Python ejecutables + definiciones para el LLM |
| `llm_client.py` | Abstracción que permite usar OpenAI o Gemini de forma intercambiable |

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar API key
cp .env.example .env
# Editar .env con tu configuración
```

### Configuración del `.env`

```bash
# Elegir proveedor: "openai" o "gemini"
LLM_PROVIDER=openai

# API Key según el proveedor elegido
OPENAI_API_KEY=sk-...      # Si usas OpenAI
GOOGLE_API_KEY=AIza...     # Si usas Gemini
```

### Modelos Disponibles

| Proveedor | Modelos |
|-----------|---------|
| OpenAI | `gpt-4o-mini` (default), `gpt-4o`, `gpt-4-turbo` |
| Gemini | `gemini-1.5-flash` (default), `gemini-1.5-pro`, `gemini-2.0-flash` |

## Uso

```bash
# Modo interactivo
python main.py

# Modo demostración (ejemplos predefinidos)
python main.py demo
```

## Herramientas Disponibles

| Herramienta | Descripción | Ejemplo de uso |
|-------------|-------------|----------------|
| `calculadora` | Operaciones matemáticas básicas | `sumar`, `restar`, `multiplicar`, `dividir` |
| `obtener_hora` | Fecha y hora actual del sistema | Sin parámetros |
| `calcular_calorias` | Análisis nutricional de ingredientes | Lista de ingredientes con gramos |
| `consultar_agente_experto` | Delega a un experto especializado | `matematicas`, `escritura`, `codigo`, `cocina` |

## El Loop del Agente

Este es el patrón fundamental que usan todos los agentes:

```python
while True:
    # 1. Enviar mensaje al LLM con herramientas disponibles
    respuesta = llm.chat(mensajes, tools=herramientas)

    # 2. ¿El LLM quiere usar herramientas?
    if respuesta.tool_calls:
        for tool_call in respuesta.tool_calls:
            # 3. Ejecutar la herramienta
            resultado = ejecutar(tool_call)
            # 4. Agregar resultado al historial
            mensajes.append(resultado)
        # Volver al paso 1
    else:
        # 5. El LLM tiene respuesta final
        return respuesta.content
```

## Próximos Pasos

Una vez entiendas este ejemplo, puedes explorar:

1. **MCP (Model Context Protocol)**: Estándar para conectar herramientas a LLMs
2. **LangChain / LangGraph**: Frameworks para construir agentes complejos
3. **CrewAI**: Framework para sistemas multi-agente
4. **Autogen**: Framework de Microsoft para agentes conversacionales

## Ejemplos de Preguntas

### Herramientas directas
- `"¿Cuánto es 25 * 4?"` → Usa `calculadora`
- `"¿Qué hora es?"` → Usa `obtener_hora`
- `"Calcula las calorías de 200g de pollo"` → Usa `calcular_calorias`

### Delegación a expertos
- `"Explícame qué es una derivada"` → Experto matemáticas
- `"Escribe un haiku sobre Python"` → Experto escritura
- `"¿Cómo hago un try/catch en Python?"` → Experto código
- `"Dame una receta de pasta para 4 personas"` → Experto cocina (flujo conversacional)
