import os
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client

from google.adk.telemetry.setup import maybe_set_otel_providers

# Initialisation globale
maybe_set_otel_providers()

app = BedrockAgentCoreApp()
log = app.logger

APP_NAME = "ADKvTROIS"

# https://google.github.io/adk-docs/agents/models/
MODEL_ID = "gemini-2.5-flash"


# Define a simple function tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers"""
    return a + b


# Get MCP Toolset
mcp_client = get_streamable_http_mcp_client()
mcp_toolset = [mcp_client] if mcp_client else []

_credentials_loaded = False

def ensure_credentials_loaded():
    global _credentials_loaded
    if not _credentials_loaded:
        load_model()
        _credentials_loaded = True


##########################
# sous agent pour les produits et les politiques de retour
##########################

PRODUCTS = {
    "PROD-001": {"name": "Wireless Headphones", "price": 79.99, "category": "audio",
                 "description": "Noise-cancelling Bluetooth headphones with 30h battery life", "warranty_months": 12},
    "PROD-002": {"name": "Smart Watch", "price": 249.99, "category": "electronics",
                 "description": "Fitness tracker with heart rate monitor, GPS, and 5-day battery", "warranty_months": 24},
    "PROD-003": {"name": "Laptop Stand", "price": 39.99, "category": "accessories",
                 "description": "Adjustable aluminum laptop stand for ergonomic desk setup", "warranty_months": 6},
    "PROD-004": {"name": "USB-C Hub", "price": 54.99, "category": "accessories",
                 "description": "7-in-1 USB-C hub with HDMI, USB-A, SD card reader, and ethernet", "warranty_months": 12},
    "PROD-005": {"name": "Mechanical Keyboard", "price": 129.99, "category": "electronics",
                 "description": "RGB mechanical keyboard with Cherry MX switches", "warranty_months": 24},
}


PRODUCTS_KB = "\n".join(
    [
        (
            f"- {product_id}: {product['name']} | ${product['price']} | "
            f"category={product['category']} | {product['description']} | "
            f"warranty={product['warranty_months']} months"
        )
        for product_id, product in PRODUCTS.items()
    ]
)


# Sub-Agent Definition
get_product_info = Agent(
    model=MODEL_ID,
    name="get_product_info",
    description="Specialist for product search by ID, name, keyword, or category.",
    instruction=(
        "You answer only product information questions. "
        "Use only the following product catalog and do not invent products.\n\n"
        f"{PRODUCTS_KB}\n\n"
        "If no match exists, reply with this exact format and replace <query> "
        "with the user's query value: "
        "No products found matching '<query>'."
    ),
)


# Agent Definition
agent = Agent(
    model=MODEL_ID,
    name="ADKvTROIS",
    description="Agent to answer questions",
    instruction="I can answer your questions using the knowledge I have!",
    tools=mcp_toolset + [add_numbers],
    sub_agents=[get_product_info],
)


# Session and Runner
async def setup_session_and_runner(user_id, session_id):
    ensure_credentials_loaded()
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


# Agent Interaction
async def call_agent_async(query, user_id, session_id):
    content = types.Content(role="user", parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner(user_id, session_id)
    events = runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    )

    final_response = None
    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text

    return final_response


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")

    # Process the user prompt
    prompt = payload.get("prompt", "What can you help me with?")
    session_id = getattr(context, "session_id", "default_session")
    user_id = payload.get("user_id", "default_user")

    # Run the agent
    result = await call_agent_async(prompt, user_id, session_id)

    # Return result
    return {"result": result}


if __name__ == "__main__":
    app.run()
