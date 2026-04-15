import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import get_streamable_http_mcp_client

app = BedrockAgentCoreApp()
log = app.logger

APP_NAME = "CustomerSupportADK"

# https://google.github.io/adk-docs/agents/models/
MODEL_ID = "gemini-2.5-flash"
_credentials_loaded = False

def ensure_credentials_loaded():
    global _credentials_loaded
    if not _credentials_loaded:
        load_model()
        _credentials_loaded = True



# --- Product & Policy Data ---

RETURN_POLICIES = {
    "electronics": {
        "window": "30 days",
        "condition": "Original packaging required, must be unused or defective",
        "refund": "Full refund to original payment method",
    },
    "accessories": {
        "window": "14 days",
        "condition": "Must be in original packaging, unused",
        "refund": "Store credit or exchange",
    },
    "audio": {
        "window": "30 days",
        "condition": "Defective items only after 15 days",
        "refund": "Full refund within 15 days, replacement after",
    },
}

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

# --- Sub-Agents ---

RETURN_POLICY_KB = "\n".join(
    [
        f"- {category}: window={policy['window']}; condition={policy['condition']}; refund={policy['refund']}"
        for category, policy in RETURN_POLICIES.items()
    ]
)

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

# Required symbol: get_return_policy is now an ADK sub-agent.
get_return_policy = LlmAgent(
    model=MODEL_ID,
    name="get_return_policy",
    description="Specialist for product return policy questions by category.",
    instruction=(
        "You answer only return-policy questions. "
        "Use only the following policy catalog and do not invent data.\n\n"
        f"{RETURN_POLICY_KB}\n\n"
        "If the category is unknown, reply with this exact format and replace <category> "
        "with the user's category value: "
        "No specific return policy found for '<category>'. Please contact support."
    ),
)

# Required symbol: get_product_info is now an ADK sub-agent.
get_product_info = LlmAgent(
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

# --- Agent Setup ---

SYSTEM_PROMPT = """You are a helpful and professional customer support assistant for an e-commerce company.

Your role is to:
- Provide accurate information using the sub-agents available to you
- Be friendly, patient, and understanding with customers
- Always offer additional help after answering questions

Always delegate the request to the appropriate sub-agent rather than guessing."""

# Agent Definition
root_agent = LlmAgent(
    model=MODEL_ID,
    name="CustomerSupportADK",
    description="Root customer support agent.",
    instruction=SYSTEM_PROMPT,
    sub_agents=[get_return_policy, get_product_info],
)


# Session and Runner
async def setup_session_and_runner(user_id, session_id):
    ensure_credentials_loaded()
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
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
    try:
        log.info("Invoking Google ADK customer-support agent...")
        ...
        result = await call_agent_async(prompt, user_id, session_id)
        return {"result": result}
    except Exception as e:
        log.exception("Fatal error in invoke")
        raise


if __name__ == "__main__":
    app.run()
