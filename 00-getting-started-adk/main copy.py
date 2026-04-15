"""
Customer Support Agent — Getting Started Sample

This is the agent code you'll copy into app/CustomerSupport/main.py after
running `agentcore create`. It replaces the generated sample with a customer
support agent that has two sub-agents: product lookup and return policy lookup.

Usage:
    1. agentcore create --name CustomerSupport --framework GoogleADK --model-provider Bedrock --defaults
  2. cd CustomerSupport
  3. Copy this file to app/CustomerSupport/main.py
  4. agentcore dev          # test locally
  5. agentcore deploy       # deploy to AWS
  6. agentcore invoke "What products do you have?" --stream
"""

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "CustomerSupportADK"
DEFAULT_USER_ID = "customer-support-user"
GOOGLE_MODEL_ID = os.getenv("GOOGLE_MODEL_ID", "gemini-2.5-flash")

app = BedrockAgentCoreApp()
log = app.logger
session_service = InMemorySessionService()

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
get_return_policy = Agent(
    model=GOOGLE_MODEL_ID,
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
get_product_info = Agent(
    model=GOOGLE_MODEL_ID,
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

root_agent = Agent(
    model=GOOGLE_MODEL_ID,
    name="customer_support_root",
    description="Root customer support agent.",
    instruction=SYSTEM_PROMPT,
    sub_agents=[get_return_policy, get_product_info],
)


def ensure_session(user_id: str, session_id: str) -> None:
    existing = session_service.get_session_sync(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not existing:
        session_service.create_session_sync(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Google ADK customer-support agent...")

    prompt = payload.get("prompt")
    if not prompt:
        raise KeyError("'prompt' field is required in payload")

    user_id = payload.get("user_id", DEFAULT_USER_ID)
    session_id = context.session_id
    if not session_id:
        raise Exception("Context session_id is not set")

    ensure_session(user_id=user_id, session_id=session_id)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text
            if text:
                yield text


if __name__ == "__main__":
    app.run()
