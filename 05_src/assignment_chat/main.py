from pathlib import Path

from langgraph.graph import StateGraph, MessagesState, START
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
import os

from assignment_chat.prompts import return_instructions
from assignment_chat.weather_service import get_weather
from assignment_chat.semantic_search_service import search_knowledge_base
from assignment_chat.tools_service import (
    calculate_statistics,
    convert_temperature,
    generate_random_fact
)
from utils.logger import get_logger


_logs = get_logger(__name__)

env_dir = Path(__file__).resolve().parent
load_dotenv(env_dir / ".env")
load_dotenv(env_dir / ".secrets")

api_gateway_key = os.getenv("API_GATEWAY_KEY")
if not api_gateway_key:
    raise ValueError(
        "Missing API_GATEWAY_KEY environment variable. "
        "Set API_GATEWAY_KEY in .env, .secrets, or export it before running the app."
    )

chat_agent = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    default_headers={"x-api-key": api_gateway_key},
)

# Define available tools
tools = [
    get_weather,
    search_knowledge_base,
    calculate_statistics,
    convert_temperature,
    generate_random_fact
]

# Get system instructions
instructions = return_instructions()


def call_model(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    response = chat_agent.bind_tools(tools).invoke(
        [SystemMessage(content=instructions)] + state["messages"]
    )
    return {
        "messages": [response]
    }


def get_graph():
    """Build and return the LangGraph state graph"""
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph
