"""
Assignment 2: AI System with Conversational Interface

This module implements a multi-service chat application using LangGraph architecture with:
- Service 1: API-based service (weather)
- Service 2: Semantic search service (knowledge base)
- Service 3: Function calling / advanced tool use (calculations, conversions, facts)
"""

import sys
from pathlib import Path

# Ensure the parent src directory is on the Python path when running app.py directly
current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from assignment_chat.main import get_graph
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv
import os

from utils.logger import get_logger

_logs = get_logger(__name__)

llm = get_graph()

load_dotenv('.secrets')


def assignment_chat(message: str, history: list[dict]) -> str:
    """
    Main chat function that processes user input and generates responses.
    Uses LangGraph to intelligently route to appropriate services.
    
    Args:
        message: The user's input message
        history: Previous chat history in Gradio message format
        
    Returns:
        The assistant's response
    """
    langchain_messages = []
    n = 0
    _logs.debug(f"History: {history}")
    
    # Convert Gradio message history to LangChain format
    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
            n += 1
    
    # Add the current user message
    langchain_messages.append(HumanMessage(content=message))

    # Create state for the graph
    state = {
        "messages": langchain_messages,
        "llm_calls": n
    }

    # Invoke the graph
    response = llm.invoke(state)
    
    # Return the last message (assistant's response)
    return response['messages'][len(response['messages']) - 1].content


# Create Gradio ChatInterface
chat = gr.ChatInterface(
    fn=assignment_chat,
    title="🤖 Shweta's Multi-Service Chat Assistant",
    description="A smart assistant with weather, knowledge base, and tool access",
    examples=[
        "What's the weather in Paris?",
        "Tell me about embeddings",
        "Calculate stats for [1, 2, 3, 4, 5]",
        "Convert 100 F to C",
        "Tell me a fun fact"
    ]
)


if __name__ == "__main__":
    _logs.info('Starting Assignment Chat App...')
    chat.launch(server_name="localhost", server_port=7861)
