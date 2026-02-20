import operator
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
import cv2
import numpy as np

# --- 1. Define State ---
class AgentState(TypedDict):
    # 'messages' will accumulate history across rounds due to operator.add
    messages: Annotated[List[BaseMessage], operator.add]
    user_text: str
    user_image: Optional[str]
    knowledge_output: Optional[str]
    vision_output: Optional[str]
    final_response: str

# --- 2. Define Nodes ---

def decision_agent(state: AgentState):
    """
    Decides routing based on LATEST input and context.
    """
    # Get the latest human message if available, or fallback to user_text
    # In a real app, you might look at conversation history to resolve pronouns (e.g., "What is *it*?")
    latest_text = state.get("user_text", "")
    current_image = state.get("user_image")
    
    # Analyze history length to show memory usage
    history_count = len(state.get("messages", []))
    print(f"\n--- [Decision Agent] Turn {history_count // 2 + 1} | Analyzing: '{latest_text}' ---")

    routes = []
    
    # Logic: If image is provided in THIS turn, use Vision.
    if current_image:
        routes.append("vision_agent")
    
    # Always check knowledge for text
    if latest_text:
        routes.append("knowledge_agent")
        
    return {"messages": [SystemMessage(content=f"Routing decision: {routes}")]}

def knowledge_agent(state: AgentState):
    print("--- [Knowledge Agent] Active ---")
    # Simulate recalling previous context
    history = state["messages"]
    query = state["user_text"]
    
    # Mock Logic: If user says "What about X?", we assume they might refer to previous context
    context_note = ""
    if len(history) > 2:
        context_note = "(Considering previous conversation context...)"
        
    return {"knowledge_output": f"{context_note} Medical guidelines for '{query}'..."}

def vision_agent(state: AgentState):
    print("--- [Vision Agent] Active ---")
    return {"vision_output": "Image analysis detected: [Condition Y]"}

def synthesizer_agent(state: AgentState):
    print("--- [Synthesizer Agent] Active ---")
    k_out = state.get("knowledge_output")
    v_out = state.get("vision_output")
    
    # Reset outputs in state so they don't persist cleanly to next turn 
    # (Optional, depending on if you want agents to remember previous outputs explicitly)
    
    final_msg = f"Response: {k_out or ''} {v_out or ''}"
    
    return {
        "final_response": final_msg,
        # Append the AI response to the message history
        "messages": [AIMessage(content=final_msg)]
    }

# --- 3. Routing Logic ---

def route_decision(state: AgentState) -> List[str]:
    routes = []
    if state.get("user_image"):
        routes.append("vision_agent")
    if state.get("user_text"):
        routes.append("knowledge_agent")
    return routes

# --- 4. Build Graph with Checkpointer ---

# Initialize Memory
memory = MemorySaver()

workflow = StateGraph(AgentState)

workflow.add_node("decision_agent", decision_agent)
workflow.add_node("knowledge_agent", knowledge_agent)
workflow.add_node("vision_agent", vision_agent)
workflow.add_node("synthesizer_agent", synthesizer_agent)

workflow.set_entry_point("decision_agent")

workflow.add_conditional_edges(
    "decision_agent",
    route_decision,
    {
        "knowledge_agent": "knowledge_agent",
        "vision_agent": "vision_agent"
    }
)

workflow.add_edge("knowledge_agent", "synthesizer_agent")
workflow.add_edge("vision_agent", "synthesizer_agent")
workflow.add_edge("synthesizer_agent", END)

# Compile with the checkpointer
app = workflow.compile(checkpointer=memory) # <--- NEW: Bind memory here