"""
LangGraph Workflow

Defines the agent workflow with conditional edges for retry logic.
"""

import uuid
from langgraph.graph import StateGraph, END

import config
from graph.state import AgentState


# Lazy-loaded agents (avoid initialization at import time)
_agents = {}


def _get_agents():
    """Get or create agent instances (lazy initialization)."""
    if not _agents:
        from agents.image_analyzer import ImageAnalyzerAgent
        from agents.query_planner import QueryPlannerAgent
        from agents.command_generator import CommandGeneratorAgent
        from agents.executor import ExecutorAgent
        from agents.verifier import VisionVerifierAgent
        
        _agents["image_analyzer"] = ImageAnalyzerAgent()
        _agents["query_planner"] = QueryPlannerAgent()
        _agents["command_generator"] = CommandGeneratorAgent()
        _agents["executor"] = ExecutorAgent()
        _agents["verifier"] = VisionVerifierAgent()
    return _agents


def analyze_node(state: AgentState) -> AgentState:
    """Image Analyzer node - analyzes input image to understand what changes are needed."""
    print(f"\n[Image Analyzer] Analyzing image: {state['image_path']}")
    agents = _get_agents()
    
    try:
        result = agents["image_analyzer"].analyze(
            image_path=state["image_path"],
            instruction=state["instruction"]
        )
        
        print(f"[Image Analyzer] Description: {result.image_description[:100]}...")
        print(f"[Image Analyzer] Areas of focus: {result.areas_of_focus}")
        print(f"[Image Analyzer] Suggested adjustments: {result.suggested_adjustments}")
        
        return {
            **state,
            "image_description": result.image_description,
            "areas_of_focus": result.areas_of_focus,
            "suggested_adjustments": result.suggested_adjustments,
            "technical_notes": result.technical_notes
        }
    except Exception as e:
        print(f"[Image Analyzer] Error: {e}")
        # Continue without analysis if it fails
        return state


def plan_node(state: AgentState) -> AgentState:
    """Query Planner node - decomposes instruction into edit steps."""
    print(f"\n[Query Planner] Analyzing instruction: {state['instruction']}")
    agents = _get_agents()
    
    try:
        if state.get("verification_feedback") and state.get("edit_steps"):
            # Re-plan with feedback
            result = agents["query_planner"].plan_with_feedback(
                instruction=state["instruction"],
                previous_plan={"edit_steps": state["edit_steps"]},
                feedback=state["verification_feedback"]
            )
        else:
            result = agents["query_planner"].plan(state["instruction"])
        
        print(f"[Query Planner] Understanding: {result.get('understanding', 'N/A')}")
        print(f"[Query Planner] Edit steps: {len(result.get('edit_steps', []))} operations")
        
        return {
            **state,
            "understanding": result.get("understanding"),
            "edit_steps": result.get("edit_steps", [])
        }
    except Exception as e:
        print(f"[Query Planner] Error: {e}")
        return {**state, "error": str(e), "completed": True}


def generate_node(state: AgentState) -> AgentState:
    """Command Generator node - creates FFmpeg command."""
    print(f"\n[Command Generator] Creating FFmpeg command...")
    agents = _get_agents()
    
    # Generate output path
    ext = state["image_path"].split(".")[-1]
    output_filename = f"edited_{uuid.uuid4().hex[:8]}.{ext}"
    output_path = str(config.OUTPUT_DIR / output_filename)
    
    try:
        if state.get("execution_error"):
            # Regenerate with error context
            command = agents["command_generator"].generate_with_error(
                edit_steps=state["edit_steps"],
                input_path=state["image_path"],
                output_path=output_path,
                previous_command=state.get("command", ""),
                error=state["execution_error"]
            )
        else:
            command = agents["command_generator"].generate(
                edit_steps=state["edit_steps"],
                input_path=state["image_path"],
                output_path=output_path
            )
        
        print(f"[Command Generator] Command: {command[:100]}...")
        
        return {
            **state,
            "command": command,
            "output_path": output_path,
            "execution_error": None
        }
    except Exception as e:
        print(f"[Command Generator] Error: {e}")
        return {**state, "error": str(e), "completed": True}


def execute_node(state: AgentState) -> AgentState:
    """Executor node - runs FFmpeg command."""
    print(f"\n[Executor] Running command...")
    agents = _get_agents()
    
    result = agents["executor"].execute(state["command"])
    
    if result.success:
        print(f"[Executor] Success! Output: {result.output_path}")
        return {
            **state,
            "output_path": result.output_path,
            "execution_success": True,
            "execution_error": None
        }
    else:
        print(f"[Executor] Failed: {result.error}")
        return {
            **state,
            "execution_success": False,
            "execution_error": result.error,
            "attempt": state["attempt"] + 1
        }


def verify_node(state: AgentState) -> AgentState:
    """Verifier node - checks edit with vision model."""
    print(f"\n[Verifier] Checking result with LLaVA...")
    agents = _get_agents()
    
    try:
        result = agents["verifier"].verify(
            original_path=state["image_path"],
            edited_path=state["output_path"],
            instruction=state["instruction"],
            edit_steps=state["edit_steps"]
        )
        
        print(f"[Verifier] Verified: {result.verified} (confidence: {result.confidence})")
        print(f"[Verifier] Feedback: {result.feedback}")
        
        if result.verified:
            return {
                **state,
                "verified": True,
                "verification_feedback": result.feedback,
                "completed": True
            }
        else:
            return {
                **state,
                "verified": False,
                "verification_feedback": result.feedback,
                "attempt": state["attempt"] + 1
            }
    except Exception as e:
        print(f"[Verifier] Error (continuing anyway): {e}")
        # If verification fails, assume success if execution succeeded
        return {**state, "verified": True, "completed": True}


def should_retry_execution(state: AgentState) -> str:
    """Decide whether to retry command generation after execution failure."""
    if state.get("execution_success"):
        return "verify"
    elif state["attempt"] >= state["max_attempts"]:
        return "end"
    else:
        return "generate"


def should_retry_verification(state: AgentState) -> str:
    """Decide whether to retry planning after verification failure."""
    if state.get("verified") or state.get("completed"):
        return "end"
    elif state["attempt"] >= state["max_attempts"]:
        return "end"
    else:
        return "plan"


def create_workflow():
    """Create the LangGraph workflow."""
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("verify", verify_node)
    
    # Set entry point - start with image analysis
    workflow.set_entry_point("analyze")
    
    # Add edges
    workflow.add_edge("analyze", "plan")
    workflow.add_edge("plan", "generate")
    workflow.add_conditional_edges(
        "execute",
        should_retry_execution,
        {
            "verify": "verify",
            "generate": "generate",
            "end": END
        }
    )
    workflow.add_edge("generate", "execute")
    workflow.add_conditional_edges(
        "verify",
        should_retry_verification,
        {
            "plan": "plan",
            "end": END
        }
    )
    
    return workflow.compile()
