"""
Agent State Definition

TypedDict defining the state that flows through the LangGraph workflow.
"""

from typing import TypedDict, Optional, List, Any


class AgentState(TypedDict):
    """State passed between agents in the workflow."""
    
    # Input
    image_path: str
    instruction: str
    
    # Image Analyzer output
    image_description: Optional[str]
    areas_of_focus: Optional[List[str]]
    suggested_adjustments: Optional[List[str]]
    technical_notes: Optional[str]
    
    # Query Planner output
    understanding: Optional[str]
    edit_steps: Optional[List[dict]]
    
    # Command Generator output  
    command: Optional[str]
    
    # Executor output
    output_path: Optional[str]
    execution_success: Optional[bool]
    execution_error: Optional[str]
    
    # Verifier output
    verified: Optional[bool]
    verification_feedback: Optional[str]
    
    # Workflow control
    attempt: int
    max_attempts: int
    error: Optional[str]
    completed: bool
