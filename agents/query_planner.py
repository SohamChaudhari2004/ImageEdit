"""
Query Planner Agent

Decomposes high-level user instructions into specific, actionable edit operations.
Example: "cinematic look" → ["increase_contrast", "warm_color_grade", "add_vignette"]
"""

from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import config


SYSTEM_PROMPT = """You are an expert image editing planner. Your job is to decompose user instructions into specific, technical image editing operations.

Given a user's request, output a JSON object with:
1. "understanding": A brief explanation of what the user wants
2. "edit_steps": A list of specific editing operations to achieve the goal

AVAILABLE EDIT OPERATIONS:
- brightness_adjust: Increase/decrease brightness (specify: "increase" or "decrease", amount: "slight", "moderate", "strong")
- contrast_adjust: Increase/decrease contrast
- saturation_adjust: Increase/decrease color saturation  
- color_temperature: Warm or cool the image (specify: "warm" or "cool")
- color_grade: Apply color grading (specify style: "cinematic", "vintage", "vibrant", etc.)
- vignette: Add dark edges vignette effect
- blur: Apply blur (specify: "gaussian", "motion", amount)
- sharpen: Sharpen the image
- grayscale: Convert to black and white
- sepia: Apply sepia tone
- crop: Crop the image (specify region or aspect ratio)
- resize: Resize image (specify dimensions or scale)
- rotate: Rotate image (specify degrees)
- flip: Flip image (specify: "horizontal" or "vertical")
- noise_reduction: Reduce noise/grain
- exposure_adjust: Adjust exposure

EXAMPLE INPUT: "make it look cinematic"
EXAMPLE OUTPUT:
{{
  "understanding": "User wants a cinematic film look with enhanced contrast, warm tones, and vignette",
  "edit_steps": [
    {{"operation": "contrast_adjust", "params": {{"direction": "increase", "amount": "moderate"}}}},
    {{"operation": "color_temperature", "params": {{"direction": "warm", "amount": "slight"}}}},
    {{"operation": "saturation_adjust", "params": {{"direction": "decrease", "amount": "slight"}}}},
    {{"operation": "vignette", "params": {{"intensity": "moderate"}}}}
  ]
}}

Always respond with valid JSON only."""


class QueryPlannerAgent:
    """Decomposes user instructions into edit operations."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=0.2
        )
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Plan the edits for this instruction: {instruction}")
        ])
        self.chain = self.prompt | self.llm | self.parser
    
    def plan(self, instruction: str) -> dict:
        """
        Decompose instruction into edit steps.
        
        Args:
            instruction: User's editing instruction
            
        Returns:
            Dict with 'understanding' and 'edit_steps'
        """
        result = self.chain.invoke({"instruction": instruction})
        return result
    
    def plan_with_feedback(self, instruction: str, previous_plan: dict, feedback: str) -> dict:
        """
        Re-plan with feedback from failed verification.
        
        Args:
            instruction: Original instruction
            previous_plan: Plan that didn't work
            feedback: Why it failed
            
        Returns:
            Updated plan
        """
        feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Plan the edits for this instruction: {instruction}"),
            ("assistant", "{previous_plan}"),
            ("human", "The previous plan didn't achieve the desired result. Feedback: {feedback}\nPlease create an improved plan.")
        ])
        
        chain = feedback_prompt | self.llm | self.parser
        result = chain.invoke({
            "instruction": instruction,
            "previous_plan": str(previous_plan),
            "feedback": feedback
        })
        return result
