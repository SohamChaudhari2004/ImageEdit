"""
Vision Verifier Agent

Uses Groq LLaVA to verify that image edits match user intent.
"""

import base64
from dataclasses import dataclass
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import config


@dataclass
class VerificationResult:
    """Result of vision verification."""
    verified: bool
    confidence: str  # high, medium, low
    feedback: str


class VisionVerifierAgent:
    """Verifies image edits using LLaVA vision model."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.VISION_MODEL,
            temperature=0.1
        )
    
    def verify(self, original_path: str, edited_path: str, 
               instruction: str, edit_steps: list) -> VerificationResult:
        """
        Verify that the edited image matches the intended edits.
        
        Args:
            original_path: Path to original image
            edited_path: Path to edited image
            instruction: User's original instruction
            edit_steps: The edit operations that were applied
            
        Returns:
            VerificationResult with verification status
        """
        # Encode edited image
        edited_b64 = self._encode_image(edited_path)
        media_type = self._get_media_type(edited_path)
        
        # Format edit steps
        steps_text = ", ".join([step["operation"] for step in edit_steps])
        
        prompt = f"""Analyze this edited image. The user requested: "{instruction}"
The following edits were applied: {steps_text}

Determine if the edits are visible and match the request.

Respond in EXACTLY this format:
VERIFIED: yes or no
CONFIDENCE: high, medium, or low
FEEDBACK: Brief explanation of what you observe"""

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{edited_b64}"}
                },
                {"type": "text", "text": prompt}
            ]
        )
        
        response = self.llm.invoke([message])
        return self._parse_response(response.content)
    
    def _encode_image(self, path: str) -> str:
        """Encode image to base64."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_media_type(self, path: str) -> str:
        """Get MIME type from extension."""
        ext = Path(path).suffix.lower()
        types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", 
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return types.get(ext, "image/jpeg")
    
    def _parse_response(self, response: str) -> VerificationResult:
        """Parse LLM response into VerificationResult."""
        lines = response.strip().split("\n")
        
        verified = False
        confidence = "low"
        feedback = response
        
        for line in lines:
            lower = line.lower()
            if lower.startswith("verified:"):
                verified = "yes" in lower
            elif lower.startswith("confidence:"):
                if "high" in lower:
                    confidence = "high"
                elif "medium" in lower:
                    confidence = "medium"
            elif lower.startswith("feedback:"):
                feedback = line.split(":", 1)[1].strip()
        
        return VerificationResult(
            verified=verified,
            confidence=confidence,
            feedback=feedback
        )
