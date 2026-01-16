"""
Image Analyzer Agent

Uses vision model to analyze the input image and determine what changes
should be made and where, based on the user's instruction.
"""

import base64
from dataclasses import dataclass
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import config


@dataclass
class AnalysisResult:
    """Result of image analysis."""
    image_description: str
    areas_of_focus: list
    suggested_adjustments: list
    technical_notes: str


class ImageAnalyzerAgent:
    """Analyzes images to determine what edits are needed."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.VISION_MODEL,
            temperature=0.2
        )
    
    def analyze(self, image_path: str, instruction: str) -> AnalysisResult:
        """
        Analyze the image and instruction to determine what changes are needed.
        
        Args:
            image_path: Path to the input image
            instruction: User's editing instruction
            
        Returns:
            AnalysisResult with detailed analysis
        """
        # Encode image
        image_b64 = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)
        
        prompt = f"""Analyze this image for editing. The user wants to: "{instruction}"

Please provide:
1. IMAGE DESCRIPTION: Briefly describe what's in the image (subject, lighting, colors, composition)
2. AREAS OF FOCUS: List specific areas/elements that need to be modified to achieve the user's goal
3. SUGGESTED ADJUSTMENTS: List specific technical adjustments (brightness, contrast, color, etc.) with approximate values
4. TECHNICAL NOTES: Any important observations about the image quality, resolution, or challenges

Format your response as:
IMAGE_DESCRIPTION: <description>
AREAS_OF_FOCUS: <comma-separated list>
SUGGESTED_ADJUSTMENTS: <comma-separated list of adjustments>
TECHNICAL_NOTES: <notes>"""

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_b64}"}
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
    
    def _parse_response(self, response: str) -> AnalysisResult:
        """Parse LLM response into AnalysisResult."""
        lines = response.strip().split("\n")
        
        image_description = ""
        areas_of_focus = []
        suggested_adjustments = []
        technical_notes = ""
        
        for line in lines:
            lower = line.lower()
            if lower.startswith("image_description:"):
                image_description = line.split(":", 1)[1].strip()
            elif lower.startswith("areas_of_focus:"):
                areas_str = line.split(":", 1)[1].strip()
                areas_of_focus = [a.strip() for a in areas_str.split(",")]
            elif lower.startswith("suggested_adjustments:"):
                adj_str = line.split(":", 1)[1].strip()
                suggested_adjustments = [a.strip() for a in adj_str.split(",")]
            elif lower.startswith("technical_notes:"):
                technical_notes = line.split(":", 1)[1].strip()
        
        # Fallback if parsing failed
        if not image_description:
            image_description = response[:200]
        
        return AnalysisResult(
            image_description=image_description,
            areas_of_focus=areas_of_focus,
            suggested_adjustments=suggested_adjustments,
            technical_notes=technical_notes
        )
