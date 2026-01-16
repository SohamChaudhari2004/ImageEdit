"""
Command Generator Agent

Generates FFmpeg commands from edit plans created by the Query Planner.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config


SYSTEM_PROMPT = """You are an FFmpeg expert. Generate the exact FFmpeg command to apply the specified image edits.

RULES:
1. Output ONLY the FFmpeg command, no explanations
2. Use INPUT_PATH and OUTPUT_PATH as placeholders
3. Always use -y flag to overwrite without asking
4. Chain multiple filters using comma separation in a single -vf
5. Use appropriate filter values for the specified intensities

FILTER REFERENCE:
- Brightness: eq=brightness=X (range: -1.0 to 1.0, slight=±0.1, moderate=±0.2, strong=±0.3)
- Contrast: eq=contrast=X (range: 0.0 to 2.0, default=1.0, slight=1.1/0.9, moderate=1.3/0.7, strong=1.5/0.5)
- Saturation: eq=saturation=X (range: 0.0 to 3.0, default=1.0, slight=1.2/0.8, moderate=1.5/0.5)
- Warm tones: colorbalance=rs=0.1:gs=0.05:bs=-0.1 (adjust values for intensity)
- Cool tones: colorbalance=rs=-0.1:gs=0:bs=0.1
- Vignette: vignette=PI/4 (adjust angle for intensity)
- Gaussian blur: gblur=sigma=X (slight=2, moderate=5, strong=10)
- Sharpen: unsharp=5:5:X:5:5:0 (X: slight=0.5, moderate=1.0, strong=1.5)
- Grayscale: format=gray
- Sepia: colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131
- Crop: crop=w:h:x:y
- Scale: scale=W:H (use -1 to maintain aspect ratio)
- Rotate: rotate=X*PI/180
- Flip: hflip or vflip

EXAMPLE:
Edit steps: increase contrast moderate, warm color temperature slight, add vignette
Command: ffmpeg -y -i INPUT_PATH -vf "eq=contrast=1.3,colorbalance=rs=0.08:gs=0.04:bs=-0.08,vignette=PI/4" OUTPUT_PATH"""


class CommandGeneratorAgent:
    """Generates FFmpeg commands from edit plans."""
    
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=0.1
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Generate FFmpeg command for these edit steps:\n{edit_steps}")
        ])
        self.chain = self.prompt | self.llm
    
    def generate(self, edit_steps: list, input_path: str, output_path: str) -> str:
        """
        Generate FFmpeg command from edit steps.
        
        Args:
            edit_steps: List of edit operations from QueryPlanner
            input_path: Path to input image
            output_path: Path for output image
            
        Returns:
            Complete FFmpeg command string
        """
        # Format edit steps for the prompt
        steps_text = "\n".join([
            f"- {step['operation']}: {step.get('params', {})}"
            for step in edit_steps
        ])
        
        response = self.chain.invoke({"edit_steps": steps_text})
        command = response.content.strip()
        
        # Clean up command (remove markdown code blocks if present)
        if command.startswith("```"):
            lines = command.split("\n")
            command = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        
        # Replace placeholders
        command = command.replace("INPUT_PATH", f'"{input_path}"')
        command = command.replace("OUTPUT_PATH", f'"{output_path}"')
        
        return command.strip()
    
    def generate_with_error(self, edit_steps: list, input_path: str, output_path: str, 
                            previous_command: str, error: str) -> str:
        """
        Regenerate command after execution error.
        
        Args:
            edit_steps: Edit operations
            input_path: Input image path
            output_path: Output image path
            previous_command: Command that failed
            error: Error message
            
        Returns:
            Fixed FFmpeg command
        """
        error_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Generate FFmpeg command for these edit steps:\n{edit_steps}"),
            ("assistant", "{previous_command}"),
            ("human", "That command failed with error: {error}\nPlease fix it.")
        ])
        
        steps_text = "\n".join([
            f"- {step['operation']}: {step.get('params', {})}"
            for step in edit_steps
        ])
        
        chain = error_prompt | self.llm
        response = chain.invoke({
            "edit_steps": steps_text,
            "previous_command": previous_command,
            "error": error
        })
        
        command = response.content.strip()
        
        # Clean up
        if command.startswith("```"):
            lines = command.split("\n")
            command = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
        
        command = command.replace("INPUT_PATH", f'"{input_path}"')
        command = command.replace("OUTPUT_PATH", f'"{output_path}"')
        
        return command.strip()
