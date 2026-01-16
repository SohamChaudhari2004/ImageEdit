"""
Agentic Image Editor - Main Entry Point

A LangGraph-powered image editing system with multi-agent orchestration.
"""

import argparse
import sys
from pathlib import Path

import config
from graph.workflow import create_workflow
from utils.image_utils import validate_image, get_image_info


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered image editor using LangGraph agents"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--instruction", "-t",
        required=True,
        help="Natural language editing instruction"
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=config.MAX_RETRIES,
        help=f"Maximum retry attempts (default: {config.MAX_RETRIES})"
    )
    
    args = parser.parse_args()
    
    # Validate input image
    image_path = str(Path(args.image).absolute())
    is_valid, error = validate_image(image_path)
    
    if not is_valid:
        print(f"Error: {error}")
        sys.exit(1)
    
    # Check API key
    if not config.GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)
    
    # Print info
    info = get_image_info(image_path)
    print(f"\n{'='*60}")
    print("AGENTIC IMAGE EDITOR")
    print(f"{'='*60}")
    print(f"Input: {args.image}")
    print(f"Size: {info['width']}x{info['height']} ({info['format']})")
    print(f"Instruction: {args.instruction}")
    print(f"Max retries: {args.max_retries}")
    print(f"{'='*60}")
    
    # Create and run workflow
    workflow = create_workflow()
    
    initial_state = {
        "image_path": image_path,
        "instruction": args.instruction,
        "understanding": None,
        "edit_steps": None,
        "command": None,
        "output_path": None,
        "execution_success": None,
        "execution_error": None,
        "verified": None,
        "verification_feedback": None,
        "attempt": 0,
        "max_attempts": args.max_retries,
        "error": None,
        "completed": False
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    
    if result.get("error"):
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    if result.get("output_path") and Path(result["output_path"]).exists():
        print(f"Success! Output saved to: {result['output_path']}")
        print(f"Attempts: {result['attempt'] + 1}")
        if result.get("verification_feedback"):
            print(f"Verification: {result['verification_feedback']}")
    else:
        print("Failed to generate output")
        if result.get("execution_error"):
            print(f"Execution error: {result['execution_error']}")
        if result.get("verification_feedback"):
            print(f"Verification feedback: {result['verification_feedback']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
