"""
Streamlit Application for Agentic Image Editor

A beautiful web interface for the LangGraph-powered image editing system.
"""

import streamlit as st
from pathlib import Path
import tempfile
import shutil
from PIL import Image

import config
from graph.workflow import create_workflow
from utils.image_utils import validate_image


# Page config
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b6d4fe;
    }
    .agent-step {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        background: #f8f9fa;
        border-radius: 0 0.25rem 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "workflow" not in st.session_state:
        st.session_state.workflow = create_workflow()
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "result" not in st.session_state:
        st.session_state.result = None
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory."""
    file_path = Path(st.session_state.temp_dir) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def run_workflow(image_path: str, instruction: str) -> dict:
    """Run the LangGraph workflow."""
    initial_state = {
        "image_path": image_path,
        "instruction": instruction,
        # Image Analyzer
        "image_description": None,
        "areas_of_focus": None,
        "suggested_adjustments": None,
        "technical_notes": None,
        # Query Planner
        "understanding": None,
        "edit_steps": None,
        "command": None,
        "output_path": None,
        "execution_success": None,
        "execution_error": None,
        "verified": None,
        "verification_feedback": None,
        "attempt": 0,
        "max_attempts": config.MAX_RETRIES,
        "error": None,
        "completed": False
    }
    return st.session_state.workflow.invoke(initial_state)


def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Editor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by LangGraph • Describe your edit in natural language</p>', unsafe_allow_html=True)
    
    # Check API key
    if not config.GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not set. Please set it in your environment variables.")
        st.code("set GROQ_API_KEY=your_key_here", language="bash")
        return
    
    # Main layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📤 Upload & Instruct")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            help="Upload the image you want to edit"
        )
        
        # Instruction input
        instruction = st.text_area(
            "Editing instruction",
            placeholder="e.g., 'Make it look cinematic with warm tones' or 'Increase brightness and add vignette'",
            height=100
        )
        
        # Example prompts
        with st.expander("💡 Example instructions"):
            examples = [
                "Make it look cinematic",
                "Add a vintage film look",
                "Make the colors more vibrant",
                "Convert to black and white with high contrast",
                "Add warm sunset tones",
                "Make it brighter and sharper",
                "Apply a cool blue mood",
                "Add a slight blur and vignette"
            ]
            for example in examples:
                if st.button(example, key=example):
                    st.session_state.example_instruction = example
                    st.rerun()
        
        # Use example if clicked
        if "example_instruction" in st.session_state:
            instruction = st.session_state.example_instruction
            del st.session_state.example_instruction
        
        # Process button
        if st.button("✨ Apply Edit", type="primary"):
            if not uploaded_file:
                st.warning("Please upload an image first")
            elif not instruction.strip():
                st.warning("Please enter an editing instruction")
            else:
                st.session_state.processing = True
                st.session_state.result = None
        
        # Show original image
        if uploaded_file:
            st.subheader("Original Image")
            st.image(uploaded_file, use_column_width=True)
    
    with col2:
        st.subheader("🖼️ Result")
        
        # Process the image
        if st.session_state.processing and uploaded_file:
            with st.spinner("Processing..."):
                # Save uploaded file
                image_path = save_uploaded_file(uploaded_file)
                
                # Status container
                status_container = st.container()
                
                with status_container:
                    st.markdown('<div class="info-box status-box">🔄 Running agent pipeline...</div>', unsafe_allow_html=True)
                    
                    # Progress steps
                    progress_placeholder = st.empty()
                    
                    progress_placeholder.markdown("""
                    <div class="agent-step">📋 <b>Query Planner</b> - Analyzing instruction...</div>
                    """, unsafe_allow_html=True)
                
                # Run workflow
                try:
                    result = run_workflow(image_path, instruction)
                    st.session_state.result = result
                    st.session_state.processing = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.processing = False
        
        # Show result
        if st.session_state.result:
            result = st.session_state.result
            
            if result.get("error"):
                st.error(f"❌ Error: {result['error']}")
            elif result.get("output_path") and Path(result["output_path"]).exists():
                st.success("✅ Edit complete!")
                
                # Show edited image
                st.image(result["output_path"], use_column_width=True)
                
                # Download button
                with open(result["output_path"], "rb") as f:
                    st.download_button(
                        "📥 Download Edited Image",
                        f,
                        file_name=f"edited_{Path(result['output_path']).name}",
                        mime="image/png"
                    )
                
                # Details expander
                with st.expander("📊 Processing Details"):
                    st.markdown(f"**Attempts:** {result.get('attempt', 0) + 1}")
                    
                    # Image Analysis section
                    if result.get("image_description"):
                        st.markdown("---")
                        st.markdown("### 🔍 Image Analysis")
                        st.markdown(f"**Description:** {result['image_description']}")
                        
                        if result.get("areas_of_focus"):
                            st.markdown("**Areas of Focus:**")
                            for area in result["areas_of_focus"]:
                                st.markdown(f"- {area}")
                        
                        if result.get("suggested_adjustments"):
                            st.markdown("**Suggested Adjustments:**")
                            for adj in result["suggested_adjustments"]:
                                st.markdown(f"- {adj}")
                        
                        if result.get("technical_notes"):
                            st.markdown(f"**Technical Notes:** {result['technical_notes']}")
                    
                    # Planning section
                    if result.get("understanding"):
                        st.markdown("---")
                        st.markdown("### 📋 Planning")
                        st.markdown(f"**Understanding:** {result['understanding']}")
                    
                    if result.get("edit_steps"):
                        st.markdown("**Edit Steps:**")
                        for step in result["edit_steps"]:
                            st.markdown(f"- `{step.get('operation', 'unknown')}`")
                    
                    if result.get("command"):
                        st.markdown("---")
                        st.markdown("### ⚙️ Execution")
                        st.markdown("**FFmpeg Command:**")
                        st.code(result["command"], language="bash")
                    
                    if result.get("verification_feedback"):
                        st.markdown("---")
                        st.markdown("### ✅ Verification")
                        st.markdown(f"**Result:** {result['verification_feedback']}")
            else:
                st.warning("⚠️ Processing completed but output file not found")
                if result.get("execution_error"):
                    st.error(f"Execution error: {result['execution_error']}")
        else:
            st.info("👆 Upload an image and describe your edit to get started")


if __name__ == "__main__":
    main()
