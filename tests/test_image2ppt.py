"""
Test for image2ppt workflow

- Input: Image file (e.g., tests/test_02.png)
- Workflow: dataflow_agent.workflow.wf_image2ppt.create_image2ppt_graph
"""

import asyncio
import pytest
import os
from pathlib import Path

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root

# Use a test image available in the repo
TEST_IMG_PATH = Path(get_project_root()) / "tests" / "test_02.png"

async def run_image2ppt_pipeline() -> Paper2FigureState:
    """
    Execute image2ppt workflow test
    """
    if not TEST_IMG_PATH.exists():
        print(f"Warning: Test image not found at {TEST_IMG_PATH}, trying alternative...")
        # Fallback to creating a dummy image if needed, or use another one
        # For now, let's assume one exists or fail gracefully
        pass

    req = Paper2FigureRequest()
    req.input_type = "FIGURE" # Single image mode
    req.input_content = str(TEST_IMG_PATH)
    req.chat_api_url = "https://api.apiyi.com/v1" # Or local/env config
    req.gen_fig_model = "gemini-3-pro-image-preview" 
    
    # Ensure API key is present (usually from env)
    if not os.getenv("DF_API_KEY"):
        print("Warning: DF_API_KEY not set in environment. VLM/AI Edit steps might fail.")

    state = Paper2FigureState(
        messages=[],
        agent_results={},
        request=req,
    )

    print(f"Starting image2ppt workflow with input: {req.input_content}")
    final_state: Paper2FigureState = await run_workflow("image2ppt", state)
    
    return final_state

@pytest.mark.asyncio
async def test_image2ppt_pipeline():
    if not TEST_IMG_PATH.exists():
        pytest.skip(f"Test image not found: {TEST_IMG_PATH}")

    final_state = await run_image2ppt_pipeline()

    assert final_state is not None, "final_state should not be None"
    assert hasattr(final_state, "ppt_path"), "state should contain ppt_path"
    
    ppt_path = getattr(final_state, "ppt_path", None)
    print(f"\n=== Generated PPT Path ===\n{ppt_path}")
    
    if ppt_path:
        assert os.path.exists(ppt_path), f"PPT file does not exist: {ppt_path}"

if __name__ == "__main__":
    # Create a dummy image if it doesn't exist for manual run
    if not TEST_IMG_PATH.exists():
        from PIL import Image
        img = Image.new('RGB', (800, 600), color = 'white')
        # Add some text to be detected
        import cv2
        img_np = np.array(img)
        cv2.putText(img_np, 'Hello World', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        Image.fromarray(img_np).save(str(TEST_IMG_PATH))
        print(f"Created dummy test image at {TEST_IMG_PATH}")
    
    import numpy as np # Re-import if needed for the block above in actual execution context
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_image2ppt_pipeline())
