"""
Minimal test for wf_paper2page_content

- Input PDF: dev_2/DataFlow-Agent/tests/2506.02454v1.pdf
- Workflow: dataflow_agent.workflow.wf_paper2page_content.create_paper2page_content_graph

Notes:
- This test focuses on graph wiring + MinerU markdown reading node.
- The outline agent name used in the workflow is "paper2page_outline_agent".
  If you haven't registered that agent role yet, this test will likely fail at outline_agent step.
  You can still verify the PDF->MinerU->markdown-read stage by temporarily bypassing the outline agent.
"""

import asyncio
import pytest
from pathlib import Path

from dataflow_agent.state import Paper2FigureState, Paper2FigureRequest
from dataflow_agent.workflow import run_workflow
from dataflow_agent.utils import get_project_root

PDF_PATH = Path(__file__).resolve().parent / "2506.02454v1.pdf"


async def run_paper2page_content_pipeline_pdf() -> Paper2FigureState:
    """
    执行 paper2page_content 工作流的测试流程（PDF 输入）
    """
    req = Paper2FigureRequest()
    req.input_type = "PDF"
    req.model = "gpt-5.1"
    req.page_count = 3
    req.chat_api_url = "https://api.apiyi.com/v1"
    req.style = "北京大学风格；英文；"
    # req.gen_fig_model = "gemini-3-pro-image-preview"
# gemini-3-pro-image-preview 
    req.all_edited_down = True
    

    state = Paper2FigureState(
        messages=[],
        agent_results={},
        request=req,
        paper_file=f"{get_project_root()}/tests/2512.16676v1.pdf",
        # paper_file=f"{get_project_root()}/tests/test.pptx",   
        # result_path = f"{get_project_root()}/outputs"
    )

#     state.text_content = """
    
#     # Technical Solutions

# The technical solutions available span the entire waste management chain:

# 1. Collection and Segregation: The foundation of effective waste management begins with reliable collection services and proper waste segregation. Without these fundamentals, downstream interventions have limited impact.

# 2. Recycling Technologies:

# Mechanical Recycling: Processing plastics through sorting, cleaning, and reprocessing into new products.   
# 。 Chemical Recycling: Breaking plastic polymers down into their chemical building blocks for reuse.

# 3. Waste-to-Energy (WtE) Systems: Converting non-recyclable plastics to energy through various thermal processes.

# 4. Cleanup Technologies: Specialized equipment for removing plastic from rivers, coastlines, and the ocean.

# Recent cost-benefit analyses provide insights into the economic viability of these solutions. For instance, a study by Agori et al. (2024) in Ughelli, Nigeria, evaluated four mitigation strategies over a 0.5-year horizon at a $1 0 \%$ discount rate:

# ![](images/7e5aaba3a3a7f79a5f67b50a91ba1b63821471b1fe77016509207ade82714e2e.jpg)  
# Cost-Benefit Analysis of Plastic Waste Management Strategies in Ugheli, Nigeria   
# Collection & recycling shows highest returns among evaluated interventions

# The results show that plastic waste collection and recycling delivered the highest returns with a Cost-Benefit Ratio (CBR) of 1.50 and Net Present Value (NPV) of ₦112,500,000, followed by household waste segregation (CBR 1.35, NPV ₦80,000,000), public awareness campaigns (CBR 1.28, NPV ₦52,500,000), and deposit-refund schemes (CBR 1.08, NPV ₦25,000,000).

# In terms of waste-to-energy systems, Khwammana & Chaiyata (2025) reported on a waste-toenergy-to-zero system that uses municipal solid waste (17.85 tonnes/day at $3 1 . 6 3 \%$ combustible) to fuel a combined cooling, heating, and power plant. The system delivers 306.98 kW at $2 2 . 3 8 \%$ efficiency, yielding a levelized energy cost of 0.15 USD/kWh, NPV of 1,634,658 USD, profitability index of 1.72, internal rate of return of $7 . 9 7 \%$ , and payback period of 9.63 years.

# Clement's (2012) Fort Bliss WtE/CSP hybrid cost-benefit study shows that NPV is highly sensitive to the gap between local tariff and WtE rate. Using EPA's WARM model for 1 million tonnes/year feedstock, the Fort Bliss WtE diversion avoids approximately 264,025 MTCO2e annually. At carbon credit prices ranging from 0.10-10 USD/MTCO2e, 20-year environmental benefits range from $\$ 0.4$ million to $\$ 36.2$ million USD.
#     """

    # 对应 wf_paper2page_content.py 中的 @register("paper2page_content")
    final_state: Paper2FigureState = await run_workflow("paper2page_content", state)
    final_state = await run_workflow("paper2ppt", final_state)
    return final_state

@pytest.mark.asyncio
async def test_paper2page_content_pipeline_pdf():
    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"

    final_state = await run_paper2page_content_pipeline_pdf()

    assert final_state is not None, "final_state 不应为 None"
    assert hasattr(final_state, "minueru_output"), "state 应包含 minueru_output"
    assert isinstance(final_state.minueru_output, str)

    # -- 调试输出，可按需保留 --
    print("\n=== minueru_output (len) ===")
    print(len(getattr(final_state, "minueru_output", "") or ""))

    print("\n=== pagecontent ===")
    print(getattr(final_state, "pagecontent", None))


if __name__ == "__main__":
    asyncio.run(run_paper2page_content_pipeline_pdf())
