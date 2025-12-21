from pathlib import Path
import json
import os

from fastapi.testclient import TestClient

from fastapi_app.main import app
import os
from dataflow_agent.utils import get_project_root   

client = TestClient(app)


def _base_form() -> dict:
    """
    这里配置你常用的公共入参，后面 3 个测试都会复用。
    你只需要改这里的值即可。
    """
    return {
        "chat_api_url": "https://api.apiyi.com/v1",
        "api_key": f"{os.environ['DF_API_KEY']}",
        "model": "gpt-5.1",
        "language": "zh",
        "style": "多啦A梦风格；英文；",
        "gen_fig_model": "gemini-2.5-flash-image-preview",
        "page_count": 2,
        "invite_code": "ABC123"
    }


def _base_form_ppt_json() -> dict:
    """
    /paper2ppt/ppt_json 公共入参
    """
    return {
        "img_gen_model_name": "gemini-2.5-flash-image-preview",
        "chat_api_url": "https://api.apiyi.com/v1",
        "api_key": f"{os.environ['DF_API_KEY']}",
        "model": "gpt-5.1",
        "language": "zh",
        "style": "多啦A梦风格；英文；",
        "aspect_ratio": "16:9",
        "invite_code": "ABC123",
    }


def test_call_pagecontent_text():
    """
    input_type = text 的调试用例。
    你只需要修改 _base_form() 和这里的 text 内容。
    """
    data = _base_form()
    data.update(
        {
            "input_type": "text",
            "text": "this is a test text for paper2ppt",
        }
    )

    resp = client.post(
        "/paper2ppt/pagecontent_json",
        data=data,
    )
    print("==== text mode status ====")
    print(resp.status_code)
    print("==== text mode body ====")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def test_call_pagecontent_pdf():
    """
    input_type = pdf 的调试用例。
    只要保证 pdf_path 指向一个存在的 pdf 文件即可。
    """
    data = _base_form()
    data.update(
        {
            "input_type": "pdf",
        }
    )

    # 使用 dev/DataFlow-Agent/tests 里的示例 pdf，按需要调整路径
    pdf_path = Path("dev/DataFlow-Agent/tests/2506.02454v1.pdf")
    assert pdf_path.exists(), f"pdf file not found: {pdf_path}"

    with pdf_path.open("rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        resp = client.post(
            "/paper2ppt/pagecontent_json",
            data=data,
            files=files,
        )

    print("==== pdf mode status ====")
    print(resp.status_code)
    print("==== pdf mode body ====")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def test_call_pagecontent_pptx():
    """
    input_type = pptx 的调试用例。
    需要你准备一个测试用 pptx 文件，并把路径改到 pptx_path。
    """
    data = _base_form()
    data.update(
        {
            "input_type": "pptx",
        }
    )

    # 自己准备一个 .pptx 放在指定位置，然后修改下面的路径
    pptx_path = Path(f"{get_project_root()}/tests/paper2ppt_editable.pptx")
    assert pptx_path.exists(), f"pptx file not found: {pptx_path}"

    with pptx_path.open("rb") as f:
        files = {
            "file": (
                "test.pptx",
                f,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        }
        resp = client.post(
            "/paper2ppt/pagecontent_json",
            data=data,
            files=files,
        )

    print("==== pptx mode status ====")
    print(resp.status_code)
    print("==== pptx mode body ====")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def test_ppt_json_with_direct_image_pagecontent():
    """
    测试 /paper2ppt/ppt_json 场景：
    - pagecontent 为直接图片路径（dict 中包含 ppt_img_path）
    - get_down = False（首次生成）
    """
    data = _base_form_ppt_json()

    # 注意：这里的 result_path 和图片路径示例需要你自己保证存在
    result_path = f"{get_project_root()}/outputs/ABC123/paper2ppt/1766070298"
    data["result_path"] = result_path

    pagecontent = [
        {"ppt_img_path": str(Path(result_path) / "ppt_images" / "slide_000.png")},
        {"ppt_img_path": str(Path(result_path) / "ppt_images" / "slide_001.png")},
    ]
    data["pagecontent"] = json.dumps(pagecontent, ensure_ascii=False)

    # get_down = False：生成模式
    data["get_down"] = "false"

    resp = client.post(
        "/paper2ppt/ppt_json",
        data=data,
    )

    print("==== ppt_json direct image status ====")
    print(resp.status_code)
    print("==== ppt_json direct image body ====")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def test_ppt_json_with_structured_pagecontent():
    """
    测试 /paper2ppt/ppt_json 场景：
    - pagecontent 为结构化内容（title/layout_description/key_points/asset_ref）
    - get_down = False（首次生成）
    """
    data = _base_form_ppt_json()

    result_path = f"{get_project_root()}/outputs/ABC123/paper2ppt/1766067301"
    data["result_path"] = result_path

    pagecontent = [
        {
            "title": "Multimodal DeepResearcher：从零生成文本-图表交织报告的智能框架",
            "layout_description": "全版居中排版，上方为大标题，下方两行小号文字分别展示作者单位与汇报人信息，不添加其他内容和图表。",
            "key_points": [
                "论文题目：Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework",
                "作者与单位：Zhaorui Yang 等，State Key Lab of CAD&CG, Zhejiang University",
                "汇报人：XXX",
            ],
            "asset_ref": None,
        },
        {
            "title": "方法概览：Formal Description of Visualization 与 Multimodal DeepResearcher",
            "layout_description": "左侧用要点概述研究任务、FDV核心思想、四阶段agent框架；右侧自上而下放两幅示意图：上方为整体框架图（Figure 2），下方为FDV文本化与重建示意图（Figure 3），两图等宽居中排列。",
            "key_points": [
                "研究任务：给定主题和带图表的专家报告示例，从零生成文本-图表交织的多模态研究报告。",
                "核心挑战：自动设计信息丰富的可视化并与长文本报告紧密结合，缺乏标准化的文本-图表交织表示形式。",
                "FDV（Formal Description of Visualization）：受Grammar of Graphics启发，从整体布局、绘图尺度、数据、marks四个视角，用结构化文本完整描述任意可视化。",
                "示例文本化流程：从人类专家报告中抽取图表→用多模态LLM生成FDV→用代码从FDV重建图表，实现高保真“图转文”。",
                "Multimodal DeepResearcher四阶段agent框架：①迭代Researching（检索+推理）；②Exemplar Textualization（图表FDV化）；③Planning（报告大纲+可视化风格指南）；④Multimodal Report Generation（草稿生成+基于D3绘图+actor-critic迭代优化）。",
                "可视化生成细节：使用D3实现高表达力的图表规格；借助浏览器控制台错误与多模态LLM视觉反馈进行最多3轮代码修正；最终由critic从候选版本中选择最佳图表。",
                "实验与效果：构建MultimodalReportBench（100个真实世界话题+5个评价指标）；在相同Claude 3.7 Sonnet条件下，相比改造后的DataNarrative基线总体胜率达到82%。",
            ],
            "asset_ref": (
                "images/98925d41396b1c5db17882d7a83faf7af0d896c6f655d6ca0e3838fc7c65d1ab.jpg,"
                "images/46f46d81324259498bf3cd7e63831f7074eac0f0b7dd8b6bd0350debf22344e7.jpg"
            ),
        },
        {
            "title": "致谢",
            "layout_description": "标题置于顶部居中，主体区域采用居中单栏布局，列出对导师、合作者及机构的致谢文字，不放图片或表格。",
            "key_points": [
                "感谢论文作者及其团队在多模态深度研究与可视化生成方向上的工作，为本次汇报提供了重要参考。",
                "感谢所在课题组/实验室及合作单位在研究过程中给予的支持与帮助。",
                "感谢各位老师和同学的聆听与宝贵意见。",
            ],
            "asset_ref": None,
        },
    ]
    data["pagecontent"] = json.dumps(pagecontent, ensure_ascii=False)
    data["get_down"] = "false"

    resp = client.post(
        "/paper2ppt/ppt_json",
        data=data,
    )

    print("==== ppt_json structured status ====")
    print(resp.status_code)
    print("==== ppt_json structured body ====")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def test_ppt_json_edit_mode():
    """
    测试 get_down = True 的编辑模式：
    - 需要 page_id + edit_prompt
    - pagecontent 中要能找出 old_path（generated_img_path 或 ppt_img_path）
    """
    data = _base_form_ppt_json()

    result_path = f"{get_project_root()}/outputs/ABC123/paper2ppt/1766070298"
    data["result_path"] = result_path

    data["get_down"] = "true"
    data["page_id"] = 0
    data["edit_prompt"] = "请把这一页的配色改成赛博朋克主题风格"

    resp = client.post(
        "/paper2ppt/ppt_json",
        data=data,
    )

    print("==== ppt_json edit mode status ====")
    print(resp.status_code)
    print("==== ppt_json edit mode body ====")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def test_pdf2ppt_generate():
    """
    测试 /api/pdf2ppt/generate 接口：

    - 上传一个 PDF（使用项目内置示例）
    - 期望返回 200 且响应体为 PPTX 二进制内容
    """
    pdf_path = Path(f"{get_project_root()}/outputs/ABC123/paper2ppt/1766067301/paper2ppt.pdf")
    assert pdf_path.exists(), f"pdf file not found: {pdf_path}"

    data = {
        "invite_code": "ABC123",
    }

    with pdf_path.open("rb") as f:
        files = {
            "pdf_file": ("test.pdf", f, "application/pdf"),
        }
        resp = client.post(
            "/api/pdf2ppt/generate",
            data=data,
            files=files,
        )

    print("==== /api/pdf2ppt/generate status ====")
    print(resp.status_code)
    print("==== /api/pdf2ppt/generate headers ====")
    print(resp.headers)

    assert resp.status_code == 200
    assert (
        resp.headers.get("content-type")
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )

    out_path = Path("dev_2/DataFlow-Agent/tests/outputs/pdf2ppt_test_output.pptx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"saved pdf2ppt output to: {out_path}")


if __name__ == "__main__":
    # test_call_pagecontent_text()
    # test_call_pagecontent_pdf()
    # test_call_pagecontent_pptx()
    # 可以按需取消下面几行的注释进行手动调试
    # test_ppt_json_with_direct_image_pagecontent()
    # test_ppt_json_with_structured_pagecontent()
    # test_ppt_json_edit_mode()
    test_pdf2ppt_generate()
