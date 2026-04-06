import asyncio

from fastapi_app.schemas import Paper2FigureRequest
from fastapi_app.workflow_adapters.wa_paper2figure import run_paper2figure_wf_api


def test_run_paper2figure_returns_failure_when_no_output_files(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    (project_root / "dataflow_agent" / "tmps").mkdir(parents=True)
    result_root = project_root / "outputs" / "paper2tec" / "case1"
    result_root.mkdir(parents=True)

    async def fake_run_workflow(_wf_name, state):
        state.ppt_path = ""
        state.svg_file_path = ""
        state.svg_img_path = ""
        state.svg_bw_file_path = ""
        state.svg_bw_img_path = ""
        state.svg_color_file_path = ""
        state.svg_color_img_path = ""
        state.output_xml_path = ""
        state.drawio_output_path = ""
        return state

    monkeypatch.setattr(
        "fastapi_app.workflow_adapters.wa_paper2figure.get_project_root",
        lambda: project_root,
    )
    monkeypatch.setattr(
        "fastapi_app.workflow_adapters.wa_paper2figure.run_workflow",
        fake_run_workflow,
    )

    req = Paper2FigureRequest(
        input_type="TEXT",
        input_content="test idea",
        graph_type="tech_route",
    )

    resp = asyncio.run(run_paper2figure_wf_api(req, result_path=result_root))

    assert resp.success is False
    assert resp.error == "生成失败：后端未产出有效文件，请检查后端日志。"
    assert resp.all_output_files == []


def test_run_paper2figure_returns_success_when_svg_exists(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    (project_root / "dataflow_agent" / "tmps").mkdir(parents=True)
    result_root = project_root / "outputs" / "paper2tec" / "case2"
    result_root.mkdir(parents=True)
    svg_path = result_root / "roadmap.svg"
    svg_path.write_text("<svg></svg>", encoding="utf-8")

    async def fake_run_workflow(_wf_name, state):
        state.svg_file_path = str(svg_path)
        state.svg_img_path = ""
        state.svg_bw_file_path = ""
        state.svg_bw_img_path = ""
        state.svg_color_file_path = ""
        state.svg_color_img_path = ""
        state.ppt_path = ""
        state.output_xml_path = ""
        state.drawio_output_path = ""
        return state

    monkeypatch.setattr(
        "fastapi_app.workflow_adapters.wa_paper2figure.get_project_root",
        lambda: project_root,
    )
    monkeypatch.setattr(
        "fastapi_app.workflow_adapters.wa_paper2figure.run_workflow",
        fake_run_workflow,
    )

    req = Paper2FigureRequest(
        input_type="TEXT",
        input_content="test idea",
        graph_type="tech_route",
    )

    resp = asyncio.run(run_paper2figure_wf_api(req, result_path=result_root))

    assert resp.success is True
    assert resp.svg_filename == str(svg_path)
    assert str(svg_path) in resp.all_output_files
