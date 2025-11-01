import click
from pathlib import Path
from datetime import datetime
from jinja2 import Template
from dataflow_agent.logger import get_logger
log = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

# ---------- util ----------
def to_snake(s: str) -> str:
    import re
    s = re.sub(r'[\- ]+', '_', s).strip('_')
    parts = re.split(r'[_]', s)
    return '_'.join(p.lower() for p in parts if p)

def to_camel(s: str) -> str:
    return ''.join(p.capitalize() for p in to_snake(s).split('_'))

# ---------- CLI ----------
@click.group()
def cli():
    """DataFlow-Agent command line."""
    pass


@cli.command("create")
@click.option("--wf_name",      help="要创建的 workflow 名称")
@click.option("--agent_name",   help="要创建的 agent 名称")
@click.option("--gradio_name",  help="要创建的 gradio page 名称")
@click.option("--prompt_name",  help="要创建的 prompt template 名称")
def create_artifact(wf_name: str | None,
                    agent_name: str | None,
                    gradio_name: str | None,
                    prompt_name: str | None):
    """
    dfa create --wf_name xxx
    dfa create --agent_name yyy
    dfa create --gradio_name zzz
    dfa create --prompt_name zzz
    """
    opts = [bool(wf_name), bool(agent_name), bool(gradio_name), bool(prompt_name)]
    if sum(opts) != 1:
        click.echo(" --wf_name / --agent_name / --gradio_name / --prompt_name 必须且只能选一个", err=True)
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # 1. Workflow
    # ------------------------------------------------------------------
    if wf_name:
        wf_name_snake = to_snake(wf_name)

        # 1.1 workflow 源码
        wf_dest      = Path(__file__).parent / "workflow" / f"wf_{wf_name_snake}.py"
        wf_tpl_path  = TEMPLATE_DIR / "workflow.py.jinja"
        wf_context   = dict(
            wf_name=wf_name,
            wf_name_snake=wf_name_snake,
            entry="step1",
            timestamp=timestamp,
        )
        _generate_file(wf_dest, wf_tpl_path, wf_context, "workflow")

        # 1.2 对应测试
        project_root = Path(__file__).parent.parent
        test_dest    = project_root / "tests" / f"test_{wf_name_snake}.py"
        test_tpl     = TEMPLATE_DIR / "test_workflow.py.jinja"
        test_ctx     = dict(
            wf_name=wf_name,
            wf_name_snake=wf_name_snake,
            timestamp=timestamp,
        )
        _generate_file(test_dest, test_tpl, test_ctx, "test")

    # ------------------------------------------------------------------
    # 2. Agent
    # ------------------------------------------------------------------
    elif agent_name:
        agent_snake = to_snake(agent_name)
        dest        = Path(__file__).parent / "agentroles" / f"{agent_snake}_agent.py"
        tpl_path    = TEMPLATE_DIR / "agent.py.jinja"
        ctx         = dict(
            agent_name=agent_name,
            agent_name_snake=agent_snake,
            agent_name_camel=to_camel(agent_name),
            timestamp=timestamp,
        )
        _generate_file(dest, tpl_path, ctx, "agent")

    # ------------------------------------------------------------------
    # 3. Gradio Page
    # ------------------------------------------------------------------
    elif gradio_name:
        page_snake   = to_snake(gradio_name)
        project_root = Path(__file__).parent.parent
        dest         = project_root / "gradio_app" / "pages" / f"page_{page_snake}.py"
        tpl_path     = TEMPLATE_DIR / "gradio_page.py.jinja"
        ctx          = dict(
            page_name=gradio_name,
            page_name_snake=page_snake,
            timestamp=timestamp,
        )
        _generate_file(dest, tpl_path, ctx, "gradio page")

    # ------------------------------------------------------------------
    # 4. Prompt Template
    # ------------------------------------------------------------------
    else:
        prompt_snake = to_snake(prompt_name)
        dest         = Path(__file__).parent / "promptstemplates" / "resources" / f"pt_{prompt_snake}_repo.py"
        tpl_path     = TEMPLATE_DIR / "prompt_repo.py.jinja"
        ctx          = dict(
            prompt_name=prompt_name,
            prompt_name_snake=prompt_snake,
            prompt_name_camel=to_camel(prompt_name),
            timestamp=timestamp,
        )
        _generate_file(dest, tpl_path, ctx, "prompt template")


# ---------- helper ----------
def _generate_file(dest: Path, tpl_path: Path, context: dict, file_type: str):
    """
    通用文件生成函数
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        # click.echo(f"  {dest} 已存在，跳过生成")
        log.error(f"  {dest} 已存在，跳过生成")
        return

    if not tpl_path.exists():
        # click.echo(f" 模板不存在: {tpl_path}", err=True)
        log.error(f" 模板不存在: {tpl_path}")
        raise SystemExit(1)

    rendered = Template(tpl_path.read_text(encoding="utf-8")).render(**context)
    dest.write_text(rendered, encoding="utf-8")

    try:
        rel_path = dest.relative_to(Path.cwd())
    except ValueError:
        rel_path = dest

    # click.echo(f" 已生成 {file_type}: {rel_path}")
    log.critical(f'生成模板文件：{file_type}: {rel_path}')


if __name__ == "__main__":
    cli()