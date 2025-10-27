import click
from pathlib import Path
from datetime import datetime
from jinja2 import Template

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
@click.option("--wf_name", help="要创建的 workflow 名称")
@click.option("--agent_name", help="要创建的 agent 名称")
def create_artifact(wf_name: str | None, agent_name: str | None):
    """
    dfa create --wf_name xxx
    dfa create --agent_name yyy
    """
    if bool(wf_name) == bool(agent_name):
        click.echo(" --wf_name 与 --agent_name 必须且只能选一个", err=True)
        raise SystemExit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if wf_name:  # ---------------- workflow ----------------
        wf_name_snake = to_snake(wf_name)
        dest = Path(__file__).parent / "workflow" / f"wf_{wf_name_snake}.py"
        tpl_path = TEMPLATE_DIR / "workflow.py.jinja"
        context = dict(
            wf_name=wf_name,
            wf_name_snake=wf_name_snake,
            entry="step1",
            timestamp=timestamp,
        )
    else:        # ---------------- agent -------------------
        agent_name_snake = to_snake(agent_name)
        dest = Path(__file__).parent / "agentroles" / f"{agent_name_snake}_agent.py"
        tpl_path = TEMPLATE_DIR / "agent.py.jinja"
        context = dict(
            agent_name=agent_name,
            agent_name_snake=agent_name_snake,
            agent_name_camel=to_camel(agent_name),
            timestamp=timestamp,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        click.echo(f" {dest} 已存在，跳过生成")
        return

    if not tpl_path.exists():
        click.echo(f"模板不存在: {tpl_path}", err=True)
        raise SystemExit(1)

    with tpl_path.open("r", encoding="utf-8") as f:
        rendered = Template(f.read()).render(**context)

    dest.write_text(rendered, encoding="utf-8")
    click.echo(f" 已生成 {dest.relative_to(Path.cwd())}")


if __name__ == "__main__":
    cli()