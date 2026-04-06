import { Handle, Position, type NodeProps } from 'reactflow';

import type { MindMapNodeData } from '../../utils/mindmapTree';

export default function MindMapNodeCard({ data, selected }: NodeProps<MindMapNodeData>) {
  const accent =
    data.branch === 'left'
      ? 'from-violet-400/30 to-fuchsia-400/10 border-violet-300/40'
      : data.branch === 'right'
        ? 'from-cyan-400/30 to-sky-400/10 border-cyan-300/40'
        : 'from-emerald-400/30 to-cyan-400/10 border-emerald-200/50';

  const shadow = selected ? 'shadow-[0_0_0_1px_rgba(255,255,255,0.18),0_24px_48px_rgba(15,23,42,0.42)]' : 'shadow-[0_18px_36px_rgba(15,23,42,0.3)]';

  return (
    <div
      className={`relative min-w-[220px] max-w-[260px] rounded-[24px] border bg-[linear-gradient(135deg,rgba(15,23,42,0.94),rgba(15,23,42,0.8))] px-4 py-3 text-left text-white backdrop-blur-xl ${accent} ${shadow}`}
    >
      <Handle id="left-target" type="target" position={Position.Left} className="!h-2.5 !w-2.5 !border-0 !bg-violet-300/70" />
      <Handle id="right-target" type="target" position={Position.Right} className="!h-2.5 !w-2.5 !border-0 !bg-cyan-300/70" />
      <Handle id="left-source" type="source" position={Position.Left} className="!h-2.5 !w-2.5 !border-0 !bg-violet-200/90" />
      <Handle id="right-source" type="source" position={Position.Right} className="!h-2.5 !w-2.5 !border-0 !bg-cyan-200/90" />

      <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-white/45">
        {data.isRoot ? 'Root' : `Depth ${data.depth}`}
      </div>
      <div className="mt-2 text-sm font-semibold leading-5 text-white">{data.label}</div>
      {data.summary ? (
        <div className="mt-2 line-clamp-3 text-xs leading-5 text-white/68">{data.summary}</div>
      ) : null}
    </div>
  );
}
