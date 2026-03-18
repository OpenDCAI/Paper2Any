import React from 'react';

interface DemoCardProps {
  title: string;
  desc: string;
  inputImg?: string;
  outputImg?: string;
}

const DemoCard: React.FC<DemoCardProps> = ({ title, desc, inputImg, outputImg }) => {
  return (
    <div className="paper2ppt-panel-soft flex flex-col gap-2 rounded-2xl p-3 transition-colors hover:bg-[rgba(255,255,255,0.84)]">
      <div className="flex gap-2">
        {/* 左侧：输入示例图片 */}
        <div className="demo-input-placeholder flex flex-1 items-center justify-center overflow-hidden rounded-xl border border-dashed border-[rgba(110,76,55,0.16)] bg-[rgba(255,255,255,0.7)]">
          {inputImg ? (
            <img
              src={inputImg}
              alt="输入示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-[rgba(103,95,88,0.82)]">输入示例图（待替换）</span>
          )}
        </div>
        {/* 右侧：输出 PPTX 示例图片 */}
        <div className="demo-output-placeholder flex flex-1 items-center justify-center overflow-hidden rounded-xl border border-dashed border-[rgba(140,29,64,0.18)] bg-[rgba(140,29,64,0.05)]">
          {outputImg ? (
            <img
              src={outputImg}
              alt="PPTX 示例图"
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-[10px] text-[#8c1d40]">PPTX 示例图（待替换）</span>
          )}
        </div>
      </div>
      <div>
        <p className="mb-1 text-[13px] font-semibold text-[#1d1c1a]">{title}</p>
        <p className="text-[11px] leading-snug text-[#675f58]">{desc}</p>
      </div>
    </div>
  );
};

export default DemoCard;
