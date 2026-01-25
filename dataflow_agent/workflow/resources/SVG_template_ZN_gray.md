import React from 'react';

const TechnicalRoadmap = () => {
  return (
    <div className="w-full max-w-4xl mx-auto p-4 bg-white">
      <svg viewBox="0 0 800 1150" className="w-full h-auto font-sans">
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#000" />
          </marker>
        </defs>

        {/* --- 标题 --- */}
        <text x="400" y="40" textAnchor="middle" fontSize="24" fontWeight="bold">
          Agent-as-a-Judge: 技术路线图
        </text>
        <text x="400" y="65" textAnchor="middle" fontSize="14" fill="#555">
          基于 arXiv:2410.10934v2
        </text>

        {/* --- 第一部分: 研究动机 (顶部虚线框) --- */}
        <rect x="50" y="90" width="700" height="120" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="115" textAnchor="middle" fontSize="16" fontWeight="bold">
          研究动机与挑战
        </text>

        {/* 动机框 */}
        <g transform="translate(70, 130)">
          <rect x="0" y="0" width="200" height="60" fill="#fff" stroke="#000" strokeWidth="1.5" />
          <text x="100" y="25" textAnchor="middle" fontSize="12" fontWeight="bold">评估方法不足</text>
          <text x="100" y="45" textAnchor="middle" fontSize="10">Pass@1 忽略中间步骤</text>
        </g>

        <g transform="translate(290, 130)">
          <rect x="0" y="0" width="220" height="60" fill="#fff" stroke="#000" strokeWidth="1.5" />
          <text x="110" y="25" textAnchor="middle" fontSize="12" fontWeight="bold">人工评估成本高</text>
          <text x="110" y="45" textAnchor="middle" fontSize="10">缓慢、昂贵、难以扩展</text>
        </g>

        <g transform="translate(530, 130)">
          <rect x="0" y="0" width="200" height="60" fill="#fff" stroke="#000" strokeWidth="1.5" />
          <text x="100" y="25" textAnchor="middle" fontSize="12" fontWeight="bold">缺乏反馈机制</text>
          <text x="100" y="45" textAnchor="middle" fontSize="10">智能体奖励信号稀疏</text>
        </g>

        {/* 向下箭头 */}
        <line x1="400" y1="210" x2="400" y2="240" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- 第二部分: 基准构建 (中间虚线框 1) --- */}
        <rect x="50" y="240" width="700" height="140" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="265" textAnchor="middle" fontSize="16" fontWeight="bold">
          基准数据集构建 (DevAI)
        </text>

        <rect x="100" y="280" width="600" height="80" fill="#fff" stroke="#000" strokeWidth="1.5" />

        {/* DevAI 详情 */}
        <g transform="translate(120, 290)">
          <text x="0" y="20" fontSize="14" fontWeight="bold">DevAI: 55个真实AI开发任务</text>
          <line x1="0" y1="30" x2="560" y2="30" stroke="#ddd" strokeWidth="1" />

          <g transform="translate(0, 45)">
            <rect x="0" y="0" width="170" height="30" fill="#f9f9f9" stroke="#333" />
            <text x="85" y="20" textAnchor="middle" fontSize="11">用户查询与标签</text>
          </g>

          <g transform="translate(195, 45)">
            <rect x="0" y="0" width="170" height="30" fill="#f9f9f9" stroke="#333" />
            <text x="85" y="20" textAnchor="middle" fontSize="11">层级化需求</text>
          </g>

          <g transform="translate(390, 45)">
            <rect x="0" y="0" width="170" height="30" fill="#f9f9f9" stroke="#333" />
            <text x="85" y="20" textAnchor="middle" fontSize="11">依赖关系DAG</text>
          </g>
        </g>

        {/* 向下箭头 */}
        <line x1="400" y1="380" x2="400" y2="410" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- 第三部分: 架构 (主虚线框) --- */}
        <rect x="50" y="410" width="700" height="420" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="435" textAnchor="middle" fontSize="16" fontWeight="bold">
          框架架构 (Agent-as-a-Judge)
        </text>

        {/* 三列布局 */}

        {/* 左列: 输入 */}
        <g transform="translate(70, 460)">
            <text x="80" y="-10" textAnchor="middle" fontSize="14" fontWeight="bold">输入</text>
            <rect x="0" y="0" width="160" height="340" fill="none" stroke="#666" strokeDasharray="4,4" />

            <rect x="20" y="40" width="120" height="50" fill="#fff" stroke="#000" />
            <text x="80" y="65" textAnchor="middle" fontSize="11">用户查询</text>
            <text x="80" y="80" textAnchor="middle" fontSize="10">(需求)</text>

            <rect x="20" y="120" width="120" height="50" fill="#fff" stroke="#000" />
            <text x="80" y="145" textAnchor="middle" fontSize="11">智能体轨迹</text>
            <text x="80" y="160" textAnchor="middle" fontSize="10">(步骤/动作)</text>

            <rect x="20" y="200" width="120" height="50" fill="#fff" stroke="#000" />
            <text x="80" y="225" textAnchor="middle" fontSize="11">工作空间</text>
            <text x="80" y="240" textAnchor="middle" fontSize="10">(代码/文件)</text>
        </g>

        {/* 从输入到核心的箭头 */}
        <line x1="230" y1="630" x2="260" y2="630" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* 中列: 评判智能体核心 */}
        <g transform="translate(260, 460)">
            <text x="140" y="-10" textAnchor="middle" fontSize="14" fontWeight="bold">评判智能体模块</text>
            <rect x="0" y="0" width="280" height="340" fill="none" stroke="#666" strokeDasharray="4,4" />

            {/* 顶部: 理解 */}
            <g transform="translate(20, 20)">
                <rect x="0" y="0" width="240" height="80" fill="#fff" stroke="#000" />
                <text x="120" y="20" textAnchor="middle" fontSize="12" fontWeight="bold">上下文理解</text>

                <rect x="10" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="62.5" y="57" textAnchor="middle" fontSize="11">图模块</text>

                <rect x="125" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="177.5" y="57" textAnchor="middle" fontSize="11">定位模块</text>
            </g>

            {/* 中部: 调查 */}
            <g transform="translate(20, 120)">
                <rect x="0" y="0" width="240" height="80" fill="#fff" stroke="#000" />
                <text x="120" y="20" textAnchor="middle" fontSize="12" fontWeight="bold">调查与取证</text>

                <rect x="10" y="35" width="65" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="42.5" y="57" textAnchor="middle" fontSize="10">读取</text>

                <rect x="87.5" y="35" width="65" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="120" y="57" textAnchor="middle" fontSize="10">搜索</text>

                <rect x="165" y="35" width="65" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="197.5" y="57" textAnchor="middle" fontSize="10">检索</text>
            </g>

            {/* 底部: 决策 */}
            <g transform="translate(20, 220)">
                <rect x="0" y="0" width="240" height="80" fill="#fff" stroke="#000" />
                <text x="120" y="20" textAnchor="middle" fontSize="12" fontWeight="bold">决策制定</text>

                <rect x="10" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="62.5" y="57" textAnchor="middle" fontSize="11">询问模块</text>

                <rect x="125" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="177.5" y="57" textAnchor="middle" fontSize="11">记忆模块</text>
            </g>

            {/* 内部箭头 */}
            <line x1="120" y1="100" x2="120" y2="120" stroke="#000" strokeWidth="1.5" markerEnd="url(#arrowhead)" />
            <line x1="120" y1="200" x2="120" y2="220" stroke="#000" strokeWidth="1.5" markerEnd="url(#arrowhead)" />
        </g>

        {/* 从核心到输出的箭头 */}
        <line x1="540" y1="630" x2="570" y2="630" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* 右列: 输出 */}
        <g transform="translate(570, 460)">
            <text x="80" y="-10" textAnchor="middle" fontSize="14" fontWeight="bold">输出</text>
            <rect x="0" y="0" width="160" height="340" fill="none" stroke="#666" strokeDasharray="4,4" />

            <rect x="20" y="100" width="120" height="60" fill="#fff" stroke="#000" />
            <text x="80" y="125" textAnchor="middle" fontSize="11">中间反馈</text>
            <text x="80" y="145" textAnchor="middle" fontSize="11"></text>

            <rect x="20" y="180" width="120" height="60" fill="#fff" stroke="#000" />
            <text x="80" y="205" textAnchor="middle" fontSize="11">最终判定</text>
            <text x="80" y="225" textAnchor="middle" fontSize="11">(满足/不满足)</text>
        </g>

        {/* 向下箭头 */}
        <line x1="400" y1="830" x2="400" y2="860" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- 第四部分: 评估协议 (底部虚线框) --- */}
        <rect x="50" y="860" width="700" height="180" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="885" textAnchor="middle" fontSize="16" fontWeight="bold">
          评估协议与结果
        </text>

        {/* 任务集 */}
        <g transform="translate(100, 900)">
            {/* 第1行 */}
            <rect x="0" y="0" width="600" height="40" fill="#fff" stroke="#000" strokeWidth="1" strokeDasharray="4,4" />
            <text x="20" y="25" fontSize="12" fontWeight="bold">评估对象:</text>
            <text x="100" y="25" fontSize="12">MetaGPT, GPT-Pilot, OpenHands (代码生成智能体)</text>

            {/* 第2行 */}
            <rect x="0" y="50" width="600" height="40" fill="#fff" stroke="#000" strokeWidth="1" strokeDasharray="4,4" />
            <text x="20" y="75" fontSize="12" fontWeight="bold">对比方法:</text>
            <text x="100" y="75" fontSize="12">Agent-as-a-Judge vs. LLM-as-a-Judge vs. Human-as-a-Judge</text>

            {/* 第3行 */}
            <rect x="0" y="100" width="600" height="40" fill="#fff" stroke="#000" strokeWidth="1" strokeDasharray="4,4" />
            <text x="20" y="125" fontSize="12" fontWeight="bold">实验结果:</text>
            <text x="100" y="125" fontSize="12">与人类共识高度一致 (90%+), 成本降低97%</text>
        </g>

        {/* 向下箭头 */}
        <line x1="400" y1="1040" x2="400" y2="1070" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- 第五部分: 结论 (底部实线框) --- */}
        <rect x="150" y="1070" width="500" height="50" fill="#fff" stroke="#000" strokeWidth="2" />
        <text x="400" y="1100" textAnchor="middle" fontSize="14" fontWeight="bold">
          结论: 可扩展的自我改进与"飞轮效应"
        </text>

      </svg>
    </div>
  );
};

export default TechnicalRoadmap;
