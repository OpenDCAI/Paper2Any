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

        {/* --- TITLE --- */}
        <text x="400" y="40" textAnchor="middle" fontSize="24" fontWeight="bold">
          Agent-as-a-Judge: Technical Roadmap
        </text>
        <text x="400" y="65" textAnchor="middle" fontSize="14" fill="#555">
          Based on arXiv:2410.10934v2
        </text>

        {/* --- SECTION 1: MOTIVATION (Top Dashed Box) --- */}
        <rect x="50" y="90" width="700" height="120" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="115" textAnchor="middle" fontSize="16" fontWeight="bold">
          Motivation & Challenges
        </text>

        {/* Motivation Boxes */}
        <g transform="translate(70, 130)">
          <rect x="0" y="0" width="200" height="60" fill="#fff" stroke="#000" strokeWidth="1.5" />
          <text x="100" y="25" textAnchor="middle" fontSize="12" fontWeight="bold">Inadequate Evaluation</text>
          <text x="100" y="45" textAnchor="middle" fontSize="10">Pass@1 ignores steps</text>
        </g>

        <g transform="translate(290, 130)">
          <rect x="0" y="0" width="220" height="60" fill="#fff" stroke="#000" strokeWidth="1.5" />
          <text x="110" y="25" textAnchor="middle" fontSize="12" fontWeight="bold">High Cost of Human Eval</text>
          <text x="110" y="45" textAnchor="middle" fontSize="10">Slow, expensive, non-scalable</text>
        </g>

        <g transform="translate(530, 130)">
          <rect x="0" y="0" width="200" height="60" fill="#fff" stroke="#000" strokeWidth="1.5" />
          <text x="100" y="25" textAnchor="middle" fontSize="12" fontWeight="bold">Lack of Feedback</text>
          <text x="100" y="45" textAnchor="middle" fontSize="10">Sparse rewards for agents</text>
        </g>

        {/* Arrow Down */}
        <line x1="400" y1="210" x2="400" y2="240" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- SECTION 2: PROPOSAL & BENCHMARK (Middle Dashed Box 1) --- */}
        <rect x="50" y="240" width="700" height="140" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="265" textAnchor="middle" fontSize="16" fontWeight="bold">
          Benchmark Construction (DevAI Dataset)
        </text>

        <rect x="100" y="280" width="600" height="80" fill="#fff" stroke="#000" strokeWidth="1.5" />
        
        {/* DevAI Details */}
        <g transform="translate(120, 290)">
          <text x="0" y="20" fontSize="14" fontWeight="bold">DevAI: 55 Realistic AI Development Tasks</text>
          <line x1="0" y1="30" x2="560" y2="30" stroke="#ddd" strokeWidth="1" />
          
          <g transform="translate(0, 45)">
            <rect x="0" y="0" width="170" height="30" fill="#f9f9f9" stroke="#333" />
            <text x="85" y="20" textAnchor="middle" fontSize="11">User Query & Tags</text>
          </g>
          
          <g transform="translate(195, 45)">
            <rect x="0" y="0" width="170" height="30" fill="#f9f9f9" stroke="#333" />
            <text x="85" y="20" textAnchor="middle" fontSize="11">Hierarchical Requirements</text>
          </g>
          
          <g transform="translate(390, 45)">
            <rect x="0" y="0" width="170" height="30" fill="#f9f9f9" stroke="#333" />
            <text x="85" y="20" textAnchor="middle" fontSize="11">Dependency DAGs</text>
          </g>
        </g>

        {/* Arrow Down */}
        <line x1="400" y1="380" x2="400" y2="410" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- SECTION 3: ARCHITECTURE (Main Dashed Box) --- */}
        <rect x="50" y="410" width="700" height="420" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="435" textAnchor="middle" fontSize="16" fontWeight="bold">
          Framework Architecture (Agent-as-a-Judge)
        </text>

        {/* 3 Columns Layout */}
        
        {/* Left Column: Inputs */}
        <g transform="translate(70, 460)">
            <text x="80" y="-10" textAnchor="middle" fontSize="14" fontWeight="bold">Inputs</text>
            <rect x="0" y="0" width="160" height="340" fill="none" stroke="#666" strokeDasharray="4,4" />
            
            <rect x="20" y="40" width="120" height="50" fill="#fff" stroke="#000" />
            <text x="80" y="65" textAnchor="middle" fontSize="11">User Query</text>
            <text x="80" y="80" textAnchor="middle" fontSize="10">(Requirements)</text>

            <rect x="20" y="120" width="120" height="50" fill="#fff" stroke="#000" />
            <text x="80" y="145" textAnchor="middle" fontSize="11">Agent Trajectory</text>
            <text x="80" y="160" textAnchor="middle" fontSize="10">(Steps/Actions)</text>

            <rect x="20" y="200" width="120" height="50" fill="#fff" stroke="#000" />
            <text x="80" y="225" textAnchor="middle" fontSize="11">Workspace</text>
            <text x="80" y="240" textAnchor="middle" fontSize="10">(Code/Files)</text>
        </g>

        {/* Arrow from Inputs to Core */}
        <line x1="230" y1="630" x2="260" y2="630" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* Middle Column: The Judge Agent Core */}
        <g transform="translate(260, 460)">
            <text x="140" y="-10" textAnchor="middle" fontSize="14" fontWeight="bold">Judge Agent Modules</text>
            <rect x="0" y="0" width="280" height="340" fill="none" stroke="#666" strokeDasharray="4,4" />

            {/* Top: Understanding */}
            <g transform="translate(20, 20)">
                <rect x="0" y="0" width="240" height="80" fill="#fff" stroke="#000" />
                <text x="120" y="20" textAnchor="middle" fontSize="12" fontWeight="bold">Context Understanding</text>
                
                <rect x="10" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="62.5" y="57" textAnchor="middle" fontSize="11">Graph Module</text>
                
                <rect x="125" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="177.5" y="57" textAnchor="middle" fontSize="11">Locate Module</text>
            </g>

            {/* Middle: Investigation */}
            <g transform="translate(20, 120)">
                <rect x="0" y="0" width="240" height="80" fill="#fff" stroke="#000" />
                <text x="120" y="20" textAnchor="middle" fontSize="12" fontWeight="bold">Investigation & Evidence</text>
                
                <rect x="10" y="35" width="65" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="42.5" y="57" textAnchor="middle" fontSize="10">Read</text>
                
                <rect x="87.5" y="35" width="65" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="120" y="57" textAnchor="middle" fontSize="10">Search</text>

                <rect x="165" y="35" width="65" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="197.5" y="57" textAnchor="middle" fontSize="10">Retrieve</text>
            </g>

            {/* Bottom: Decision */}
            <g transform="translate(20, 220)">
                <rect x="0" y="0" width="240" height="80" fill="#fff" stroke="#000" />
                <text x="120" y="20" textAnchor="middle" fontSize="12" fontWeight="bold">Decision Making</text>
                
                <rect x="10" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="62.5" y="57" textAnchor="middle" fontSize="11">Ask Module</text>
                
                <rect x="125" y="35" width="105" height="35" fill="#f0f0f0" stroke="#333" />
                <text x="177.5" y="57" textAnchor="middle" fontSize="11">Memory</text>
            </g>

            {/* Internal Arrows */}
            <line x1="120" y1="100" x2="120" y2="120" stroke="#000" strokeWidth="1.5" markerEnd="url(#arrowhead)" />
            <line x1="120" y1="200" x2="120" y2="220" stroke="#000" strokeWidth="1.5" markerEnd="url(#arrowhead)" />
        </g>

        {/* Arrow from Core to Output */}
        <line x1="540" y1="630" x2="570" y2="630" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* Right Column: Outputs */}
        <g transform="translate(570, 460)">
            <text x="80" y="-10" textAnchor="middle" fontSize="14" fontWeight="bold">Outputs</text>
            <rect x="0" y="0" width="160" height="340" fill="none" stroke="#666" strokeDasharray="4,4" />
            
            <rect x="20" y="100" width="120" height="60" fill="#fff" stroke="#000" />
            <text x="80" y="125" textAnchor="middle" fontSize="11">Intermediate</text>
            <text x="80" y="145" textAnchor="middle" fontSize="11">Feedback</text>

            <rect x="20" y="180" width="120" height="60" fill="#fff" stroke="#000" />
            <text x="80" y="205" textAnchor="middle" fontSize="11">Final Verdict</text>
            <text x="80" y="225" textAnchor="middle" fontSize="11">(Satisfied / Not)</text>
        </g>

        {/* Arrow Down */}
        <line x1="400" y1="830" x2="400" y2="860" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- SECTION 4: EVALUATION PROTOCOL (Bottom Dashed Box) --- */}
        <rect x="50" y="860" width="700" height="180" fill="none" stroke="#000" strokeWidth="2" strokeDasharray="8,8" />
        <text x="400" y="885" textAnchor="middle" fontSize="16" fontWeight="bold">
          Evaluation Protocol & Results
        </text>

        {/* Task Sets */}
        <g transform="translate(100, 900)">
            {/* Row 1 */}
            <rect x="0" y="0" width="600" height="40" fill="#fff" stroke="#000" strokeWidth="1" strokeDasharray="4,4" />
            <text x="20" y="25" fontSize="12" fontWeight="bold">Subjects:</text>
            <text x="100" y="25" fontSize="12">MetaGPT, GPT-Pilot, OpenHands (Code Generation Agents)</text>

            {/* Row 2 */}
            <rect x="0" y="50" width="600" height="40" fill="#fff" stroke="#000" strokeWidth="1" strokeDasharray="4,4" />
            <text x="20" y="75" fontSize="12" fontWeight="bold">Comparison:</text>
            <text x="100" y="75" fontSize="12">Agent-as-a-Judge vs. LLM-as-a-Judge vs. Human-as-a-Judge</text>

            {/* Row 3 */}
            <rect x="0" y="100" width="600" height="40" fill="#fff" stroke="#000" strokeWidth="1" strokeDasharray="4,4" />
            <text x="20" y="125" fontSize="12" fontWeight="bold">Results:</text>
            <text x="100" y="125" fontSize="12">High alignment with Human consensus (90%+), 97% Cost Reduction</text>
        </g>

        {/* Arrow Down */}
        <line x1="400" y1="1040" x2="400" y2="1070" stroke="#000" strokeWidth="2" markerEnd="url(#arrowhead)" />

        {/* --- SECTION 5: CONCLUSION (Solid Bottom Box) --- */}
        <rect x="150" y="1070" width="500" height="50" fill="#fff" stroke="#000" strokeWidth="2" />
        <text x="400" y="1100" textAnchor="middle" fontSize="14" fontWeight="bold">
          Conclusion: Scalable Self-Improvement & "Flywheel Effect"
        </text>

      </svg>
    </div>
  );
};

export default TechnicalRoadmap;