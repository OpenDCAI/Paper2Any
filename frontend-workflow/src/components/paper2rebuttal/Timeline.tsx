import { useState } from 'react';
import { CheckCircle, Circle, Clock, FileText, Search, Filter, BookOpen } from 'lucide-react';

interface TimelineNode {
  id: string;
  title: string;
  description: string;
  status: 'completed' | 'current' | 'pending';
  timestamp?: string;
  data?: any;
}

interface TimelineProps {
  nodes: TimelineNode[];
  currentIndex: number;
  onNodeClick?: (index: number) => void;
  horizontal?: boolean;
}

const Timeline = ({ nodes, currentIndex, onNodeClick, horizontal = false }: TimelineProps) => {
  if (nodes.length === 0) return null;

  // Horizontal timeline
  if (horizontal) {
    return (
      <div className="relative w-full">
        {/* Timeline line */}
        <div className="absolute top-6 left-0 right-0 h-0.5 bg-white/20" />
        
        <div className="relative flex items-start justify-between gap-2">
          {nodes.map((node, index) => {
            const isCompleted = node.status === 'completed' || index < currentIndex;
            const isCurrent = node.status === 'current' || index === currentIndex;
            const isPending = node.status === 'pending' || index > currentIndex;
            
            return (
              <div
                key={node.id}
                className={`relative flex-1 flex flex-col items-center ${
                  onNodeClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''
                }`}
                onClick={() => onNodeClick && onNodeClick(index)}
              >
                {/* Node icon */}
                <div className={`relative z-10 flex items-center justify-center w-12 h-12 rounded-full transition-colors mb-2 ${
                  isCompleted ? 'bg-green-500' : isCurrent ? 'bg-blue-500 animate-pulse' : 'bg-gray-500'
                }`}>
                  {isCompleted ? (
                    <CheckCircle className="w-6 h-6 text-white" />
                  ) : isCurrent ? (
                    <Clock className="w-6 h-6 text-white" />
                  ) : (
                    <Circle className="w-6 h-6 text-white" />
                  )}
                </div>
                
                {/* Content */}
                <div className="text-center max-w-[120px]">
                  <div className={`font-semibold text-sm ${
                    isCompleted ? 'text-green-400' : isCurrent ? 'text-blue-400' : 'text-gray-400'
                  }`}>
                    {node.title}
                  </div>
                  <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                    {node.description}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Vertical timeline (original)
  return (
    <div className="relative">
      {/* Timeline line */}
      <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-white/20" />
      
      <div className="space-y-4">
        {nodes.map((node, index) => {
          const isCompleted = node.status === 'completed' || index < currentIndex;
          const isCurrent = node.status === 'current' || index === currentIndex;
          const isPending = node.status === 'pending' || index > currentIndex;
          
          return (
            <div
              key={node.id}
              className={`relative flex items-start gap-4 ${
                onNodeClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''
              }`}
              onClick={() => onNodeClick && onNodeClick(index)}
            >
              {/* Node icon */}
              <div className={`relative z-10 flex items-center justify-center w-12 h-12 rounded-full transition-colors ${
                isCompleted ? 'bg-green-500' : isCurrent ? 'bg-blue-500 animate-pulse' : 'bg-gray-500'
              }`}>
                {isCompleted ? (
                  <CheckCircle className="w-6 h-6 text-white" />
                ) : isCurrent ? (
                  <Clock className="w-6 h-6 text-white" />
                ) : (
                  <Circle className="w-6 h-6 text-white" />
                )}
              </div>
              
              {/* Content */}
              <div className="flex-1 pt-1">
                <div className={`font-semibold ${
                  isCompleted ? 'text-green-400' : isCurrent ? 'text-blue-400' : 'text-gray-400'
                }`}>
                  {node.title}
                </div>
                <div className="text-sm text-gray-400 mt-1">
                  {node.description}
                </div>
                {node.timestamp && (
                  <div className="text-xs text-gray-500 mt-1">
                    {node.timestamp}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Timeline;
