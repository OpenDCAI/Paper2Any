import { Headphones } from 'lucide-react';

export const PodcastTool = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center opacity-50">
      <Headphones size={48} className="mb-4 text-primary-400" />
      <h3 className="text-lg font-medium">知识播客生成</h3>
      <p className="text-sm text-gray-500 mt-2">Coming Soon...</p>
    </div>
  );
};
