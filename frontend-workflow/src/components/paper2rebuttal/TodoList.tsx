import { CheckCircle, Circle, FlaskConical, BarChart3, FileText, GitCompare, TestTube } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface TodoItem {
  id: number;
  title: string;
  description: string;
  type: 'experiment' | 'analysis' | 'clarification' | 'comparison' | 'ablation';
  status: 'pending' | 'completed' | 'in_progress';
  related_papers?: string[];
}

interface TodoListProps {
  todos: TodoItem[];
  onToggle?: (id: number) => void;
}

const typeIcons = {
  experiment: FlaskConical,
  analysis: BarChart3,
  clarification: FileText,
  comparison: GitCompare,
  ablation: TestTube,
};

const typeColors = {
  experiment: 'bg-purple-500/20 text-purple-300 border-purple-500/50',
  analysis: 'bg-blue-500/20 text-blue-300 border-blue-500/50',
  clarification: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50',
  comparison: 'bg-green-500/20 text-green-300 border-green-500/50',
  ablation: 'bg-pink-500/20 text-pink-300 border-pink-500/50',
};

const TodoList = ({ todos, onToggle }: TodoListProps) => {
  const { t } = useTranslation(['paper2rebuttal']);

  if (!todos || todos.length === 0) {
    return (
      <div className="py-4 text-sm text-[var(--text-secondary)]">
        {t('paper2rebuttal:todo.noTodos')}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {todos.map((todo) => {
        const Icon = typeIcons[todo.type] || FileText;
        const colorClass = typeColors[todo.type] || typeColors.clarification;
        
        return (
          <div
            key={todo.id}
            className={`p-4 rounded-lg border ${
              todo.status === 'completed' 
                ? 'bg-green-500/10 border-green-500/30' 
                : 'bg-white/82 border-primary-100'
            } hover:bg-white transition-colors`}
          >
            <div className="flex items-start gap-3">
              <button
                onClick={() => onToggle && onToggle(todo.id)}
                className="mt-1 flex-shrink-0"
              >
                {todo.status === 'completed' ? (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                ) : (
                  <Circle className="w-5 h-5 text-gray-400" />
                )}
              </button>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="w-4 h-4" />
                  <h4 className={`font-semibold ${
                    todo.status === 'completed' ? 'text-green-700 line-through' : 'text-[var(--text-primary)]'
                  }`}>
                    {todo.title}
                  </h4>
                  <span className={`px-2 py-0.5 rounded text-xs border ${colorClass}`}>
                    {todo.type}
                  </span>
                </div>
                
                <p className={`text-sm ${
                  todo.status === 'completed' ? 'text-[#8b726b]' : 'text-[var(--text-secondary)]'
                } whitespace-pre-wrap`}>
                  {todo.description}
                </p>
                
                {todo.related_papers && todo.related_papers.length > 0 && (
                  <div className="mt-2 border-t border-primary-100 pt-2">
                    <div className="mb-1 text-xs text-[var(--text-secondary)]">{t('paper2rebuttal:todo.relatedPapers')}</div>
                    <div className="flex flex-wrap gap-1">
                      {todo.related_papers.map((paper, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded text-xs"
                        >
                          {paper}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default TodoList;
