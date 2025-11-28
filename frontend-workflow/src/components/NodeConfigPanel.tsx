import { useState, useEffect } from 'react';
import { X, Settings, Trash2, Save, RotateCcw, Plus, Minus } from 'lucide-react';
import * as Icons from 'lucide-react';
import { WorkflowNode, AgentConfig } from '../types';

interface NodeConfigPanelProps {
  node: WorkflowNode | null;
  onClose: () => void;
  onDelete: (nodeId: string) => void;
  onUpdate: (nodeId: string, config: AgentConfig) => void;
}

const NodeConfigPanel = ({ node, onClose, onDelete, onUpdate }: NodeConfigPanelProps) => {
  const [config, setConfig] = useState<AgentConfig>({});
  const [schemaKey, setSchemaKey] = useState('');
  const [schemaValue, setSchemaValue] = useState('');

  useEffect(() => {
    if (node) {
      // 合并默认配置和当前配置
      const defaultConfig = node.data.agentType.defaultConfig || {};
      const currentConfig = node.data.config || {};
      setConfig({ ...defaultConfig, ...currentConfig });
    }
  }, [node]);

  if (!node) return null;

  const { agentType } = node.data;
  const IconComponent = Icons[agentType.icon as keyof typeof Icons] as any;

  const handleSave = () => {
    onUpdate(node.id, config);
  };

  const handleReset = () => {
    const defaultConfig = agentType.defaultConfig || {};
    setConfig(defaultConfig);
  };

  const addSchemaField = () => {
    if (schemaKey && schemaValue) {
      setConfig({
        ...config,
        response_schema: {
          ...(config.response_schema || {}),
          [schemaKey]: schemaValue,
        },
      });
      setSchemaKey('');
      setSchemaValue('');
    }
  };

  const removeSchemaField = (key: string) => {
    const newSchema = { ...(config.response_schema || {}) };
    delete newSchema[key];
    setConfig({
      ...config,
      response_schema: newSchema,
    });
  };

  const addRequiredField = (field: string) => {
    if (field && !config.required_fields?.includes(field)) {
      setConfig({
        ...config,
        required_fields: [...(config.required_fields || []), field],
      });
    }
  };

  const removeRequiredField = (field: string) => {
    setConfig({
      ...config,
      required_fields: config.required_fields?.filter(f => f !== field) || [],
    });
  };

  return (
    <div className="w-96 h-full glass-dark border-l border-white/10 flex flex-col animate-slide-in">
      {/* 头部 */}
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white">节点配置</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X size={20} className="text-gray-400" />
          </button>
        </div>

        {/* 节点信息 */}
        <div className="flex items-center gap-4">
          <div
            className="p-3 rounded-xl"
            style={{
              background: `${agentType.color}20`,
              color: agentType.color,
            }}
          >
            {IconComponent && <IconComponent size={28} />}
          </div>
          <div>
            <div className="font-semibold text-white">
              {agentType.displayName}
            </div>
            <div className="text-sm text-gray-400">{agentType.name}</div>
          </div>
        </div>
      </div>

      {/* 配置表单 */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* 基础参数 */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-4 flex items-center gap-2">
            <Settings size={16} />
            基础参数
          </h3>

          <div className="space-y-4">
            {/* 模型名称 */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">模型名称</label>
              <input
                type="text"
                value={config.model_name || ''}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500"
                placeholder="gpt-4"
              />
            </div>

            {/* API URL */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Chat API URL (可选)</label>
              <input
                type="text"
                value={config.chat_api_url || ''}
                onChange={(e) => setConfig({ ...config, chat_api_url: e.target.value })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500"
                placeholder="https://api.openai.com/v1"
              />
            </div>

            {/* Temperature */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Temperature: {config.temperature?.toFixed(1) || '0.0'}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.temperature || 0}
                onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>

            {/* Max Tokens */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Max Tokens</label>
              <input
                type="number"
                value={config.max_tokens || 16384}
                onChange={(e) => setConfig({ ...config, max_tokens: parseInt(e.target.value) })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500"
              />
            </div>

            {/* Tool Mode */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Tool Mode</label>
              <select
                value={config.tool_mode || 'auto'}
                onChange={(e) => setConfig({ ...config, tool_mode: e.target.value as any })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500"
              >
                <option value="auto">Auto</option>
                <option value="none">None</option>
                <option value="required">Required</option>
              </select>
            </div>

            {/* Parser Type */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Parser Type</label>
              <select
                value={config.parser_type || 'json'}
                onChange={(e) => setConfig({ ...config, parser_type: e.target.value as any })}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500"
              >
                <option value="json">JSON</option>
                <option value="xml">XML</option>
                <option value="text">Text</option>
              </select>
            </div>

            {/* 模式特定参数 */}
            {agentType.mode === 'react' && (
              <div>
                <label className="block text-xs text-gray-500 mb-1">Max Retries</label>
                <input
                  type="number"
                  value={config.max_retries || 3}
                  onChange={(e) => setConfig({ ...config, max_retries: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                           text-white text-sm focus:outline-none focus:border-primary-500"
                />
              </div>
            )}

            {agentType.mode === 'parallel' && (
              <div>
                <label className="block text-xs text-gray-500 mb-1">Concurrency Limit</label>
                <input
                  type="number"
                  value={config.concurrency_limit || 5}
                  onChange={(e) => setConfig({ ...config, concurrency_limit: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                           text-white text-sm focus:outline-none focus:border-primary-500"
                />
              </div>
            )}

            {agentType.mode === 'vlm' && (
              <>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">VLM Mode</label>
                  <select
                    value={config.vlm_mode || 'understanding'}
                    onChange={(e) => setConfig({ ...config, vlm_mode: e.target.value as any })}
                    className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                             text-white text-sm focus:outline-none focus:border-primary-500"
                  >
                    <option value="understanding">Understanding</option>
                    <option value="generation">Generation</option>
                    <option value="edit">Edit</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Image Detail</label>
                  <select
                    value={config.image_detail || 'auto'}
                    onChange={(e) => setConfig({ ...config, image_detail: e.target.value as any })}
                    className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                             text-white text-sm focus:outline-none focus:border-primary-500"
                  >
                    <option value="auto">Auto</option>
                    <option value="low">Low</option>
                    <option value="high">High</option>
                  </select>
                </div>
              </>
            )}
          </div>
        </div>

        {/* 提示词配置 */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-4">提示词配置</h3>
          
          <div className="space-y-4">
            {/* System Prompt */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">System Prompt</label>
              <textarea
                value={config.system_prompt || ''}
                onChange={(e) => setConfig({ ...config, system_prompt: e.target.value })}
                rows={4}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500 resize-none"
                placeholder="定义Agent的角色和行为..."
              />
            </div>

            {/* Task Prompt */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Task Prompt</label>
              <textarea
                value={config.task_prompt || ''}
                onChange={(e) => setConfig({ ...config, task_prompt: e.target.value })}
                rows={4}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500 resize-none"
                placeholder="描述具体任务..."
              />
            </div>
          </div>
        </div>

        {/* JSON Schema配置 */}
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-4">JSON Schema配置</h3>
          
          <div className="space-y-4">
            {/* Response Schema */}
            <div>
              <label className="block text-xs text-gray-500 mb-2">Response Schema</label>
              
              {/* 现有字段 */}
              {config.response_schema && Object.keys(config.response_schema).length > 0 && (
                <div className="space-y-2 mb-2">
                  {Object.entries(config.response_schema).map(([key, value]) => (
                    <div key={key} className="flex items-center gap-2 p-2 bg-white/5 rounded">
                      <span className="text-xs text-white flex-1">{key}: {value}</span>
                      <button
                        onClick={() => removeSchemaField(key)}
                        className="p-1 hover:bg-red-500/20 rounded transition-colors"
                      >
                        <Minus size={14} className="text-red-400" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* 添加新字段 */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={schemaKey}
                  onChange={(e) => setSchemaKey(e.target.value)}
                  placeholder="字段名"
                  className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                           text-white text-sm focus:outline-none focus:border-primary-500"
                />
                <input
                  type="text"
                  value={schemaValue}
                  onChange={(e) => setSchemaValue(e.target.value)}
                  placeholder="类型"
                  className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                           text-white text-sm focus:outline-none focus:border-primary-500"
                />
                <button
                  onClick={addSchemaField}
                  className="p-2 bg-primary-500/20 hover:bg-primary-500/30 rounded-lg transition-colors"
                >
                  <Plus size={18} className="text-primary-400" />
                </button>
              </div>
            </div>

            {/* Schema Description */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Schema Description</label>
              <textarea
                value={config.response_schema_description || ''}
                onChange={(e) => setConfig({ ...config, response_schema_description: e.target.value })}
                rows={3}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500 resize-none"
                placeholder="描述返回格式..."
              />
            </div>

            {/* Response Example */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Response Example (JSON)</label>
              <textarea
                value={config.response_example || ''}
                onChange={(e) => setConfig({ ...config, response_example: e.target.value })}
                rows={4}
                className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                         text-white text-sm focus:outline-none focus:border-primary-500 resize-none font-mono"
                placeholder='{"key": "value"}'
              />
            </div>

            {/* Required Fields */}
            <div>
              <label className="block text-xs text-gray-500 mb-2">Required Fields</label>
              
              {/* 现有必填字段 */}
              {config.required_fields && config.required_fields.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {config.required_fields.map((field) => (
                    <div
                      key={field}
                      className="flex items-center gap-1 px-2 py-1 bg-primary-500/20 rounded text-xs text-primary-300"
                    >
                      <span>{field}</span>
                      <button
                        onClick={() => removeRequiredField(field)}
                        className="hover:text-red-400 transition-colors"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* 从Schema中选择 */}
              {config.response_schema && Object.keys(config.response_schema).length > 0 && (
                <select
                  onChange={(e) => {
                    if (e.target.value) {
                      addRequiredField(e.target.value);
                      e.target.value = '';
                    }
                  }}
                  className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg
                           text-white text-sm focus:outline-none focus:border-primary-500"
                >
                  <option value="">选择必填字段...</option>
                  {Object.keys(config.response_schema).map((key) => (
                    <option key={key} value={key}>{key}</option>
                  ))}
                </select>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="p-6 border-t border-white/10 space-y-3">
        <div className="flex gap-3">
          <button
            onClick={handleSave}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-3 
                     bg-primary-500 hover:bg-primary-600 rounded-lg text-white 
                     transition-colors font-medium"
          >
            <Save size={18} />
            保存配置
          </button>
          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 px-4 py-3 
                     bg-white/5 hover:bg-white/10 rounded-lg text-gray-300 
                     transition-colors"
          >
            <RotateCcw size={18} />
          </button>
        </div>
        
        <button
          onClick={() => onDelete(node.id)}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 
                   bg-red-500/20 hover:bg-red-500/30 border border-red-500/50
                   rounded-lg text-red-400 transition-colors"
        >
          <Trash2 size={18} />
          删除节点
        </button>
      </div>
    </div>
  );
};

export default NodeConfigPanel;
