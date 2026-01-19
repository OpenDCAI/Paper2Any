import { useState } from 'react';
import { MessageSquare, Send, Bot, User, Loader2 } from 'lucide-react';
import { ChatMessage, KnowledgeFile } from '../types';

interface ChatToolProps {
  files: KnowledgeFile[];
  selectedIds: Set<string>;
}

export const ChatTool = ({ files, selectedIds }: ChatToolProps) => {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: '你好！我是你的知识库助手。请在“我的知识库”中勾选素材，然后在此处进行提问。',
      time: new Date().toLocaleTimeString()
    }
  ]);
  const [inputMsg, setInputMsg] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!inputMsg.trim()) return;
    
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMsg,
      time: new Date().toLocaleTimeString()
    };
    
    setChatMessages(prev => [...prev, userMsg]);
    setInputMsg('');
    setIsChatLoading(true);

    try {
      if (selectedIds.size === 0) {
        const botMsg: ChatMessage = {
            id: Date.now().toString(),
            role: 'assistant',
            content: '请先在中间的知识库列表中勾选至少一个文件，我才能基于这些资料回答您的问题。',
            time: new Date().toLocaleTimeString()
        };
        setChatMessages(prev => [...prev, botMsg]);
        setIsChatLoading(false);
        return;
      }

      const selectedFiles = files.filter(f => selectedIds.has(f.id)).map(f => f.url);
      
      // Construct history for API
      const history = chatMessages.filter(m => m.id !== 'welcome').map(m => ({
          role: m.role,
          content: m.content
      }));

      const res = await fetch('/api/v1/kb/chat', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({
              files: selectedFiles,
              query: userMsg.content,
              history: history
          })
      });

      if (!res.ok) {
          throw new Error("Chat request failed");
      }

      const data = await res.json();
      
      const botMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || "抱歉，我无法回答这个问题。",
        time: new Date().toLocaleTimeString()
      };
      setChatMessages(prev => [...prev, botMsg]);
      
    } catch (err) {
      console.error("Chat error:", err);
      const errorMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: "发生错误，请稍后重试。",
          time: new Date().toLocaleTimeString()
      };
      setChatMessages(prev => [...prev, errorMsg]);
    } finally {
        setIsChatLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0a0a1a]">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {selectedIds.size === 0 && (
          <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs flex items-start gap-2">
            <div className="mt-0.5"><Bot size={14} /></div>
            <p>请先在中间的知识库列表中勾选至少一个文件，我才能基于这些资料回答您的问题。</p>
          </div>
        )}

        {chatMessages.map(msg => (
          <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
              msg.role === 'assistant' ? 'bg-primary-500/20 text-primary-400' : 'bg-white/10 text-gray-400'
            }`}>
              {msg.role === 'assistant' ? <Bot size={16} /> : <User size={16} />}
            </div>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
              msg.role === 'assistant' ? 'bg-white/5 text-gray-200' : 'bg-primary-600 text-white'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        {isChatLoading && (
          <div className="flex gap-3 animate-pulse">
            <div className="w-8 h-8 rounded-full bg-primary-500/20 text-primary-400 flex items-center justify-center"><Bot size={16} /></div>
            <div className="bg-white/5 rounded-2xl px-4 py-3 text-sm flex items-center gap-2 text-gray-400">
              <Loader2 size={14} className="animate-spin" /> 思考中...
            </div>
          </div>
        )}
      </div>
      
      <div className="p-4 border-t border-white/5 bg-[#0a0a1a]">
        <div className="relative">
          <input
            type="text"
            value={inputMsg}
            onChange={e => setInputMsg(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
            placeholder={selectedIds.size > 0 ? "有问题尽管问我..." : "请先勾选素材..."}
            disabled={selectedIds.size === 0}
            className="w-full bg-white/5 border border-white/10 rounded-xl pl-4 pr-12 py-3.5 text-sm text-gray-200 outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMsg.trim() || isChatLoading || selectedIds.size === 0}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};
