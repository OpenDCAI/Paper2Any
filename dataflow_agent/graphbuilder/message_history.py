from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Literal
from datetime import datetime, timedelta
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    ToolMessage,
    RemoveMessage
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langchain_core.messages.utils import trim_messages
import hashlib

# ==================== 核心封装类 ====================
@dataclass
class AdvancedMessageHistory:
    """
    高级消息历史管理器 - 封装 LangGraph 原生能力
    
    功能：
    1. 消息合并（多源、去重）
    2. 消息过滤（类型、时间、内容）
    3. 消息清理（批量、压缩）
    4. 统一接口（屏蔽底层复杂性）
    """
    
    # ===== 核心配置 =====
    checkpointer: BaseCheckpointSaver = field(default_factory=MemorySaver)
    thread_id: str = "default"
    
    # ===== 历史管理配置 =====
    max_messages: int = 100
    max_tokens: Optional[int] = None
    max_age_hours: Optional[int] = None  # 消息最大保留时间
    
    # ===== 消息处理配置 =====
    auto_deduplicate: bool = True  # 自动去重
    keep_system_messages: bool = True  # 始终保留系统消息
    
    # ===== 内部缓存 =====
    _message_cache: Dict[str, BaseMessage] = field(default_factory=dict, init=False)
    _metadata_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """初始化配置"""
        self._ensure_checkpointer()
    
    def _ensure_checkpointer(self):
        """确保 Checkpointer 已初始化"""
        if self.checkpointer is None:
            self.checkpointer = MemorySaver()
    
    # ==================== 核心方法：消息操作 ====================
    
    def add_messages(
        self,
        messages: List[BaseMessage],
        deduplicate: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加消息到历史（支持去重）
        
        Args:
            messages: 要添加的消息列表
            deduplicate: 是否去重（None 使用默认配置）
            metadata: 消息元数据
        
        Example:
            >>> history.add_messages([
            ...     HumanMessage(content="Hello"),
            ...     AIMessage(content="Hi there!")
            ... ])
        """
        deduplicate = deduplicate if deduplicate is not None else self.auto_deduplicate
        
        # 去重处理
        if deduplicate:
            messages = self._deduplicate_messages(messages)
        
        # 添加元数据
        if metadata:
            for msg in messages:
                msg_id = self._get_message_id(msg)
                self._metadata_cache[msg_id] = metadata
        
        # 使用 LangGraph 原生机制保存
        # 这里我们不直接使用 checkpointer，而是返回更新指令
        # 让 LangGraph 的状态管理系统处理
        return messages
    
    def merge_histories(
        self,
        *histories: List[BaseMessage],
        strategy: Literal["chronological", "interleave", "priority"] = "chronological"
    ) -> List[BaseMessage]:
        """
        合并多个消息历史
        
        Args:
            histories: 多个消息历史列表
            strategy: 合并策略
                - chronological: 按时间顺序
                - interleave: 交替合并
                - priority: 按优先级（第一个列表优先）
        
        Example:
            >>> history1 = [HumanMessage(content="Q1"), AIMessage(content="A1")]
            >>> history2 = [HumanMessage(content="Q2"), AIMessage(content="A2")]
            >>> merged = manager.merge_histories(history1, history2)
        """
        if not histories:
            return []
        
        if strategy == "chronological":
            return self._merge_chronological(*histories)
        elif strategy == "interleave":
            return self._merge_interleave(*histories)
        elif strategy == "priority":
            return self._merge_priority(*histories)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
    
    def filter_messages(
        self,
        messages: List[BaseMessage],
        message_types: Optional[List[type]] = None,
        content_pattern: Optional[str] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        custom_filter: Optional[Callable[[BaseMessage], bool]] = None
    ) -> List[BaseMessage]:
        """
        过滤消息
        
        Args:
            messages: 要过滤的消息列表
            message_types: 保留的消息类型（如 [HumanMessage, AIMessage]）
            content_pattern: 内容匹配模式（正则表达式）
            time_range: 时间范围 (start, end)
            custom_filter: 自定义过滤函数
        
        Example:
            >>> # 只保留人类和 AI 消息
            >>> filtered = manager.filter_messages(
            ...     messages,
            ...     message_types=[HumanMessage, AIMessage]
            ... )
        """
        filtered = messages
        
        # 按类型过滤
        if message_types:
            filtered = [m for m in filtered if type(m) in message_types]
        
        # 按内容过滤
        if content_pattern:
            import re
            pattern = re.compile(content_pattern)
            filtered = [m for m in filtered if pattern.search(m.content)]
        
        # 按时间过滤
        if time_range:
            filtered = self._filter_by_time(filtered, time_range)
        
        # 自定义过滤
        if custom_filter:
            filtered = [m for m in filtered if custom_filter(m)]
        
        return filtered
    
    def clean_messages(
        self,
        messages: List[BaseMessage],
        remove_duplicates: bool = True,
        remove_empty: bool = True,
        compress_consecutive: bool = True,
        max_length: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        清理消息历史
        
        Args:
            messages: 要清理的消息列表
            remove_duplicates: 移除重复消息
            remove_empty: 移除空消息
            compress_consecutive: 压缩连续的同类型消息
            max_length: 最大保留数量
        
        Example:
            >>> cleaned = manager.clean_messages(
            ...     messages,
            ...     remove_duplicates=True,
            ...     compress_consecutive=True
            ... )
        """
        result = list(messages)
        
        # 移除空消息
        if remove_empty:
            result = [m for m in result if m.content and m.content.strip()]
        
        # 去重
        if remove_duplicates:
            result = self._deduplicate_messages(result)
        
        # 压缩连续消息
        if compress_consecutive:
            result = self._compress_consecutive_messages(result)
        
        # 长度限制
        if max_length and len(result) > max_length:
            # 保留系统消息
            if self.keep_system_messages:
                system_msgs = [m for m in result if isinstance(m, SystemMessage)]
                other_msgs = [m for m in result if not isinstance(m, SystemMessage)]
                result = system_msgs + other_msgs[-(max_length - len(system_msgs)):]
            else:
                result = result[-max_length:]
        
        return result
    
    def trim_messages_smart(
        self,
        messages: List[BaseMessage],
        max_tokens: Optional[int] = None,
        strategy: Literal["last", "first", "summary"] = "last"
    ) -> List[BaseMessage]:
        """
        智能消息修剪（基于 LangGraph 的 trim_messages）
        
        Args:
            messages: 要修剪的消息列表
            max_tokens: 最大 token 数
            strategy: 修剪策略
        
        Example:
            >>> trimmed = manager.trim_messages_smart(
            ...     messages,
            ...     max_tokens=1000,
            ...     strategy="last"
            ... )
        """
        max_tokens = max_tokens or self.max_tokens
        
        if not max_tokens:
            return messages
        
        if strategy == "summary":
            # 使用摘要策略
            return self._trim_with_summary(messages, max_tokens)
        else:
            # 使用 LangGraph 原生 trim_messages
            return trim_messages(
                messages,
                strategy=strategy,
                max_tokens=max_tokens,
                token_counter=len,  # 可替换为更精确的计数器
                start_on="human",
                end_on=("human", "tool")
            )
    
    # ==================== 辅助方法 ====================
    
    def _get_message_id(self, message: BaseMessage) -> str:
        """生成消息唯一 ID"""
        if hasattr(message, 'id') and message.id:
            return message.id
        
        # 基于内容生成 ID
        content = f"{message.type}:{message.content}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _deduplicate_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """去重消息"""
        seen = set()
        result = []
        
        for msg in messages:
            msg_id = self._get_message_id(msg)
            if msg_id not in seen:
                seen.add(msg_id)
                result.append(msg)
        
        return result
    
    def _compress_consecutive_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """压缩连续的同类型消息"""
        if not messages:
            return []
        
        result = []
        current_group = [messages[0]]
        
        for msg in messages[1:]:
            if type(msg) == type(current_group[0]):
                current_group.append(msg)
            else:
                # 合并当前组
                if len(current_group) > 1:
                    merged_content = "\n\n".join(m.content for m in current_group)
                    merged_msg = type(current_group[0])(content=merged_content)
                    result.append(merged_msg)
                else:
                    result.append(current_group[0])
                
                current_group = [msg]
        
        # 处理最后一组
        if len(current_group) > 1:
            merged_content = "\n\n".join(m.content for m in current_group)
            merged_msg = type(current_group[0])(content=merged_content)
            result.append(merged_msg)
        else:
            result.append(current_group[0])
        
        return result
    
    def _merge_chronological(self, *histories: List[BaseMessage]) -> List[BaseMessage]:
        """按时间顺序合并"""
        all_messages = []
        for history in histories:
            all_messages.extend(history)
        
        # 假设消息有时间戳，否则保持原顺序
        return sorted(
            all_messages,
            key=lambda m: getattr(m, 'timestamp', datetime.now())
        )
    
    def _merge_interleave(self, *histories: List[BaseMessage]) -> List[BaseMessage]:
        """交替合并"""
        result = []
        max_len = max(len(h) for h in histories)
        
        for i in range(max_len):
            for history in histories:
                if i < len(history):
                    result.append(history[i])
        
        return result
    
    def _merge_priority(self, *histories: List[BaseMessage]) -> List[BaseMessage]:
        """优先级合并（去重时保留第一个）"""
        result = []
        seen = set()
        
        for history in histories:
            for msg in history:
                msg_id = self._get_message_id(msg)
                if msg_id not in seen:
                    seen.add(msg_id)
                    result.append(msg)
        
        return result
    
    def _filter_by_time(
        self,
        messages: List[BaseMessage],
        time_range: tuple[datetime, datetime]
    ) -> List[BaseMessage]:
        """按时间过滤"""
        start, end = time_range
        return [
            m for m in messages
            if hasattr(m, 'timestamp') and start <= m.timestamp <= end
        ]
    
    def _trim_with_summary(
        self,
        messages: List[BaseMessage],
        max_tokens: int
    ) -> List[BaseMessage]:
        """使用摘要策略修剪"""
        # 这里可以集成 LangMem 的 SummarizationNode
        # 简化版本：只保留最近的消息 + 一个摘要
        
        from langchain_core.messages.utils import count_tokens_approximately
        
        current_tokens = count_tokens_approximately(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        # 保留系统消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # 简单策略：保留最后的消息 + 摘要前面的内容
        # 实际应该调用 LLM 生成摘要
        summary_content = f"[Earlier conversation summarized: {len(other_msgs) - 10} messages]"
        summary_msg = SystemMessage(content=summary_content)
        
        return system_msgs + [summary_msg] + other_msgs[-10:]
    

    
    def get_messages(
        self, 
        thread_id: Optional[str] = None,
        limit: Optional[int] = None,
        before: Optional[str] = None
    ) -> List[BaseMessage]:
        """
        获取消息历史
        
        Args:
            thread_id: 线程ID，如果为None则使用默认线程
            limit: 限制返回的消息数量
            before: 获取指定 checkpoint_id 之前的消息
            
        Returns:
            消息列表
            
        Example:
            >>> # 获取默认线程的所有消息
            >>> messages = manager.get_messages()
            >>> 
            >>> # 获取指定线程的最新10条消息
            >>> messages = manager.get_messages(thread_id="session_1", limit=10)
        """
        # 使用提供的 thread_id 或默认值
        tid = thread_id or self.thread_id
        
        # 构建配置
        config = {"configurable": {"thread_id": tid}}
        
        try:
            # 如果指定了 before，添加到配置中
            if before:
                config["configurable"]["checkpoint_id"] = before
            
            # 从 checkpointer 获取最新状态
            checkpoint = self.checkpointer.get(config)
            
            if checkpoint is None:
                return []
            
            # 从 checkpoint 中提取 messages
            messages = []
            if hasattr(checkpoint, 'values'):
                # checkpoint.values 是一个字典，包含完整状态
                state = checkpoint.values
                if isinstance(state, dict) and 'messages' in state:
                    messages = state['messages']
                elif isinstance(state, dict):
                    # 尝试从其他可能的键获取
                    for key in ['message', 'msg', 'history']:
                        if key in state:
                            messages = state[key]
                            break
            
            # 确保返回的是列表
            if not isinstance(messages, list):
                messages = [messages] if messages else []
            
            # 应用限制
            if limit and len(messages) > limit:
                messages = messages[-limit:]  # 取最新的 N 条
            
            return messages
            
        except Exception as e:
            # 如果获取失败，返回空列表
            # 在生产环境中应该记录日志
            import logging
            logging.warning(f"Failed to get messages for thread {tid}: {e}")
            return []

    def save_messages(
        self,
        messages: List[BaseMessage],
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        保存消息到 checkpointer
        
        Args:
            messages: 要保存的消息列表
            thread_id: 线程ID
            metadata: 额外的元数据
            
        Returns:
            是否保存成功
            
        Example:
            >>> success = manager.save_messages([
            ...     HumanMessage(content="Hello"),
            ...     AIMessage(content="Hi!")
            ... ], thread_id="session_1")
        """
        tid = thread_id or self.thread_id
        config = {"configurable": {"thread_id": tid}}
        
        try:
            # 构建要保存的状态
            state = {"messages": messages}
            if metadata:
                state["metadata"] = metadata
            
            # 使用 checkpointer 的 put 方法保存
            # 注意：不同的 checkpointer 实现可能有不同的接口
            # 这里提供一个通用的实现
            from langgraph.checkpoint.base import Checkpoint
            
            checkpoint = Checkpoint(
                v=1,
                ts=datetime.now().isoformat(),
                id=hashlib.md5(f"{tid}:{datetime.now()}".encode()).hexdigest(),
                channel_values=state,
                channel_versions={},
                versions_seen={}
            )
            
            self.checkpointer.put(config, checkpoint, metadata or {})
            return True
            
        except Exception as e:
            import logging
            logging.error(f"Failed to save messages for thread {tid}: {e}")
            return False
    # 新增方法===========================================
    def get_message_count(self, thread_id: Optional[str] = None) -> int:
        """
        获取消息数量
        
        Args:
            thread_id: 线程ID
            
        Returns:
            消息数量
            
        Example:
            >>> count = manager.get_message_count("session_1")
            >>> print(f"Total messages: {count}")
        """
        messages = self.get_messages(thread_id)
        return len(messages)

    def delete_messages(
        self,
        thread_id: Optional[str] = None,
        before: Optional[datetime] = None
    ) -> bool:
        """
        删除消息历史
        
        Args:
            thread_id: 线程ID，如果为None则删除默认线程
            before: 删除此时间之前的消息（如果为None则删除全部）
            
        Returns:
            是否删除成功
            
        Example:
            >>> # 删除整个线程的历史
            >>> manager.delete_messages("session_1")
            >>> 
            >>> # 删除7天前的消息
            >>> from datetime import datetime, timedelta
            >>> week_ago = datetime.now() - timedelta(days=7)
            >>> manager.delete_messages("session_1", before=week_ago)
        """
        tid = thread_id or self.thread_id
        
        try:
            if before:
                # 获取现有消息
                messages = self.get_messages(tid)
                
                # 过滤保留的消息
                kept_messages = [
                    m for m in messages
                    if not hasattr(m, 'timestamp') or m.timestamp >= before
                ]
                
                # 保存过滤后的消息
                return self.save_messages(kept_messages, tid)
            else:
                # 删除整个线程
                config = {"configurable": {"thread_id": tid}}
                
                # 保存空消息列表
                return self.save_messages([], tid)
                
        except Exception as e:
            import logging
            logging.error(f"Failed to delete messages for thread {tid}: {e}")
            return False

    def get_all_threads(self) -> List[str]:
        """
        获取所有线程ID
        
        Returns:
            线程ID列表
            
        Example:
            >>> threads = manager.get_all_threads()
            >>> for thread in threads:
            ...     print(f"Thread: {thread}")
        """
        try:
            # 这个方法依赖于 checkpointer 的实现
            # MemorySaver 可能需要遍历内部存储
            # PostgresSaver 可以查询数据库
            
            # 对于 MemorySaver
            if hasattr(self.checkpointer, 'storage'):
                storage = self.checkpointer.storage
                threads = set()
                for key in storage.keys():
                    # key 格式通常是 (thread_id, checkpoint_ns, checkpoint_id)
                    if isinstance(key, tuple) and len(key) >= 1:
                        threads.add(key[0])
                return list(threads)
            
            # 对于其他类型的 checkpointer，可能需要不同的实现
            return []
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to get all threads: {e}")
            return []

    def get_latest_checkpoint_id(self, thread_id: Optional[str] = None) -> Optional[str]:
        """
        获取最新的 checkpoint ID
        
        Args:
            thread_id: 线程ID
            
        Returns:
            最新的 checkpoint ID，如果不存在则返回 None
            
        Example:
            >>> checkpoint_id = manager.get_latest_checkpoint_id("session_1")
            >>> if checkpoint_id:
            ...     print(f"Latest checkpoint: {checkpoint_id}")
        """
        tid = thread_id or self.thread_id
        config = {"configurable": {"thread_id": tid}}
        
        try:
            checkpoint = self.checkpointer.get(config)
            if checkpoint and hasattr(checkpoint, 'id'):
                return checkpoint.id
            return None
        except:
            return None

    def get_message_history(
        self,
        thread_id: Optional[str] = None,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        获取详细的消息历史（包含元数据）
        
        Args:
            thread_id: 线程ID
            limit: 限制返回数量
            include_metadata: 是否包含元数据
            
        Returns:
            消息历史列表，每个元素包含消息和可选的元数据
            
        Example:
            >>> history = manager.get_message_history(
            ...     thread_id="session_1",
            ...     limit=10,
            ...     include_metadata=True
            ... )
            >>> for item in history:
            ...     print(f"Message: {item['message'].content}")
            ...     if 'metadata' in item:
            ...         print(f"Metadata: {item['metadata']}")
        """
        messages = self.get_messages(thread_id, limit)
        
        result = []
        for msg in messages:
            item = {"message": msg}
            
            if include_metadata:
                msg_id = self._get_message_id(msg)
                if msg_id in self._metadata_cache:
                    item["metadata"] = self._metadata_cache[msg_id]
            
            result.append(item)
        
        return result

    def clear_cache(self):
        """
        清除内部缓存
        
        Example:
            >>> manager.clear_cache()
        """
        self._message_cache.clear()
        self._metadata_cache.clear()

    def export_history(
        self,
        thread_id: Optional[str] = None,
        format: Literal["json", "dict", "markdown"] = "dict"
    ) -> Any:
        """
        导出消息历史
        
        Args:
            thread_id: 线程ID
            format: 导出格式
                - json: JSON 字符串
                - dict: Python 字典
                - markdown: Markdown 格式文本
            
        Returns:
            导出的数据
            
        Example:
            >>> # 导出为字典
            >>> data = manager.export_history("session_1", format="dict")
            >>> 
            >>> # 导出为 JSON
            >>> json_str = manager.export_history("session_1", format="json")
            >>> 
            >>> # 导出为 Markdown
            >>> md = manager.export_history("session_1", format="markdown")
        """
        messages = self.get_messages(thread_id)
        
        if format == "dict":
            return {
                "thread_id": thread_id or self.thread_id,
                "message_count": len(messages),
                "messages": [
                    {
                        "type": msg.type,
                        "content": msg.content,
                        "id": self._get_message_id(msg)
                    }
                    for msg in messages
                ]
            }
        
        elif format == "json":
            import json
            data = self.export_history(thread_id, format="dict")
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            lines = [f"# Chat History - {thread_id or self.thread_id}\n"]
            for i, msg in enumerate(messages, 1):
                role = msg.type.upper()
                lines.append(f"## Message {i} - {role}")
                lines.append(f"{msg.content}\n")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        

        
    #     # ==================== 场景：清理和优化历史 ====================
    # # 1. 获取消息
    # messages = history_manager.get_messages("session_1")

    # # 2. 清理消息（去重、移除空消息）
    # cleaned = history_manager.clean_messages(
    #     messages,
    #     remove_duplicates=True,
    #     remove_empty=True,
    #     compress_consecutive=True
    # )

    # # 3. 过滤只保留对话消息
    # dialogue_only = history_manager.filter_messages(
    #     cleaned,
    #     message_types=[HumanMessage, AIMessage]
    # )

    # # 4. 智能修剪
    # trimmed = history_manager.trim_messages_smart(
    #     dialogue_only,
    #     max_tokens=2000,
    #     strategy="last"
    # )

    # # 5. 保存优化后的历史
    # history_manager.save_messages(trimmed, "session_1")

    # print(f"优化完成: {len(messages)} → {len(trimmed)} 条消息")