from dataflow.core.prompt import DIYPromptABC

__all__ = ['FinancialDataFilterPrompt']

class FinancialDataFilterPrompt(DIYPromptABC):
    """
    The prompt for filtering financial data.
    """
    def __init__(self):
        pass
    
    def build_prompt(self, financial_data: str) -> str:
        """
        Constructs a prompt to evaluate the correctness and validity of financial data.
        """
        prompt = f"""
        # 角色：
        你是一个金融数据分析助手，负责检查和过滤金融数据。
        
        # 任务
        你的任务是检查给定的金融数据是否符合以下标准：
        1. 格式是否规范，包括日期格式、货币符号等。
        2. 数据是否合理，例如数值是否在合理范围内。
        3. 数据是否完整，是否缺少关键字段。
        4. 数据是否一致，是否存在矛盾信息。
        
        # 工作步骤
        1. 检查数据格式是否正确。
        2. 验证数据的合理性。
        3. 确认数据的完整性。
        4. 检查数据的一致性。
        
        # 输出格式
        在完成检查后，输出结果应为JSON格式，包含以下键：
        {{
            "judgement_test": true/false,
            "error_type": "<错误描述或null>"
        }}
        
        你可以包含你的思考过程，但最终答案必须是上述JSON对象。
        
        以下是需要评估的金融数据：
        -------------------------------
        {financial_data}
        -------------------------------
        """
        return prompt