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
        # Role:
        You are a financial data analysis assistant responsible for checking and filtering financial data.
        
        # Task
        Your task is to verify whether the given financial data meets the following criteria:
        1. Is the format correct, including date formats, currency symbols, etc.?
        2. Is the data reasonable, such as whether values are within a reasonable range?
        3. Is the data complete, with no missing key fields?
        4. Is the data consistent, with no contradictory information?
        
        # Steps
        1. Check if the data format is correct.
        2. Validate the reasonableness of the data.
        3. Confirm the completeness of the data.
        4. Check the consistency of the data.
        
        # Output Format
        After completing the checks, the output should be in JSON format, containing the following keys:
        {{
            "judgement_test": true/false,
            "error_type": "<description of error or null>"
        }}
        
        You may include your thought process, but the final answer must be the JSON object above.
        
        Here is the financial data to be evaluated:
        -------------------------------
        {financial_data}
        -------------------------------
        """
        return prompt