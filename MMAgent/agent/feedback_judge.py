
from prompt.llmina_template import AGENT_FEEDBACK_JUDGE_PROMPT

class ProblemJudge:
    """
    判别型智能体：判断问题描述是否清晰，无歧义。
    generate方法返回True/False，True表示问题描述清晰。
    """
    def __init__(self, llm):
        self.llm = llm

    def judge_actor(self, agent_feedback: str):
        """
        根据提示词判断问题描述是否清晰。
        返回True/False。
        """
        # 构造判别提示词
        prompt = AGENT_FEEDBACK_JUDGE_PROMPT.format(agent_feedback = agent_feedback).strip()
        result = self.llm.generate(prompt)
        if isinstance(result, str):
            result = result.strip().lower()
            if result in ["True", "true", "1"]:
                return True
            elif result in ["False", "false", "0"]:
                return False
        return False