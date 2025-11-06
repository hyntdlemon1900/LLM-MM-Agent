"""
LLMINA问题澄清总结智能体
用于在交互式问题澄清过程中总结用户提供的新增有效信息
"""
from prompt.llmina_template import CLARIFICATION_ROUND_SUMMARY_PROMPT, CLARIFICATION_FINAL_SUMMARY_PROMPT


class ProblemClarificationSummary:
    """
    总结型智能体：在问题澄清过程中提炼用户新增的关键信息
    提供两种总结能力：
    1. 单轮总结：提炼本轮用户补充中的有效信息
    2. 全局总结：汇总整个澄清历史中所有新增的关键信息
    """
    def __init__(self, llm):
        self.llm = llm

    def summary_actor(self, interaction_data):
        """
        根据输入类型自动选择单轮总结或全局总结
        
        Args:
            interaction_data: 可以是:
                - dict: 单轮交互数据 {'problem': str, 'agent_feedback': str, 'user_reply': str}
                - list: 多轮交互历史 [{'round': int, 'problem': str, 'agent_feedback': str, 'user_reply': str, ...}, ...]
        
        Returns:
            str: 总结后的文本
        """
        if isinstance(interaction_data, dict):
            # 单轮总结
            return self.summarize_round(
                problem_desc=interaction_data.get('problem', ''),
                agent_feedback=interaction_data.get('agent_feedback', ''),
                user_reply=interaction_data.get('user_reply', '')
            )
        elif isinstance(interaction_data, list):
            # 全局总结
            return self.summarize_history(interaction_data)
        else:
            return ''

    def summarize_round(self, problem_desc: str, agent_feedback: str, user_reply: str):
        """
        单轮总结：提炼本轮用户回复中的新增有效信息
        
        Args:
            problem_desc: 当前问题描述
            agent_feedback: 智能体的反馈/提问
            user_reply: 用户的补充/回复
        
        Returns:
            str: 提炼后的关键信息，可直接附加到问题描述末尾
        """
        # 若用户未提供任何信息，直接返回空
        if not user_reply or not user_reply.strip():
            return ''
        
        # 构造单轮总结提示词
        prompt = CLARIFICATION_ROUND_SUMMARY_PROMPT.format(
            problem_description=problem_desc,
            agent_feedback=agent_feedback,
            user_reply=user_reply
        ).strip()
        
        result = self.llm.generate(prompt)
        
        return result

    def summarize_history(self, history: list):
        """
        全局总结：汇总整个澄清历史中所有新增的关键信息
        
        Args:
            history: 交互历史列表，每个元素为一轮交互记录
                [{'round': int, 'problem': str, 'agent_feedback': str, 'user_reply': str, 'round_summary': str}, ...]
        
        Returns:
            str: 汇总后的自洽补充说明
        """
        if not history:
            return ''
        
        # 提取所有轮次的摘要和用户回复
        rounds_info = []
        for record in history:
            round_num = record.get('round', 0)
            user_reply = record.get('user_reply', '').strip()
            round_summary = record.get('round_summary', '').strip()
            
            if user_reply or round_summary:
                rounds_info.append({
                    'round': round_num,
                    'user_reply': user_reply,
                    'round_summary': round_summary
                })
        
        if not rounds_info:
            return ''
        
        # 构造全局总结提示词
        prompt = CLARIFICATION_FINAL_SUMMARY_PROMPT.format(
            history_content=self._format_history(rounds_info)
        ).strip()
        
        result = self.llm.generate(prompt)
        
        return result

    def _format_history(self, rounds_info: list):
        """
        格式化历史记录为易读文本
        
        Args:
            rounds_info: 轮次信息列表
        
        Returns:
            str: 格式化后的文本
        """
        formatted = []
        for info in rounds_info:
            round_num = info['round']
            user_reply = info['user_reply']
            round_summary = info['round_summary']
            
            formatted.append(f"第{round_num}轮:")
            if user_reply:
                formatted.append(f"  用户回复: {user_reply}")
            if round_summary:
                formatted.append(f"  本轮摘要: {round_summary}")
            formatted.append("")
        
        return "\n".join(formatted)
