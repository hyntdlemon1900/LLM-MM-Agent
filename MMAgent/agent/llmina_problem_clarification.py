from .base_agent import BaseAgent
from prompt.llmina_template import PROBLEM_CLARIFACATION_PROMPT

class ProblemClarification(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
    
    def clarification_actor(self, llm_judge, llm_summary, problem_str, user_input_func, max_rounds: int = 5):
        """
        交互式问题澄清流程
        llm_understand: 理解智能体，输入为问题描述，输出为澄清建议或确认
        llm_judge: 判断智能体，输入为当前问题描述和澄清历史，输出为True/False
        problem_str: 初始问题描述
        user_input_func: 用户输入函数，参数为智能体的反馈，返回用户补充/澄清
        llm_summary: 总结智能体（可选），输入交互历史或本轮信息，输出总结文本
        max_rounds: 最大交互轮次上限，避免死循环
        """
        current_problem = problem_str  # 保存未装饰的原始问题+用户补充
        history = []
        final_summary = ''

        def _summarize_round(problem_desc: str, agent_fb: str, user_rep: str) -> str:
            """对本轮新增的用户补充进行提炼，返回可直接附加的问题补充文本。"""
            return (llm_summary.summary_actor({
                'problem': problem_desc,
                'agent_feedback': agent_fb,
                'user_reply': user_rep
            }) or '').strip()
                    
        round_idx = 1
        while round_idx <= max_rounds:
            # 1) 用PROBLEM_CLARIFACATION_PROMPT装饰当前问题，向LLM询问
            decorated_problem = PROBLEM_CLARIFACATION_PROMPT.format(problem_str=current_problem).strip()
            agent_feedback = self.llm.generate(decorated_problem)

            # 2) 记录历史
            record = {
                'round': round_idx,
                'problem': current_problem,
                'agent_feedback': agent_feedback
            }

            # 3) 获取用户回复
            user_reply = user_input_func(agent_feedback)
            record['user_reply'] = user_reply

            # 4) 本轮摘要提炼，仅提取用户新增有效信息
            round_summary = _summarize_round(current_problem, agent_feedback, user_reply)
            record['round_summary'] = round_summary
            history.append(record)

            # 5) 将新增有效信息附加到问题描述末尾，形成新的问题描述
            if round_summary:
                current_problem = (current_problem + "\n" + round_summary).strip()

            # 6) 由判别智能体判断是否已清晰
            judge_result = llm_judge.judge_actor(agent_feedback)
            if isinstance(judge_result, str):
                judge_result = judge_result.strip().lower() in ['True', 'yes', '1']
            if judge_result:
                break

            # 若用户未提供任何信息，且仍未通过判别，则提前结束避免空转
            if not (user_reply and user_reply.strip()):
                break

            round_idx += 1

        # 7) 生成整个交互过程的最终总结
        try:    
            final_summary = (llm_summary.summary_actor(history) or '').strip()
        except Exception:
            # 兜底确保返回值存在
            final_summary = ''

        return current_problem, history, final_summary
